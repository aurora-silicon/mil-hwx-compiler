#import "HWXObjectWriter.h"

#import "HWXMachOFormat.h"

static NSString *const HWXObjectWriterErrorDomain = @"ANE.HWX.ObjectWriter";

static NSUInteger alignUp(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1) / alignment * alignment;
}
static BOOL multiplyWithoutOverflow(NSUInteger left, NSUInteger right,
                                    NSUInteger *result) {
    return !__builtin_mul_overflow(left, right, result);
}
static BOOL addWithoutOverflow(uint64_t left, uint64_t right,
                               uint64_t *result) {
    return !__builtin_add_overflow(left, right, result);
}
static BOOL alignUpWithoutOverflow(uint64_t value, uint64_t alignment,
                                   uint64_t *result) {
    uint64_t adjusted = 0;
    if (alignment == 0 ||
        !addWithoutOverflow(value, alignment - 1, &adjusted)) return NO;
    *result = adjusted / alignment * alignment;
    return YES;
}
static void setName(char destination[16], const char *name) {
    memset(destination, 0, 16);
    memcpy(destination, name, MIN(strlen(name), (size_t)15));
}
static void writeU32(NSMutableData *data, NSUInteger offset, uint32_t value) {
    [data replaceBytesInRange:NSMakeRange(offset,4) withBytes:&value];
}
static void writeU64(NSMutableData *data, NSUInteger offset, uint64_t value) {
    [data replaceBytesInRange:NSMakeRange(offset,8) withBytes:&value];
}
static void writeString(NSMutableData *data, NSUInteger offset, NSString *value) {
    NSData *bytes = [value dataUsingEncoding:NSUTF8StringEncoding];
    [data replaceBytesInRange:NSMakeRange(offset, bytes.length) withBytes:bytes.bytes];
}
static struct section_64 sectionRecord(const char *sectionName,
                                       const char *segmentName,
                                       uint64_t address, uint64_t size,
                                       uint32_t fileOffset, uint32_t alignment,
                                       uint32_t flags) {
    struct section_64 section = {};
    setName(section.sectname, sectionName); setName(section.segname, segmentName);
    section.addr=address; section.size=size; section.offset=fileOffset;
    section.align=alignment; section.flags=flags; return section;
}
static NSData *segmentRecord(const char *name, uint64_t vmAddress,
                             uint64_t vmSize, uint64_t fileOffset,
                             uint64_t fileSize, int32_t maxProtection,
                             int32_t initialProtection, uint32_t flags,
                             const struct section_64 *sections,
                             uint32_t sectionCount) {
    NSMutableData *data = [NSMutableData dataWithLength:
        sizeof(struct segment_command_64)+sectionCount*sizeof(struct section_64)];
    struct segment_command_64 segment = {};
    segment.cmd=LC_SEGMENT_64; segment.cmdsize=(uint32_t)data.length;
    setName(segment.segname,name); segment.vmaddr=vmAddress; segment.vmsize=vmSize;
    segment.fileoff=fileOffset; segment.filesize=fileSize;
    segment.maxprot=maxProtection; segment.initprot=initialProtection;
    segment.nsects=sectionCount; segment.flags=flags;
    [data replaceBytesInRange:NSMakeRange(0,sizeof(segment)) withBytes:&segment];
    if (sectionCount) [data replaceBytesInRange:NSMakeRange(sizeof(segment),
        sectionCount*sizeof(struct section_64)) withBytes:sections];
    return data;
}
static NSUInteger elementBytes(ANEElementType type) {
    switch(type) {
        case ANEElementTypeFP16:return 2; case ANEElementTypeFP32:return 4;
        case ANEElementTypeInt8:return 1; case ANEElementTypeInt32:return 4;
        case ANEElementTypeUInt64:return 8; case ANEElementTypeBool:return 1;
        case ANEElementTypeString: case ANEElementTypeInvalid:return 0;
    }
}
static uint32_t elementCode(ANEElementType type) {
    if(type==ANEElementTypeFP16)return 5; if(type==ANEElementTypeInt8)return 2;
    if(type==ANEElementTypeFP32)return 6; return 0;
}
static BOOL descriptorNameValid(NSString *value) {
    NSData *bytes = [value dataUsingEncoding:NSASCIIStringEncoding];
    if (!bytes.length) return NO;
    const uint8_t *p = (const uint8_t *)bytes.bytes;
    for (NSUInteger i = 0; i < bytes.length; ++i)
        if (p[i] < 0x21 || p[i] > 0x7e) return NO;
    return YES;
}
static void shapeDimensions(NSArray<NSNumber *> *s, NSUInteger *n,
                            NSUInteger *c, NSUInteger *h, NSUInteger *w) {
    *n = 1; *c = 1; *h = 1; *w = 1;
    if (s.count == 4) {
        *n=s[0].unsignedIntegerValue; *c=s[1].unsignedIntegerValue;
        *h=s[2].unsignedIntegerValue; *w=s[3].unsignedIntegerValue;
    } else if (s.count == 3) {
        *n=s[0].unsignedIntegerValue; *h=s[1].unsignedIntegerValue;
        *w=s[2].unsignedIntegerValue;
    } else if (s.count == 2) {
        *h=s[0].unsignedIntegerValue; *w=s[1].unsignedIntegerValue;
    } else if (s.count == 1) {
        *w=s[0].unsignedIntegerValue;
    }
}
static BOOL logicalBytes(HWXObjectBinding *binding, NSUInteger *result) {
    NSUInteger count = 1;
    for (NSNumber *dimension in binding.shape)
        if (!multiplyWithoutOverflow(count, dimension.unsignedIntegerValue,
                                     &count)) return NO;
    return multiplyWithoutOverflow(count, elementBytes(binding.elementType),
                                   result);
}
static NSData *bufferReference(HWXObjectBinding *binding, uint64_t vmAddress) {
    NSMutableData *data=[NSMutableData dataWithLength:0x20];
    writeU32(data,0,0x40); writeU32(data,4,0x20); writeU64(data,8,0x18);
    writeU64(data,0x10,vmAddress);
    NSData *name=[binding.shortName dataUsingEncoding:NSUTF8StringEncoding];
    NSUInteger length=MIN((NSUInteger)8,name.length);
    [data replaceBytesInRange:NSMakeRange(0x18,length) withBytes:name.bytes];
    return data;
}
static NSData *programDescriptor(uint64_t textAddress,uint64_t textConstAddress,
                                 uint64_t scratchAddress,
                                 NSArray<NSNumber *> *resourceAddresses,
                                 uint32_t taskWordCount,
                                 HWXObjectProgramInfo *info,
                                 HWXObjectArchitecture architecture) {
    if (architecture == HWXObjectArchitectureH13) {
        NSMutableData *d = [NSMutableData dataWithLength:0x880];
        writeU32(d,0,4); writeU32(d,4,0x880); writeU32(d,8,1);
        writeU32(d,0xc,0x21a); writeU64(d,0x10,textAddress);
        writeU64(d,0x18,textConstAddress);
        for (NSUInteger index = 0; index < MIN((NSUInteger)5,
                                               resourceAddresses.count); ++index)
            writeU64(d,0x30 + index * 8,
                     resourceAddresses[index].unsignedLongLongValue);
        writeU64(d,0x810,textAddress);
        writeU32(d,0x818,taskWordCount - 1);
        writeU32(d,0x81c,(uint32_t)info.taskCount);
        writeU32(d,0x824,0xffff); writeU32(d,0x83c,0x878);
        writeU32(d,0x850,0x11); writeU32(d,0x858,4);
        writeU32(d,0x860,1); writeU32(d,0x868,9); writeU32(d,0x86c,8);
        writeString(d,0x878,@"net");
        return d;
    }
    if (architecture == HWXObjectArchitectureH14) {
        // Decoded from research/oracles/h14: kind 4 in a 0x890-byte command,
        // text_const at 0x20, resource slots 0..3 zero with the scratch base
        // in slot 4, and a 32-word trailer whose function name is "main".
        // The word at 0xc is not decoded by the campaign; it follows the
        // other kind-4 descriptor.
        NSMutableData *d = [NSMutableData dataWithLength:0x890];
        writeU32(d,0,4); writeU32(d,4,0x890); writeU32(d,8,4);
        writeU32(d,0xc,0x222); writeU64(d,0x10,textAddress);
        writeU64(d,0x20,textConstAddress);
        writeU64(d,0x50,scratchAddress);
        writeU64(d,0x810,textAddress);
        writeU32(d,0x820,4); writeU32(d,0x824,taskWordCount);
        writeU32(d,0x830,(uint32_t)info.taskCount);
        writeU32(d,0x838,UINT32_MAX); writeU32(d,0x83c,UINT32_MAX);
        writeU32(d,0x844,0xffff); writeU32(d,0x848,0x888);
        writeU32(d,0x858,info.h14ScratchDescriptorWord);
        writeU32(d,0x860,(uint32_t)info.recordCount); writeU32(d,0x868,3);
        writeU32(d,0x870,1); writeU32(d,0x878,9); writeU32(d,0x87c,8);
        writeU32(d,0x880,info.unresolvedDescriptorWord);
        writeString(d,0x888,@"main");
        return d;
    }
    BOOL mixed = info.descriptorLayout ==
        HWXProgramDescriptorLayoutScratchBackedMixed;
    NSUInteger length = mixed ? 0x8c0 : 0x8a0;
    NSMutableData *d=[NSMutableData dataWithLength:length];
    writeU32(d,0,0x04);writeU32(d,4,(uint32_t)length);writeU32(d,8,4);
    writeU32(d,0xc,mixed ? 0x22a : 0x222);
    writeU64(d,0x10,textAddress);writeU64(d,0x20,textConstAddress);
    if (mixed) writeU64(d,0x40,scratchAddress);
    for (NSUInteger index = 0; index < MIN((NSUInteger)5,
                                           resourceAddresses.count); ++index)
        writeU64(d,0x70 + index * 0x10,
                 resourceAddresses[index].unsignedLongLongValue);
    writeU64(d,0x810,textAddress);
    writeU32(d,0x820,4);writeU32(d,0x824,taskWordCount);
    writeU32(d,0x830,(uint32_t)info.taskCount);
    writeU32(d,0x838,UINT32_MAX);writeU32(d,0x83c,UINT32_MAX);
    writeU32(d,0x844,0xffff);
    writeU32(d,0x860,(uint32_t)info.recordCount);
    if (mixed) {
        writeU32(d,0x848,0x8b8);writeU32(d,0x858,(uint32_t)info.scratchByteLength);
        writeU32(d,0x868,4);writeU32(d,0x870,4);writeU32(d,0x878,2);
        writeU32(d,0x87c,8);writeU32(d,0x880,(uint32_t)info.scratchByteLength);
        writeU32(d,0x888,3);writeU32(d,0x88c,8);writeU32(d,0x890,0x08020004);
        writeU32(d,0x894,8);writeU32(d,0x898,7);writeU32(d,0x89c,8);
        writeU32(d,0x8a0,0x461c4000);writeU32(d,0x8a8,9);writeU32(d,0x8ac,8);
        writeU32(d,0x8b0,info.formatCode);writeString(d,0x8b8,@"main");
    } else {
        writeU32(d,0x848,0x898);writeU32(d,0x868,4);writeU32(d,0x870,2);
        writeU32(d,0x878,7);writeU32(d,0x87c,8);writeU32(d,0x880,0x461c4000);
        writeU32(d,0x888,9);writeU32(d,0x88c,8);writeU32(d,0x890,info.formatCode);
        writeString(d,0x898,@"main");
    }
    return d;
}
static NSData *tensorDescriptor(HWXObjectBinding *binding, uint32_t bindingIndex) {
    NSUInteger n=1,c=1,h=1,w=1;
    shapeDimensions(binding.shape, &n, &c, &h, &w);
    NSUInteger eb=elementBytes(binding.elementType);
    NSUInteger row=binding.rowStrideBytes, plane=binding.planeStrideBytes;
    NSUInteger batch=binding.batchStrideBytes, total=binding.storageByteLength;
    NSData *symbol=[binding.symbol dataUsingEncoding:NSUTF8StringEncoding];
    NSData *shortName=[binding.shortName dataUsingEncoding:NSUTF8StringEncoding];
    uint32_t symbolOffset=0xd3d;
    uint32_t shortOffset=(uint32_t)(symbolOffset+symbol.length+1);
    NSUInteger recordBytes=MAX((NSUInteger)0xd48,
        alignUp(shortOffset+shortName.length+1,8));
    NSMutableData *d=[NSMutableData dataWithLength:recordBytes];
    writeU32(d,0,4);writeU32(d,4,(uint32_t)recordBytes);writeU32(d,8,3);writeU32(d,0xc,0x34a);
    writeU32(d,0x10,3);writeU32(d,0x14,bindingIndex);writeU32(d,0x18,0xd38);
    writeU32(d,0x20,symbolOffset);writeU32(d,0x24,elementCode(binding.elementType));
    writeU32(d,0x28,(uint32_t)n);writeU32(d,0x2c,(uint32_t)c);
    writeU32(d,0x30,(uint32_t)h);writeU32(d,0x34,(uint32_t)w);writeU32(d,0x38,1);
    writeU64(d,0x50,batch);writeU64(d,0x58,plane);writeU64(d,0x60,row);
    writeU64(d,0x68,eb);writeU64(d,0x70,total);writeU32(d,0x78,1);
    writeU32(d,0x7c,shortOffset);writeU64(d,0x80,total);
    writeString(d,0xd38,@"main");writeString(d,symbolOffset,binding.symbol);
    writeString(d,shortOffset,binding.shortName);return d;
}
static NSString *architectureName(HWXObjectArchitecture architecture) {
    switch (architecture) {
        case HWXObjectArchitectureH13: return @"h13";
        case HWXObjectArchitectureH14: return @"h14";
        case HWXObjectArchitectureH16G: return @"h16g";
        case HWXObjectArchitectureH17: return @"h17";
        case HWXObjectArchitectureH18: return @"h18";
    }
}
static NSData *compilerMetadata(HWXObjectArchitecture architecture) {
    NSString *text = [NSString stringWithFormat:
        @"ANEC v1\nresearch_hwx_compiler v1\n\n Module ANEC:\n\t Target: %@\n",
        architectureName(architecture)];
    NSData *bytes = [text dataUsingEncoding:NSUTF8StringEncoding];
    NSUInteger length = architecture == HWXObjectArchitectureH13
        ? 0x728 : alignUp(8 + bytes.length + 1, 8);
    NSMutableData *record = [NSMutableData dataWithLength:length];
    writeU32(record, 0, 0x08);
    writeU32(record, 4, (uint32_t)record.length);
    [record replaceBytesInRange:NSMakeRange(8, bytes.length)
                      withBytes:bytes.bytes];
    return record;
}
static uint32_t appendSymbolName(NSMutableData *strings, NSString *name) {
    uint32_t offset = (uint32_t)strings.length;
    NSData *bytes = [name dataUsingEncoding:NSUTF8StringEncoding];
    [strings appendData:bytes];
    const uint8_t zero = 0;
    [strings appendBytes:&zero length:1];
    return offset;
}
static void appendSymbol(NSMutableData *symbols, NSMutableData *strings,
                         NSString *name, uint8_t section, uint16_t descriptor,
                         uint64_t value) {
    struct nlist_64 symbol = {};
    symbol.n_un.n_strx = appendSymbolName(strings, name);
    symbol.n_type = N_SECT | N_EXT;
    symbol.n_sect = section;
    symbol.n_desc = descriptor;
    symbol.n_value = value;
    [symbols appendBytes:&symbol length:sizeof(symbol)];
}
static void buildSymbolTables(NSArray<HWXObjectBinding *> *bindings,
                              NSArray<NSNumber *> *bindingAddresses,
                              uint64_t kernelAddress,
                              NSArray<NSNumber *> *kernelSymbolOffsets,
                              uint8_t firstBindingSection,
                              uint8_t kernelSection,
                              NSMutableData **symbolsOut,
                              NSMutableData **stringsOut) {
    NSMutableData *symbols = [NSMutableData data];
    NSMutableData *strings = [NSMutableData dataWithBytes:"\0" length:1];
    for (NSUInteger i = 0; i < bindings.count; ++i)
        appendSymbol(symbols, strings, bindings[i].shortName,
                     (uint8_t)(firstBindingSection + i), 0,
                     bindingAddresses[i].unsignedLongLongValue);
    for (NSUInteger i = 0; i < kernelSymbolOffsets.count; ++i) {
        NSUInteger kernelOffset = kernelSymbolOffsets[i].unsignedIntegerValue;
        appendSymbol(symbols, strings,
            [NSString stringWithFormat:@"__kern_0_tile_%lu", (unsigned long)i],
            kernelSection, i == 0 ? 2 : 0, kernelAddress + kernelOffset);
    }
    *symbolsOut = symbols;
    *stringsOut = strings;
}
static NSData *symbolTableCommand(uint32_t symbolOffset,
                                  uint32_t symbolCount,
                                  uint32_t stringOffset,
                                  uint32_t stringBytes) {
    struct symtab_command command = {};
    command.cmd = LC_SYMTAB;
    command.cmdsize = sizeof(command);
    command.symoff = symbolOffset;
    command.nsyms = symbolCount;
    command.stroff = stringOffset;
    command.strsize = stringBytes;
    return [NSData dataWithBytes:&command length:sizeof(command)];
}

@implementation HWXObjectBinding
- (instancetype)initWithSymbol:(NSString *)symbol shortName:(NSString *)shortName
                           role:(HWXObjectBindingRole)role
                    elementType:(ANEElementType)elementType
                          shape:(NSArray<NSNumber *> *)shape {
    NSUInteger n=1,c=1,h=1,w=1;
    shapeDimensions(shape, &n, &c, &h, &w);
    NSUInteger elementByteCount = elementBytes(elementType);
    NSUInteger row = w * elementByteCount;
    NSUInteger plane = h * row;
    NSUInteger batch = c * plane;
    return [self initWithSymbol:symbol shortName:shortName role:role
        elementType:elementType shape:shape rowStrideBytes:row
        planeStrideBytes:plane batchStrideBytes:batch
        storageByteLength:n * batch];
}

- (instancetype)initWithSymbol:(NSString *)symbol shortName:(NSString *)shortName
                           role:(HWXObjectBindingRole)role
                    elementType:(ANEElementType)elementType
                          shape:(NSArray<NSNumber *> *)shape
                 rowStrideBytes:(NSUInteger)rowStrideBytes
               planeStrideBytes:(NSUInteger)planeStrideBytes
               batchStrideBytes:(NSUInteger)batchStrideBytes
              storageByteLength:(NSUInteger)storageByteLength {
    self=[super init];
    if(self){
        _symbol=[symbol copy]; _shortName=[shortName copy]; _role=role;
        _elementType=elementType; _shape=[shape copy];
        _rowStrideBytes=rowStrideBytes; _planeStrideBytes=planeStrideBytes;
        _batchStrideBytes=batchStrideBytes; _storageByteLength=storageByteLength;
        _allocationByteLength=storageByteLength;
    }
    return self;
}
@end

@implementation HWXObjectProgramInfo
- (instancetype)initWithTaskCount:(NSUInteger)taskCount
                       recordCount:(NSUInteger)recordCount
                        formatCode:(uint32_t)formatCode
                 scratchByteLength:(NSUInteger)scratchByteLength
                  descriptorLayout:(HWXProgramDescriptorLayout)descriptorLayout {
    return [self initWithTaskCount:taskCount firstTaskByteLength:0
        recordCount:recordCount formatCode:formatCode
        scratchByteLength:scratchByteLength
        descriptorLayout:descriptorLayout];
}

- (instancetype)initWithTaskCount:(NSUInteger)taskCount
               firstTaskByteLength:(NSUInteger)firstTaskByteLength
                       recordCount:(NSUInteger)recordCount
                        formatCode:(uint32_t)formatCode
                 scratchByteLength:(NSUInteger)scratchByteLength
                  descriptorLayout:(HWXProgramDescriptorLayout)descriptorLayout {
    self = [super init];
    if (self) {
        _taskCount = taskCount; _firstTaskByteLength = firstTaskByteLength;
        _recordCount = recordCount; _formatCode = formatCode;
        _scratchByteLength = scratchByteLength;
        _scratchAllocationByteLength = scratchByteLength;
        _descriptorLayout = descriptorLayout;
    }
    return self;
}

- (instancetype)initWithTaskCount:(NSUInteger)taskCount
                       recordCount:(NSUInteger)recordCount
                        formatCode:(uint32_t)formatCode
                 scratchByteLength:(NSUInteger)scratchByteLength
       scratchAllocationByteLength:(NSUInteger)scratchAllocationByteLength
                  descriptorLayout:(HWXProgramDescriptorLayout)descriptorLayout {
    self = [super init];
    if (self) {
        _taskCount = taskCount; _firstTaskByteLength = 0;
        _recordCount = recordCount; _formatCode = formatCode;
        _scratchByteLength = scratchByteLength;
        _scratchAllocationByteLength = scratchAllocationByteLength;
        _descriptorLayout = descriptorLayout;
    }
    return self;
}
@end

@implementation HWXObjectWriter
+ (NSData *)buildObjectWithTaskDescriptor:(NSData *)taskDescriptor
                             constantRegion:(NSData *)constantRegion
                                    bindings:(NSArray<HWXObjectBinding *> *)bindings
                                       error:(NSError **)error {
    return [self buildObjectWithTaskDescriptor:taskDescriptor
        constantRegion:constantRegion bindings:bindings
        kernelRelocationOffsets:@[@0x1a8] programRecordCount:0x10
        programFormatCode:0x0f error:error];
}

+ (NSData *)buildObjectWithTaskDescriptor:(NSData *)taskDescriptor
                             constantRegion:(NSData *)constantRegion
                                    bindings:(NSArray<HWXObjectBinding *> *)bindings
                     kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
                          programRecordCount:(NSUInteger)programRecordCount
                           programFormatCode:(uint32_t)programFormatCode
                                       error:(NSError **)error {
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:kernelRelocationOffsets.count
        recordCount:programRecordCount formatCode:programFormatCode
        scratchByteLength:0 descriptorLayout:HWXProgramDescriptorLayoutLinear];
    return [self buildObjectWithTaskDescriptor:taskDescriptor
        constantRegion:constantRegion bindings:bindings
        kernelRelocationOffsets:kernelRelocationOffsets programInfo:info
        error:error];
}
+ (NSData *)buildObjectWithTaskDescriptor:(NSData *)taskDescriptor
                             constantRegion:(NSData *)constantRegion
                                    bindings:(NSArray<HWXObjectBinding *> *)bindings
                     kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
                                 programInfo:(HWXObjectProgramInfo *)programInfo
                                       error:(NSError **)error {
    return [self buildObjectForArchitecture:HWXObjectArchitectureH16G
        taskDescriptor:taskDescriptor constantRegion:constantRegion
        bindings:bindings kernelRelocationOffsets:kernelRelocationOffsets
        programInfo:programInfo error:error];
}

+ (NSData *)buildObjectForArchitecture:(HWXObjectArchitecture)architecture
                             taskDescriptor:(NSData *)taskDescriptor
                              constantRegion:(NSData *)constantRegion
                                     bindings:(NSArray<HWXObjectBinding *> *)bindings
                      kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
                                 programInfo:(HWXObjectProgramInfo *)programInfo
                                       error:(NSError **)error {
    if (architecture == HWXObjectArchitectureH16G) {
        // A short reference has eight bytes including its NUL. Keep semantic
        // tensor symbols intact and give longer MIL identifiers a local name.
        // This does not affect the pinned owner fixture's x/z/y identities.
        NSMutableSet<NSString *> *symbols = [NSMutableSet set];
        NSMutableSet<NSString *> *shortNames = [NSMutableSet set];
        for (HWXObjectBinding *binding in bindings) {
            BOOL valid = descriptorNameValid(binding.symbol) &&
                descriptorNameValid(binding.shortName) &&
                ![symbols containsObject:binding.symbol] &&
                ![shortNames containsObject:binding.shortName] &&
                (binding.role == HWXObjectBindingRoleInput ||
                 binding.role == HWXObjectBindingRoleOutput);
            for (NSNumber *dimension in binding.shape)
                valid = valid && dimension.longLongValue > 0 &&
                    dimension.unsignedLongLongValue <= UINT32_MAX;
            if (!valid) {
                if (error) *error = [NSError errorWithDomain:HWXObjectWriterErrorDomain
                    code:1 userInfo:@{NSLocalizedDescriptionKey:
                        @"h16g.invalid-binding-identity: unique printable names, valid directions and positive u32 dimensions required"}];
                return nil;
            }
            [symbols addObject:binding.symbol];
            [shortNames addObject:binding.shortName];
        }
        NSMutableArray<HWXObjectBinding *> *normalized = [NSMutableArray array];
        NSUInteger nameIndex = 0;
        for (HWXObjectBinding *binding in bindings) {
            NSString *shortName = binding.shortName;
            if ([shortName lengthOfBytesUsingEncoding:NSASCIIStringEncoding] > 7) {
                do {
                    shortName = [NSString stringWithFormat:@"_io%lu", (unsigned long)nameIndex++];
                } while ([shortNames containsObject:shortName]);
                [shortNames addObject:shortName];
            }
            // SPEC-r3 +0x70/+0x80 describe one batch span, not the backing
            // allocation. Preserve allocation padding independently.
            HWXObjectBinding *copy = [[HWXObjectBinding alloc]
                initWithSymbol:binding.symbol shortName:shortName role:binding.role
                elementType:binding.elementType shape:binding.shape
                rowStrideBytes:binding.rowStrideBytes
                planeStrideBytes:binding.planeStrideBytes
                batchStrideBytes:binding.batchStrideBytes
                storageByteLength:binding.batchStrideBytes];
            copy.allocationByteLength = binding.allocationByteLength;
            [normalized addObject:copy];
        }
        bindings = normalized;
    }
    NSMutableArray<HWXObjectBinding *> *inputs = [NSMutableArray array];
    NSMutableArray<HWXObjectBinding *> *outputs = [NSMutableArray array];
    for (HWXObjectBinding *binding in bindings) {
        if (binding.role == HWXObjectBindingRoleInput)
            [inputs addObject:binding];
        else
            [outputs addObject:binding];
    }
    // H13, H14, H17 and H18 keep constants in __TEXT/__const; H16G uses a
    // __KERN_0 segment with relocated kernel-table addends. H17 and H18 put
    // only their kernel table (programInfo.kernelTable) in __KERN_0.
    BOOL textConstants = architecture == HWXObjectArchitectureH13 ||
        architecture == HWXObjectArchitectureH14 ||
        architecture == HWXObjectArchitectureH17 ||
        architecture == HWXObjectArchitectureH18;
    BOOL supportedArchitecture = textConstants ||
        architecture == HWXObjectArchitectureH16G;
    BOOL separateKernelTable = architecture == HWXObjectArchitectureH17 ||
        architecture == HWXObjectArchitectureH18;
    NSData *kernelData = textConstants ? programInfo.kernelTable : constantRegion;
    BOOL hasKernel = kernelData.length != 0;
    BOOL hasConstants = constantRegion.length != 0;
    BOOL isThreeSurfaceProgram = bindings.count == 3 && inputs.count == 2 &&
        outputs.count == 1;
    BOOL resourceCountValid = inputs.count >= 1 && outputs.count == 1 &&
        bindings.count <= 4 &&
        bindings.count + (hasKernel ? 1 : 0) <= 5;
    BOOL bindingLayoutsValid = YES;
    for (HWXObjectBinding *binding in bindings) {
        NSUInteger n=1,c=1,h=1,w=1;
        shapeDimensions(binding.shape, &n, &c, &h, &w);
        NSUInteger denseRow = 0, minimumPlane = 0;
        NSUInteger minimumBatch = 0, minimumStorage = 0;
        BOOL productsValid = binding.shape.count >= 1 &&
            binding.shape.count <= 4 && n != 0 && c != 0 && h != 0 && w != 0 &&
            multiplyWithoutOverflow(w, elementBytes(binding.elementType),
                                    &denseRow) &&
            multiplyWithoutOverflow(h, binding.rowStrideBytes,
                                    &minimumPlane) &&
            multiplyWithoutOverflow(c, binding.planeStrideBytes,
                                    &minimumBatch) &&
            multiplyWithoutOverflow(n, binding.batchStrideBytes,
                                    &minimumStorage);
        // Apple's batched surfaces record one batch element's stride as the
        // descriptor's total size, so the whole-batch span is checked against
        // the VM allocation and the descriptor's own size against one element.
        bindingLayoutsValid = bindingLayoutsValid && productsValid &&
            denseRow != 0 &&
            binding.rowStrideBytes >= denseRow &&
            binding.planeStrideBytes >= minimumPlane &&
            binding.batchStrideBytes >= minimumBatch &&
            binding.storageByteLength >= minimumBatch &&
            binding.allocationByteLength >= minimumStorage;
    }
    BOOL relocationsValid = (textConstants ? hasConstants : hasKernel) ||
        kernelRelocationOffsets.count == 0;
    NSMutableSet<NSNumber *> *relocationSites = [NSMutableSet set];
    for (NSNumber *offset in kernelRelocationOffsets) {
        NSUInteger value = offset.unsignedIntegerValue;
        relocationsValid = relocationsValid &&
            (textConstants ? hasConstants : hasKernel) &&
            value % 4 == 0 &&
            value <= taskDescriptor.length &&
            sizeof(uint32_t) <= taskDescriptor.length - value;
        if (architecture == HWXObjectArchitectureH16G) {
            relocationsValid = relocationsValid &&
                ![relocationSites containsObject:@(value)];
            [relocationSites addObject:@(value)];
        }
        if (relocationsValid) {
            uint32_t addend = 0;
            [taskDescriptor getBytes:&addend
                               range:NSMakeRange(value, sizeof(addend))];
            relocationsValid = addend < constantRegion.length;
        }
    }
    NSString *rejection = nil;
    if (!supportedArchitecture)
        rejection = @"writer supports only H13, H14, H16G, H17 and H18 architectures";
    else if (!programInfo || programInfo.taskCount == 0)
        rejection = @"writer requires program info with at least one task";
    else if (programInfo.kernelTable.length && !separateKernelTable)
        rejection = @"only H17 and H18 objects carry a separate kernel table";
    else if (textConstants && (taskDescriptor.length == 0 ||
                     taskDescriptor.length % sizeof(uint32_t) ||
                     programInfo.descriptorLayout != HWXProgramDescriptorLayoutLinear ||
                     (programInfo.firstTaskByteLength &&
                      (programInfo.firstTaskByteLength > taskDescriptor.length ||
                       programInfo.firstTaskByteLength % sizeof(uint32_t)))))
        rejection = @"text-constant HWX requires a nonempty word-aligned linear task stream";
    else if (!resourceCountValid)
        rejection = @"writer requires one output, at least one input, at most four surfaces and five total resources";
    // Apple partitions one program into up to 129 tasks rather than emitting
    // a second program; the largest decoded single-batch stream is 80,888
    // bytes for [512,8192] x const[8192,8192]. The batched matmul oracles
    // run 26·B tasks in one stream — 324,212 bytes at B=16 — so the cap
    // covers that decoded maximum.
    else if (taskDescriptor.length == 0 || taskDescriptor.length > 0x60000)
        rejection = @"writer requires a nonempty task descriptor of at most 0x60000 bytes";
    else if (architecture == HWXObjectArchitectureH16G &&
             (taskDescriptor.length % 4 || programInfo.taskCount > UINT32_MAX ||
              programInfo.recordCount > UINT32_MAX ||
              programInfo.scratchByteLength > UINT32_MAX ||
              (programInfo.firstTaskByteLength &&
               programInfo.firstTaskByteLength != taskDescriptor.length)))
        rejection = @"h16g.invalid-program-width: H16 word count covers the complete stream and descriptor fields must fit u32";
    // Apple keeps a whole matmul weight in one program's __TEXT/__const: the
    // largest decoded section is 134,217,728 bytes, for [512,8192] x
    // const[8192,8192]. The cap is twice that.
    else if (constantRegion.length > 0x10000000)
        rejection = @"writer requires a constant region of at most 256 MiB";
    else if (programInfo.descriptorLayout == HWXProgramDescriptorLayoutLinear &&
             programInfo.scratchByteLength != 0)
        rejection = @"a linear program cannot declare scratch";
    else if (programInfo.descriptorLayout == HWXProgramDescriptorLayoutScratchBackedMixed &&
             (programInfo.scratchByteLength == 0 ||
              (!isThreeSurfaceProgram &&
               programInfo.scratchAllocationByteLength < programInfo.scratchByteLength)))
        rejection = @"a scratch-backed program needs scratch that its allocation covers";
    else if (!relocationsValid)
        rejection = @"kernel relocations must be aligned, inside the descriptor, and address the kernel table";
    else if (!bindingLayoutsValid)
        rejection = @"binding shapes and strides must be nonzero, rank 1 to 4, and cover the tensor";
    if (rejection) {
        if(error)*error=[NSError errorWithDomain:HWXObjectWriterErrorDomain code:1
            userInfo:@{NSLocalizedDescriptionKey:rejection}];
        return nil;
    }
    uint64_t scratchVM=0x30000000;
    uint64_t nextBindingVM = 0;
    if (!addWithoutOverflow(scratchVM,
            (uint64_t)programInfo.scratchAllocationByteLength,
            &nextBindingVM)) {
        if (error) *error=[NSError errorWithDomain:HWXObjectWriterErrorDomain
            code:1 userInfo:@{NSLocalizedDescriptionKey:
                @"writer VM layout overflows"}];
        return nil;
    }
    NSMutableArray<NSNumber *> *bindingVMs = [NSMutableArray array];
    NSMutableArray<NSNumber *> *bindingSizes = [NSMutableArray array];
    for (HWXObjectBinding *binding in bindings) {
        uint64_t size = 0, next = 0;
        if (!alignUpWithoutOverflow(binding.allocationByteLength, 0x4000, &size) ||
            !addWithoutOverflow(nextBindingVM, size, &next)) {
            if (error) *error=[NSError errorWithDomain:HWXObjectWriterErrorDomain
                code:1 userInfo:@{NSLocalizedDescriptionKey:
                    @"writer binding VM layout overflows"}];
            return nil;
        }
        [bindingVMs addObject:@(nextBindingVM)];
        [bindingSizes addObject:@(size)];
        nextBindingVM = next;
    }
    uint64_t textVM=nextBindingVM;
    uint64_t textEndVM = 0, textConstVM = 0;
    if (!addWithoutOverflow(textVM, taskDescriptor.length, &textEndVM) ||
        !alignUpWithoutOverflow(textEndVM, 0x40, &textConstVM)) {
        if (error) *error=[NSError errorWithDomain:HWXObjectWriterErrorDomain
            code:1 userInfo:@{NSLocalizedDescriptionKey:
                @"writer text VM layout overflows"}];
        return nil;
    }
    NSUInteger firstBindingLogicalBytes = 0;
    BOOL firstBindingLogicalValid = bindings.count != 0 &&
        logicalBytes(bindings[0], &firstBindingLogicalBytes);
    BOOL compactThreeSurface = bindings.count == 3 &&
        programInfo.descriptorLayout == HWXProgramDescriptorLayoutLinear &&
        firstBindingLogicalValid && firstBindingLogicalBytes <= 0x100000;
    uint32_t textFileOffset = textConstants ? 0x4000 :
        ((kernelRelocationOffsets.count <= 1 &&
          (bindings.count == 2 || compactThreeSurface)) ? 0x4000 : 0x8000);
    uint64_t textConstOffset = textConstVM - textVM;
    uint64_t textSegmentSize = textConstants
        ? alignUp(textConstOffset + constantRegion.length, 0x4000) : 0x8000;
    // __KERN_0 follows __TEXT directly: 0x8000 past the text for H16G, one
    // 16 KiB-aligned text-and-constant span for H17 and H18.
    uint64_t kernelVM = 0;
    if (hasKernel && !addWithoutOverflow(textVM, textSegmentSize, &kernelVM)) {
        if (error) *error=[NSError errorWithDomain:HWXObjectWriterErrorDomain
            code:1 userInfo:@{NSLocalizedDescriptionKey:
                @"writer kernel VM layout overflows"}];
        return nil;
    }
    uint32_t kernelFileOffset = textFileOffset + (uint32_t)textSegmentSize;
    uint64_t kernelSegmentSize = hasKernel
        ? alignUp(kernelData.length, 0x4000) : 0;
    uint32_t textConstFile = textConstants ? textFileOffset + (uint32_t)textConstOffset :
        (uint32_t)alignUp(textFileOffset + taskDescriptor.length, 0x40);
    NSMutableArray<NSData *> *commands = [NSMutableArray array];
    [commands addObject:segmentRecord("__PAGEZERO",0,0x4000,0,0,0,0,4,NULL,0)];
    BOOL hasScratch = programInfo.scratchAllocationByteLength != 0;
    if (hasScratch) {
        struct section_64 scratchSection=sectionRecord("__bss","__DATA",
            scratchVM,programInfo.scratchAllocationByteLength,0,6,0x25);
        [commands addObject:segmentRecord("__DATA",scratchVM,
            programInfo.scratchAllocationByteLength,0,0,3,3,4,&scratchSection,1)];
    }
    for (NSUInteger i = 0; i < bindings.count; ++i) {
        HWXObjectBinding *binding = bindings[i];
        BOOL isInput = binding.role == HWXObjectBindingRoleInput;
        uint64_t address = bindingVMs[i].unsignedLongLongValue;
        uint64_t sectionSize = textConstants ? binding.storageByteLength
                                    : bindingSizes[i].unsignedLongLongValue;
        uint64_t segmentSize = bindingSizes[i].unsignedLongLongValue;
        struct section_64 section=sectionRecord(
            isInput ? "__const" : "__data", "__FVMLIB",
            address,sectionSize,0,14,isInput ? 0x21 : 0x23);
        [commands addObject:segmentRecord("__FVMLIB",address,segmentSize,0,0,
            isInput ? 1 : 2,isInput ? 1 : 2,6,&section,1)];
    }
    uint32_t textFlags = kernelRelocationOffsets.count == 0 ? 0x28 : 0x128;
    struct section_64 textSecs[2]={
        sectionRecord("__text","__TEXT",textVM,taskDescriptor.length,textFileOffset,14,textFlags),
        sectionRecord("__const","__TEXT",textConstVM,
            textConstants ? constantRegion.length : 0x4000,textConstFile,6,0x26)};
    // Apple's H13 objects keep __TEXT flags at zero even when a __DATA/__bss
    // scratch segment is present; H16G sets bit 2.
    uint32_t textSegmentFlags = (hasScratch && !textConstants) ? 4 : 0;
    [commands addObject:segmentRecord("__TEXT",textVM,textSegmentSize,
        textFileOffset,textSegmentSize,5,5,textSegmentFlags,textSecs,2)];
    if (hasKernel) {
        struct section_64 kernSec=sectionRecord("__kern_0","__KERN_0",kernelVM,
            kernelData.length,kernelFileOffset,6,0x26);
        [commands addObject:segmentRecord("__KERN_0",kernelVM,kernelSegmentSize,
            kernelFileOffset,kernelSegmentSize,1,1,4,&kernSec,1)];
    }
    for (NSUInteger i = 0; i < bindings.count; ++i)
        [commands addObject:bufferReference(bindings[i],
            bindingVMs[i].unsignedLongLongValue)];
    NSMutableArray<NSNumber *> *resourceAddresses = [bindingVMs mutableCopy];
    if (hasKernel) [resourceAddresses addObject:@(kernelVM)];
    NSUInteger firstTaskBytes = programInfo.firstTaskByteLength ?: taskDescriptor.length;
    [commands addObject:programDescriptor(textVM,textConstVM,scratchVM,
        resourceAddresses,(uint32_t)(firstTaskBytes / 4),programInfo,
        architecture)];
    for (HWXObjectBinding *input in inputs)
        [commands addObject:tensorDescriptor(input, 1)];
    for (HWXObjectBinding *output in outputs)
        [commands addObject:tensorDescriptor(output, 2)];
    [commands addObject:compilerMetadata(architecture)];

    NSMutableData *symbols = nil;
    NSMutableData *strings = nil;
    NSMutableOrderedSet<NSNumber *> *kernelOffsets =
        [NSMutableOrderedSet orderedSet];
    if (hasKernel) [kernelOffsets addObject:@0];
    if (!textConstants) for (NSNumber *relocationOffsetNumber in kernelRelocationOffsets) {
        uint32_t addend = 0;
        [taskDescriptor getBytes:&addend range:NSMakeRange(
            relocationOffsetNumber.unsignedIntegerValue, sizeof(addend))];
        [kernelOffsets addObject:@(addend)];
    }
    uint8_t firstBindingSection = hasScratch ? 2 : 1;
    uint8_t kernelSection = (uint8_t)((hasScratch ? 1 : 0) +
        bindings.count + 3);
    uint8_t textConstSection = (uint8_t)((hasScratch ? 1 : 0) +
        bindings.count + 2);
    buildSymbolTables(bindings, bindingVMs, kernelVM,
                      kernelOffsets.array, firstBindingSection, kernelSection,
                      &symbols, &strings);
    NSUInteger commandBytes=sizeof(struct symtab_command);
    for(NSData *c in commands)commandBytes+=c.length;
    NSUInteger commandEnd = sizeof(struct mach_header_64) + commandBytes;
    uint32_t symbolOffset = (uint32_t)alignUp(commandEnd, 8);
    uint32_t stringOffset = symbolOffset + (uint32_t)symbols.length;
    uint32_t relocationOffset = (uint32_t)alignUp(
        stringOffset + strings.length, 8);
    [commands addObject:symbolTableCommand(symbolOffset,
        (uint32_t)(symbols.length / sizeof(struct nlist_64)), stringOffset,
        (uint32_t)strings.length)];
    textSecs[0].reloff = relocationOffset;
    textSecs[0].nreloc = (uint32_t)kernelRelocationOffsets.count;
    NSUInteger textCommandIndex = 1 + (hasScratch ? 1 : 0) + bindings.count;
    commands[textCommandIndex] = segmentRecord("__TEXT",textVM,textSegmentSize,
        textFileOffset,textSegmentSize,5,5,textSegmentFlags,textSecs,2);
    if(sizeof(struct mach_header_64)+commandBytes>textFileOffset){
        if(error) {
            *error=[NSError errorWithDomain:HWXObjectWriterErrorDomain code:2
                userInfo:@{NSLocalizedDescriptionKey:@"generated commands cross __TEXT"}];
        }
        return nil;
    }
    NSUInteger relocationBytes = kernelRelocationOffsets.count * 8;
    if (relocationOffset > textFileOffset - relocationBytes) {
        if (error) {
            *error=[NSError errorWithDomain:HWXObjectWriterErrorDomain code:3
                userInfo:@{NSLocalizedDescriptionKey:
                    @"generated symbols and relocation cross __TEXT"}];
        }
        return nil;
    }
    NSUInteger imageLength = hasKernel ? kernelFileOffset + kernelSegmentSize
        : textFileOffset + textSegmentSize;
    NSMutableData *image=[NSMutableData dataWithLength:imageLength];
    struct mach_header_64 header={};header.magic=0xBEEFFACE;header.cputype=0x80;
    header.cpusubtype=(uint32_t)architecture;header.filetype=2;
    header.ncmds=(uint32_t)commands.count;
    header.sizeofcmds=(uint32_t)commandBytes;header.flags=0x00200000;
    [image replaceBytesInRange:NSMakeRange(0,sizeof(header)) withBytes:&header];
    NSUInteger offset=sizeof(header);for(NSData *c in commands){
        [image replaceBytesInRange:NSMakeRange(offset,c.length) withBytes:c.bytes];offset+=c.length;}
    [image replaceBytesInRange:NSMakeRange(symbolOffset,symbols.length)
                     withBytes:symbols.bytes];
    [image replaceBytesInRange:NSMakeRange(stringOffset,strings.length)
                     withBytes:strings.bytes];
    for (NSUInteger i = 0; i < kernelRelocationOffsets.count; ++i) {
        const uint32_t relocation[2] = {
            kernelRelocationOffsets[i].unsignedIntValue,
            (textConstants ? 0x05000000u | textConstSection
                 : 0x07000000u | kernelSection)};
        [image replaceBytesInRange:NSMakeRange(relocationOffset + i * 8,
            sizeof(relocation)) withBytes:relocation];
    }
    [image replaceBytesInRange:NSMakeRange(textFileOffset,taskDescriptor.length)
                     withBytes:taskDescriptor.bytes];
    if (textConstants)
        [image replaceBytesInRange:NSMakeRange(textConstFile,constantRegion.length)
                         withBytes:constantRegion.bytes];
    if (hasKernel)
        [image replaceBytesInRange:NSMakeRange(kernelFileOffset,kernelData.length)
                         withBytes:kernelData.bytes];
    return image;
}
@end
