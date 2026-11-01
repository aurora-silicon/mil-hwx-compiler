#import <Foundation/Foundation.h>
#import "HWXMachOFormat.h"

#import "H16GConvEncoder.h"
#import "H16GConvChainEncoder.h"
#import "H16GLayoutEncoder.h"
#import "HWXImage.h"
#import "HWXObjectWriter.h"

#include <stdio.h>

static int failures = 0;
static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    failures++;
}

static NSArray<HWXObjectBinding *> *bindings(void) {
    return @[
        [[HWXObjectBinding alloc] initWithSymbol:@"x" shortName:@"x"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@64,@64,@64]],
        [[HWXObjectBinding alloc] initWithSymbol:@"y@output" shortName:@"y"
            role:HWXObjectBindingRoleOutput elementType:ANEElementTypeFP16
            shape:@[@1,@64,@64,@64]],
    ];
}

static BOOL commandExists(NSData *imageData, uint32_t wanted) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == wanted) return YES;
        offset += command->cmdsize;
    }
    return NO;
}

static BOOL segmentExists(NSData *imageData, const char *wanted) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *segment =
                (const struct segment_command_64 *)command;
            if (strncmp(segment->segname, wanted, 16) == 0) return YES;
        }
        offset += command->cmdsize;
    }
    return NO;
}

static uint64_t segmentVMSize(NSData *imageData, const char *wanted) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *segment =
                (const struct segment_command_64 *)command;
            if (strncmp(segment->segname, wanted, 16) == 0)
                return segment->vmsize;
        }
        offset += command->cmdsize;
    }
    return UINT64_MAX;
}

static NSArray<NSNumber *> *segmentVMAddresses(NSData *imageData,
                                                const char *wanted) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    NSMutableArray<NSNumber *> *addresses = [NSMutableArray array];
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *segment =
                (const struct segment_command_64 *)command;
            if (strncmp(segment->segname, wanted, 16) == 0)
                [addresses addObject:@(segment->vmaddr)];
        }
        offset += command->cmdsize;
    }
    return addresses;
}

static NSArray<NSNumber *> *segmentVMSizes(NSData *imageData,
                                            const char *wanted) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    NSMutableArray<NSNumber *> *sizes = [NSMutableArray array];
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *segment =
                (const struct segment_command_64 *)command;
            if (strncmp(segment->segname, wanted, 16) == 0)
                [sizes addObject:@(segment->vmsize)];
        }
        offset += command->cmdsize;
    }
    return sizes;
}

static const struct section_64 *textSection(NSData *imageData) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *segment =
                (const struct segment_command_64 *)command;
            const struct section_64 *sections =
                (const struct section_64 *)(segment + 1);
            for (uint32_t j = 0; j < segment->nsects; ++j)
                if (strncmp(sections[j].sectname, "__text", 16) == 0)
                    return &sections[j];
        }
        offset += command->cmdsize;
    }
    return NULL;
}

static NSString *firstFVMLIBSectionName(NSData *imageData) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *segment =
                (const struct segment_command_64 *)command;
            if (strncmp(segment->segname, "__FVMLIB", 16) == 0 &&
                segment->nsects == 1) {
                const struct section_64 *section =
                    (const struct section_64 *)(segment + 1);
                return [[NSString alloc] initWithBytes:section->sectname
                    length:strnlen(section->sectname, 16)
                    encoding:NSUTF8StringEncoding];
            }
        }
        offset += command->cmdsize;
    }
    return nil;
}

static NSArray<NSString *> *tensorSymbols(NSData *imageData) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    NSMutableArray<NSString *> *symbols = [NSMutableArray array];
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == 0x04 && command->cmdsize >= 0x24 &&
            *(const uint32_t *)(bytes + offset + 8) == 3) {
            uint32_t symbolOffset = *(const uint32_t *)(bytes + offset + 0x20);
            const char *symbol = (const char *)(bytes + offset + symbolOffset);
            [symbols addObject:[NSString stringWithUTF8String:symbol]];
        }
        offset += command->cmdsize;
    }
    return symbols;
}

static NSArray<NSNumber *> *tensorBindingIndices(NSData *imageData) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    NSMutableArray<NSNumber *> *indices = [NSMutableArray array];
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == 0x04 && command->cmdsize >= 0x18 &&
            *(const uint32_t *)(bytes + offset + 8) == 3)
            [indices addObject:@(*(const uint32_t *)(bytes + offset + 0x14))];
        offset += command->cmdsize;
    }
    return indices;
}

static NSArray<NSArray<NSNumber *> *> *tensorShapes(NSData *imageData) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    NSMutableArray<NSArray<NSNumber *> *> *shapes = [NSMutableArray array];
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == 0x04 && command->cmdsize >= 0x38 &&
            *(const uint32_t *)(bytes + offset + 8) == 3) {
            const uint32_t *shape =
                (const uint32_t *)(bytes + offset + 0x28);
            [shapes addObject:@[@(shape[0]),@(shape[1]),@(shape[2]),@(shape[3])]];
        }
        offset += command->cmdsize;
    }
    return shapes;
}

static NSArray<NSArray<NSNumber *> *> *tensorPhysicalLayouts(NSData *imageData) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    NSMutableArray<NSArray<NSNumber *> *> *layouts = [NSMutableArray array];
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == 0x04 && command->cmdsize >= 0x88 &&
            *(const uint32_t *)(bytes + offset + 8) == 3) {
            [layouts addObject:@[
                @(*(const uint64_t *)(bytes + offset + 0x60)),
                @(*(const uint64_t *)(bytes + offset + 0x58)),
                @(*(const uint64_t *)(bytes + offset + 0x50)),
                @(*(const uint64_t *)(bytes + offset + 0x70)),
            ]];
        }
        offset += command->cmdsize;
    }
    return layouts;
}

static NSUInteger commandCount(NSData *imageData, uint32_t wanted) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header), count = 0;
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == wanted) count++;
        offset += command->cmdsize;
    }
    return count;
}

static uint32_t programDescriptorWord(NSData *imageData,
                                      NSUInteger relativeOffset) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == 0x04 && command->cmdsize >= relativeOffset + 4 &&
            *(const uint32_t *)(bytes + offset + 8) == 4)
            return *(const uint32_t *)(bytes + offset + relativeOffset);
        offset += command->cmdsize;
    }
    return UINT32_MAX;
}

static uint64_t programDescriptorQWord(NSData *imageData,
                                       NSUInteger relativeOffset) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == 0x04 && command->cmdsize >= relativeOffset + 8 &&
            *(const uint32_t *)(bytes + offset + 8) == 4)
            return *(const uint64_t *)(bytes + offset + relativeOffset);
        offset += command->cmdsize;
    }
    return UINT64_MAX;
}

static void testObjectIsBuiltFromInputs(void) {
    NSError *error = nil;
    NSData *td = [H16GConvEncoder encodeConv1x1WithInputChannels:64
        outputChannels:64 spatial:64 bytesPerWeight:2
        numericMode:ANELegalNumericModeFP16 reluEpilogue:YES error:&error];
    NSMutableData *weights = [NSMutableData dataWithLength:8192];
    uint8_t *bytes = (uint8_t *)weights.mutableBytes;
    for (NSUInteger i=0; i<weights.length; ++i) bytes[i] = (uint8_t)(i*17+3);
    NSData *imageData = [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:weights bindings:bindings() error:&error];
    HWXImage *image = [HWXImage imageWithData:imageData error:&error];
    expect(image != nil, @"newly constructed object reparses");
    expect(image.magic == 0xBEEFFACE && image.cpuType == 0x80 &&
           image.cpuSubtype == 0x07, @"object targets H16G");
    expect([[image firstSectionNamed:@"__text" inSegment:@"__TEXT"].data
            isEqualToData:td], @"encoded TD is placed as __TEXT.__text");
    expect([[image firstSectionNamed:@"__kern_0" inSegment:@"__KERN_0"].data
            isEqualToData:weights], @"supplied constants are placed without an oracle file");
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)imageData.bytes;
    expect(header->ncmds == 12,
           @"object includes generated compiler metadata and linker records");
    expect(commandExists(imageData, 0x08) &&
           commandExists(imageData, LC_SYMTAB),
           @"object carries self-described metadata and a symbol table");
    const struct section_64 *text = textSection(imageData);
    expect(text && text->nreloc == 1 && text->reloff != 0,
           @"text section carries the local kernel relocation used by the TD");
    if (text && text->nreloc == 1) {
        const uint32_t *relocation = (const uint32_t *)
            ((const uint8_t *)imageData.bytes + text->reloff);
        expect(relocation[0] == 0x1a8 && relocation[1] == 0x07000005,
               @"kernel relocation targets the decoded TD field and section");
    }
    expect(header->sizeofcmds < 0x4000-sizeof(*header),
           @"generated command envelope fits before __TEXT");
}

static void testEmissionIsDeterministicAndDataDependent(void) {
    NSError *error = nil;
    NSData *td = [H16GConvEncoder encodeConv1x1WithInputChannels:64
        outputChannels:64 spatial:64 bytesPerWeight:2
        numericMode:ANELegalNumericModeFP16 reluEpilogue:YES error:&error];
    NSMutableData *a = [NSMutableData dataWithLength:8192];
    NSMutableData *b = [a mutableCopy];
    ((uint8_t *)b.mutableBytes)[77] = 9;
    NSData *a1 = [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:a bindings:bindings() error:&error];
    NSData *a2 = [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:a bindings:bindings() error:&error];
    NSData *b1 = [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:b bindings:bindings() error:&error];
    expect([a1 isEqualToData:a2], @"same typed inputs emit identical objects");
    expect(![a1 isEqualToData:b1], @"constant data changes the emitted object");
}

static void testMultiTaskObjectDerivesLayoutAndRelocations(void) {
    NSError *error = nil;
    H16GEncodedTDProgram *program = [H16GConvChainEncoder
        encodeW8A8C64S64WithDepth:4 error:&error];
    NSMutableData *constants = [NSMutableData dataWithLength:0x4000];
    uint8_t *constantBytes = (uint8_t *)constants.mutableBytes;
    for (NSUInteger i = 0; i < constants.length; ++i)
        constantBytes[i] = (uint8_t)(i * 29 + 7);
    NSArray<HWXObjectBinding *> *multiTaskBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:@"c3@output" shortName:@"c3"
            role:HWXObjectBindingRoleOutput elementType:ANEElementTypeFP16
            shape:@[@1,@64,@64,@64]],
        [[HWXObjectBinding alloc] initWithSymbol:@"x" shortName:@"x"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@64,@64,@64]],
    ];
    NSData *imageData = [HWXObjectWriter
        buildObjectWithTaskDescriptor:program.data
                       constantRegion:constants
                             bindings:multiTaskBindings
              kernelRelocationOffsets:program.kernelRelocationOffsets
                  programRecordCount:program.programRecordCount
                   programFormatCode:program.programFormatCode
                                error:&error];
    HWXImage *image = [HWXImage imageWithData:imageData error:&error];
    expect(image != nil && imageData.length == 0x14000,
           @"multi-task object selects a generated layout large enough for its payloads");
    expect([[image firstSectionNamed:@"__text" inSegment:@"__TEXT"].data
            isEqualToData:program.data],
           @"multi-task TD is installed without a template object");
    expect([[image firstSectionNamed:@"__kern_0" inSegment:@"__KERN_0"].data
            isEqualToData:constants],
           @"multi-task constants are installed without a template object");
    const struct section_64 *text = textSection(imageData);
    expect(text && text->offset == 0x8000 && text->nreloc == 4,
           @"multi-task text section records every kernel relocation");
    expect([firstFVMLIBSectionName(imageData) isEqualToString:@"__data"],
           @"descriptor binding order controls the generated external slots");
    expect([tensorSymbols(imageData) isEqualToArray:@[@"x", @"c3@output"]],
           @"tensor metadata remains in semantic input-output order");
    expect(programDescriptorWord(imageData, 0x824) == 0x18c &&
           programDescriptorWord(imageData, 0x830) == 4,
           @"program metadata derives TD word and task counts");
    expect(programDescriptorWord(imageData, 0x860) == 0x3b &&
           programDescriptorWord(imageData, 0x890) == 7,
           @"program metadata carries the encoder-provided runtime fields");
    if (text && text->nreloc == 4) {
        const uint32_t *relocations = (const uint32_t *)
            ((const uint8_t *)imageData.bytes + text->reloff);
        for (NSUInteger i = 0; i < 4; ++i) {
            expect(relocations[i * 2] ==
                       program.kernelRelocationOffsets[i].unsignedIntValue &&
                   relocations[i * 2 + 1] == 0x07000005,
                   @"each generated relocation targets the kernel section");
        }
    }
}

static void testScratchBackedMixedProgramHasIndependentEnvelopeMetadata(void) {
    NSMutableData *td = [NSMutableData dataWithLength:0x948];
    NSMutableData *constants = [NSMutableData dataWithLength:0x80];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:10 recordCount:0x75 formatCode:1
        scratchByteLength:0x10000
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSArray<HWXObjectBinding *> *mixedBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:@"x" shortName:@"x"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@64,@4,@192]],
        [[HWXObjectBinding alloc] initWithSymbol:@"y@output" shortName:@"y"
            role:HWXObjectBindingRoleOutput elementType:ANEElementTypeFP16
            shape:@[@1,@64,@4,@64]],
    ];
    NSError *error = nil;
    NSData *imageData = [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:constants bindings:mixedBindings
        kernelRelocationOffsets:@[@0x5b4] programInfo:info error:&error];
    HWXImage *image = [HWXImage imageWithData:imageData error:&error];
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)imageData.bytes;
    const struct section_64 *text = textSection(imageData);
    expect(image != nil && header->ncmds == 13 && segmentExists(imageData,"__DATA"),
           @"scratch-backed programs receive a generated BSS segment");
    expect(programDescriptorWord(imageData,0x004) == 0x8c0 &&
           programDescriptorWord(imageData,0x830) == 10 &&
           programDescriptorWord(imageData,0x858) == 0x10000 &&
           programDescriptorWord(imageData,0x860) == 0x75 &&
           programDescriptorWord(imageData,0x8b0) == 1,
           @"mixed program descriptor fields come from explicit encoder metadata");
    if (text && text->nreloc == 1) {
        const uint32_t *relocation = (const uint32_t *)
            ((const uint8_t *)imageData.bytes + text->reloff);
        expect(relocation[0] == 0x5b4 && relocation[1] == 0x07000006,
               @"scratch segment shifts the local kernel relocation section");
    }
}

static void testScratchBackedLayoutProgramNeedsNoKernelSegment(void) {
    NSError *error = nil;
    H16GEncodedTDProgram *program = [H16GLayoutEncoder
        encodeOperationName:@"depth_to_space"
        inputShape:@[@1,@256,@32,@32]
        outputShape:@[@1,@16,@128,@128]
        blockSize:4 strategy:ANETileStrategyLayoutDMA3 error:&error];
    NSArray<HWXObjectBinding *> *layoutBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:@"x" shortName:@"x"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@256,@32,@32]],
        [[HWXObjectBinding alloc] initWithSymbol:@"y@output" shortName:@"y"
            role:HWXObjectBindingRoleOutput elementType:ANEElementTypeFP16
            shape:@[@1,@16,@128,@128]],
    ];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:3 recordCount:program.programRecordCount
        formatCode:program.programFormatCode
        scratchByteLength:program.scratchByteLength
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSData *imageData = [HWXObjectWriter
        buildObjectWithTaskDescriptor:program.data
        constantRegion:[NSData data] bindings:layoutBindings
        kernelRelocationOffsets:program.kernelRelocationOffsets
        programInfo:info error:&error];
    HWXImage *image = [HWXImage imageWithData:imageData error:&error];
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)imageData.bytes;
    const struct section_64 *text = textSection(imageData);
    expect(image != nil && error == nil && imageData.length == 0xc000,
           @"constant-free layout object reparses with the measured file envelope");
    expect(header->ncmds == 12 && segmentExists(imageData,"__DATA") &&
           !segmentExists(imageData,"__KERN_0"),
           @"layout object carries scratch but no invented kernel segment");
    expect(text && text->offset == 0x4000 && text->flags == 0x28 &&
           text->nreloc == 0,
           @"constant-free TD text uses the measured no-relocation section form");
    expect(programDescriptorWord(imageData,0x830) == 3 &&
           programDescriptorWord(imageData,0x858) == 0x1000000 &&
           programDescriptorWord(imageData,0x090) == 0 &&
           programDescriptorWord(imageData,0x094) == 0,
           @"layout metadata records three tasks, scratch size, and no kernel address");
}

static void testScratchAllocationIsIndependentFromUsableScratch(void) {
    NSMutableData *td = [NSMutableData dataWithLength:0x8ac];
    const uint32_t addends[] = {0, 0x2000, 0x4000, 0x6000};
    const NSUInteger relocations[] = {0x41c, 0x594, 0x704, 0x890};
    for (NSUInteger i = 0; i < 4; ++i)
        [td replaceBytesInRange:NSMakeRange(relocations[i], 4)
                      withBytes:&addends[i]];
    NSMutableData *constants = [NSMutableData dataWithLength:0x8000];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:7 recordCount:62 formatCode:0x8a
        scratchByteLength:0x800000
        scratchAllocationByteLength:0x900000
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSError *error = nil;
    NSData *imageData = [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:constants bindings:bindings()
        kernelRelocationOffsets:@[@0x41c,@0x594,@0x704,@0x890]
        programInfo:info error:&error];
    expect(imageData != nil && error == nil,
           @"fused object accepts distinct allocated and usable scratch sizes");
    expect(segmentVMSize(imageData, "__DATA") == 0x900000 &&
           programDescriptorWord(imageData, 0x858) == 0x800000,
           @"Mach-O BSS allocation and program scratch field remain independent");
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)imageData.bytes;
    expect(header->sizeofcmds < 0x8000 - sizeof(*header),
           @"kernel symbols follow referenced tiles rather than every 512-byte chunk");
}

static void testThreeSurfaceMatmulEnvelopeIsGenerated(void) {
    NSMutableData *td = [NSMutableData dataWithLength:0x6d4];
    NSArray<HWXObjectBinding *> *matmulBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:@"a" shortName:@"a"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@768,@768]],
        [[HWXObjectBinding alloc] initWithSymbol:@"b" shortName:@"b"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@768,@768]],
        [[HWXObjectBinding alloc] initWithSymbol:@"y@output" shortName:@"y"
            role:HWXObjectBindingRoleOutput elementType:ANEElementTypeFP16
            shape:@[@1,@768,@768]],
    ];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:4 recordCount:46 formatCode:0
        scratchByteLength:0x1e0000
        scratchAllocationByteLength:0x120000
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSError *error = nil;
    NSData *imageData = [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:[NSData data] bindings:matmulBindings
        kernelRelocationOffsets:@[] programInfo:info error:&error];
    HWXImage *image = [HWXImage imageWithData:imageData error:&error];
    const struct mach_header_64 *header = imageData.length
        ? (const struct mach_header_64 *)imageData.bytes : NULL;
    const struct section_64 *text = imageData.length
        ? textSection(imageData) : NULL;
    NSArray<NSNumber *> *surfaceAddresses = imageData.length
        ? segmentVMAddresses(imageData, "__FVMLIB") : @[];
    expect(image != nil && error == nil,
           @"clean writer accepts two inputs and one output without a kernel");
    expect(header && header->ncmds == 15 &&
           commandCount(imageData, 0x40) == 3 &&
           surfaceAddresses.count == 3,
           @"matmul envelope has three surface segments and references");
    expect(text && text->offset == 0x8000 && text->nreloc == 0 &&
           !segmentExists(imageData, "__KERN_0"),
           @"matmul TD uses the measured 0x8000 text envelope with no kernel");
    expect([tensorSymbols(imageData) isEqualToArray:
            @[@"a", @"b", @"y@output"]] &&
           [tensorBindingIndices(imageData) isEqualToArray:@[@1,@1,@2]] &&
           [tensorShapes(imageData) isEqualToArray:@[
                @[@1,@1,@768,@768],@[@1,@1,@768,@768],
                @[@1,@1,@768,@768]]],
           @"matmul metadata exposes both inputs and the output semantically");
    if (surfaceAddresses.count == 3) {
        expect(programDescriptorQWord(imageData, 0x70) ==
                   surfaceAddresses[0].unsignedLongLongValue &&
               programDescriptorQWord(imageData, 0x80) ==
                   surfaceAddresses[1].unsignedLongLongValue &&
               programDescriptorQWord(imageData, 0x90) ==
                   surfaceAddresses[2].unsignedLongLongValue,
               @"program slots 0x70/0x80/0x90 bind A, B, and output");
    }
    expect(programDescriptorWord(imageData, 0x830) == 4 &&
           programDescriptorWord(imageData, 0x858) == 0x1e0000 &&
           programDescriptorWord(imageData, 0x860) == 46 &&
           programDescriptorWord(imageData, 0x8b0) == 0 &&
           segmentVMSize(imageData,"__DATA") == 0x120000,
           @"matmul metadata separates its working set from BSS allocation");
}

static void testBindingCanSeparateSemanticShapeFromPhysicalStorage(void) {
    NSMutableData *td = [NSMutableData dataWithLength:0x100];
    NSArray<HWXObjectBinding *> *paddedBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:@"x" shortName:@"x"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@32,@8,@8]],
        [[HWXObjectBinding alloc] initWithSymbol:@"y@output" shortName:@"y"
            role:HWXObjectBindingRoleOutput elementType:ANEElementTypeFP16
            shape:@[@1,@1,@8,@8] rowStrideBytes:64 planeStrideBytes:512
            batchStrideBytes:512 storageByteLength:0x5000],
    ];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:1 recordCount:16 formatCode:0 scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    NSError *error = nil;
    NSData *imageData = [HWXObjectWriter
        buildObjectWithTaskDescriptor:td constantRegion:[NSData data]
        bindings:paddedBindings kernelRelocationOffsets:@[] programInfo:info
        error:&error];
    NSArray<NSArray<NSNumber *> *> *layouts = imageData.length
        ? tensorPhysicalLayouts(imageData) : @[];
    NSArray<NSNumber *> *surfaceSizes = imageData.length
        ? segmentVMSizes(imageData, "__FVMLIB") : @[];
    expect(imageData != nil && error == nil,
           @"writer accepts a measured padded physical output layout");
    expect([tensorShapes(imageData) isEqualToArray:@[
                @[@1,@32,@8,@8], @[@1,@1,@8,@8]]],
           @"physical padding does not alter the semantic tensor shape");
    expect(layouts.count == 2 &&
           [layouts[1] isEqualToArray:@[@64,@512,@512,@512]],
           @"H16 tensor span equals one batch stride independently of allocation padding");
    expect(surfaceSizes.count == 2 && surfaceSizes[1].unsignedLongLongValue == 0x8000,
           @"padded storage participates in independent surface allocation");
}

static void testThreeSurfaceProgramCanReferenceKernelTable(void) {
    NSMutableData *td = [NSMutableData dataWithLength:0x1ec];
    NSMutableData *table = [NSMutableData dataWithLength:128];
    NSArray<HWXObjectBinding *> *threeBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:@"denominator"
            shortName:@"denominator" role:HWXObjectBindingRoleInput
            elementType:ANEElementTypeFP16 shape:@[@1,@1,@128,@1]
            rowStrideBytes:64 planeStrideBytes:8192
            batchStrideBytes:8192 storageByteLength:8192],
        [[HWXObjectBinding alloc] initWithSymbol:@"numerator"
            shortName:@"numerator" role:HWXObjectBindingRoleInput
            elementType:ANEElementTypeFP16 shape:@[@1,@1,@128,@128]],
        [[HWXObjectBinding alloc] initWithSymbol:@"normalized@output"
            shortName:@"normalized" role:HWXObjectBindingRoleOutput
            elementType:ANEElementTypeFP16 shape:@[@1,@1,@128,@128]],
    ];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:2 recordCount:31 formatCode:0 scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    NSError *error = nil;
    NSData *imageData = [HWXObjectWriter
        buildObjectWithTaskDescriptor:td constantRegion:table
        bindings:threeBindings kernelRelocationOffsets:@[@0xd0]
        programInfo:info error:&error];
    NSArray<NSNumber *> *surfaces = imageData.length
        ? segmentVMAddresses(imageData, "__FVMLIB") : @[];
    NSArray<NSNumber *> *kernels = imageData.length
        ? segmentVMAddresses(imageData, "__KERN_0") : @[];
    expect(imageData != nil && error == nil && surfaces.count == 3 &&
           kernels.count == 1,
           @"writer accepts two inputs, one output, and one kernel table");
    if (surfaces.count == 3 && kernels.count == 1)
        expect(programDescriptorQWord(imageData, 0x70) ==
                   surfaces[0].unsignedLongLongValue &&
               programDescriptorQWord(imageData, 0x80) ==
                   surfaces[1].unsignedLongLongValue &&
               programDescriptorQWord(imageData, 0x90) ==
                   surfaces[2].unsignedLongLongValue &&
               programDescriptorQWord(imageData, 0xa0) ==
                   kernels[0].unsignedLongLongValue,
               @"program resource slots retain all surfaces before the kernel");
}

static void testFourSurfaceScratchProgramUsesMeasuredResourceLayout(void) {
    NSMutableData *td = [NSMutableData dataWithLength:0x7d4];
    NSMutableData *table = [NSMutableData dataWithLength:128];
    NSArray<HWXObjectBinding *> *fourBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:@"q" shortName:@"q"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@1,@128,@128]],
        [[HWXObjectBinding alloc] initWithSymbol:@"k" shortName:@"k"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@1,@128,@128]],
        [[HWXObjectBinding alloc] initWithSymbol:@"v" shortName:@"v"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@1,@128,@128]],
        [[HWXObjectBinding alloc] initWithSymbol:@"y@output" shortName:@"y"
            role:HWXObjectBindingRoleOutput elementType:ANEElementTypeFP16
            shape:@[@1,@1,@128,@128]],
    ];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:8 recordCount:115 formatCode:1
        scratchByteLength:32768 scratchAllocationByteLength:32768
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSError *error = nil;
    NSData *imageData = [HWXObjectWriter
        buildObjectWithTaskDescriptor:td constantRegion:table
        bindings:fourBindings kernelRelocationOffsets:@[@0x48c]
        programInfo:info error:&error];
    NSArray<NSNumber *> *surfaces = imageData.length
        ? segmentVMAddresses(imageData, "__FVMLIB") : @[];
    NSArray<NSNumber *> *kernels = imageData.length
        ? segmentVMAddresses(imageData, "__KERN_0") : @[];
    const struct section_64 *text = imageData.length
        ? textSection(imageData) : NULL;
    expect(imageData != nil && error == nil && surfaces.count == 4 &&
           kernels.count == 1,
           @"writer accepts three inputs, one output, and one kernel table");
    if (surfaces.count == 4 && kernels.count == 1)
        expect(programDescriptorQWord(imageData, 0x70) ==
                   surfaces[0].unsignedLongLongValue &&
               programDescriptorQWord(imageData, 0x80) ==
                   surfaces[1].unsignedLongLongValue &&
               programDescriptorQWord(imageData, 0x90) ==
                   surfaces[2].unsignedLongLongValue &&
               programDescriptorQWord(imageData, 0xa0) ==
                   surfaces[3].unsignedLongLongValue &&
               programDescriptorQWord(imageData, 0xb0) ==
                   kernels[0].unsignedLongLongValue,
               @"program descriptor records all four surfaces before the kernel");
    if (text && text->nreloc == 1) {
        const uint32_t *relocation = (const uint32_t *)
            ((const uint8_t *)imageData.bytes + text->reloff);
        expect(text->offset == 0x8000 && relocation[0] == 0x48c &&
               relocation[1] == 0x07000008,
               @"four-surface relocation targets the measured kernel section");
    }
}

static void testWriterRejectsUnrepresentableResourcesAndRelocations(void) {
    NSMutableArray<HWXObjectBinding *> *fiveBindings = [NSMutableArray array];
    for (NSUInteger index = 0; index < 4; ++index)
        [fiveBindings addObject:[[HWXObjectBinding alloc]
            initWithSymbol:[NSString stringWithFormat:@"i%lu",
                (unsigned long)index]
            shortName:[NSString stringWithFormat:@"i%lu",
                (unsigned long)index]
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@1,@8,@8]]];
    [fiveBindings addObject:[[HWXObjectBinding alloc]
        initWithSymbol:@"y@output" shortName:@"y"
        role:HWXObjectBindingRoleOutput elementType:ANEElementTypeFP16
        shape:@[@1,@1,@8,@8]]];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:1 recordCount:1 formatCode:0 scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    NSError *error = nil;
    expect([HWXObjectWriter
        buildObjectWithTaskDescriptor:[NSMutableData dataWithLength:0x100]
        constantRegion:[NSData data] bindings:fiveBindings
        kernelRelocationOffsets:@[] programInfo:info error:&error] == nil &&
        error != nil, @"writer rejects a fifth surface");

    NSMutableData *td = [NSMutableData dataWithLength:0x200];
    uint32_t outOfRangeAddend = 0x80;
    [td replaceBytesInRange:NSMakeRange(0x40, sizeof(outOfRangeAddend))
                  withBytes:&outOfRangeAddend];
    error = nil;
    expect([HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:[NSMutableData dataWithLength:0x80] bindings:bindings()
        kernelRelocationOffsets:@[@0x40] programInfo:info error:&error] == nil &&
        error != nil, @"writer rejects a relocation addend past its kernel table");

    HWXObjectBinding *validOutput = bindings()[1];
    for (NSArray<NSNumber *> *shape in @[@[], @[@1,@1,@1,@1,@1]]) {
        HWXObjectBinding *badRank = [[HWXObjectBinding alloc]
            initWithSymbol:@"x" shortName:@"x"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:shape rowStrideBytes:64 planeStrideBytes:64
            batchStrideBytes:64 storageByteLength:64];
        error = nil;
        expect([HWXObjectWriter
            buildObjectWithTaskDescriptor:[NSMutableData dataWithLength:0x100]
            constantRegion:[NSData data] bindings:@[badRank, validOutput]
            kernelRelocationOffsets:@[] programInfo:info error:&error] == nil &&
            error != nil, @"writer rejects a binding rank outside one through four");
    }

    NSNumber *largest = [NSNumber numberWithUnsignedLongLong:UINT64_MAX];
    HWXObjectBinding *overflowShape = [[HWXObjectBinding alloc]
        initWithSymbol:@"x" shortName:@"x"
        role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
        shape:@[largest] rowStrideBytes:NSUIntegerMax
        planeStrideBytes:NSUIntegerMax batchStrideBytes:NSUIntegerMax
        storageByteLength:NSUIntegerMax];
    error = nil;
    expect([HWXObjectWriter
        buildObjectWithTaskDescriptor:[NSMutableData dataWithLength:0x100]
        constantRegion:[NSData data] bindings:@[overflowShape, validOutput]
        kernelRelocationOffsets:@[] programInfo:info error:&error] == nil &&
        error != nil, @"writer rejects an overflowing tensor shape");

    HWXObjectBinding *overflowVM = [[HWXObjectBinding alloc]
        initWithSymbol:@"x" shortName:@"x"
        role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
        shape:@[@1] rowStrideBytes:2 planeStrideBytes:2
        batchStrideBytes:2 storageByteLength:NSUIntegerMax];
    error = nil;
    expect([HWXObjectWriter
        buildObjectWithTaskDescriptor:[NSMutableData dataWithLength:0x100]
        constantRegion:[NSData data] bindings:@[overflowVM, validOutput]
        kernelRelocationOffsets:@[] programInfo:info error:&error] == nil &&
        error != nil, @"writer rejects an overflowing surface VM layout");
}

static NSData *smallRelocatableObject(void) {
    NSMutableData *td = [NSMutableData dataWithLength:0x200];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:1 recordCount:1 formatCode:0 scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    return [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:[NSMutableData dataWithLength:0x80] bindings:bindings()
        kernelRelocationOffsets:@[@0x40] programInfo:info error:nil];
}

static void testH16BindingWidthsAndIdentities(void) {
    NSData *td = [NSMutableData dataWithLength:0x100];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:1 recordCount:16 formatCode:0 scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    HWXObjectBinding *out = [[HWXObjectBinding alloc]
        initWithSymbol:@"y@output" shortName:@"y"
        role:HWXObjectBindingRoleOutput elementType:ANEElementTypeFP16
        shape:@[@3,@1,@8,@8]];
    HWXObjectBinding *input = [[HWXObjectBinding alloc]
        initWithSymbol:@"camera_frame" shortName:@"camera_frame"
        role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
        shape:@[@3,@1,@8,@8]];
    NSError *error = nil;
    NSData *image = [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:[NSData data] bindings:@[input, out]
        kernelRelocationOffsets:@[] programInfo:info error:&error];
    expect(image != nil && error == nil,
           @"H16 accepts long semantic identifiers and batched allocations");
    if (image) {
        const uint8_t *bytes = (const uint8_t *)image.bytes;
        const struct mach_header_64 *header = (const struct mach_header_64 *)bytes;
        NSUInteger pos = sizeof(*header);
        BOOL shortReference = NO;
        for (uint32_t i = 0; i < header->ncmds; ++i) {
            const struct load_command *c = (const struct load_command *)(bytes + pos);
            if (c->cmd == 0x40 && !shortReference) {
                expect(memcmp(bytes + pos + 0x18, "_io0\0\0\0\0", 8) == 0,
                       @"long names get a NUL-terminated local reference without truncation");
                shortReference = YES;
            }
            pos += c->cmdsize;
        }
        NSArray *layouts = tensorPhysicalLayouts(image);
        expect(layouts.count == 2 && [layouts[0] isEqualToArray:@[@16,@128,@128,@128]],
               @"H16 descriptor records one batch while allocation covers three");
        expect([tensorSymbols(image)[0] isEqualToString:@"camera_frame"],
               @"short references preserve the semantic symbol");
    }
    for (HWXObjectBinding *bad in @[
        [[HWXObjectBinding alloc] initWithSymbol:@"y@output" shortName:@"x"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16 shape:@[@1]],
        [[HWXObjectBinding alloc] initWithSymbol:@"x" shortName:@"y"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16 shape:@[@1]],
        [[HWXObjectBinding alloc] initWithSymbol:@"x" shortName:@"x"
            role:HWXObjectBindingRoleInput elementType:ANEElementTypeFP16
            shape:@[@1,@1,@1,@0x100000000ULL] rowStrideBytes:0x200000000ULL
            planeStrideBytes:0x200000000ULL batchStrideBytes:0x200000000ULL
            storageByteLength:0x200000000ULL],
    ]) {
        error = nil;
        expect([HWXObjectWriter buildObjectWithTaskDescriptor:td
            constantRegion:[NSData data] bindings:@[bad, out]
            kernelRelocationOffsets:@[] programInfo:info error:&error] == nil && error != nil,
               @"H16 rejects duplicate identities and dimensions that truncate to u32");
    }
    error = nil;
    expect([HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:[NSMutableData dataWithLength:16] bindings:bindings()
        kernelRelocationOffsets:@[@0x40, @0x40] programInfo:info error:&error] == nil && error != nil,
           @"H16 rejects duplicate relocation patch sites");
}

static void testParserRejectsInvalidSymbolAndRelocationTables(void) {
    NSData *valid = smallRelocatableObject();
    NSError *error = nil;
    expect(valid != nil && [HWXImage imageWithData:valid error:&error] != nil,
           @"parser validation test starts from a valid object");

    NSMutableData *badSymbols = [valid mutableCopy];
    struct mach_header_64 *header =
        (struct mach_header_64 *)badSymbols.mutableBytes;
    NSUInteger commandOffset = sizeof(*header);
    for (uint32_t index = 0; index < header->ncmds; ++index) {
        struct load_command *command = (struct load_command *)
            ((uint8_t *)badSymbols.mutableBytes + commandOffset);
        if (command->cmd == LC_SYMTAB) {
            ((struct symtab_command *)command)->symoff =
                (uint32_t)badSymbols.length;
            break;
        }
        commandOffset += command->cmdsize;
    }
    error = nil;
    expect([HWXImage imageWithData:badSymbols error:&error] == nil &&
           error != nil, @"parser rejects a symbol table outside the file");

    NSMutableData *badRelocationEnvelope = [valid mutableCopy];
    struct section_64 *badText = (struct section_64 *)
        textSection(badRelocationEnvelope);
    if (badText) badText->reloff = (uint32_t)badRelocationEnvelope.length - 4;
    error = nil;
    expect(badText != NULL &&
           [HWXImage imageWithData:badRelocationEnvelope error:&error] == nil &&
           error != nil, @"parser rejects a relocation table outside the file");

    NSMutableData *badRelocation = [valid mutableCopy];
    const struct section_64 *text = textSection(badRelocation);
    if (text && text->nreloc == 1) {
        uint32_t invalidTarget = 0x070000ff;
        [badRelocation replaceBytesInRange:
            NSMakeRange(text->reloff + sizeof(uint32_t), sizeof(invalidTarget))
                                  withBytes:&invalidTarget];
    }
    error = nil;
    expect(text != NULL && [HWXImage imageWithData:badRelocation
                                             error:&error] == nil &&
           error != nil, @"parser rejects an invalid relocation section ordinal");
}

static void testConvolutionTextGrowsBeyondCorpusCap(void) {
    // Larger than the old 0x60000 corpus-based policy cap. No executable
    // meaning is claimed for these zero bytes: this checks container bounds.
    NSData *task = [NSMutableData dataWithLength:0x60004];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:1 firstTaskByteLength:task.length recordCount:16
        formatCode:0 scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    info.kernelRegionConstants = YES;
    for (NSNumber *architecture in @[@(HWXObjectArchitectureH16G), @(HWXObjectArchitectureH17)]) {
        NSError *error = nil;
        NSData *image = [HWXObjectWriter buildObjectForArchitecture:
            (HWXObjectArchitecture)architecture.unsignedIntegerValue
            taskDescriptor:task constantRegion:[NSMutableData dataWithLength:64]
            bindings:bindings() kernelRelocationOffsets:@[] programInfo:info error:&error];
        expect(image != nil && error == nil,
            @"convolution container accepts a stream above the old corpus cap");
        const struct section_64 *text = image ? textSection(image) : NULL;
        expect(text != NULL && text->size == task.length,
            @"large convolution text retains its complete stream");
        if (image) expect([HWXImage imageWithData:image error:&error] != nil,
            @"large convolution text and zero const area fit their segment");
    }
    NSError *error = nil;
    expect([HWXObjectWriter buildObjectForArchitecture:HWXObjectArchitectureH13
        taskDescriptor:task constantRegion:[NSMutableData dataWithLength:64]
        bindings:bindings() kernelRelocationOffsets:@[] programInfo:info error:&error] == nil,
        @"H16/H17 convolution kernel placement cannot be selected for H13");
}

int main(void) {
    @autoreleasepool {
        testObjectIsBuiltFromInputs();
        testEmissionIsDeterministicAndDataDependent();
        testMultiTaskObjectDerivesLayoutAndRelocations();
        testScratchBackedMixedProgramHasIndependentEnvelopeMetadata();
        testScratchBackedLayoutProgramNeedsNoKernelSegment();
        testScratchAllocationIsIndependentFromUsableScratch();
        testThreeSurfaceMatmulEnvelopeIsGenerated();
        testBindingCanSeparateSemanticShapeFromPhysicalStorage();
        testThreeSurfaceProgramCanReferenceKernelTable();
        testFourSurfaceScratchProgramUsesMeasuredResourceLayout();
        testWriterRejectsUnrepresentableResourcesAndRelocations();
        testParserRejectsInvalidSymbolAndRelocationTables();
        testConvolutionTextGrowsBeyondCorpusCap();
        testH16BindingWidthsAndIdentities();
        printf("HWX object writer: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
