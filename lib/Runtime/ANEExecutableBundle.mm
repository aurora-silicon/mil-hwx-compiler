#import "ANEExecutableBundle.h"

#import "HWXImage.h"
#import "HWXMachOFormat.h"

static NSString *const ANEBundleErrorDomain = @"dev.maderix.ANEBundle";

static ANEExecutableBundle *bundleFailure(NSError **error, NSInteger code,
                                          NSString *message) {
    if (error)
        *error = [NSError errorWithDomain:ANEBundleErrorDomain code:code
            userInfo:@{NSLocalizedDescriptionKey: message}];
    return nil;
}

static NSString *roleName(ANESurfaceRole role) {
    switch (role) {
        case ANESurfaceRoleInput: return @"input";
        case ANESurfaceRoleWeight: return @"constant";
        case ANESurfaceRoleIntermediate: return @"intermediate";
        case ANESurfaceRoleOutput: return @"output";
    }
}

static BOOL parseRole(NSString *name, ANESurfaceRole *role) {
    if ([name isEqualToString:@"input"]) *role = ANESurfaceRoleInput;
    else if ([name isEqualToString:@"constant"]) *role = ANESurfaceRoleWeight;
    else if ([name isEqualToString:@"intermediate"])
        *role = ANESurfaceRoleIntermediate;
    else if ([name isEqualToString:@"output"]) *role = ANESurfaceRoleOutput;
    else return NO;
    return YES;
}

static BOOL unsignedValue(id object, NSUInteger *value) {
    if (![object isKindOfClass:[NSNumber class]]) return NO;
    long long signedValue = [object longLongValue];
    unsigned long long raw = [object unsignedLongLongValue];
    if (signedValue < 0 || raw > NSUIntegerMax) return NO;
    *value = (NSUInteger)raw;
    return YES;
}

static NSArray<NSString *> *sharedSurfaceIdentifiers(
    NSArray<ANEHWXArtifact *> *artifacts) {
    NSMutableSet<NSString *> *inputs = [NSMutableSet set];
    NSMutableSet<NSString *> *outputs = [NSMutableSet set];
    NSMutableArray<NSString *> *outputOrder = [NSMutableArray array];
    for (ANEHWXArtifact *artifact in artifacts) {
        for (ANEHWXBinding *binding in artifact.bindings) {
            if (binding.role == ANESurfaceRoleInput)
                [inputs addObject:binding.identifier];
            if (binding.role == ANESurfaceRoleOutput) {
                [outputs addObject:binding.identifier];
                if (![outputOrder containsObject:binding.identifier])
                    [outputOrder addObject:binding.identifier];
            }
        }
    }
    NSMutableArray<NSString *> *shared = [NSMutableArray array];
    for (NSString *identifier in outputOrder)
        if ([inputs containsObject:identifier] &&
            [outputs containsObject:identifier])
            [shared addObject:identifier];
    return [shared copy];
}

// Compile-time geometry only. The consumer must cross-check this against the
// immutable HWX and supply owned mappings; VM aperture bases are never DVAs.
static NSDictionary *h16ExecuteResources(NSData *image) {
    // Parse the actual section layout: aperture order can interleave input and
    // output surfaces, independently of the runtime binding array's order.
    if (![HWXImage imageWithData:image error:nil]) return nil;
    struct mach_header_64 header;
    memcpy(&header, image.bytes, sizeof(header));
    if (header.cpusubtype != 7) return nil;
    const uint8_t *bytes = (const uint8_t *)image.bytes;
    NSUInteger offset = sizeof(header), inputIndex = 0, outputIndex = 0;
    NSMutableArray *buffers = [NSMutableArray array];
    for (NSUInteger i = 0; i < header.ncmds; ++i) {
        struct load_command command;
        memcpy(&command, bytes + offset, sizeof(command));
        if (command.cmd == LC_SEGMENT_64) {
            struct segment_command_64 segment;
            memcpy(&segment, bytes + offset, sizeof(segment));
            if (strncmp(segment.segname, "__FVMLIB", 16) == 0) {
                if (segment.initprot != 1 && segment.initprot != 2) return nil;
                BOOL output = segment.initprot == 2;
                [buffers addObject:@{
                    @"apertureSlot": @(buffers.count),
                    @"index": @(output ? outputIndex++ : inputIndex++),
                    @"class": @(output ? 1 : 0), @"offsetBytes": @0,
                    @"sizeBytes": @(segment.vmsize), @"mappingBytes": @(segment.vmsize),
                }];
            }
        }
        offset += command.cmdsize;
    }
    // The current writer emits no custom-BAR descriptions. Never reinterpret
    // aperture records, embedded kernels or scratch as custom BAR records.
    return @{@"version": @1, @"commandBytes": @0xca8,
             @"bufferOffset": @0x60, @"bufferStride": @0x30,
             @"customBarOffset": @0xca8, @"customBarStride": @0x18,
             @"buffers": buffers, @"customBars": @[]};
}

@implementation ANEExecutableBundle
- (instancetype)initWithTarget:(NSString *)target
                      artifacts:(NSArray<ANEHWXArtifact *> *)artifacts
                   dispatchPlan:(NSArray<NSNumber *> *)dispatchPlan
                      passTrace:(NSArray<NSString *> *)passTrace {
    return [self initWithTarget:target artifacts:artifacts
                   dispatchPlan:dispatchPlan passTrace:passTrace
               compositionTrace:@[]];
}

- (instancetype)initWithTarget:(NSString *)target
                      artifacts:(NSArray<ANEHWXArtifact *> *)artifacts
                   dispatchPlan:(NSArray<NSNumber *> *)dispatchPlan
                      passTrace:(NSArray<NSString *> *)passTrace
               compositionTrace:(NSArray<NSString *> *)compositionTrace {
    self = [super init];
    if (self) {
        _target = [target copy];
        _artifacts = [artifacts copy];
        _dispatchPlan = [dispatchPlan copy];
        _sharedSurfaceIdentifiers = sharedSurfaceIdentifiers(_artifacts);
        _passTrace = [passTrace copy];
        _compositionTrace = [compositionTrace copy];
    }
    return self;
}

- (BOOL)writeToDirectory:(NSURL *)directory error:(NSError **)error {
    NSFileManager *fileManager = NSFileManager.defaultManager;
    if (![fileManager createDirectoryAtURL:directory
               withIntermediateDirectories:YES attributes:nil error:error])
        return NO;
    NSMutableArray *artifactRecords = [NSMutableArray array];
    for (NSUInteger index = 0; index < self.artifacts.count; ++index) {
        ANEHWXArtifact *artifact = self.artifacts[index];
        NSString *fileName = [NSString stringWithFormat:@"program-%lu.hwx",
                                                       (unsigned long)index];
        NSURL *fileURL = [directory URLByAppendingPathComponent:fileName];
        if (![artifact.image writeToURL:fileURL options:NSDataWritingAtomic
                                   error:error]) return NO;
        NSMutableArray *bindings = [NSMutableArray array];
        for (ANEHWXBinding *binding in artifact.bindings) {
            [bindings addObject:@{
                @"identifier": binding.identifier,
                @"role": roleName(binding.role),
                @"logicalBytes": @(binding.logicalByteLength),
                @"allocationBytes": @(binding.allocationByteLength),
                @"ioSurfaceIndex": @(binding.ioSurfaceIndex),
                @"rowStrideBytes": @(binding.rowStrideBytes),
                @"planeStrideBytes": @(binding.planeStrideBytes),
                @"batchStrideBytes": @(binding.batchStrideBytes),
            }];
        }
        NSMutableDictionary *record = [@{
            @"file": fileName,
            @"bytes": @(artifact.image.length),
            @"bindings": bindings,
            @"operations": artifact.operations,
        } mutableCopy];
        if ([self.target isEqualToString:@"H16G"]) {
            NSDictionary *resources = h16ExecuteResources(artifact.image);
            if (resources) record[@"h16Execute"] = resources;
        }
        [artifactRecords addObject:record];
    }
    NSDictionary *manifest = @{
        @"formatVersion": @2,
        @"target": self.target,
        @"artifacts": artifactRecords,
        @"dispatchPlan": self.dispatchPlan,
        @"sharedSurfaceIdentifiers": self.sharedSurfaceIdentifiers,
        @"passTrace": self.passTrace,
        @"compositionTrace": self.compositionTrace,
    };
    NSData *manifestData = [NSJSONSerialization dataWithJSONObject:manifest
        options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
        error:error];
    return manifestData && [manifestData writeToURL:
        [directory URLByAppendingPathComponent:@"manifest.json"]
        options:NSDataWritingAtomic error:error];
}

+ (instancetype)bundleWithContentsOfDirectory:(NSURL *)directory
                                          error:(NSError **)error {
    NSURL *manifestURL =
        [directory URLByAppendingPathComponent:@"manifest.json"];
    NSData *manifestData = [NSData dataWithContentsOfURL:manifestURL
                                                options:0 error:error];
    if (!manifestData) return nil;
    id root = [NSJSONSerialization JSONObjectWithData:manifestData
                                               options:0 error:error];
    if (!root) return nil;
    if (![root isKindOfClass:[NSDictionary class]])
        return bundleFailure(error, 1, @"bundle manifest must be an object");
    NSDictionary *manifest = (NSDictionary *)root;
    BOOL version1 = [manifest[@"formatVersion"] isEqual:@1];
    BOOL version2 = [manifest[@"formatVersion"] isEqual:@2];
    if ((!version1 && !version2) ||
        ![manifest[@"target"] isKindOfClass:[NSString class]] ||
        ![manifest[@"artifacts"] isKindOfClass:[NSArray class]] ||
        ![manifest[@"dispatchPlan"] isKindOfClass:[NSArray class]] ||
        (version2 && ![manifest[@"sharedSurfaceIdentifiers"]
                         isKindOfClass:[NSArray class]]) ||
        ![manifest[@"passTrace"] isKindOfClass:[NSArray class]])
        return bundleFailure(error, 2,
            @"bundle manifest has missing or unsupported top-level fields");

    NSArray *records = manifest[@"artifacts"];
    NSMutableArray<ANEHWXArtifact *> *artifacts = [NSMutableArray array];
    for (id rawRecord in records) {
        if (![rawRecord isKindOfClass:[NSDictionary class]])
            return bundleFailure(error, 3,
                                 @"artifact record must be an object");
        NSDictionary *record = (NSDictionary *)rawRecord;
        NSString *fileName = record[@"file"];
        NSArray *rawBindings = record[@"bindings"];
        if (![fileName isKindOfClass:[NSString class]] ||
            ![rawBindings isKindOfClass:[NSArray class]] ||
            fileName.length == 0 ||
            ![fileName.lastPathComponent isEqualToString:fileName])
            return bundleFailure(error, 4,
                @"artifact record has an invalid file or bindings field");
        NSURL *imageURL = [directory URLByAppendingPathComponent:fileName];
        NSData *imageData = [NSData dataWithContentsOfURL:imageURL
                                                  options:0 error:error];
        if (!imageData) return nil;
        NSError *parseError = nil;
        if (![HWXImage imageWithData:imageData error:&parseError]) {
            if (error) *error = parseError;
            return nil;
        }

        NSMutableArray<ANEHWXBinding *> *bindings = [NSMutableArray array];
        for (id rawBinding in rawBindings) {
            if (![rawBinding isKindOfClass:[NSDictionary class]])
                return bundleFailure(error, 5,
                                     @"binding record must be an object");
            NSDictionary *binding = (NSDictionary *)rawBinding;
            NSString *identifier = binding[@"identifier"];
            NSString *rawRole = binding[@"role"];
            ANESurfaceRole role;
            NSUInteger logicalBytes = 0;
            NSUInteger allocationBytes = 0;
            NSUInteger rowStride = 0;
            NSUInteger planeStride = 0;
            NSUInteger batchStride = 0;
            id rawIndex = binding[@"ioSurfaceIndex"];
            if (![identifier isKindOfClass:[NSString class]] ||
                identifier.length == 0 ||
                ![rawRole isKindOfClass:[NSString class]] ||
                !parseRole(rawRole, &role) ||
                !unsignedValue(binding[@"logicalBytes"], &logicalBytes) ||
                !unsignedValue(binding[@"allocationBytes"], &allocationBytes) ||
                !unsignedValue(binding[@"rowStrideBytes"], &rowStride) ||
                !unsignedValue(binding[@"planeStrideBytes"], &planeStride) ||
                !unsignedValue(binding[@"batchStrideBytes"], &batchStride) ||
                ![rawIndex isKindOfClass:[NSNumber class]] ||
                logicalBytes == 0 || allocationBytes < logicalBytes)
                return bundleFailure(error, 6,
                    @"binding record has invalid role, size, stride or index");
            long long index = [rawIndex longLongValue];
            if (index < -1 ||
                (index >= 0 && [rawIndex unsignedLongLongValue] > NSIntegerMax))
                return bundleFailure(error, 7,
                                     @"binding IOSurface index is out of range");
            [bindings addObject:[[ANEHWXBinding alloc]
                initWithIdentifier:identifier role:role
                logicalByteLength:logicalBytes
                allocationByteLength:allocationBytes
                ioSurfaceIndex:(NSInteger)index rowStrideBytes:rowStride
                planeStrideBytes:planeStride batchStrideBytes:batchStride]];
        }
        if (bindings.count == 0)
            return bundleFailure(error, 8,
                                 @"artifact has no binding records");
        NSNumber *recordedBytes = record[@"bytes"];
        NSUInteger byteLength = 0;
        if (!unsignedValue(recordedBytes, &byteLength) ||
            byteLength != imageData.length)
            return bundleFailure(error, 9,
                                 @"artifact byte count does not match the file");
        NSArray *rawOperations = record[@"operations"] ?: @[];
        if (![rawOperations isKindOfClass:[NSArray class]])
            return bundleFailure(error, 16,
                                 @"artifact operations must be an array");
        for (id operation in rawOperations)
            if (![operation isKindOfClass:[NSString class]])
                return bundleFailure(error, 16,
                                     @"artifact operations must be strings");
        [artifacts addObject:[[ANEHWXArtifact alloc] initWithImage:imageData
                                                         bindings:bindings
                                                       operations:rawOperations]];
    }
    if (artifacts.count == 0)
        return bundleFailure(error, 10, @"bundle has no artifacts");

    NSArray *rawPlan = manifest[@"dispatchPlan"];
    NSMutableArray<NSNumber *> *plan = [NSMutableArray array];
    for (id item in rawPlan) {
        NSUInteger index = 0;
        if (!unsignedValue(item, &index) || index >= artifacts.count)
            return bundleFailure(error, 11,
                                 @"dispatch plan references an invalid artifact");
        [plan addObject:@(index)];
    }
    if (plan.count == 0)
        return bundleFailure(error, 12, @"bundle has an empty dispatch plan");

    NSArray *rawTrace = manifest[@"passTrace"];
    for (id item in rawTrace)
        if (![item isKindOfClass:[NSString class]])
            return bundleFailure(error, 13,
                                 @"pass trace entries must be strings");
    NSArray *rawCompositionTrace = manifest[@"compositionTrace"] ?: @[];
    if (![rawCompositionTrace isKindOfClass:[NSArray class]])
        return bundleFailure(error, 17,
                             @"composition trace must be an array");
    for (id item in rawCompositionTrace)
        if (![item isKindOfClass:[NSString class]])
            return bundleFailure(error, 17,
                                 @"composition trace entries must be strings");
    ANEExecutableBundle *bundle = [[self alloc] initWithTarget:manifest[@"target"]
        artifacts:artifacts dispatchPlan:plan passTrace:rawTrace
        compositionTrace:rawCompositionTrace];
    if (version2) {
        NSArray *recordedShared = manifest[@"sharedSurfaceIdentifiers"];
        for (id identifier in recordedShared)
            if (![identifier isKindOfClass:[NSString class]])
                return bundleFailure(error, 14,
                    @"shared surface identifiers must be strings");
        if (![recordedShared isEqualToArray:bundle.sharedSurfaceIdentifiers])
            return bundleFailure(error, 15,
                @"shared surface identifiers do not match artifact bindings");
    }
    return bundle;
}
@end
