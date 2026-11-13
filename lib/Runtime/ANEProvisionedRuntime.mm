#import "ANEProvisionedRuntime.h"

#import "ANEExecutableBundle.h"
#import "ANEHWXArtifact.h"

#import <dlfcn.h>
#import <mach/mach_time.h>
#import <objc/message.h>

#include <cstring>

NSErrorDomain const ANERuntimeErrorDomain = @"dev.maderix.ANERuntime";

static double microsecondsBetween(uint64_t start, uint64_t stop) {
    static mach_timebase_info_data_t timebase = {};
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)(stop - start) * (double)timebase.numer /
           (double)timebase.denom / 1000.0;
}

@interface ANEDispatchProfileEntry ()
- (instancetype)initWithArtifactIndex:(NSUInteger)artifactIndex
                     inputSurfaceCount:(NSUInteger)inputSurfaceCount
                    outputSurfaceCount:(NSUInteger)outputSurfaceCount
               surfaceWrapMicroseconds:(double)surfaceWrapMicroseconds
              requestBuildMicroseconds:(double)requestBuildMicroseconds
                    submitMicroseconds:(double)submitMicroseconds
          hardwareExecutionNanoseconds:(uint64_t)hardwareExecutionNanoseconds;
@end

@implementation ANEDispatchProfileEntry
- (instancetype)initWithArtifactIndex:(NSUInteger)artifactIndex
                     inputSurfaceCount:(NSUInteger)inputSurfaceCount
                    outputSurfaceCount:(NSUInteger)outputSurfaceCount
               surfaceWrapMicroseconds:(double)surfaceWrapMicroseconds
              requestBuildMicroseconds:(double)requestBuildMicroseconds
                    submitMicroseconds:(double)submitMicroseconds
          hardwareExecutionNanoseconds:(uint64_t)hardwareExecutionNanoseconds {
    self = [super init];
    if (self) {
        _artifactIndex = artifactIndex;
        _inputSurfaceCount = inputSurfaceCount;
        _outputSurfaceCount = outputSurfaceCount;
        _surfaceWrapMicroseconds = surfaceWrapMicroseconds;
        _requestBuildMicroseconds = requestBuildMicroseconds;
        _submitMicroseconds = submitMicroseconds;
        _hardwareExecutionNanoseconds = hardwareExecutionNanoseconds;
    }
    return self;
}
@end

@interface ANEEvaluationProfile ()
@property(nonatomic, readwrite, copy) NSArray<ANEDispatchProfileEntry *> *entries;
@property(nonatomic, readwrite) double totalMicroseconds;
@end

@implementation ANEEvaluationProfile
- (instancetype)init {
    self = [super init];
    if (self) _entries = @[];
    return self;
}
@end

static NSError *runtimeError(ANERuntimeError code, NSString *message) {
    return [NSError errorWithDomain:ANERuntimeErrorDomain code:code
        userInfo:@{NSLocalizedDescriptionKey: message}];
}

static BOOL fail(NSError **error, ANERuntimeError code, NSString *message) {
    if (error) *error = runtimeError(code, message);
    return NO;
}

static BOOL validateBindingManifest(NSArray<ANEHWXBinding *> *bindings,
                                    BOOL isH13,
                                    NSError **error) {
    NSMutableSet<NSString *> *identifiers = [NSMutableSet set];
    NSMutableIndexSet *ioIndices = [NSMutableIndexSet indexSet];
    NSUInteger ioBindingCount = 0;
    NSUInteger inputCount = 0;
    NSUInteger outputCount = 0;
    BOOL outputChannel4 = NO;
    BOOL inputChannel5 = NO;
    BOOL inputChannel6 = NO;
    for (ANEHWXBinding *binding in bindings) {
        if (binding.identifier.length == 0 ||
            [identifiers containsObject:binding.identifier])
            return fail(error, ANERuntimeErrorUnsupportedBundle,
                        @"binding identifiers must be nonempty and unique");
        [identifiers addObject:binding.identifier];
        if (binding.logicalByteLength == 0 ||
            binding.allocationByteLength < binding.logicalByteLength)
            return fail(error, ANERuntimeErrorUnsupportedBundle,
                        @"binding allocation must cover a nonempty logical tensor");

        BOOL isIO = binding.role == ANESurfaceRoleInput ||
                    binding.role == ANESurfaceRoleOutput;
        if (!isIO) {
            if (binding.ioSurfaceIndex != -1)
                return fail(error, ANERuntimeErrorUnsupportedBundle,
                            @"non-I/O bindings cannot claim IOSurface indices");
            continue;
        }
        if (binding.ioSurfaceIndex < 0 ||
            [ioIndices containsIndex:(NSUInteger)binding.ioSurfaceIndex])
            return fail(error, ANERuntimeErrorUnsupportedBundle,
                        @"I/O bindings require unique nonnegative IOSurface indices");
        if (binding.role == ANESurfaceRoleInput) {
            ++inputCount;
            inputChannel5 |= binding.ioSurfaceIndex == 5;
            inputChannel6 |= binding.ioSurfaceIndex == 6;
        }
        if (binding.role == ANESurfaceRoleOutput) {
            ++outputCount;
            outputChannel4 |= binding.ioSurfaceIndex == 4;
        }
        [ioIndices addIndex:(NSUInteger)binding.ioSurfaceIndex];
        ++ioBindingCount;
        if (binding.rowStrideBytes == 0 ||
            binding.planeStrideBytes < binding.rowStrideBytes ||
            binding.batchStrideBytes < binding.planeStrideBytes ||
            binding.batchStrideBytes > binding.allocationByteLength)
            return fail(error, ANERuntimeErrorUnsupportedBundle,
                        @"I/O binding strides must be monotonic and cover the logical tensor");
    }
    if (isH13) {
        BOOL channelsMatch = outputCount == 1 && outputChannel4 &&
            inputCount >= 1 && inputCount <= 2 && inputChannel5 &&
            (inputCount == 1 || inputChannel6);
        if (!channelsMatch || ioBindingCount != inputCount + outputCount)
            return fail(error, ANERuntimeErrorUnsupportedBundle,
                        @"H13 requires output channel 4 and input channels 5 then 6");
    } else if (ioBindingCount == 0 || ioIndices.count != ioBindingCount ||
               ioIndices.firstIndex != 0 || ioIndices.lastIndex != ioBindingCount - 1) {
        return fail(error, ANERuntimeErrorUnsupportedBundle,
                    @"I/O binding indices must form one contiguous zero-based range");
    }
    return YES;
}

static BOOL matchingSurfaceLayout(ANEHWXBinding *left,
                                  ANEHWXBinding *right) {
    return left.logicalByteLength == right.logicalByteLength &&
        left.allocationByteLength == right.allocationByteLength &&
        left.rowStrideBytes == right.rowStrideBytes &&
        left.planeStrideBytes == right.planeStrideBytes &&
        left.batchStrideBytes == right.batchStrideBytes;
}

static BOOL validateDispatchBundle(
    ANEExecutableBundle *bundle,
    NSArray<ANEHWXBinding *> **graphInputs,
    NSArray<ANEHWXBinding *> **graphOutputs,
    NSDictionary<NSString *, ANEHWXBinding *> **surfaceBindings,
    NSError **error) {
    BOOL isH13 = [bundle.target isEqualToString:@"H13"];
    if ((!isH13 && ![bundle.target isEqualToString:@"H16G"]) ||
        bundle.artifacts.count == 0 ||
        bundle.dispatchPlan.count != bundle.artifacts.count)
        return fail(error, ANERuntimeErrorUnsupportedBundle,
                    @"dispatch plan must cover every supported artifact exactly once");

    NSMutableDictionary<NSString *, NSNumber *> *producerByIdentifier =
        [NSMutableDictionary dictionary];
    NSMutableSet<NSString *> *consumedIdentifiers = [NSMutableSet set];
    NSMutableDictionary<NSString *, ANEHWXBinding *> *canonicalBindings =
        [NSMutableDictionary dictionary];
    for (NSUInteger artifactIndex = 0;
         artifactIndex < bundle.artifacts.count; ++artifactIndex) {
        ANEHWXArtifact *artifact = bundle.artifacts[artifactIndex];
        if (!validateBindingManifest(artifact.bindings, isH13, error)) return NO;
        for (ANEHWXBinding *binding in artifact.bindings) {
            if (binding.role == ANESurfaceRoleWeight) continue;
            ANEHWXBinding *canonical = canonicalBindings[binding.identifier];
            if (canonical && !matchingSurfaceLayout(canonical, binding))
                return fail(error, ANERuntimeErrorUnsupportedBundle,
                            @"shared surface layouts differ across artifacts");
            if (!canonical) canonicalBindings[binding.identifier] = binding;
            if (binding.role == ANESurfaceRoleInput)
                [consumedIdentifiers addObject:binding.identifier];
            if (binding.role == ANESurfaceRoleOutput) {
                if (producerByIdentifier[binding.identifier])
                    return fail(error, ANERuntimeErrorUnsupportedBundle,
                                @"a surface may have only one producing artifact");
                producerByIdentifier[binding.identifier] = @(artifactIndex);
            }
        }
    }

    NSMutableIndexSet *dispatched = [NSMutableIndexSet indexSet];
    NSMutableSet<NSString *> *available = [NSMutableSet set];
    NSMutableArray<ANEHWXBinding *> *inputs = [NSMutableArray array];
    NSMutableSet<NSString *> *seenInputs = [NSMutableSet set];
    for (ANEHWXArtifact *artifact in bundle.artifacts)
        for (ANEHWXBinding *binding in artifact.bindings)
            if (binding.role == ANESurfaceRoleInput &&
                !producerByIdentifier[binding.identifier] &&
                ![seenInputs containsObject:binding.identifier]) {
                [seenInputs addObject:binding.identifier];
                [available addObject:binding.identifier];
                [inputs addObject:canonicalBindings[binding.identifier]];
            }

    for (NSNumber *rawIndex in bundle.dispatchPlan) {
        NSUInteger artifactIndex = rawIndex.unsignedIntegerValue;
        if (artifactIndex >= bundle.artifacts.count ||
            [dispatched containsIndex:artifactIndex])
            return fail(error, ANERuntimeErrorUnsupportedBundle,
                        @"dispatch plan contains an invalid or repeated artifact");
        ANEHWXArtifact *artifact = bundle.artifacts[artifactIndex];
        for (ANEHWXBinding *binding in artifact.bindings)
            if (binding.role == ANESurfaceRoleInput &&
                ![available containsObject:binding.identifier])
                return fail(error, ANERuntimeErrorUnsupportedBundle,
                            @"dispatch consumes a surface before its producer");
        for (ANEHWXBinding *binding in artifact.bindings)
            if (binding.role == ANESurfaceRoleOutput)
                [available addObject:binding.identifier];
        [dispatched addIndex:artifactIndex];
    }

    NSMutableArray<ANEHWXBinding *> *outputs = [NSMutableArray array];
    for (NSNumber *rawIndex in bundle.dispatchPlan) {
        ANEHWXArtifact *artifact = bundle.artifacts[rawIndex.unsignedIntegerValue];
        for (ANEHWXBinding *binding in artifact.bindings)
            if (binding.role == ANESurfaceRoleOutput &&
                ![consumedIdentifiers containsObject:binding.identifier])
                [outputs addObject:canonicalBindings[binding.identifier]];
    }
    if (inputs.count == 0 || outputs.count == 0)
        return fail(error, ANERuntimeErrorUnsupportedBundle,
                    @"dispatch must expose graph inputs and outputs");
    *graphInputs = [inputs copy];
    *graphOutputs = [outputs copy];
    *surfaceBindings = [canonicalBindings copy];
    return YES;
}

@interface ANEIOSurfaceBuffer ()
- (nullable instancetype)initWithBinding:(ANEHWXBinding *)binding
                                    error:(NSError **)error;
- (BOOL)zeroAllocationWithError:(NSError **)error;
@end

@implementation ANEIOSurfaceBuffer {
    IOSurfaceRef _ioSurface;
}

- (instancetype)initWithBinding:(ANEHWXBinding *)binding
                            error:(NSError **)error {
    self = [super init];
    if (!self) return nil;
    NSDictionary *properties = @{
        (id)kIOSurfaceWidth: @(binding.allocationByteLength),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(binding.allocationByteLength),
        (id)kIOSurfaceAllocSize: @(binding.allocationByteLength),
        (id)kIOSurfacePixelFormat: @0,
    };
    _ioSurface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
    if (!_ioSurface) {
        if (error) *error = runtimeError(ANERuntimeErrorSurfaceAllocation,
            [NSString stringWithFormat:@"cannot allocate IOSurface for '%@'",
                                       binding.identifier]);
        return nil;
    }
    _identifier = [binding.identifier copy];
    _logicalByteLength = binding.logicalByteLength;
    _allocationByteLength = binding.allocationByteLength;
    return self;
}

- (void)dealloc {
    if (_ioSurface) CFRelease(_ioSurface);
}

- (BOOL)zeroAllocationWithError:(NSError **)error {
    IOReturn locked=IOSurfaceLock(_ioSurface,0,nullptr);
    if(locked!=0) {
        if(error) *error=runtimeError(ANERuntimeErrorSurfaceAllocation,
            [NSString stringWithFormat:@"cannot clear IOSurface for '%@'",
                                       self.identifier]);
        return NO;
    }
    std::memset(IOSurfaceGetBaseAddress(_ioSurface),0,
                IOSurfaceGetAllocSize(_ioSurface));
    IOSurfaceUnlock(_ioSurface,0,nullptr);
    return YES;
}

@end

@implementation ANEProvisionedRuntime {
    NSArray<ANEHWXBinding *> *_inputBindings;
    NSArray<ANEHWXBinding *> *_outputBindings;
    NSDictionary<NSString *, ANEHWXBinding *> *_surfaceBindings;
    NSMutableDictionary<NSString *, ANEIOSurfaceBuffer *> *_surfaceBuffers;
    id _client;
    NSMutableArray *_models;
    NSMutableArray<NSDictionary *> *_loadOptions;
}

- (instancetype)initWithBundle:(ANEExecutableBundle *)bundle
                       modelHash:(NSString *)modelHash
                             qos:(unsigned int)qos
                           error:(NSError **)error {
    return [self initWithBundle:bundle modelHashes:@[modelHash ?: @""]
                           qos:qos error:error];
}

- (instancetype)initWithBundle:(ANEExecutableBundle *)bundle
                     modelHashes:(NSArray<NSString *> *)modelHashes
                             qos:(unsigned int)qos
                           error:(NSError **)error {
    if (modelHashes.count != bundle.artifacts.count) {
        fail(error, ANERuntimeErrorUnsupportedBundle,
             @"one provisioned cache identity is required per artifact");
        return nil;
    }
    for (NSString *modelHash in modelHashes)
        if (![modelHash isKindOfClass:[NSString class]] ||
            modelHash.length == 0) {
            fail(error, ANERuntimeErrorUnsupportedBundle,
                 @"provisioned cache identities must not be empty");
            return nil;
        }
    NSArray<ANEHWXBinding *> *inputs = nil;
    NSArray<ANEHWXBinding *> *outputs = nil;
    NSDictionary<NSString *, ANEHWXBinding *> *surfaces = nil;
    if (!validateDispatchBundle(bundle, &inputs, &outputs, &surfaces, error))
        return nil;
    self = [super init];
    if (!self) return nil;
    _bundle = bundle;
    _modelHashes = [modelHashes copy];
    _modelHash = [_modelHashes.firstObject copy];
    _qos = qos;
    _inputBindings = inputs;
    _outputBindings = outputs;
    _surfaceBindings = surfaces;
    _surfaceBuffers = [NSMutableDictionary dictionary];
    _models = [NSMutableArray array];
    _loadOptions = [NSMutableArray array];
    return self;
}

- (BOOL)ensureSurfaceBuffersWithError:(NSError **)error {
    for (NSString *identifier in _surfaceBindings) {
        if (_surfaceBuffers[identifier]) continue;
        ANEIOSurfaceBuffer *buffer = [[ANEIOSurfaceBuffer alloc]
            initWithBinding:_surfaceBindings[identifier] error:error];
        if (!buffer) return NO;
        _surfaceBuffers[identifier] = buffer;
    }
    return YES;
}

- (NSArray<ANEIOSurfaceBuffer *> *)buffersForBindings:
    (NSArray<ANEHWXBinding *> *)bindings error:(NSError **)error {
    if (![self ensureSurfaceBuffersWithError:error]) return nil;
    NSMutableArray<ANEIOSurfaceBuffer *> *buffers = [NSMutableArray array];
    for (ANEHWXBinding *binding in bindings)
        [buffers addObject:_surfaceBuffers[binding.identifier]];
    return [buffers copy];
}

- (NSArray<ANEIOSurfaceBuffer *> *)createInputBuffersWithError:
    (NSError **)error {
    return [self buffersForBindings:_inputBindings error:error];
}

- (NSArray<ANEIOSurfaceBuffer *> *)createOutputBuffersWithError:
    (NSError **)error {
    return [self buffersForBindings:_outputBindings error:error];
}

- (NSArray<ANEIOSurfaceBuffer *> *)surfaceBuffersForArtifactAtIndex:
    (NSUInteger)artifactIndex error:(NSError **)error {
    if (artifactIndex >= _bundle.artifacts.count) {
        fail(error, ANERuntimeErrorUnsupportedBundle,
             @"artifact surface request is out of range");
        return nil;
    }
    NSMutableArray<ANEHWXBinding *> *ioBindings = [NSMutableArray array];
    for (ANEHWXBinding *binding in _bundle.artifacts[artifactIndex].bindings)
        if (binding.role == ANESurfaceRoleInput ||
            binding.role == ANESurfaceRoleOutput)
            [ioBindings addObject:binding];
    [ioBindings sortUsingComparator:^NSComparisonResult(
        ANEHWXBinding *left, ANEHWXBinding *right) {
        if (left.ioSurfaceIndex < right.ioSurfaceIndex) return NSOrderedAscending;
        if (left.ioSurfaceIndex > right.ioSurfaceIndex) return NSOrderedDescending;
        return NSOrderedSame;
    }];
    return [self buffersForBindings:ioBindings error:error];
}

- (void)unloadPartiallyLoadedModels {
    for (NSInteger index = (NSInteger)_models.count - 1; index >= 0; --index) {
        NSError *ignored = nil;
        ((BOOL (*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            _client, @selector(unloadModel:options:qos:error:), _models[index],
            _loadOptions[index], _qos, &ignored);
    }
    [_models removeAllObjects];
    [_loadOptions removeAllObjects];
}

- (BOOL)loadWithError:(NSError **)error {
    if (_loaded) return YES;
    NSString *framework =
        @"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
         "AppleNeuralEngine";
    if (!dlopen(framework.fileSystemRepresentation, RTLD_NOW | RTLD_LOCAL))
        return fail(error, ANERuntimeErrorFrameworkUnavailable,
                    @"AppleNeuralEngine runtime framework is unavailable");

    Class clientClass = NSClassFromString(@"_ANEClient");
    Class modelClass = NSClassFromString(@"_ANEModel");
    if (!clientClass || !modelClass)
        return fail(error, ANERuntimeErrorFrameworkUnavailable,
                    @"required ANE runtime classes are unavailable");
    @try {
        _client = ((id (*)(id, SEL))objc_msgSend)(
            clientClass, @selector(sharedConnection));
        for (NSString *modelHash in _modelHashes) {
            BOOL exists = ((BOOL (*)(id, SEL, id))objc_msgSend)(
                _client, @selector(compiledModelExistsMatchingHash:), modelHash);
            if (!exists) {
                [self unloadPartiallyLoadedModels];
                return fail(error, ANERuntimeErrorCacheMiss,
                    [NSString stringWithFormat:
                        @"no provisioned HWX for cache identity '%@' in this executable namespace",
                        modelHash]);
            }
            NSURL *modelURL = [NSURL fileURLWithPath:modelHash];
            id model = ((id (*)(id, SEL, id, id))objc_msgSend)(
                modelClass, @selector(modelAtURL:key:), modelURL, @"");
            NSDictionary *options = @{
                @"kANEFInMemoryModelIsCachedKey": @YES,
                @"kANEFIsInMemoryModelTypeKey": modelHash,
                @"kANEFModelType": @"kANEFModelMIL",
            };
            NSError *loadError = nil;
            BOOL loaded = ((BOOL (*)(id, SEL, id, id, unsigned int, NSError **))
                           objc_msgSend)(_client,
                @selector(loadModel:options:qos:error:), model, options,
                _qos, &loadError);
            uint64_t handle = ((uint64_t (*)(id, SEL))objc_msgSend)(
                model, @selector(programHandle));
            if (loaded) {
                [_models addObject:model];
                [_loadOptions addObject:options];
            }
            if (!loaded || handle == 0) {
                NSError *failure = loadError ?: runtimeError(
                    ANERuntimeErrorLoadFailed,
                    @"ANE model load returned no handle");
                [self unloadPartiallyLoadedModels];
                if (error) *error = failure;
                return NO;
            }
        }
        _loaded = YES;
        return YES;
    } @catch (NSException *exception) {
        [self unloadPartiallyLoadedModels];
        return fail(error, ANERuntimeErrorPrivateAPIException,
            [NSString stringWithFormat:@"ANE load raised %@: %@",
                                       exception.name, exception.reason]);
    }
}

- (BOOL)validateBuffers:(NSArray<ANEIOSurfaceBuffer *> *)buffers
                bindings:(NSArray<ANEHWXBinding *> *)bindings
                    error:(NSError **)error {
    if (buffers.count != bindings.count)
        return fail(error, ANERuntimeErrorBindingMismatch,
                    @"runtime surface count does not match the binding manifest");
    for (NSUInteger index = 0; index < buffers.count; ++index) {
        ANEIOSurfaceBuffer *buffer = buffers[index];
        ANEHWXBinding *binding = bindings[index];
        if (![buffer.identifier isEqualToString:binding.identifier] ||
            buffer.logicalByteLength != binding.logicalByteLength ||
            buffer.allocationByteLength < binding.allocationByteLength ||
            IOSurfaceGetAllocSize(buffer.ioSurface) <
                binding.allocationByteLength) {
            return fail(error, ANERuntimeErrorBindingMismatch,
                [NSString stringWithFormat:@"surface does not match binding '%@' (%lu bytes)",
                    binding.identifier,
                    (unsigned long)binding.allocationByteLength]);
        }
    }
    return YES;
}

- (BOOL)evaluateInputs:(NSArray<ANEIOSurfaceBuffer *> *)inputs
                outputs:(NSArray<ANEIOSurfaceBuffer *> *)outputs
                  error:(NSError **)error {
    return [self evaluateInputs:inputs outputs:outputs profile:nil
                          error:error];
}

- (BOOL)evaluateInputs:(NSArray<ANEIOSurfaceBuffer *> *)inputs
                outputs:(NSArray<ANEIOSurfaceBuffer *> *)outputs
                profile:(ANEEvaluationProfile *)profile
                  error:(NSError **)error {
    if (profile) {
        profile.entries = @[];
        profile.totalMicroseconds = 0.0;
    }
    if (!_loaded)
        return fail(error, ANERuntimeErrorNotLoaded,
                    @"load the provisioned model before evaluation");
    if (![self validateBuffers:inputs bindings:_inputBindings error:error] ||
        ![self validateBuffers:outputs bindings:_outputBindings error:error])
        return NO;
    if (![self ensureSurfaceBuffersWithError:error]) return NO;
    for (ANEIOSurfaceBuffer *buffer in inputs)
        _surfaceBuffers[buffer.identifier] = buffer;
    for (ANEIOSurfaceBuffer *buffer in outputs)
        _surfaceBuffers[buffer.identifier] = buffer;
    // Programs are allowed to expose a logical tensor inside a larger
    // physical surface.  Clear shared intermediates before every dispatch
    // sequence so padded planes and rows have deterministic zero semantics
    // instead of retaining data from a previous evaluation.
    for(NSString *identifier in _bundle.sharedSurfaceIdentifiers)
        if(![_surfaceBuffers[identifier] zeroAllocationWithError:error])
            return NO;
    Class requestClass = NSClassFromString(@"_ANERequest");
    Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!requestClass || !surfaceClass)
        return fail(error, ANERuntimeErrorFrameworkUnavailable,
                    @"required ANE request classes are unavailable");
    NSMutableArray<ANEDispatchProfileEntry *> *entries =
        profile ? [NSMutableArray array] : nil;
    uint64_t evaluationStarted = profile ? mach_continuous_time() : 0;
    BOOL isH13 = [_bundle.target isEqualToString:@"H13"];
    @try {
        for (NSNumber *rawArtifactIndex in _bundle.dispatchPlan) {
            NSUInteger artifactIndex = rawArtifactIndex.unsignedIntegerValue;
            ANEHWXArtifact *artifact = _bundle.artifacts[artifactIndex];
            uint64_t wrapStarted = profile ? mach_continuous_time() : 0;
            NSMutableArray<ANEHWXBinding *> *inputBindings =
                [NSMutableArray array];
            NSMutableArray<ANEHWXBinding *> *outputBindings =
                [NSMutableArray array];
            for (ANEHWXBinding *binding in artifact.bindings) {
                if (binding.role == ANESurfaceRoleInput)
                    [inputBindings addObject:binding];
                if (binding.role == ANESurfaceRoleOutput)
                    [outputBindings addObject:binding];
            }
            NSComparator byIndex = ^NSComparisonResult(
                ANEHWXBinding *left, ANEHWXBinding *right) {
                if (left.ioSurfaceIndex < right.ioSurfaceIndex)
                    return NSOrderedAscending;
                if (left.ioSurfaceIndex > right.ioSurfaceIndex)
                    return NSOrderedDescending;
                return NSOrderedSame;
            };
            [inputBindings sortUsingComparator:byIndex];
            [outputBindings sortUsingComparator:byIndex];
            NSMutableArray *inputObjects = [NSMutableArray array];
            NSMutableArray *outputObjects = [NSMutableArray array];
            NSMutableArray<NSNumber *> *inputIndices = [NSMutableArray array];
            NSMutableArray<NSNumber *> *outputIndices = [NSMutableArray array];
            for (NSUInteger index = 0; index < inputBindings.count; ++index) {
                ANEIOSurfaceBuffer *buffer =
                    _surfaceBuffers[inputBindings[index].identifier];
                [inputObjects addObject:
                    ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(surfaceClass,
                        @selector(objectWithIOSurface:), buffer.ioSurface)];
                [inputIndices addObject:isH13
                    ? @(inputBindings[index].ioSurfaceIndex) : @(index)];
            }
            for (NSUInteger index = 0; index < outputBindings.count; ++index) {
                ANEIOSurfaceBuffer *buffer =
                    _surfaceBuffers[outputBindings[index].identifier];
                [outputObjects addObject:
                    ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(surfaceClass,
                        @selector(objectWithIOSurface:), buffer.ioSurface)];
                [outputIndices addObject:isH13
                    ? @(outputBindings[index].ioSurfaceIndex) : @(index)];
            }
            uint64_t requestStarted = profile ? mach_continuous_time() : 0;
            id model = _models[artifactIndex];
            if (profile &&
                [model respondsToSelector:@selector(setPerfStatsMask:)])
                ((void (*)(id, SEL, unsigned int))objc_msgSend)(
                    model, @selector(setPerfStatsMask:),
                    profile.performanceStatisticsMask);
            id request =
                ((id (*)(id, SEL, id, id, id, id, id, id, id))objc_msgSend)(
                    requestClass,
                    @selector(requestWithInputs:inputIndices:outputs:
                              outputIndices:weightsBuffer:perfStats:
                              procedureIndex:),
                    inputObjects, inputIndices, outputObjects, outputIndices,
                    nil, nil, @0);
            uint64_t submitStarted = profile ? mach_continuous_time() : 0;
            NSError *evaluateError = nil;
            BOOL evaluated =
                ((BOOL (*)(id, SEL, id, id, id, unsigned int, NSError **))
                 objc_msgSend)(_client,
                    @selector(evaluateWithModel:options:request:qos:error:),
                    model, @{}, request, _qos, &evaluateError);
            uint64_t submitStopped = profile ? mach_continuous_time() : 0;
            if (!evaluated) {
                if (error) *error = evaluateError ?: runtimeError(
                    ANERuntimeErrorEvaluationFailed, @"ANE evaluation failed");
                return NO;
            }
            if (profile) {
                uint64_t hardwareNanoseconds = 0;
                if ([request respondsToSelector:@selector(perfStats)]) {
                    id stats = ((id (*)(id, SEL))objc_msgSend)(
                        request, @selector(perfStats));
                    if (stats && [stats respondsToSelector:
                                  @selector(hwExecutionTime)])
                        hardwareNanoseconds =
                            ((uint64_t (*)(id, SEL))objc_msgSend)(
                                stats, @selector(hwExecutionTime));
                }
                [entries addObject:[[ANEDispatchProfileEntry alloc]
                    initWithArtifactIndex:artifactIndex
                    inputSurfaceCount:inputBindings.count
                    outputSurfaceCount:outputBindings.count
                    surfaceWrapMicroseconds:microsecondsBetween(
                        wrapStarted, requestStarted)
                    requestBuildMicroseconds:microsecondsBetween(
                        requestStarted, submitStarted)
                    submitMicroseconds:microsecondsBetween(
                        submitStarted, submitStopped)
                    hardwareExecutionNanoseconds:hardwareNanoseconds]];
            }
        }
        if (profile) {
            profile.entries = entries;
            profile.totalMicroseconds = microsecondsBetween(
                evaluationStarted, mach_continuous_time());
        }
        return YES;
    } @catch (NSException *exception) {
        return fail(error, ANERuntimeErrorPrivateAPIException,
            [NSString stringWithFormat:@"ANE evaluation raised %@: %@",
                                       exception.name, exception.reason]);
    }
}

- (BOOL)unloadWithError:(NSError **)error {
    if (!_loaded) return YES;
    @try {
        for (NSInteger index = (NSInteger)_models.count - 1;
             index >= 0; --index) {
            NSError *unloadError = nil;
            BOOL unloaded =
                ((BOOL (*)(id, SEL, id, id, unsigned int, NSError **))
                 objc_msgSend)(_client,
                    @selector(unloadModel:options:qos:error:), _models[index],
                    _loadOptions[index], _qos, &unloadError);
            if (!unloaded) {
                if (error) *error = unloadError ?: runtimeError(
                    ANERuntimeErrorUnloadFailed, @"ANE model unload failed");
                return NO;
            }
        }
        _loaded = NO;
        [_models removeAllObjects];
        [_loadOptions removeAllObjects];
        return YES;
    } @catch (NSException *exception) {
        return fail(error, ANERuntimeErrorPrivateAPIException,
            [NSString stringWithFormat:@"ANE unload raised %@: %@",
                                       exception.name, exception.reason]);
    }
}

- (void)dealloc {
    [self unloadWithError:nil];
}

@end
