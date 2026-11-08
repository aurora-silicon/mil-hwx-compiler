#import "H16GTarget.h"

@implementation H16GOptionalUInt
- (instancetype)initUnavailableWithProvenance:(NSString *)provenance {
    self = [super init];
    if (self) {
        _available = NO;
        _provenance = [provenance copy];
    }
    return self;
}
- (instancetype)initWithValue:(NSUInteger)value provenance:(NSString *)provenance {
    self = [super init];
    if (self) {
        _available = YES;
        _value = value;
        _provenance = [provenance copy];
    }
    return self;
}
@end

@implementation H16GTaskCapability
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage
                            packetFamily:(H16GTaskPacketFamily)packetFamily
                                geometry:(NSUInteger)geometry {
    return [self initWithStageOperations:stageOperations
        inputShapes:inputShapes outputShape:outputShape numericMode:numericMode
        outputStorage:outputStorage packetFamily:packetFamily geometry:geometry
        programTaskCount:1 standaloneTaskCount:1
        semanticConstraint:H16GTaskSemanticConstraintNone];
}
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage
                            packetFamily:(H16GTaskPacketFamily)packetFamily
                                geometry:(NSUInteger)geometry
                        programTaskCount:(NSUInteger)programTaskCount
                      semanticConstraint:(H16GTaskSemanticConstraint)semanticConstraint {
    return [self initWithStageOperations:stageOperations
        inputShapes:inputShapes outputShape:outputShape numericMode:numericMode
        outputStorage:outputStorage packetFamily:packetFamily geometry:geometry
        programTaskCount:programTaskCount standaloneTaskCount:programTaskCount
        semanticConstraint:semanticConstraint];
}
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage
                            packetFamily:(H16GTaskPacketFamily)packetFamily
                                geometry:(NSUInteger)geometry
                        programTaskCount:(NSUInteger)programTaskCount
                     standaloneTaskCount:(NSUInteger)standaloneTaskCount
                      semanticConstraint:(H16GTaskSemanticConstraint)semanticConstraint {
    self = [super init];
    if (self) {
        _stageOperations = [stageOperations copy];
        _inputShapes = [inputShapes copy];
        _outputShape = [outputShape copy];
        _numericMode = numericMode;
        _outputStorage = outputStorage;
        _packetFamily = packetFamily;
        _geometry = geometry;
        _programTaskCount = programTaskCount;
        _standaloneTaskCount = standaloneTaskCount;
        _semanticConstraint = semanticConstraint;
    }
    return self;
}
@end

@implementation H16GProgramCompositionCapability
- (instancetype)initWithProducerPacketFamily:
    (H16GTaskPacketFamily)producerPacketFamily
    consumerPacketFamily:(H16GTaskPacketFamily)consumerPacketFamily
    producerOperationName:(NSString *)producerOperationName
    consumerOperationName:(NSString *)consumerOperationName
    producerGeometry:(NSUInteger)producerGeometry
    consumerGeometry:(NSUInteger)consumerGeometry
    bridgeStorage:(ANEScheduledBridgeStorage)bridgeStorage
    action:(H16GProgramCompositionAction)action
    provenance:(NSString *)provenance {
    return [self initWithProducerPacketFamily:producerPacketFamily
        consumerPacketFamily:consumerPacketFamily
        producerOperationName:producerOperationName
        consumerOperationName:consumerOperationName
        producerGeometry:producerGeometry consumerGeometry:consumerGeometry
        bridgeStorage:bridgeStorage action:action
        consumerTaskCountContribution:1 provenance:provenance];
}
- (instancetype)initWithProducerPacketFamily:
    (H16GTaskPacketFamily)producerPacketFamily
    consumerPacketFamily:(H16GTaskPacketFamily)consumerPacketFamily
    producerOperationName:(NSString *)producerOperationName
    consumerOperationName:(NSString *)consumerOperationName
    producerGeometry:(NSUInteger)producerGeometry
    consumerGeometry:(NSUInteger)consumerGeometry
    bridgeStorage:(ANEScheduledBridgeStorage)bridgeStorage
    action:(H16GProgramCompositionAction)action
    consumerTaskCountContribution:(NSUInteger)consumerTaskCountContribution
    provenance:(NSString *)provenance {
    self = [super init];
    if (self) {
        _producerPacketFamily = producerPacketFamily;
        _consumerPacketFamily = consumerPacketFamily;
        _producerOperationName = [producerOperationName copy];
        _consumerOperationName = [consumerOperationName copy];
        _producerGeometry = producerGeometry;
        _consumerGeometry = consumerGeometry;
        _bridgeStorage = bridgeStorage;
        _action = action;
        _consumerTaskCountContribution = consumerTaskCountContribution;
        _provenance = [provenance copy];
    }
    return self;
}
@end

static NSArray<H16GTaskCapability *> *taskCapabilityRows(void) {
    static NSArray<H16GTaskCapability *> *rows = [] {
        NSMutableArray<H16GTaskCapability *> *result = [NSMutableArray array];
        for (NSNumber *sizeNumber in @[@128, @256]) {
            NSUInteger size = sizeNumber.unsignedIntegerValue;
            NSArray<NSNumber *> *matrix = @[@1, @(size), @(size)];
            NSArray<NSNumber *> *image = @[@1, @1, @(size), @(size)];
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[@"matmul", @"reshape"]
                inputShapes:@[matrix, matrix] outputShape:image
                numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilySquareMatmul geometry:size
                programTaskCount:2 semanticConstraint:
                    H16GTaskSemanticConstraintNonTransposedMatmul]];
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[@"matmul", @"reshape"]
                inputShapes:@[matrix, matrix] outputShape:image
                numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageSRAM
                packetFamily:H16GTaskPacketFamilySquareMatmul geometry:size
                programTaskCount:2 semanticConstraint:
                    H16GTaskSemanticConstraintNonTransposedMatmul]];
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[@"gelu"] inputShapes:@[image]
                outputShape:image numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilyUnaryLUT geometry:size]];
        }
        NSArray<NSNumber *> *matrix128 = @[@1, @1, @128, @128];
        NSArray<NSNumber *> *row128 = @[@1, @1, @128, @1];
        [result addObject:[[H16GTaskCapability alloc]
            initWithStageOperations:@[@"matmul"]
            inputShapes:@[matrix128, matrix128] outputShape:matrix128
            numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageExternal
            packetFamily:H16GTaskPacketFamilySquareMatmul geometry:128
            programTaskCount:2 standaloneTaskCount:2 semanticConstraint:
                H16GTaskSemanticConstraintNonTransposedMatmul]];
        [result addObject:[[H16GTaskCapability alloc]
            initWithStageOperations:@[@"exp"] inputShapes:@[matrix128]
            outputShape:matrix128 numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageExternal
            packetFamily:H16GTaskPacketFamilyUnaryLUT geometry:128]];
        [result addObject:[[H16GTaskCapability alloc]
            initWithStageOperations:@[@"mul"] inputShapes:@[matrix128]
            outputShape:matrix128 numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageExternal
            packetFamily:H16GTaskPacketFamilyScalarScale geometry:128
            programTaskCount:1 semanticConstraint:
                H16GTaskSemanticConstraintScalarOperand]];
        for (NSString *operation in @[@"sub", @"mul"])
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[operation]
                inputShapes:@[matrix128, row128] outputShape:matrix128
                numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilyMatrixRowALU geometry:128]];
        for (NSString *operation in @[@"add", @"mul", @"max"])
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[operation]
                inputShapes:@[row128, row128] outputShape:row128
                numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilyRowStateALU geometry:128]];
        for (NSString *operation in @[@"reduce_max", @"reduce_sum"])
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[operation] inputShapes:@[matrix128]
                outputShape:row128 numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilyReduction geometry:128
                programTaskCount:1 semanticConstraint:
                    H16GTaskSemanticConstraintLastAxis]];
        [result addObject:[[H16GTaskCapability alloc]
            initWithStageOperations:@[@"real_div"]
            inputShapes:@[matrix128, row128] outputShape:matrix128
            numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageExternal
            packetFamily:H16GTaskPacketFamilyMatrixRowDivision geometry:128
            programTaskCount:2 standaloneTaskCount:2
            semanticConstraint:H16GTaskSemanticConstraintNone]];
        for (NSString *operation in @[@"add", @"mul"])
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[operation]
                inputShapes:@[matrix128, matrix128] outputShape:matrix128
                numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilySquareALU geometry:128]];
        [result addObject:[[H16GTaskCapability alloc]
            initWithStageOperations:@[@"sub", @"exp"]
            inputShapes:@[matrix128, matrix128] outputShape:matrix128
            numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageSRAM
            packetFamily:H16GTaskPacketFamilyPrimitiveFallback geometry:128
            programTaskCount:2 semanticConstraint:
                H16GTaskSemanticConstraintNone]];
        for (NSString *operation in @[@"reduce_max", @"reduce_sum"])
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[operation]
                inputShapes:@[matrix128] outputShape:matrix128
                numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageSRAM
                packetFamily:H16GTaskPacketFamilyPrimitiveFallback
                geometry:128 programTaskCount:1
                semanticConstraint:H16GTaskSemanticConstraintLastAxis]];
        [result addObject:[[H16GTaskCapability alloc]
            initWithStageOperations:@[@"reciprocal"]
            inputShapes:@[matrix128] outputShape:matrix128
            numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageSRAM
            packetFamily:H16GTaskPacketFamilyPrimitiveFallback geometry:128
            programTaskCount:2 semanticConstraint:
                H16GTaskSemanticConstraintNone]];
        return [result copy];
    }();
    return rows;
}

static NSArray<H16GProgramCompositionCapability *> *
programCompositionCapabilityRows(void) {
    static NSArray<H16GProgramCompositionCapability *> *rows = [] {
        NSMutableArray<H16GProgramCompositionCapability *> *result =
            [NSMutableArray array];
        for (NSNumber *sizeNumber in @[@128, @256]) {
            NSUInteger size = sizeNumber.unsignedIntegerValue;
            [result addObject:[[H16GProgramCompositionCapability alloc]
                initWithProducerPacketFamily:H16GTaskPacketFamilySquareMatmul
                consumerPacketFamily:H16GTaskPacketFamilyUnaryLUT
                producerOperationName:@"matmul"
                consumerOperationName:@"gelu"
                producerGeometry:size consumerGeometry:size
                bridgeStorage:ANEScheduledBridgeStorageSRAM
                action:H16GProgramCompositionActionTiledPostOperation
                consumerTaskCountContribution:0
                provenance:@"controlled M4 matmul and unary graph, 2026-09-02"]];
        }
        [result addObject:[[H16GProgramCompositionCapability alloc]
            initWithProducerPacketFamily:
                H16GTaskPacketFamilyPrimitiveFallback
            consumerPacketFamily:H16GTaskPacketFamilySquareALU
            producerOperationName:@"reciprocal"
            consumerOperationName:@"mul"
            producerGeometry:128 consumerGeometry:128
            bridgeStorage:ANEScheduledBridgeStorageSRAM
            action:H16GProgramCompositionActionPrimitiveFallback
            consumerTaskCountContribution:0
            provenance:@"decoded real_div primitive pair, 2026-09-02"]];
        // A constant scalar multiply after a square matmul is one fp16 word
        // in the matmul's output-scale field. Apple-compiled matmul-then-mul
        // oracles differ from the plain matmul oracle only in that word, for
        // n=128 (0.0884 and 0.25) and n=256 (0.0884).
        // The scalar-scale task row is measured at n=128 only, so the
        // composition row stops there even though the n=256 word is decoded.
        [result addObject:[[H16GProgramCompositionCapability alloc]
            initWithProducerPacketFamily:H16GTaskPacketFamilySquareMatmul
            consumerPacketFamily:H16GTaskPacketFamilyScalarScale
            producerOperationName:@"matmul"
            consumerOperationName:@"mul"
            producerGeometry:128 consumerGeometry:128
            bridgeStorage:ANEScheduledBridgeStorageSRAM
            action:H16GProgramCompositionActionMatmulOutputScale
            consumerTaskCountContribution:0
            provenance:@"Apple matmul-then-scalar-mul oracle word diff, 2026-09-03"]];
        // A constant scalar multiply before a table unary is one fp16 word in
        // the table's input-scale field. Apple-compiled mul-then-exp oracles
        // (0.0884, 0.25) and mul-then-sigmoid (0.25) differ from the plain
        // unary oracle only in that word. Only exp has a standalone task row
        // measured on M4, so only exp gets a composition row.
        [result addObject:[[H16GProgramCompositionCapability alloc]
            initWithProducerPacketFamily:H16GTaskPacketFamilyScalarScale
            consumerPacketFamily:H16GTaskPacketFamilyUnaryLUT
            producerOperationName:@"mul"
            consumerOperationName:@"exp"
            producerGeometry:128 consumerGeometry:128
            bridgeStorage:ANEScheduledBridgeStorageSRAM
            action:H16GProgramCompositionActionLUTInputScale
            consumerTaskCountContribution:1
            provenance:@"Apple scalar-mul-then-exp oracle word diff, 2026-09-03"]];
        [result addObject:[[H16GProgramCompositionCapability alloc]
            initWithProducerPacketFamily:H16GTaskPacketFamilyUnaryLUT
            consumerPacketFamily:H16GTaskPacketFamilyReduction
            producerOperationName:@"exp"
            consumerOperationName:@"reduce_sum"
            producerGeometry:128 consumerGeometry:128
            bridgeStorage:ANEScheduledBridgeStorageSRAM
            action:H16GProgramCompositionActionSRAMTaskChain
            consumerTaskCountContribution:1
            provenance:@"M4 unary-to-reduction SRAM program, 2026-09-03"]];
        [result addObject:[[H16GProgramCompositionCapability alloc]
            initWithProducerPacketFamily:
                H16GTaskPacketFamilyPrimitiveFallback
            consumerPacketFamily:H16GTaskPacketFamilyPrimitiveFallback
            producerOperationName:@"reduce_max"
            consumerOperationName:@"exp"
            producerGeometry:128 consumerGeometry:128
            bridgeStorage:ANEScheduledBridgeStorageSRAM
            action:H16GProgramCompositionActionSRAMTaskChain
            consumerTaskCountContribution:2
            provenance:@"M4 row normalization SRAM program, 2026-09-03"]];
        [result addObject:[[H16GProgramCompositionCapability alloc]
            initWithProducerPacketFamily:
                H16GTaskPacketFamilyPrimitiveFallback
            consumerPacketFamily:H16GTaskPacketFamilyPrimitiveFallback
            producerOperationName:@"exp"
            consumerOperationName:@"reduce_sum"
            producerGeometry:128 consumerGeometry:128
            bridgeStorage:ANEScheduledBridgeStorageSRAM
            action:H16GProgramCompositionActionSRAMTaskChain
            consumerTaskCountContribution:1
            provenance:@"M4 row normalization SRAM program, 2026-09-03"]];
        [result addObject:[[H16GProgramCompositionCapability alloc]
            initWithProducerPacketFamily:
                H16GTaskPacketFamilyPrimitiveFallback
            consumerPacketFamily:H16GTaskPacketFamilyPrimitiveFallback
            producerOperationName:@"reduce_sum"
            consumerOperationName:@"reciprocal"
            producerGeometry:128 consumerGeometry:128
            bridgeStorage:ANEScheduledBridgeStorageSRAM
            action:H16GProgramCompositionActionSRAMTaskChain
            consumerTaskCountContribution:2
            provenance:@"M4 row normalization SRAM program, 2026-09-03"]];
        return [result copy];
    }();
    return rows;
}

@implementation H16GTarget
+ (instancetype)currentTarget {
    static H16GTarget *target = [[H16GTarget alloc] init];
    return target;
}
- (instancetype)init {
    self = [super init];
    if (self) {
        _identifier = @"H16G";
        _sramBankCount = 64;
        _sramGranuleBytes = 16;
        _computeSetCount = 8;
        _ioSurfaceAlignment = 16384;
        _outputSlabBytes = 1024 * 1024;
        _workingSetBytes = 2 * 1024 * 1024;
        _maximumTensorDimension = 65536;
        _minimumSurfaceRowBytes = 64;
        _layoutDMAProgramMaxInputBytes = 1024 * 1024;
        _layoutBlockSizes = @[@2, @4, @8];
        _maximumProgramInputCount = [[H16GOptionalUInt alloc]
            initWithValue:3
            provenance:@"conservative five-resource object bound with output and kernel"];
        _maximumProgramTaskCount = [[H16GOptionalUInt alloc]
            initWithValue:8
            provenance:@"conservative bound from an M4-validated eight-task program"];
        _maximumTaskDescriptorByteLength = 0x3fc0;
    }
    return self;
}
- (BOOL)validateSpaceDepthInputShape:(NSArray<NSNumber *> *)inputShape
                         outputShape:(NSArray<NSNumber *> *)outputShape
                           blockSize:(NSUInteger)blockSize
                        depthToSpace:(BOOL)depthToSpace
                              reason:(NSString **)reason {
    BOOL supportedBlock = [_layoutBlockSizes containsObject:@(blockSize)];
    if (inputShape.count != 4 || outputShape.count != 4 || !supportedBlock) {
        if (reason) *reason = @"layout transform requires rank-4 NCHW and a measured block size";
        return NO;
    }
    NSUInteger n = inputShape[0].unsignedIntegerValue;
    NSUInteger channels = inputShape[1].unsignedIntegerValue;
    NSUInteger height = inputShape[2].unsignedIntegerValue;
    NSUInteger width = inputShape[3].unsignedIntegerValue;
    NSUInteger blockArea = blockSize * blockSize;
    if (n != 1 || outputShape[0].unsignedIntegerValue != 1 ||
        channels == 0 || height == 0 || width == 0 ||
        height > _maximumTensorDimension || width > _maximumTensorDimension) {
        if (reason) *reason = @"layout transform requires one positive, bounded NCHW batch";
        return NO;
    }
    NSUInteger expectedChannels = 0;
    NSUInteger expectedHeight = 0;
    NSUInteger expectedWidth = 0;
    if (depthToSpace) {
        if (channels % blockArea != 0) {
            if (reason) *reason = @"depth_to_space input channels must be divisible by block squared";
            return NO;
        }
        expectedChannels = channels / blockArea;
        expectedHeight = height * blockSize;
        expectedWidth = width * blockSize;
    } else {
        if (height % blockSize != 0 || width % blockSize != 0) {
            if (reason) *reason = @"space_to_depth spatial dimensions must be divisible by block";
            return NO;
        }
        expectedChannels = channels * blockArea;
        expectedHeight = height / blockSize;
        expectedWidth = width / blockSize;
    }
    BOOL shapeMatches = outputShape[1].unsignedIntegerValue == expectedChannels &&
        outputShape[2].unsignedIntegerValue == expectedHeight &&
        outputShape[3].unsignedIntegerValue == expectedWidth;
    if (!shapeMatches || expectedChannels > _maximumTensorDimension ||
        expectedHeight > _maximumTensorDimension ||
        expectedWidth > _maximumTensorDimension) {
        if (reason) *reason = @"layout output type does not match the block transform";
        return NO;
    }
    return YES;
}
- (BOOL)supportsOperationKind:(ANEOperationKind)kind {
    switch (kind) {
        case ANEOperationKindConv:
        case ANEOperationKindMatmul:
        case ANEOperationKindALU:
        case ANEOperationKindLUT:
        case ANEOperationKindReduce:
        case ANEOperationKindLayout:
        case ANEOperationKindQuantize:
        case ANEOperationKindDequantize:
            return YES;
        case ANEOperationKindConstant:
        case ANEOperationKindHighLevel:
        case ANEOperationKindMask:
        case ANEOperationKindUnsupported:
            return NO;
    }
}
- (BOOL)supportsFusedEpilogue:(NSString *)operationName
                 producerKind:(ANEOperationKind)producerKind {
    return ([@[@"relu",@"gelu"] containsObject:operationName]) &&
        (producerKind == ANEOperationKindConv ||
         producerKind == ANEOperationKindMatmul);
}
- (H16GTaskCapability *)taskCapabilityForStageOperations:
    (NSArray<NSString *> *)stageOperations
    inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
    outputShape:(NSArray<NSNumber *> *)outputShape
    numericMode:(ANELegalNumericMode)numericMode
    outputStorage:(ANEScheduledBridgeStorage)outputStorage {
    for (H16GTaskCapability *row in taskCapabilityRows())
        if ([row.stageOperations isEqualToArray:stageOperations] &&
            [row.inputShapes isEqualToArray:inputShapes] &&
            [row.outputShape isEqualToArray:outputShape] &&
            row.numericMode == numericMode &&
            row.outputStorage == outputStorage) return row;
    return nil;
}
- (H16GProgramCompositionCapability *)
    programCompositionCapabilityFrom:(H16GTaskPacketFamily)producerPacketFamily
    to:(H16GTaskPacketFamily)consumerPacketFamily
    producerOperationName:(NSString *)producerOperationName
    consumerOperationName:(NSString *)consumerOperationName
    producerGeometry:(NSUInteger)producerGeometry
    consumerGeometry:(NSUInteger)consumerGeometry
    bridgeStorage:(ANEScheduledBridgeStorage)bridgeStorage {
    for (H16GProgramCompositionCapability *row in
             programCompositionCapabilityRows())
        if (row.producerPacketFamily == producerPacketFamily &&
            row.consumerPacketFamily == consumerPacketFamily &&
            [row.producerOperationName isEqualToString:producerOperationName] &&
            [row.consumerOperationName isEqualToString:consumerOperationName] &&
            row.producerGeometry == producerGeometry &&
            row.consumerGeometry == consumerGeometry &&
            row.bridgeStorage == bridgeStorage) return row;
    return nil;
}
@end
