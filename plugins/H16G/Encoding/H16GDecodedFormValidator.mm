#import "H16GDecodedFormValidator.h"

#import "H16GConvEncoder.h"
#import "H16GALUEncoder.h"
#import "H16GDepthwiseEncoder.h"
#import "H16GMatmulEncoder.h"
#import "H16GLUTEncoder.h"
#import "H16GRegularConvEncoder.h"
#import "H16GReduceEncoder.h"
#import "H16GTDWriter.h"

NSUInteger H16GElementCount(NSArray<NSNumber *> *shape) {
    NSUInteger count = shape.count ? 1 : 0;
    for (NSNumber *dimension in shape) {
        NSUInteger value = dimension.unsignedIntegerValue;
        if (value == 0 || count > NSUIntegerMax / value) return 0;
        count *= value;
    }
    return count;
}

static BOOL reject(NSError **error, NSString *message) {
    if (error) *error = [NSError errorWithDomain:H16GTDWriterErrorDomain code:5
        userInfo:@{NSLocalizedDescriptionKey:message}];
    return NO;
}

static BOOL typeIs(ANEValueType *type, ANEElementType element,
                   NSArray<NSNumber *> *shape) {
    return type.elementType == element && [type.shape isEqualToArray:shape];
}

static ANEOperationNode *operandNode(ANEOperationGraph *graph,
                                     ANEOperationNode *node,
                                     NSString *name) {
    ANEGraphValue *value = node.sourceOperation.operands[name].value;
    return value ? [graph nodeForValueName:value.name] : nil;
}

static ANEGraphArgument *singleValue(ANEGraphArgument *argument) {
    while (argument.kind == ANEGraphArgumentKindCall &&
           argument.callArguments.count == 1) {
        argument = argument.callArguments[0].value;
    }
    return argument;
}

static NSString *scalarText(ANEGraphArgument *argument) {
    argument = singleValue(argument);
    return argument.text;
}

static NSArray<NSString *> *listTexts(ANEGraphArgument *argument) {
    argument = singleValue(argument);
    if (argument.kind != ANEGraphArgumentKindList) return nil;
    NSMutableArray<NSString *> *texts = [NSMutableArray array];
    for (ANEGraphArgument *element in argument.elements) {
        NSString *text = scalarText(element);
        if (!text) return nil;
        [texts addObject:text];
    }
    return texts;
}

static ANEGraphArgument *constantValue(ANEOperationNode *node) {
    return node.sourceOperation.attributes[@"val"];
}

static BOOL constantScalarIs(ANEOperationNode *node, NSString *expected) {
    return node && [scalarText(constantValue(node)) isEqualToString:expected];
}

static BOOL constantListIs(ANEOperationNode *node,
                           NSArray<NSString *> *expected) {
    return node && [listTexts(constantValue(node)) isEqualToArray:expected];
}

static BOOL attributeScalarIs(ANEOperationNode *node, NSString *name,
                              NSString *expected) {
    return node.sourceOperation &&
        [scalarText(node.sourceOperation.attributes[name]) isEqualToString:expected];
}

static BOOL taskHasOperations(ANEScheduledTask *task,
                              ANEOperationGraph *graph,
                              NSArray<NSString *> *names) {
    if (task.sourceNodeIdentifiers.count != names.count) return NO;
    for (NSUInteger i = 0; i < names.count; ++i) {
        ANEOperationNode *node =
            [graph nodeForValueName:task.sourceNodeIdentifiers[i]];
        if (!node || ![node.operationName isEqualToString:names[i]]) return NO;
    }
    return YES;
}

static BOOL validateConvCommon(ANEOperationGraph *graph,
                               ANEOperationNode *conv,
                               BOOL quantized,
                               NSError **error) {
    NSArray<NSNumber *> *activationShape = @[@1,@64,@64,@64];
    NSArray<NSNumber *> *weightShape = @[@64,@64,@1,@1];
    ANEOperationNode *weight = operandNode(graph,conv,@"weight");
    ANEOperationNode *activation = operandNode(graph,conv,@"x");
    BOOL externalActivation = [conv.externalValueNames containsObject:
        graph.sourceFunction.inputs.firstObject.name];
    BOOL activationLegal = externalActivation ||
        (activation && typeIs(activation.outputType,ANEElementTypeFP16,
                              activationShape));
    if (![conv.operationName isEqualToString:@"conv"] ||
        !typeIs(conv.outputType,ANEElementTypeFP16,activationShape) ||
        !activationLegal || !weight ||
        !typeIs(weight.outputType,ANEElementTypeFP16,weightShape) ||
        !constantScalarIs(operandNode(graph,conv,@"pad_type"),@"valid") ||
        !constantListIs(operandNode(graph,conv,@"strides"),@[@"1",@"1"]) ||
        !constantListIs(operandNode(graph,conv,@"pad"),@[@"0",@"0",@"0",@"0"]) ||
        !constantListIs(operandNode(graph,conv,@"dilations"),@[@"1",@"1"]) ||
        !constantScalarIs(operandNode(graph,conv,@"groups"),@"1"))
        return reject(error,@"decoded Conv form requires fp16 NCHW C64/S64, 1x1 weights, stride/dilation one, zero pad, and one group");
    BOOL weightKind = quantized
        ? [weight.operationName isEqualToString:@"constexpr_affine_dequantize"]
        : [weight.operationName isEqualToString:@"const"];
    if (!weightKind) return reject(error,@"Conv weight kind does not match its numeric mode");
    return YES;
}

static BOOL validateQuantizedWeight(ANEOperationNode *weight, NSError **error) {
    if (!attributeScalarIs(weight,@"axis",@"0") ||
        !attributeScalarIs(weight,@"scale",@"0x1p-3") ||
        !attributeScalarIs(weight,@"zero_point",@"0"))
        return reject(error,@"decoded W8A8 weights require axis 0, scale 0x1p-3, and zero point 0");
    ANEGraphArgument *quantized = weight.sourceOperation.attributes[@"quantized_data"];
    if (!quantized || quantized.calleeValueType.elementType != ANEElementTypeInt8 ||
        ![quantized.calleeValueType.shape isEqualToArray:@[@64,@64,@1,@1]])
        return reject(error,@"decoded W8A8 weight payload must be int8[64,64,1,1]");
    return YES;
}

@implementation H16GDecodedFormValidator
+ (BOOL)validateFP16ConvGraph:(ANEOperationGraph *)graph
                     scheduled:(ANEScheduledGraph *)scheduled
                         error:(NSError **)error {
    if (graph.sourceFunction.inputs.count != 1 ||
        graph.outputValueNames.count != 1 || scheduled.tasks.count != 1 ||
        scheduled.tasks[0].numericMode != ANELegalNumericModeFP16)
        return reject(error,@"fp16 Conv form requires one input, one output, and one hardware task");
    ANEScheduledTask *task = scheduled.tasks[0];
    BOOL hasRelu = taskHasOperations(task,graph,@[@"conv",@"relu"]);
    BOOL plainConv = taskHasOperations(task,graph,@[@"conv"]);
    if (!hasRelu && !plainConv)
        return reject(error,@"fp16 Conv task may contain Conv1x1 with an optional fused ReLU");
    NSUInteger active = 0;
    for (ANEOperationNode *node in graph.nodes)
        if (node.kind != ANEOperationKindConstant) ++active;
    if (active != (hasRelu ? 2u : 1u))
        return reject(error,@"fp16 Conv form contains an extra compute operation");
    ANEOperationNode *conv = [graph nodeForValueName:
        task.sourceNodeIdentifiers[0]];
    ANEGraphValue *externalInput = graph.sourceFunction.inputs.firstObject;
    ANEGraphValue *activation = conv.sourceOperation.operands[@"x"].value;
    ANEOperationNode *weight = operandNode(graph,conv,@"weight");
    NSArray<NSNumber *> *inputShape = externalInput.type.shape;
    NSArray<NSNumber *> *outputShape = conv.outputType.shape;
    NSArray<NSNumber *> *weightShape = weight.outputType.shape;
    BOOL rankAndType = externalInput.type.elementType == ANEElementTypeFP16 &&
        conv.outputType.elementType == ANEElementTypeFP16 &&
        inputShape.count == 4 && outputShape.count == 4 &&
        weightShape.count == 4;
    if (!rankAndType || !activation ||
        ![activation.name isEqualToString:externalInput.name] ||
        !weight || ![weight.operationName isEqualToString:@"const"] ||
        weight.outputType.elementType != ANEElementTypeFP16)
        return reject(error,@"fp16 Conv1x1 requires one external fp16 NCHW activation and one fp16 constant weight");
    NSUInteger inputChannels = inputShape[1].unsignedIntegerValue;
    NSUInteger outputChannels = outputShape[1].unsignedIntegerValue;
    NSUInteger spatial = outputShape[3].unsignedIntegerValue;
    BOOL geometryMatches = inputShape[0].unsignedIntegerValue == 1 &&
        outputShape[0].unsignedIntegerValue == 1 &&
        inputShape[2].unsignedIntegerValue == spatial &&
        inputShape[3].unsignedIntegerValue == spatial &&
        outputShape[2].unsignedIntegerValue == spatial &&
        weightShape[0].unsignedIntegerValue == outputChannels &&
        weightShape[1].unsignedIntegerValue == inputChannels &&
        weightShape[2].unsignedIntegerValue == 1 &&
        weightShape[3].unsignedIntegerValue == 1 &&
        [H16GConvEncoder supportsConv1x1WithInputChannels:inputChannels
            outputChannels:outputChannels spatial:spatial];
    if (!geometryMatches ||
        !constantScalarIs(operandNode(graph,conv,@"pad_type"),@"valid") ||
        !constantListIs(operandNode(graph,conv,@"strides"),@[@"1",@"1"]) ||
        !constantListIs(operandNode(graph,conv,@"pad"),@[@"0",@"0",@"0",@"0"]) ||
        !constantListIs(operandNode(graph,conv,@"dilations"),@[@"1",@"1"]) ||
        !constantScalarIs(operandNode(graph,conv,@"groups"),@"1"))
        return reject(error,@"Conv1x1 geometry or attributes have no decoded H16G schedule");
    if (plainConv) {
        if (![graph.outputValueNames.firstObject isEqualToString:conv.identifier])
            return reject(error,@"plain Conv must produce the graph result");
        return YES;
    }
    ANEOperationNode *relu = [graph nodeForValueName:
        task.sourceNodeIdentifiers[1]];
    if (relu.inputs.count != 1 || relu.inputs[0] != conv ||
        ![relu.outputType.shape isEqualToArray:outputShape] ||
        relu.outputType.elementType != ANEElementTypeFP16 ||
        ![graph.outputValueNames.firstObject isEqualToString:relu.identifier])
        return reject(error,@"fused ReLU must consume the Conv and produce the graph result");
    return YES;
}

+ (BOOL)validateW8A8ConvGraph:(ANEOperationGraph *)graph
                      scheduled:(ANEScheduledGraph *)scheduled
                          error:(NSError **)error {
    if (graph.sourceFunction.inputs.count != 1 || graph.outputValueNames.count != 1 ||
        !typeIs(graph.sourceFunction.inputs[0].type,ANEElementTypeFP16,
                @[@1,@64,@64,@64]) || scheduled.tasks.count != 4)
        return reject(error,@"decoded W8A8 form is exactly four C64/S64 Conv tasks");
    NSUInteger convCount=0,quantizeCount=0,dequantizeCount=0,weightCount=0;
    for (ANEOperationNode *node in graph.nodes) {
        if ([node.operationName isEqualToString:@"conv"]) ++convCount;
        else if ([node.operationName isEqualToString:@"quantize"]) ++quantizeCount;
        else if ([node.operationName isEqualToString:@"dequantize"]) ++dequantizeCount;
        else if ([node.operationName isEqualToString:@"constexpr_affine_dequantize"])
            ++weightCount;
        else if (node.kind != ANEOperationKindConstant)
            return reject(error,@"W8A8 chain contains an unsupported compute operation");
    }
    if (convCount!=4 || quantizeCount!=3 || dequantizeCount!=3 || weightCount!=4)
        return reject(error,@"W8A8 chain requires four weights and three Q/DQ bridges");
    for (NSUInteger i=0;i<4;++i) {
        ANEScheduledTask *task=scheduled.tasks[i];
        ANELegalNumericMode expected=i==0?ANELegalNumericModeW8A8InputBoundary:
            (i==3?ANELegalNumericModeW8A8OutputBoundary:ANELegalNumericModeW8A8Packed);
        if (task.operationKind!=ANEOperationKindConv || task.numericMode!=expected ||
            (i>0 && ![task.dependencies isEqualToArray:@[@(i-1)]]))
            return reject(error,@"W8A8 tasks are not one legal boundary-to-boundary chain");
        ANEOperationNode *conv=[graph nodeForValueName:task.sourceNodeIdentifiers.firstObject];
        if (!validateConvCommon(graph,conv,YES,error)) return NO;
        ANEOperationNode *weight=operandNode(graph,conv,@"weight");
        if (!validateQuantizedWeight(weight,error)) return NO;
    }
    for (ANEOperationNode *node in graph.nodes) {
        if ([node.operationName isEqualToString:@"quantize"] &&
            (!constantScalarIs(operandNode(graph,node,@"output_dtype"),@"int8") ||
             !constantScalarIs(operandNode(graph,node,@"scale"),@"0x1p-3")))
            return reject(error,@"activation quantize requires int8 and scale 0x1p-3");
        if ([node.operationName isEqualToString:@"dequantize"] &&
            !constantScalarIs(operandNode(graph,node,@"scale"),@"0x1p-3"))
            return reject(error,@"activation dequantize requires scale 0x1p-3");
    }
    return [graph.outputValueNames.firstObject
        isEqualToString:scheduled.tasks.lastObject.nodeIdentifier] ||
        reject(error,@"W8A8 final task does not produce the graph result");
}

+ (BOOL)validateFP16SquareMatmulGraph:(ANEOperationGraph *)graph
                              scheduled:(ANEScheduledGraph *)scheduled
                                  error:(NSError **)error {
    if (graph.sourceFunction.inputs.count != 2 ||
        graph.outputValueNames.count != 1 || scheduled.tasks.count != 1 ||
        scheduled.tasks[0].operationKind != ANEOperationKindMatmul ||
        scheduled.tasks[0].numericMode != ANELegalNumericModeFP16 ||
        !taskHasOperations(scheduled.tasks[0],graph,@[@"matmul"]))
        return reject(error,@"square matmul requires two inputs, one output, and one fp16 hardware task");
    NSUInteger active = 0;
    for (ANEOperationNode *node in graph.nodes)
        if (node.kind != ANEOperationKindConstant) ++active;
    if (active != 1)
        return reject(error,@"standalone square matmul contains an extra compute operation");
    ANEOperationNode *matmul = [graph nodeForValueName:
        scheduled.tasks[0].sourceNodeIdentifiers.firstObject];
    ANEGraphValue *left = matmul.sourceOperation.operands[@"x"].value;
    ANEGraphValue *right = matmul.sourceOperation.operands[@"y"].value;
    if (!left || !right || left.producer || right.producer ||
        left.type.elementType != ANEElementTypeFP16 ||
        right.type.elementType != ANEElementTypeFP16 ||
        matmul.outputType.elementType != ANEElementTypeFP16 ||
        left.type.shape.count != 3 || right.type.shape.count != 3 ||
        matmul.outputType.shape.count != 3 ||
        ![left.type.shape isEqualToArray:right.type.shape] ||
        ![left.type.shape isEqualToArray:matmul.outputType.shape])
        return reject(error,@"square matmul requires matching external fp16[1,N,N] tensors");
    NSUInteger size = left.type.shape[1].unsignedIntegerValue;
    BOOL square = left.type.shape[0].unsignedIntegerValue == 1 &&
        left.type.shape[2].unsignedIntegerValue == size;
    if (!square || ![H16GMatmulEncoder supportsSquareSize:size] ||
        !constantScalarIs(operandNode(graph,matmul,@"transpose_x"),@"false") ||
        !constantScalarIs(operandNode(graph,matmul,@"transpose_y"),@"false") ||
        ![graph.outputValueNames.firstObject isEqualToString:matmul.identifier])
        return reject(error,@"matmul geometry, transpose flags, or result have no decoded H16G schedule");
    ANETilePlan *tile = scheduled.tasks[0].tilePlan;
    if (tile.count != [H16GMatmulEncoder tileCountForSquareSize:size] ||
        tile.rows != [H16GMatmulEncoder rowsPerTileForSquareSize:size])
        return reject(error,@"planned matmul row tiles do not match the decoded packet family");
    return YES;
}

+ (BOOL)validateFP16BinaryALUGraph:(ANEOperationGraph *)graph
                           scheduled:(ANEScheduledGraph *)scheduled
                               error:(NSError **)error {
    if (graph.sourceFunction.inputs.count != 2 ||
        graph.outputValueNames.count != 1 || scheduled.tasks.count != 1 ||
        scheduled.tasks[0].operationKind != ANEOperationKindALU ||
        scheduled.tasks[0].sourceNodeIdentifiers.count != 1)
        return reject(error,@"binary ALU requires two inputs, one output, and one hardware task");
    ANEOperationNode *operation=[graph nodeForValueName:
        scheduled.tasks[0].sourceNodeIdentifiers.firstObject];
    ANEGraphValue *left=operation.sourceOperation.operands[@"x"].value;
    ANEGraphValue *right=operation.sourceOperation.operands[@"y"].value;
    if (!left||!right||left.producer||right.producer||
        left.type.elementType!=ANEElementTypeFP16||
        right.type.elementType!=ANEElementTypeFP16||
        operation.outputType.elementType!=ANEElementTypeFP16||
        left.type.shape.count<1||left.type.shape.count>4||
        ![left.type.shape isEqualToArray:right.type.shape]||
        ![left.type.shape isEqualToArray:operation.outputType.shape])
        return reject(error,@"binary ALU requires matching external fp16 tensors of rank 1 to 4");
    NSUInteger size=[H16GALUEncoder squareViewSizeForElementCount:
        H16GElementCount(left.type.shape)];
    if (size==NSNotFound||
        ![H16GALUEncoder supportsOperationName:operation.operationName
            squareSize:size]||
        ![graph.outputValueNames.firstObject isEqualToString:operation.identifier])
        return reject(error,@"binary ALU operation or geometry has no decoded H16G packet");
    NSUInteger active=0;
    for(ANEOperationNode *node in graph.nodes)
        if(node.kind!=ANEOperationKindConstant)++active;
    return active==1||reject(error,@"standalone binary ALU contains an extra compute operation");
}

+ (BOOL)validateFP16UnaryPointwiseGraph:(ANEOperationGraph *)graph
                                scheduled:(ANEScheduledGraph *)scheduled
                                    error:(NSError **)error {
    if (graph.sourceFunction.inputs.count != 1 ||
        graph.outputValueNames.count != 1 || scheduled.tasks.count != 1 ||
        scheduled.tasks[0].sourceNodeIdentifiers.count != 1 ||
        scheduled.tasks[0].numericMode != ANELegalNumericModeFP16)
        return reject(error,@"unary pointwise form requires one input, one output, and one fp16 hardware task");
    ANEOperationNode *operation=[graph nodeForValueName:
        scheduled.tasks[0].sourceNodeIdentifiers.firstObject];
    if (!operation || (operation.kind != ANEOperationKindALU &&
                       operation.kind != ANEOperationKindLUT))
        return reject(error,@"unary pointwise task has no decoded ALU/LUT operation");
    ANEGraphValue *input=operation.sourceOperation.operands[@"x"].value;
    if (!input || input.producer || input.type.elementType!=ANEElementTypeFP16 ||
        operation.outputType.elementType!=ANEElementTypeFP16 ||
        input.type.shape.count < 1 || input.type.shape.count > 4 ||
        ![input.type.shape isEqualToArray:operation.outputType.shape] ||
        [H16GLUTEncoder squareViewSizeForOperationName:operation.operationName
            elementCount:H16GElementCount(input.type.shape)] == NSNotFound ||
        ![graph.outputValueNames.firstObject isEqualToString:operation.identifier])
        return reject(error,@"unary pointwise operation, type, shape, or result has no rule-built H16G packet");
    NSUInteger active=0;
    for (ANEOperationNode *node in graph.nodes)
        if (node.kind!=ANEOperationKindConstant) ++active;
    return active==1 || reject(error,@"standalone unary pointwise graph contains an extra compute operation");
}

+ (BOOL)validateFP16ReductionGraph:(ANEOperationGraph *)graph
                           scheduled:(ANEScheduledGraph *)scheduled
                               error:(NSError **)error {
    if (graph.sourceFunction.inputs.count != 1 ||
        graph.outputValueNames.count != 1 || scheduled.tasks.count != 1 ||
        scheduled.tasks[0].operationKind != ANEOperationKindReduce ||
        scheduled.tasks[0].sourceNodeIdentifiers.count != 1 ||
        scheduled.tasks[0].numericMode != ANELegalNumericModeFP16)
        return reject(error,@"reduction requires one input, one output, and one fp16 scheduled operation");
    ANEOperationNode *operation=[graph nodeForValueName:
        scheduled.tasks[0].sourceNodeIdentifiers.firstObject];
    ANEGraphValue *input=operation.sourceOperation.operands[@"x"].value;
    ANEOperationNode *axes=operandNode(graph,operation,@"axes");
    NSArray<NSString *> *axisTexts=axes ? listTexts(constantValue(axes)) : nil;
    if (!operation || operation.kind != ANEOperationKindReduce || !input ||
        input.producer || input.type.elementType != ANEElementTypeFP16 ||
        operation.outputType.elementType != ANEElementTypeFP16 ||
        axisTexts.count != 1 ||
        ![scalarText(operation.sourceOperation.arguments[@"keep_dims"])
            isEqualToString:@"true"] ||
        ![graph.outputValueNames.firstObject isEqualToString:operation.identifier])
        return reject(error,@"reduction requires external fp16 NCHW input, one constant axis, keep_dims=true, and a direct result");
    NSUInteger axis=(NSUInteger)axisTexts.firstObject.integerValue;
    if (![H16GReduceEncoder supportsOperationName:operation.operationName
        inputShape:input.type.shape axis:axis])
        return reject(error,@"reduction operation, axis, or geometry has no rule-built H16G packet");
    NSMutableArray<NSNumber *> *expected=[input.type.shape mutableCopy];
    if (axis >= expected.count) return reject(error,@"reduction axis is outside the input rank");
    expected[axis]=@1;
    if (![operation.outputType.shape isEqualToArray:expected])
        return reject(error,@"reduction output shape does not retain the reduced dimension");
    NSUInteger active=0;
    for (ANEOperationNode *node in graph.nodes)
        if (node.kind != ANEOperationKindConstant) ++active;
    return active == 1 || reject(error,@"standalone reduction graph contains an extra compute operation");
}

+ (BOOL)validateFP16DepthwiseGraph:(ANEOperationGraph *)graph
                           scheduled:(ANEScheduledGraph *)scheduled
                               error:(NSError **)error {
    if (graph.sourceFunction.inputs.count != 1 ||
        graph.outputValueNames.count != 1 || scheduled.tasks.count != 1 ||
        scheduled.tasks[0].numericMode != ANELegalNumericModeFP16 ||
        !taskHasOperations(scheduled.tasks[0],graph,@[@"conv"]))
        return reject(error,@"depthwise 3x3 requires one plain fp16 Conv task");
    NSUInteger active = 0;
    for (ANEOperationNode *node in graph.nodes)
        if (node.kind != ANEOperationKindConstant) ++active;
    if (active != 1)
        return reject(error,@"depthwise 3x3 form contains an extra compute operation");
    ANEOperationNode *conv = [graph nodeForValueName:
        scheduled.tasks[0].sourceNodeIdentifiers.firstObject];
    ANEGraphValue *input = graph.sourceFunction.inputs.firstObject;
    ANEGraphValue *activation = conv.sourceOperation.operands[@"x"].value;
    ANEOperationNode *weight = operandNode(graph,conv,@"weight");
    NSArray<NSNumber *> *inputShape = input.type.shape;
    NSArray<NSNumber *> *outputShape = conv.outputType.shape;
    NSArray<NSNumber *> *weightShape = weight.outputType.shape;
    BOOL ranks = inputShape.count == 4 && outputShape.count == 4 &&
        weightShape.count == 4;
    NSUInteger channels = ranks ? inputShape[1].unsignedIntegerValue : 0;
    NSArray<NSNumber *> *expectedActivation = @[@1,@(channels),@64,@64];
    NSArray<NSNumber *> *expectedWeight = @[@(channels),@1,@3,@3];
    if (!ranks || input.type.elementType != ANEElementTypeFP16 ||
        conv.outputType.elementType != ANEElementTypeFP16 ||
        !activation || ![activation.name isEqualToString:input.name] ||
        ![inputShape isEqualToArray:expectedActivation] ||
        ![outputShape isEqualToArray:expectedActivation] ||
        !weight || ![weight.operationName isEqualToString:@"const"] ||
        weight.outputType.elementType != ANEElementTypeFP16 ||
        ![weightShape isEqualToArray:expectedWeight] ||
        ![H16GDepthwiseEncoder supportsChannels:channels spatial:64] ||
        !constantScalarIs(operandNode(graph,conv,@"pad_type"),@"same") ||
        !constantListIs(operandNode(graph,conv,@"strides"),@[@"1",@"1"]) ||
        !constantListIs(operandNode(graph,conv,@"pad"),@[@"0",@"0",@"0",@"0"]) ||
        !constantListIs(operandNode(graph,conv,@"dilations"),@[@"1",@"1"]) ||
        !constantScalarIs(operandNode(graph,conv,@"groups"),
            [NSString stringWithFormat:@"%lu",(unsigned long)channels]) ||
        ![graph.outputValueNames.firstObject isEqualToString:conv.identifier])
        return reject(error,@"depthwise 3x3 geometry or attributes have no decoded H16G schedule");
    return YES;
}

+ (BOOL)validateFP16RegularConvGraph:(ANEOperationGraph *)graph
                            scheduled:(ANEScheduledGraph *)scheduled
                                error:(NSError **)error {
    if (graph.sourceFunction.inputs.count != 1 ||
        graph.outputValueNames.count != 1 || scheduled.tasks.count != 1 ||
        scheduled.tasks[0].numericMode != ANELegalNumericModeFP16 ||
        !taskHasOperations(scheduled.tasks[0],graph,@[@"conv"]))
        return reject(error,@"regular 3x3/5x5 requires one plain fp16 Conv task");
    NSUInteger active = 0;
    for (ANEOperationNode *node in graph.nodes)
        if (node.kind != ANEOperationKindConstant) ++active;
    if (active != 1)
        return reject(error,@"regular Conv form contains an extra compute operation");
    ANEOperationNode *conv = [graph nodeForValueName:
        scheduled.tasks[0].sourceNodeIdentifiers.firstObject];
    ANEGraphValue *input = graph.sourceFunction.inputs.firstObject;
    ANEGraphValue *activation = conv.sourceOperation.operands[@"x"].value;
    ANEOperationNode *weight = operandNode(graph,conv,@"weight");
    NSArray<NSNumber *> *inputShape = input.type.shape;
    NSArray<NSNumber *> *outputShape = conv.outputType.shape;
    NSArray<NSNumber *> *weightShape = weight.outputType.shape;
    BOOL ranks = inputShape.count == 4 && outputShape.count == 4 &&
        weightShape.count == 4;
    NSUInteger channels = ranks ? inputShape[1].unsignedIntegerValue : 0;
    NSUInteger spatial = ranks ? inputShape[3].unsignedIntegerValue : 0;
    NSUInteger kernel = ranks ? weightShape[2].unsignedIntegerValue : 0;
    NSArray<NSNumber *> *expectedActivation =
        @[@1,@(channels),@(spatial),@(spatial)];
    NSArray<NSNumber *> *expectedWeight =
        @[@(channels),@(channels),@(kernel),@(kernel)];
    if (!ranks || input.type.elementType != ANEElementTypeFP16 ||
        conv.outputType.elementType != ANEElementTypeFP16 ||
        !activation || ![activation.name isEqualToString:input.name] ||
        ![inputShape isEqualToArray:expectedActivation] ||
        ![outputShape isEqualToArray:expectedActivation] ||
        !weight || ![weight.operationName isEqualToString:@"const"] ||
        weight.outputType.elementType != ANEElementTypeFP16 ||
        ![weightShape isEqualToArray:expectedWeight] ||
        ![H16GRegularConvEncoder supportsChannels:channels
            spatial:spatial kernel:kernel] ||
        !constantScalarIs(operandNode(graph,conv,@"pad_type"),@"same") ||
        !constantListIs(operandNode(graph,conv,@"strides"),@[@"1",@"1"]) ||
        !constantListIs(operandNode(graph,conv,@"pad"),@[@"0",@"0",@"0",@"0"]) ||
        !constantListIs(operandNode(graph,conv,@"dilations"),@[@"1",@"1"]) ||
        !constantScalarIs(operandNode(graph,conv,@"groups"),@"1") ||
        ![graph.outputValueNames.firstObject isEqualToString:conv.identifier])
        return reject(error,@"regular Conv geometry or attributes have no decoded H16G schedule");
    return YES;
}

+ (BOOL)validateLayoutConvChainGraph:(ANEOperationGraph *)graph
                             scheduled:(ANEScheduledGraph *)scheduled
                                 error:(NSError **)error {
    if (graph.sourceFunction.inputs.count != 1 ||
        graph.outputValueNames.count != 1 || scheduled.tasks.count != 3 ||
        !taskHasOperations(scheduled.tasks[0],graph,@[@"space_to_depth"]) ||
        !taskHasOperations(scheduled.tasks[1],graph,@[@"conv"]) ||
        !taskHasOperations(scheduled.tasks[2],graph,@[@"depth_to_space"]))
        return reject(error,@"fused layout-compute form requires one S2D, one Conv1x1, and one D2S task");
    if (scheduled.tasks[0].operationKind != ANEOperationKindLayout ||
        scheduled.tasks[1].operationKind != ANEOperationKindConv ||
        scheduled.tasks[2].operationKind != ANEOperationKindLayout ||
        scheduled.tasks[0].numericMode != ANELegalNumericModeFP16 ||
        scheduled.tasks[1].numericMode != ANELegalNumericModeFP16 ||
        scheduled.tasks[2].numericMode != ANELegalNumericModeFP16 ||
        scheduled.tasks[0].dependencies.count != 0 ||
        ![scheduled.tasks[1].dependencies isEqualToArray:@[@0]] ||
        ![scheduled.tasks[2].dependencies isEqualToArray:@[@1]] ||
        ![scheduled.tasks[0].regionIdentifier
            isEqualToString:scheduled.tasks[1].regionIdentifier] ||
        ![scheduled.tasks[1].regionIdentifier
            isEqualToString:scheduled.tasks[2].regionIdentifier])
        return reject(error,@"S2D, Conv and D2S must be one linear fp16 hardware region");

    ANEOperationNode *s2d = [graph nodeForValueName:
        scheduled.tasks[0].sourceNodeIdentifiers.firstObject];
    ANEOperationNode *conv = [graph nodeForValueName:
        scheduled.tasks[1].sourceNodeIdentifiers.firstObject];
    ANEOperationNode *d2s = [graph nodeForValueName:
        scheduled.tasks[2].sourceNodeIdentifiers.firstObject];
    ANEGraphValue *externalInput = graph.sourceFunction.inputs.firstObject;
    NSArray<NSNumber *> *inputShape = externalInput.type.shape;
    if (inputShape.count != 4 || externalInput.type.elementType != ANEElementTypeFP16)
        return reject(error,@"fused layout-compute input must be fp16 NCHW");
    NSUInteger channels = inputShape[1].unsignedIntegerValue;
    NSArray<NSNumber *> *packedShape = @[@1,@(channels*16),@32,@32];
    BOOL measuredGeometry = inputShape[0].unsignedIntegerValue == 1 &&
        channels >= 8 && channels <= 32 && channels % 8 == 0 &&
        inputShape[2].unsignedIntegerValue == 128 &&
        inputShape[3].unsignedIntegerValue == 128;
    ANEGraphValue *s2dInput = s2d.sourceOperation.operands[@"x"].value;
    ANEOperationNode *convInput = operandNode(graph,conv,@"x");
    ANEOperationNode *d2sInput = operandNode(graph,d2s,@"x");
    ANEOperationNode *weight = operandNode(graph,conv,@"weight");
    NSArray<NSNumber *> *weightShape =
        @[@(channels*16),@(channels*16),@1,@1];
    if (!measuredGeometry || !s2dInput ||
        ![s2dInput.name isEqualToString:externalInput.name] ||
        !typeIs(s2d.outputType,ANEElementTypeFP16,packedShape) ||
        convInput != s2d || d2sInput != conv ||
        !typeIs(conv.outputType,ANEElementTypeFP16,packedShape) ||
        !typeIs(d2s.outputType,ANEElementTypeFP16,inputShape) ||
        !weight || ![weight.operationName isEqualToString:@"const"] ||
        !typeIs(weight.outputType,ANEElementTypeFP16,weightShape) ||
        !constantScalarIs(operandNode(graph,s2d,@"block_size"),@"4") ||
        !constantScalarIs(operandNode(graph,d2s,@"block_size"),@"4") ||
        !constantScalarIs(operandNode(graph,conv,@"pad_type"),@"valid") ||
        !constantListIs(operandNode(graph,conv,@"strides"),@[@"1",@"1"]) ||
        !constantListIs(operandNode(graph,conv,@"pad"),@[@"0",@"0",@"0",@"0"]) ||
        !constantListIs(operandNode(graph,conv,@"dilations"),@[@"1",@"1"]) ||
        !constantScalarIs(operandNode(graph,conv,@"groups"),@"1") ||
        ![graph.outputValueNames.firstObject isEqualToString:d2s.identifier])
        return reject(error,@"fused layout-compute geometry is outside the measured B4 S128 Conv1x1 family");
    NSUInteger active = 0;
    for (ANEOperationNode *node in graph.nodes)
        if (node.kind != ANEOperationKindConstant) ++active;
    return active == 3 || reject(error,@"fused layout-compute region contains an extra operation");
}

+ (BOOL)validateMixedGraph:(ANEOperationGraph *)graph
                   scheduled:(ANEScheduledGraph *)scheduled
                       error:(NSError **)error {
    NSArray<NSArray<NSString *> *> *signatures=@[
        @[@"slice_by_size",@"slice_by_size",@"slice_by_size"],
        @[@"transpose",@"transpose"],@[@"matmul"],@[@"mul"],
        @[@"reduce_max"],@[@"sub",@"exp"],@[@"reduce_sum"],
        @[@"reciprocal"],@[@"mul",@"transpose"],
        @[@"matmul",@"transpose",@"reshape"],
    ];
    if (scheduled.tasks.count!=10 || graph.sourceFunction.inputs.count!=1 ||
        graph.outputValueNames.count!=1 ||
        !typeIs(graph.sourceFunction.inputs[0].type,ANEElementTypeFP16,
                @[@1,@64,@4,@192]))
        return reject(error,@"decoded mixed form requires one fp16[1,64,4,192] input and ten tasks");
    for(NSUInteger i=0;i<10;++i)
        if(!taskHasOperations(scheduled.tasks[i],graph,signatures[i]))
            return reject(error,@"mixed task signature is outside the decoded packet grammar");
    NSArray<NSArray<NSNumber *> *> *begins=@[
        @[@0,@0,@0,@0],@[@0,@0,@0,@64],@[@0,@0,@0,@128]];
    for(NSUInteger i=0;i<3;++i){
        ANEOperationNode *slice=[graph nodeForValueName:scheduled.tasks[0].sourceNodeIdentifiers[i]];
        NSArray<NSString *> *begin=@[[begins[i][0] stringValue],[begins[i][1] stringValue],
            [begins[i][2] stringValue],[begins[i][3] stringValue]];
        if(!typeIs(slice.outputType,ANEElementTypeFP16,@[@1,@64,@4,@64]) ||
           !constantListIs(operandNode(graph,slice,@"begin"),begin) ||
           !constantListIs(operandNode(graph,slice,@"size"),@[@"1",@"64",@"4",@"64"]))
            return reject(error,@"split task requires the decoded Q/K/V slice bounds");
    }
    NSArray<NSArray<NSString *> *> *perms=@[
        @[@"0",@"2",@"3",@"1"],@[@"0",@"2",@"1",@"3"]];
    for(NSUInteger i=0;i<2;++i){
        ANEOperationNode *transpose=[graph nodeForValueName:scheduled.tasks[1].sourceNodeIdentifiers[i]];
        if(!typeIs(transpose.outputType,ANEElementTypeFP16,@[@1,@4,@64,@64]) ||
           !constantListIs(operandNode(graph,transpose,@"perm"),perms[i]))
            return reject(error,@"Q/K transpose permutation is outside the decoded form");
    }
    ANEOperationNode *scores=[graph nodeForValueName:scheduled.tasks[2].sourceNodeIdentifiers[0]];
    ANEOperationNode *scaled=[graph nodeForValueName:scheduled.tasks[3].sourceNodeIdentifiers[0]];
    if(!typeIs(scores.outputType,ANEElementTypeFP16,@[@1,@4,@64,@64]) ||
       !constantScalarIs(operandNode(graph,scores,@"transpose_x"),@"false") ||
       !constantScalarIs(operandNode(graph,scores,@"transpose_y"),@"false") ||
       !constantScalarIs(operandNode(graph,scaled,@"y"),@"0.125"))
        return reject(error,@"score matmul requires untransposed inputs and scale 0.125");
    for(NSUInteger i=4;i<=8;++i)
        for(NSString *identifier in scheduled.tasks[i].sourceNodeIdentifiers){
            ANEOperationNode *node=[graph nodeForValueName:identifier];
            if(!typeIs(node.outputType,ANEElementTypeFP16,@[@1,@4,@64,@64]))
                return reject(error,@"softmax intermediates require fp16[1,4,64,64]");
            if(node.sourceOperation &&
               [node.sourceOperation.operationName isEqualToString:@"softmax"] &&
               !constantScalarIs(operandNode(graph,node,@"axis"),@"-1"))
                return reject(error,@"decoded softmax reduction axis is -1");
        }
    ANEOperationNode *valueTranspose=[graph nodeForValueName:scheduled.tasks[8].sourceNodeIdentifiers[1]];
    if(!constantListIs(operandNode(graph,valueTranspose,@"perm"),@[@"0",@"2",@"3",@"1"]))
        return reject(error,@"V transpose permutation is outside the decoded form");
    ANEOperationNode *context=[graph nodeForValueName:scheduled.tasks[9].sourceNodeIdentifiers[0]];
    ANEOperationNode *outputTranspose=[graph nodeForValueName:scheduled.tasks[9].sourceNodeIdentifiers[1]];
    ANEOperationNode *reshape=[graph nodeForValueName:scheduled.tasks[9].sourceNodeIdentifiers[2]];
    if(!constantScalarIs(operandNode(graph,context,@"transpose_x"),@"false") ||
       !constantScalarIs(operandNode(graph,context,@"transpose_y"),@"false") ||
       !constantListIs(operandNode(graph,outputTranspose,@"perm"),@[@"0",@"3",@"1",@"2"]) ||
       !constantListIs(operandNode(graph,reshape,@"shape"),@[@"1",@"64",@"4",@"64"]) ||
       !typeIs(reshape.outputType,ANEElementTypeFP16,@[@1,@64,@4,@64]) ||
       ![graph.outputValueNames.firstObject isEqualToString:reshape.identifier])
        return reject(error,@"final matmul/transpose/reshape boundary is outside the decoded form");
    NSArray<NSArray<NSNumber *> *> *dependencies=@[
        @[],@[@0],@[@1],@[@2],@[@3],@[@3,@4],@[@5],@[@6],@[@0,@5,@7],@[@8]];
    for(NSUInteger i=0;i<10;++i)
        if(![scheduled.tasks[i].dependencies isEqualToArray:dependencies[i]])
            return reject(error,@"mixed task dependencies do not match the decoded schedule");
    return YES;
}
@end
