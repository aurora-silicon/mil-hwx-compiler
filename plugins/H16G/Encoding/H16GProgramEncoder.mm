#import "H16GProgramEncoder.h"

#import "ANEBlobResolver.h"
#import "ANEHWXArtifact.h"
#import "H16GConstantPacker.h"
#import "H16GALUEncoder.h"
#import "H16GConvChainEncoder.h"
#import "H16GConvEncoder.h"
#import "H16GDecodedFormValidator.h"
#import "H16GDepthwiseEncoder.h"
#import "H16GLayoutEncoder.h"
#import "H16GLayoutConvChainEncoder.h"
#import "H16GLUTEncoder.h"
#import "H16GMatmulEncoder.h"
#import "H16GMixedTaskEncoder.h"
#import "H16GRegularConvEncoder.h"
#import "H16GReduceEncoder.h"
#import "H16GTarget.h"
#import "HWXObjectWriter.h"

#include "H16ConvProgram.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

static ANESourceRange sourceRange(ANEOperationNode *node) {
    if (node.sourceOperation) return node.sourceOperation.range;
    ANESourceLocation location = ANESourceLocationMake(0, 1, 1);
    return ANESourceRangeMake(location, location);
}

static void emitError(ANEDiagnosticEngine *diagnostics, NSString *code,
                      NSString *message, ANEOperationNode *node) {
    [diagnostics emitSeverity:ANEDiagnosticSeverityError code:code
                      message:message range:sourceRange(node)];
}

static NSUInteger elementBytes(ANEElementType type) {
    switch (type) {
        case ANEElementTypeFP16: return 2;
        case ANEElementTypeInt8: return 1;
        case ANEElementTypeFP32:
        case ANEElementTypeInt32: return 4;
        case ANEElementTypeUInt64: return 8;
        case ANEElementTypeBool: return 1;
        case ANEElementTypeString:
        case ANEElementTypeInvalid: return 0;
    }
}

static NSUInteger tensorBytes(ANEValueType *type) {
    NSUInteger count = 1;
    for (NSNumber *dimension in type.shape)
        count *= dimension.unsignedIntegerValue;
    return count * elementBytes(type.elementType);
}

static NSUInteger alignUp(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static ANEOperationNode *nodeNamed(ANEOperationGraph *graph, NSString *name);

static ANEOperationNode *producerForOperand(ANEOperationGraph *graph,
                                            ANEOperationNode *operation,
                                            NSString *name) {
    ANEGraphValue *value=operation.sourceOperation.operands[name].value;
    return value ? [graph nodeForValueName:value.name] : nil;
}

static NSString *scalarLiteral(ANEGraphArgument *argument) {
    if (!argument) return nil;
    if (argument.text) return argument.text;
    if (argument.kind==ANEGraphArgumentKindCall &&
        argument.callArguments.count==1)
        return scalarLiteral(argument.callArguments.firstObject.value);
    if ((argument.kind==ANEGraphArgumentKindList ||
         argument.kind==ANEGraphArgumentKindTuple) &&
        argument.elements.count==1)
        return scalarLiteral(argument.elements.firstObject);
    return nil;
}

static BOOL fp16ScalarFromConstant(ANEOperationNode *node,
                                   uint16_t *bits) {
    NSString *text=scalarLiteral(node.sourceOperation.attributes[@"val"]);
    if (!text) return NO;
    char *end=nullptr;
    double value=strtod(text.UTF8String,&end);
    if (!end || *end || !isfinite(value)) return NO;
    _Float16 half=(_Float16)value;
    memcpy(bits,&half,sizeof(*bits));
    return isfinite((double)half);
}

static BOOL fp16ScalarFromArgument(ANEOperationGraph *graph,
                                   ANEGraphArgument *argument,
                                   uint16_t *bits) {
    NSString *text=scalarLiteral(argument);
    if (text) {
        char *end=nullptr;
        double value=strtod(text.UTF8String,&end);
        if (end && !*end && isfinite(value)) {
            _Float16 half=(_Float16)value;
            memcpy(bits,&half,sizeof(*bits));
            return isfinite((double)half);
        }
    }
    ANEOperationNode *constant=argument.value
        ? [graph nodeForValueName:argument.value.name] : nil;
    return constant && fp16ScalarFromConstant(constant,bits);
}

static ANEHWXArtifact *encodeTransformerLinearProgram(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    NSURL *modelRoot, ANEDiagnosticEngine *diagnostics) {
    (void)scheduled;
    ANEOperationNode *linear=nil;
    for (ANEOperationNode *node in graph.nodes)
        if ([node.operationName isEqualToString:@"linear"]) linear=node;
    if (![linear.operationName isEqualToString:@"linear"] ||
        graph.sourceFunction.inputs.count!=1 ||
        linear.outputType.shape.count!=3) return nil;
    ANEGraphValue *input=graph.sourceFunction.inputs.firstObject;
    NSArray<NSNumber *> *inputShape=input.type.shape;
    NSArray<NSNumber *> *outputShape=linear.outputType.shape;
    if (inputShape.count!=3 || inputShape[0].unsignedIntegerValue!=1 ||
        outputShape[0].unsignedIntegerValue!=1 ||
        inputShape[1].unsignedIntegerValue!=outputShape[1].unsignedIntegerValue)
        return nil;
    ANEOperationNode *weight=producerForOperand(graph,linear,@"weight");
    ANEOperationNode *bias=producerForOperand(graph,linear,@"bias");
    if (!weight.sourceOperation || weight.outputType.shape.count!=2)
        return nil;
    NSUInteger tokens=inputShape[1].unsignedIntegerValue;
    NSUInteger cin=inputShape[2].unsignedIntegerValue;
    NSUInteger cout=outputShape[2].unsignedIntegerValue;
    if (![weight.outputType.shape isEqualToArray:@[@(cout),@(cin)]]) return nil;
    BOOL hasBias=bias!=nil;
    if (hasBias && (!bias.sourceOperation ||
        ![bias.outputType.shape isEqualToArray:@[@(cout)]])) return nil;
    NSData *rawWeights=[ANEBlobResolver loadConstantForOperation:weight.sourceOperation
        expectedBytes:cin*cout*2 modelRoot:modelRoot diagnostics:diagnostics];
    NSData *rawBias=hasBias ? [ANEBlobResolver
        loadConstantForOperation:bias.sourceOperation expectedBytes:cout*2
        modelRoot:modelRoot diagnostics:diagnostics] : nil;
    if (!rawWeights || (hasBias && !rawBias)) return nil;
    ANEOperationNode *result=linear;
    ANEOperationNode *post=linear.users.count==1 ? linear.users.firstObject : nil;
    BOOL gelu=[post.operationName isEqualToString:@"gelu"];
    BOOL scale=[post.operationName isEqualToString:@"mul"] ||
               [post.operationName isEqualToString:@"multiply"];
    if (post && !gelu && !scale) return nil;
    if (post) result=post;
    NSData *geluTable=nil;
    ane::h16::TransformerLinearOptions options;
    options.bias=hasBias;
    if (gelu) {
        NSError *tableError=nil;
        geluTable=[H16GLUTEncoder constantRegionForOperationName:@"gelu"
                                                          error:&tableError];
        if (!geluTable || geluTable.length!=128) {
            emitError(diagnostics,@"h16g.encode.transformer-linear-gelu",
                tableError.localizedDescription ?: @"GELU table is unavailable",post);
            return nil;
        }
        options.gelu=true;
        options.geluTable=static_cast<const uint8_t *>(geluTable.bytes);
        options.geluTableBytes=geluTable.length;
    }
    if (scale) {
        ANEGraphArgument *x=post.sourceOperation.operands[@"x"];
        ANEGraphArgument *y=post.sourceOperation.operands[@"y"];
        ANEGraphArgument *scalar=[x.value.name isEqualToString:linear.identifier]
            ? y : x;
        uint16_t bits=0;
        if (!scalar || !fp16ScalarFromArgument(graph,scalar,&bits)) return nil;
        options.outputScale=bits;
    }
    ane::h16::TransformerLinearProgram encoded;
    try {
        encoded=ane::h16::encodeTransformerLinear((uint32_t)tokens,
            (uint32_t)cin,(uint32_t)cout,
            static_cast<const uint8_t *>(rawWeights.bytes),rawWeights.length,
            hasBias ? static_cast<const uint8_t *>(rawBias.bytes) : nullptr,
            hasBias ? rawBias.length : 0,options);
    } catch (const std::exception &exception) {
        emitError(diagnostics,@"h16g.encode.transformer-linear-fields",
            [NSString stringWithUTF8String:exception.what()],linear);
        return nil;
    }
    NSData *td=[NSData dataWithBytes:encoded.program.task.data()
        length:encoded.program.task.size()];
    NSData *constants=[NSData dataWithBytes:encoded.program.constants.data()
        length:encoded.program.constants.size()];
    NSMutableArray<NSNumber *> *relocations=[NSMutableArray array];
    for (std::size_t offset : encoded.program.kernelRelocations)
        [relocations addObject:@(offset)];
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:input.name shortName:input.name
            role:HWXObjectBindingRoleInput elementType:input.type.elementType
            shape:inputShape],
        [[HWXObjectBinding alloc]
            initWithSymbol:[result.identifier stringByAppendingString:@"@output"]
            shortName:result.identifier role:HWXObjectBindingRoleOutput
            elementType:result.outputType.elementType shape:result.outputType.shape],
    ];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:encoded.program.taskCount
        recordCount:encoded.programRecordCount
        formatCode:encoded.programFormatCode
        scratchByteLength:encoded.scratchBytes
        scratchAllocationByteLength:encoded.scratchBytes
        descriptorLayout:encoded.scratchBacked
            ? HWXProgramDescriptorLayoutScratchBackedMixed
            : HWXProgramDescriptorLayoutLinear];
    info.kernelRegionConstants=YES;
    NSError *error=nil;
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:constants bindings:objectBindings
        kernelRelocationOffsets:relocations programInfo:info error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.transformer-linear-object",
            error.localizedDescription ?: @"cannot construct transformer linear HWX",
            linear);
        return nil;
    }
    NSUInteger inputBytes=tensorBytes(input.type);
    NSUInteger outputBytes=tensorBytes(result.outputType);
    NSArray<ANEHWXBinding *> *runtimeBindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:inputBytes
            allocationByteLength:alignUp(inputBytes,0x4000) ioSurfaceIndex:0
            rowStrideBytes:cin*2 planeStrideBytes:inputBytes
            batchStrideBytes:inputBytes],
        [[ANEHWXBinding alloc]initWithIdentifier:weight.identifier
            role:ANESurfaceRoleWeight logicalByteLength:constants.length
            allocationByteLength:constants.length ioSurfaceIndex:-1],
        [[ANEHWXBinding alloc]initWithIdentifier:result.identifier
            role:ANESurfaceRoleOutput logicalByteLength:outputBytes
            allocationByteLength:alignUp(outputBytes,0x4000) ioSurfaceIndex:1
            rowStrideBytes:cout*2 planeStrideBytes:outputBytes
            batchStrideBytes:outputBytes],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeWhisperStem1Program(
    ANEOperationGraph *graph, NSURL *modelRoot,
    ANEDiagnosticEngine *diagnostics) {
    ANEOperationNode *conv=nil;
    for (ANEOperationNode *node in graph.nodes)
        if ([node.operationName isEqualToString:@"conv"] &&
            node.outputType.shape.count==3) conv=node;
    if (!conv || graph.sourceFunction.inputs.count!=1) return nil;
    ANEGraphValue *input=graph.sourceFunction.inputs.firstObject;
    NSArray<NSNumber *> *in=input.type.shape,*out=conv.outputType.shape;
    if (in.count!=3 || in[0].unsignedIntegerValue!=1 ||
        in[1].unsignedIntegerValue!=80 ||
        out[0].unsignedIntegerValue!=1 ||
        out[2].unsignedIntegerValue!=in[2].unsignedIntegerValue)
        return nil;
    NSUInteger tokens=in[2].unsignedIntegerValue;
    NSUInteger channels=out[1].unsignedIntegerValue;
    ANEOperationNode *weight=producerForOperand(graph,conv,@"weight");
    ANEOperationNode *bias=producerForOperand(graph,conv,@"bias");
    if (!weight.sourceOperation || !bias.sourceOperation ||
        ![weight.outputType.shape isEqualToArray:@[@(channels),@80,@3]] ||
        ![bias.outputType.shape isEqualToArray:@[@(channels)]]) return nil;
    ANEOperationNode *result=conv;
    ANEOperationNode *post=conv.users.count==1 ? conv.users.firstObject : nil;
    BOOL gelu=[post.operationName isEqualToString:@"gelu"];
    if (post && !gelu) return nil;
    if (post) result=post;
    NSData *rawWeights=[ANEBlobResolver loadConstantForOperation:weight.sourceOperation
        expectedBytes:channels*80*3*2 modelRoot:modelRoot diagnostics:diagnostics];
    NSData *rawBias=[ANEBlobResolver loadConstantForOperation:bias.sourceOperation
        expectedBytes:channels*2 modelRoot:modelRoot diagnostics:diagnostics];
    if (!rawWeights || !rawBias) return nil;
    ane::h16::TransformerLinearOptions options;
    options.bias=true;
    NSData *geluTable=nil;
    if (gelu) {
        NSError *tableError=nil;
        geluTable=[H16GLUTEncoder constantRegionForOperationName:@"gelu"
                                                          error:&tableError];
        if (!geluTable || geluTable.length!=128) {
            emitError(diagnostics,@"h16g.encode.whisper-stem-gelu",
                tableError.localizedDescription ?: @"GELU table is unavailable",post);
            return nil;
        }
        options.gelu=true;
        options.geluTable=static_cast<const uint8_t *>(geluTable.bytes);
        options.geluTableBytes=geluTable.length;
    }
    ane::h16::TransformerLinearProgram encoded;
    try {
        encoded=ane::h16::encodeWhisperStem1((uint32_t)tokens,
            (uint32_t)channels,
            static_cast<const uint8_t *>(rawWeights.bytes),rawWeights.length,
            static_cast<const uint8_t *>(rawBias.bytes),rawBias.length,options);
    } catch (const std::exception &exception) {
        emitError(diagnostics,@"h16g.encode.whisper-stem-fields",
            [NSString stringWithUTF8String:exception.what()],conv);
        return nil;
    }
    NSData *td=[NSData dataWithBytes:encoded.program.task.data()
        length:encoded.program.task.size()];
    NSData *constants=[NSData dataWithBytes:encoded.program.constants.data()
        length:encoded.program.constants.size()];
    NSMutableArray<NSNumber *> *relocations=[NSMutableArray array];
    for (std::size_t offset : encoded.program.kernelRelocations)
        [relocations addObject:@(offset)];
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:input.name shortName:input.name
            role:HWXObjectBindingRoleInput elementType:input.type.elementType
            shape:in],
        [[HWXObjectBinding alloc]
            initWithSymbol:[result.identifier stringByAppendingString:@"@output"]
            shortName:result.identifier role:HWXObjectBindingRoleOutput
            elementType:result.outputType.elementType shape:result.outputType.shape],
    ];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:1 recordCount:0x10 formatCode:0 scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    info.kernelRegionConstants=YES;
    NSError *error=nil;
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:constants bindings:objectBindings
        kernelRelocationOffsets:relocations programInfo:info error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.whisper-stem-object",
            error.localizedDescription ?: @"cannot construct Whisper stem HWX",conv);
        return nil;
    }
    NSUInteger inputBytes=tensorBytes(input.type);
    NSUInteger outputBytes=tensorBytes(result.outputType);
    NSArray<ANEHWXBinding *> *runtimeBindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:inputBytes
            allocationByteLength:alignUp(inputBytes,0x4000) ioSurfaceIndex:0
            rowStrideBytes:tokens*2 planeStrideBytes:inputBytes
            batchStrideBytes:inputBytes],
        [[ANEHWXBinding alloc]initWithIdentifier:weight.identifier
            role:ANESurfaceRoleWeight logicalByteLength:constants.length
            allocationByteLength:constants.length ioSurfaceIndex:-1],
        [[ANEHWXBinding alloc]initWithIdentifier:result.identifier
            role:ANESurfaceRoleOutput logicalByteLength:outputBytes
            allocationByteLength:alignUp(outputBytes,0x4000) ioSurfaceIndex:1
            rowStrideBytes:tokens*2 planeStrideBytes:outputBytes
            batchStrideBytes:outputBytes],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:runtimeBindings];
}

typedef struct {
    NSUInteger rowBytes;
    NSUInteger planeBytes;
    NSUInteger batchBytes;
    NSUInteger storageBytes;
} H16GNCHWSurfaceLayout;

static H16GNCHWSurfaceLayout nchwSurfaceLayout(ANEValueType *type) {
    NSArray<NSNumber *> *shape=type.shape;
    NSUInteger n=shape[0].unsignedIntegerValue;
    NSUInteger channels=shape[1].unsignedIntegerValue;
    NSUInteger height=shape[2].unsignedIntegerValue;
    NSUInteger width=shape[3].unsignedIntegerValue;
    NSUInteger minimum=[H16GTarget currentTarget].minimumSurfaceRowBytes;
    NSUInteger row=alignUp(width*elementBytes(type.elementType),minimum);
    NSUInteger plane=height*row;
    NSUInteger batch=channels*plane;
    H16GNCHWSurfaceLayout result={row,plane,batch,n*batch};
    return result;
}

static ANEOperationNode *nodeNamed(ANEOperationGraph *graph, NSString *name) {
    return [graph nodeForValueName:name];
}

static ANEOperationNode *convSourceForTask(ANEOperationGraph *graph,
                                           ANEScheduledTask *task) {
    for (NSString *identifier in task.sourceNodeIdentifiers) {
        ANEOperationNode *node = nodeNamed(graph, identifier);
        if (node.kind == ANEOperationKindConv) return node;
    }
    return nil;
}

static ANEOperationNode *layoutSourceForTask(ANEOperationGraph *graph,
                                             ANEScheduledTask *task) {
    for (NSString *identifier in task.sourceNodeIdentifiers) {
        ANEOperationNode *node = nodeNamed(graph, identifier);
        if ([node.operationName isEqualToString:@"space_to_depth"] ||
            [node.operationName isEqualToString:@"depth_to_space"])
            return node;
    }
    return nil;
}

static BOOL hasLayoutConvChainSignature(ANEOperationGraph *graph,
                                        ANEScheduledGraph *scheduled) {
    if (scheduled.tasks.count != 3) return NO;
    NSArray<NSString *> *names =
        @[@"space_to_depth", @"conv", @"depth_to_space"];
    for (NSUInteger i = 0; i < names.count; ++i) {
        ANEScheduledTask *task = scheduled.tasks[i];
        if (task.sourceNodeIdentifiers.count != 1) return NO;
        ANEOperationNode *node = nodeNamed(
            graph, task.sourceNodeIdentifiers.firstObject);
        if (!node || ![node.operationName isEqualToString:names[i]]) return NO;
    }
    return YES;
}

static ANEGraphArgument *singleValue(ANEGraphArgument *argument) {
    while ((argument.kind == ANEGraphArgumentKindCall &&
            argument.callArguments.count == 1) ||
           ((argument.kind == ANEGraphArgumentKindList ||
             argument.kind == ANEGraphArgumentKindTuple) &&
            argument.elements.count == 1)) {
        argument = argument.kind == ANEGraphArgumentKindCall
            ? argument.callArguments[0].value : argument.elements[0];
    }
    return argument;
}

static NSUInteger constantOperand(ANEOperationNode *node, NSString *name) {
    ANEGraphValue *value = node.sourceOperation.operands[name].value;
    if (!value) return 0;
    for (ANEOperationNode *input in node.inputs) {
        if (![input.identifier isEqualToString:value.name] ||
            !input.sourceOperation) continue;
        NSString *text = singleValue(
            input.sourceOperation.attributes[@"val"]).text;
        return text ? (NSUInteger)text.longLongValue : 0;
    }
    return 0;
}

static ANEOperationNode *convWeight(ANEOperationNode *conv,
                                    BOOL quantized) {
    for (ANEOperationNode *input in conv.inputs) {
        NSArray<NSNumber *> *shape = input.outputType.shape;
        BOOL constantKind = input.kind == ANEOperationKindConstant ||
            (quantized && [input.operationName isEqualToString:
                           @"constexpr_affine_dequantize"]);
        if (constantKind && shape.count == 4) return input;
    }
    return nil;
}

static ANEHWXArtifact *encodeDepthwiseProgram(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    NSURL *modelRoot, ANEDiagnosticEngine *diagnostics) {
    NSError *error = nil;
    if (![H16GDecodedFormValidator validateFP16DepthwiseGraph:graph
        scheduled:scheduled error:&error]) {
        emitError(diagnostics,@"h16g.encode.illegal-depthwise-form",
                  error.localizedDescription,graph.nodes.firstObject);
        return nil;
    }
    ANEScheduledTask *task = scheduled.tasks.firstObject;
    ANEOperationNode *conv = convSourceForTask(graph,task);
    ANEOperationNode *weight = convWeight(conv,NO);
    ANEGraphValue *input = graph.sourceFunction.inputs.firstObject;
    ANEOperationNode *result = nodeNamed(graph,task.nodeIdentifier);
    NSUInteger channels = input.type.shape[1].unsignedIntegerValue;
    NSData *raw = [ANEBlobResolver loadConstantForOperation:weight.sourceOperation
        expectedBytes:channels*9*2 modelRoot:modelRoot diagnostics:diagnostics];
    NSData *packed = raw ? [H16GConstantPacker
        packDepthwise3x3Weights:raw channels:channels error:&error] : nil;
    NSData *td = packed ? [H16GDepthwiseEncoder
        encode3x3WithChannels:channels spatial:64 error:&error] : nil;
    if (!td) {
        if (raw) emitError(diagnostics,@"h16g.encode.depthwise-fields",
            error.localizedDescription ?: @"cannot encode depthwise 3x3",conv);
        return nil;
    }
    NSArray<HWXObjectBinding *> *objectBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:input.name
            shortName:input.name role:HWXObjectBindingRoleInput
            elementType:input.type.elementType shape:input.type.shape],
        [[HWXObjectBinding alloc]
            initWithSymbol:[result.identifier stringByAppendingString:@"@output"]
            shortName:result.identifier role:HWXObjectBindingRoleOutput
            elementType:result.outputType.elementType shape:result.outputType.shape],
    ];
    NSData *image = [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:packed bindings:objectBindings
        kernelRelocationOffsets:@[@0x1b0] programRecordCount:0x10
        programFormatCode:0x0f error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.depthwise-object",
            error.localizedDescription ?: @"cannot construct depthwise HWX object",conv);
        return nil;
    }
    NSUInteger inputBytes = tensorBytes(input.type);
    NSUInteger outputBytes = tensorBytes(result.outputType);
    NSUInteger width = input.type.shape[3].unsignedIntegerValue;
    NSUInteger height = input.type.shape[2].unsignedIntegerValue;
    NSArray<ANEHWXBinding *> *runtimeBindings = @[
        [[ANEHWXBinding alloc] initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:inputBytes
            allocationByteLength:alignUp(inputBytes,0x4000) ioSurfaceIndex:0
            rowStrideBytes:width*2 planeStrideBytes:width*height*2
            batchStrideBytes:inputBytes],
        [[ANEHWXBinding alloc] initWithIdentifier:weight.identifier
            role:ANESurfaceRoleWeight logicalByteLength:packed.length
            allocationByteLength:packed.length ioSurfaceIndex:-1],
        [[ANEHWXBinding alloc] initWithIdentifier:result.identifier
            role:ANESurfaceRoleOutput logicalByteLength:outputBytes
            allocationByteLength:alignUp(outputBytes,0x4000) ioSurfaceIndex:1
            rowStrideBytes:width*2 planeStrideBytes:width*height*2
            batchStrideBytes:outputBytes],
    ];
    return [[ANEHWXArtifact alloc] initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeRegularConvProgram(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    NSURL *modelRoot, ANEDiagnosticEngine *diagnostics) {
    NSError *error = nil;
    if (![H16GDecodedFormValidator validateFP16RegularConvGraph:graph
        scheduled:scheduled error:&error]) {
        emitError(diagnostics,@"h16g.encode.illegal-regular-conv-form",
                  error.localizedDescription,graph.nodes.firstObject);
        return nil;
    }
    ANEScheduledTask *task = scheduled.tasks.firstObject;
    ANEOperationNode *conv = convSourceForTask(graph,task);
    ANEOperationNode *weight = convWeight(conv,NO);
    ANEGraphValue *input = graph.sourceFunction.inputs.firstObject;
    ANEOperationNode *result = nodeNamed(graph,task.nodeIdentifier);
    NSUInteger channels = input.type.shape[1].unsignedIntegerValue;
    NSUInteger spatial = input.type.shape[3].unsignedIntegerValue;
    NSUInteger kernel = weight.outputType.shape[2].unsignedIntegerValue;
    NSUInteger rawBytes = channels*channels*kernel*kernel*2;
    NSData *raw = [ANEBlobResolver loadConstantForOperation:weight.sourceOperation
        expectedBytes:rawBytes modelRoot:modelRoot diagnostics:diagnostics];
    NSData *packed = raw ? [H16GConstantPacker packRegularConvWeights:raw
        inputChannels:channels outputChannels:channels
        kernelHeight:kernel kernelWidth:kernel error:&error] : nil;
    NSData *td = packed ? [H16GRegularConvEncoder
        encodeWithChannels:channels spatial:spatial kernel:kernel error:&error] : nil;
    if (!td) {
        if (raw) emitError(diagnostics,@"h16g.encode.regular-conv-fields",
            error.localizedDescription ?: @"cannot encode regular Conv",conv);
        return nil;
    }
    NSArray<HWXObjectBinding *> *objectBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:input.name
            shortName:input.name role:HWXObjectBindingRoleInput
            elementType:input.type.elementType shape:input.type.shape],
        [[HWXObjectBinding alloc]
            initWithSymbol:[result.identifier stringByAppendingString:@"@output"]
            shortName:result.identifier role:HWXObjectBindingRoleOutput
            elementType:result.outputType.elementType shape:result.outputType.shape],
    ];
    NSUInteger relocation = [H16GRegularConvEncoder
        kernelRelocationOffsetForChannels:channels];
    NSData *image = [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:packed bindings:objectBindings
        kernelRelocationOffsets:@[@(relocation)] programRecordCount:0x10
        programFormatCode:0x0f error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.regular-conv-object",
            error.localizedDescription ?: @"cannot construct regular Conv HWX object",conv);
        return nil;
    }
    NSUInteger inputBytes = tensorBytes(input.type);
    NSUInteger outputBytes = tensorBytes(result.outputType);
    NSArray<ANEHWXBinding *> *runtimeBindings = @[
        [[ANEHWXBinding alloc] initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:inputBytes
            allocationByteLength:alignUp(inputBytes,0x4000) ioSurfaceIndex:0
            rowStrideBytes:spatial*2 planeStrideBytes:spatial*spatial*2
            batchStrideBytes:inputBytes],
        [[ANEHWXBinding alloc] initWithIdentifier:weight.identifier
            role:ANESurfaceRoleWeight logicalByteLength:packed.length
            allocationByteLength:packed.length ioSurfaceIndex:-1],
        [[ANEHWXBinding alloc] initWithIdentifier:result.identifier
            role:ANESurfaceRoleOutput logicalByteLength:outputBytes
            allocationByteLength:alignUp(outputBytes,0x4000) ioSurfaceIndex:1
            rowStrideBytes:spatial*2 planeStrideBytes:spatial*spatial*2
            batchStrideBytes:outputBytes],
    ];
    return [[ANEHWXArtifact alloc] initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeSquareMatmulProgram(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    ANEDiagnosticEngine *diagnostics) {
    NSError *error = nil;
    if (![H16GDecodedFormValidator validateFP16SquareMatmulGraph:graph
        scheduled:scheduled error:&error]) {
        emitError(diagnostics,@"h16g.encode.illegal-square-matmul-form",
            error.localizedDescription,graph.nodes.firstObject);
        return nil;
    }
    ANEScheduledTask *task = scheduled.tasks.firstObject;
    ANEOperationNode *matmul = nodeNamed(graph,task.nodeIdentifier);
    ANEGraphValue *left = matmul.sourceOperation.operands[@"x"].value;
    ANEGraphValue *right = matmul.sourceOperation.operands[@"y"].value;
    NSUInteger size = matmul.outputType.shape[1].unsignedIntegerValue;
    H16GEncodedTDProgram *program = [H16GMatmulEncoder
        encodeSquareSize:size error:&error];
    if (!program) {
        emitError(diagnostics,@"h16g.encode.matmul-fields",
            error.localizedDescription ?: @"cannot encode square matmul",matmul);
        return nil;
    }
    NSArray<HWXObjectBinding *> *objectBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:left.name shortName:left.name
            role:HWXObjectBindingRoleInput elementType:left.type.elementType
            shape:left.type.shape],
        [[HWXObjectBinding alloc] initWithSymbol:right.name shortName:right.name
            role:HWXObjectBindingRoleInput elementType:right.type.elementType
            shape:right.type.shape],
        [[HWXObjectBinding alloc]
            initWithSymbol:[matmul.identifier stringByAppendingString:@"@output"]
            shortName:matmul.identifier role:HWXObjectBindingRoleOutput
            elementType:matmul.outputType.elementType shape:matmul.outputType.shape],
    ];
    NSUInteger tensorByteLength = size*size*2;
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:[H16GMatmulEncoder tileCountForSquareSize:size]+1
        recordCount:program.programRecordCount
        formatCode:program.programFormatCode
        scratchByteLength:program.scratchByteLength
        scratchAllocationByteLength:MAX(tensorByteLength,program.scratchByteLength)
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSData *image = [HWXObjectWriter
        buildObjectWithTaskDescriptor:program.data constantRegion:[NSData data]
        bindings:objectBindings kernelRelocationOffsets:@[] programInfo:info
        error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.matmul-object",
            error.localizedDescription ?: @"cannot construct matmul HWX object",
            matmul);
        return nil;
    }
    NSUInteger rowStride = size*2;
    NSArray<ANEHWXBinding *> *runtimeBindings = @[
        [[ANEHWXBinding alloc] initWithIdentifier:left.name
            role:ANESurfaceRoleInput logicalByteLength:tensorByteLength
            allocationByteLength:alignUp(tensorByteLength,0x4000)
            ioSurfaceIndex:0 rowStrideBytes:rowStride
            planeStrideBytes:tensorByteLength batchStrideBytes:tensorByteLength],
        [[ANEHWXBinding alloc] initWithIdentifier:right.name
            role:ANESurfaceRoleInput logicalByteLength:tensorByteLength
            allocationByteLength:alignUp(tensorByteLength,0x4000)
            ioSurfaceIndex:1 rowStrideBytes:rowStride
            planeStrideBytes:tensorByteLength batchStrideBytes:tensorByteLength],
        [[ANEHWXBinding alloc] initWithIdentifier:matmul.identifier
            role:ANESurfaceRoleOutput logicalByteLength:tensorByteLength
            allocationByteLength:alignUp(tensorByteLength,0x4000)
            ioSurfaceIndex:2 rowStrideBytes:rowStride
            planeStrideBytes:tensorByteLength batchStrideBytes:tensorByteLength],
    ];
    return [[ANEHWXArtifact alloc] initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeBinaryALUProgram(
    ANEOperationGraph *graph,ANEScheduledGraph *scheduled,
    ANEDiagnosticEngine *diagnostics) {
    NSError *error=nil;
    if(![H16GDecodedFormValidator validateFP16BinaryALUGraph:graph
        scheduled:scheduled error:&error]){
        emitError(diagnostics,@"h16g.encode.illegal-binary-alu-form",
            error.localizedDescription,graph.nodes.firstObject);return nil;
    }
    ANEOperationNode *operation=nodeNamed(graph,scheduled.tasks[0].nodeIdentifier);
    ANEGraphValue *left=operation.sourceOperation.operands[@"x"].value;
    ANEGraphValue *right=operation.sourceOperation.operands[@"y"].value;
    NSUInteger elements=H16GElementCount(operation.outputType.shape);
    NSUInteger size=[H16GALUEncoder squareViewSizeForElementCount:elements];
    NSArray<NSNumber *> *view=@[@1,@1,@(size),@(size)];
    H16GEncodedTDProgram *program=[H16GALUEncoder
        encodeOperationName:operation.operationName squareSize:size error:&error];
    if(!program){emitError(diagnostics,@"h16g.encode.alu-fields",
        error.localizedDescription ?: @"cannot encode binary ALU",operation);return nil;}
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:left.name shortName:left.name
            role:HWXObjectBindingRoleInput elementType:left.type.elementType
            shape:view],
        [[HWXObjectBinding alloc]
            initWithSymbol:[operation.identifier stringByAppendingString:@"@output"]
            shortName:operation.identifier role:HWXObjectBindingRoleOutput
            elementType:operation.outputType.elementType shape:view],
        [[HWXObjectBinding alloc]initWithSymbol:right.name shortName:right.name
            role:HWXObjectBindingRoleInput elementType:right.type.elementType
            shape:view],
    ];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:1 recordCount:program.programRecordCount
        formatCode:program.programFormatCode scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:program.data
        constantRegion:[NSData data] bindings:objectBindings
        kernelRelocationOffsets:@[] programInfo:info error:&error];
    if(!image){emitError(diagnostics,@"h16g.encode.alu-object",
        error.localizedDescription ?: @"cannot construct binary ALU HWX",operation);
        return nil;}
    // A non-square operand runs as a contiguous [1,1,N,N] view of its
    // bytes: logical bytes cover the tensor, the allocation covers the view.
    NSUInteger bytes=size*size*2,rowBytes=size*2,logical=elements*2;
    NSArray<ANEHWXBinding *> *runtimeBindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:left.name role:ANESurfaceRoleInput
            logicalByteLength:logical allocationByteLength:alignUp(bytes,0x4000)
            ioSurfaceIndex:0 rowStrideBytes:rowBytes planeStrideBytes:bytes
            batchStrideBytes:bytes],
        [[ANEHWXBinding alloc]initWithIdentifier:right.name role:ANESurfaceRoleInput
            logicalByteLength:logical allocationByteLength:alignUp(bytes,0x4000)
            ioSurfaceIndex:1 rowStrideBytes:rowBytes planeStrideBytes:bytes
            batchStrideBytes:bytes],
        [[ANEHWXBinding alloc]initWithIdentifier:operation.identifier
            role:ANESurfaceRoleOutput logicalByteLength:logical
            allocationByteLength:alignUp(bytes,0x4000) ioSurfaceIndex:2
            rowStrideBytes:rowBytes planeStrideBytes:bytes batchStrideBytes:bytes],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeUnaryPointwiseProgram(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    ANEDiagnosticEngine *diagnostics) {
    NSError *error=nil;
    if (![H16GDecodedFormValidator validateFP16UnaryPointwiseGraph:graph
        scheduled:scheduled error:&error]) {
        emitError(diagnostics,@"h16g.encode.illegal-unary-pointwise-form",
            error.localizedDescription,graph.nodes.firstObject);
        return nil;
    }
    ANEOperationNode *operation=nodeNamed(graph,
        scheduled.tasks[0].sourceNodeIdentifiers.firstObject);
    ANEGraphValue *input=operation.sourceOperation.operands[@"x"].value;
    NSUInteger elements=H16GElementCount(input.type.shape);
    NSUInteger size=[H16GLUTEncoder
        squareViewSizeForOperationName:operation.operationName elementCount:elements];
    NSArray<NSNumber *> *view=@[@1,@1,@(size),@(size)];
    H16GEncodedTDProgram *program=[H16GLUTEncoder
        encodeOperationName:operation.operationName inputShape:view
        error:&error];
    NSData *constants=[H16GLUTEncoder
        constantRegionForOperationName:operation.operationName error:&error];
    if (!program || !constants) {
        emitError(diagnostics,@"h16g.encode.unary-pointwise-fields",
            error.localizedDescription ?: @"cannot encode unary pointwise task",
            operation);
        return nil;
    }
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:input.name shortName:input.name
            role:HWXObjectBindingRoleInput elementType:input.type.elementType
            shape:view],
        [[HWXObjectBinding alloc]
            initWithSymbol:[operation.identifier stringByAppendingString:@"@output"]
            shortName:operation.identifier role:HWXObjectBindingRoleOutput
            elementType:operation.outputType.elementType shape:view],
    ];
    NSUInteger taskCount=[operation.operationName isEqualToString:@"log"] ? 2 : 1;
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:taskCount recordCount:program.programRecordCount
        formatCode:program.programFormatCode scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:program.data
        constantRegion:constants bindings:objectBindings
        kernelRelocationOffsets:program.kernelRelocationOffsets
        programInfo:info error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.unary-pointwise-object",
            error.localizedDescription ?: @"cannot construct unary pointwise HWX",
            operation);
        return nil;
    }
    // Contiguous [1,1,N,N] view, as for the binary ALU form.
    NSUInteger bytes=size*size*2,logical=elements*2;
    NSUInteger rowBytes=size*2;
    NSUInteger planeBytes=bytes;
    NSArray<ANEHWXBinding *> *runtimeBindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:logical
            allocationByteLength:alignUp(bytes,0x4000) ioSurfaceIndex:0
            rowStrideBytes:rowBytes planeStrideBytes:planeBytes
            batchStrideBytes:bytes],
        [[ANEHWXBinding alloc]initWithIdentifier:operation.identifier
            role:ANESurfaceRoleOutput logicalByteLength:logical
            allocationByteLength:alignUp(bytes,0x4000) ioSurfaceIndex:1
            rowStrideBytes:rowBytes planeStrideBytes:planeBytes
            batchStrideBytes:bytes],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeReductionProgram(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    ANEDiagnosticEngine *diagnostics) {
    NSError *error=nil;
    if (![H16GDecodedFormValidator validateFP16ReductionGraph:graph
        scheduled:scheduled error:&error]) {
        emitError(diagnostics,@"h16g.encode.illegal-reduction-form",
            error.localizedDescription,graph.nodes.firstObject);
        return nil;
    }
    ANEOperationNode *operation=nodeNamed(graph,
        scheduled.tasks[0].sourceNodeIdentifiers.firstObject);
    ANEGraphValue *input=operation.sourceOperation.operands[@"x"].value;
    NSUInteger axis=constantOperand(operation,@"axes");
    H16GReduceEncoding *encoding=[H16GReduceEncoder
        encodeOperationName:operation.operationName inputShape:input.type.shape
        axis:axis error:&error];
    if (!encoding) {
        emitError(diagnostics,@"h16g.encode.reduction-fields",
            error.localizedDescription ?: @"cannot encode reduction task",operation);
        return nil;
    }
    H16GEncodedTDProgram *program=encoding.tdProgram;
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:input.name shortName:input.name
            role:HWXObjectBindingRoleInput elementType:input.type.elementType
            shape:input.type.shape
            rowStrideBytes:encoding.inputRowStrideBytes
            planeStrideBytes:encoding.inputPlaneStrideBytes
            batchStrideBytes:encoding.inputBatchStrideBytes
            storageByteLength:encoding.inputStorageByteLength],
        [[HWXObjectBinding alloc]
            initWithSymbol:[operation.identifier stringByAppendingString:@"@output"]
            shortName:operation.identifier role:HWXObjectBindingRoleOutput
            elementType:operation.outputType.elementType shape:encoding.outputShape
            rowStrideBytes:encoding.outputRowStrideBytes
            planeStrideBytes:encoding.outputPlaneStrideBytes
            batchStrideBytes:encoding.outputBatchStrideBytes
            storageByteLength:encoding.outputStorageByteLength],
    ];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:encoding.taskCount
        recordCount:program.programRecordCount
        formatCode:program.programFormatCode scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:program.data
        constantRegion:[NSData data] bindings:objectBindings
        kernelRelocationOffsets:@[] programInfo:info error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.reduction-object",
            error.localizedDescription ?: @"cannot construct reduction HWX",operation);
        return nil;
    }
    NSUInteger inputBytes=tensorBytes(input.type);
    NSUInteger outputLogicalBytes=tensorBytes(operation.outputType);
    NSArray<ANEHWXBinding *> *runtimeBindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:inputBytes
            allocationByteLength:alignUp(encoding.inputStorageByteLength,0x4000)
            ioSurfaceIndex:0 rowStrideBytes:encoding.inputRowStrideBytes
            planeStrideBytes:encoding.inputPlaneStrideBytes
            batchStrideBytes:encoding.inputBatchStrideBytes],
        [[ANEHWXBinding alloc]initWithIdentifier:operation.identifier
            role:ANESurfaceRoleOutput logicalByteLength:outputLogicalBytes
            allocationByteLength:alignUp(encoding.outputStorageByteLength,0x4000)
            ioSurfaceIndex:1 rowStrideBytes:encoding.outputRowStrideBytes
            planeStrideBytes:encoding.outputPlaneStrideBytes
            batchStrideBytes:encoding.outputBatchStrideBytes],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeQuantizedConvChain(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    NSURL *modelRoot, ANEDiagnosticEngine *diagnostics) {
    NSUInteger depth = scheduled.tasks.count;
    NSMutableArray<ANEOperationNode *> *convs = [NSMutableArray array];
    NSMutableArray<ANEOperationNode *> *weights = [NSMutableArray array];
    NSMutableData *constantRegion = [NSMutableData data];
    NSError *error = nil;
    if (![H16GDecodedFormValidator validateW8A8ConvGraph:graph
        scheduled:scheduled error:&error]) {
        emitError(diagnostics,@"h16g.encode.illegal-w8a8-form",
                  error.localizedDescription,graph.nodes.firstObject);
        return nil;
    }
    for (NSUInteger i = 0; i < depth; ++i) {
        ANEScheduledTask *task = scheduled.tasks[i];
        ANELegalNumericMode expectedMode = i == 0
            ? ANELegalNumericModeW8A8InputBoundary
            : (i + 1 == depth ? ANELegalNumericModeW8A8OutputBoundary
                              : ANELegalNumericModeW8A8Packed);
        if (task.operationKind != ANEOperationKindConv ||
            task.numericMode != expectedMode ||
            (i > 0 && ![task.dependencies isEqualToArray:@[@(i - 1)]])) {
            emitError(diagnostics, @"h16g.encode.invalid-conv-chain",
                      @"quantized Conv tasks must form one linear boundary-to-boundary chain",
                      graph.nodes.firstObject);
            return nil;
        }
        ANEOperationNode *conv = convSourceForTask(graph, task);
        ANEOperationNode *weight = conv ? convWeight(conv, YES) : nil;
        NSArray<NSNumber *> *shape = weight.outputType.shape;
        NSArray<NSNumber *> *outputShape = conv.outputType.shape;
        BOOL decodedGeometry = conv && weight && weight.sourceOperation &&
            shape.count == 4 && outputShape.count == 4 &&
            shape[0].unsignedIntegerValue == 64 &&
            shape[1].unsignedIntegerValue == 64 &&
            outputShape[1].unsignedIntegerValue == 64 &&
            outputShape[2].unsignedIntegerValue == 64 &&
            outputShape[3].unsignedIntegerValue == 64;
        if (!decodedGeometry) {
            emitError(diagnostics, @"h16g.encode.unsupported-conv-chain-geometry",
                      @"decoded quantized Conv-chain fields require C64 and S64",
                      conv ?: graph.nodes.firstObject);
            return nil;
        }
        NSData *raw = [ANEBlobResolver
            loadConstantForOperation:weight.sourceOperation
            expectedBytes:64 * 64 modelRoot:modelRoot
            diagnostics:diagnostics];
        NSData *packed = raw ? [H16GConstantPacker packConv1x1Weights:raw
            inputChannels:64 outputChannels:64 bytesPerWeight:1
            packingFormat:H16GConvWeightPackingFormatW8A8 error:&error] : nil;
        if (!packed) {
            if (raw) emitError(diagnostics, @"h16g.encode.conv-chain-weights",
                error.localizedDescription ?: @"cannot pack quantized Conv weights",
                weight);
            return nil;
        }
        [convs addObject:conv];
        [weights addObject:weight];
        [constantRegion appendData:packed];
    }

    H16GEncodedTDProgram *program = [H16GConvChainEncoder
        encodeW8A8C64S64WithDepth:depth error:&error];
    ANEGraphValue *inputValue = graph.sourceFunction.inputs.firstObject;
    ANEOperationNode *result = convs.lastObject;
    if (!program || !inputValue || !result) {
        emitError(diagnostics, @"h16g.encode.conv-chain-fields",
                  error.localizedDescription ?: @"cannot encode quantized Conv chain",
                  result ?: graph.nodes.firstObject);
        return nil;
    }
    NSArray<HWXObjectBinding *> *objectBindings = @[
        [[HWXObjectBinding alloc]
            initWithSymbol:[result.identifier stringByAppendingString:@"@output"]
            shortName:result.identifier role:HWXObjectBindingRoleOutput
            elementType:result.outputType.elementType shape:result.outputType.shape],
        [[HWXObjectBinding alloc] initWithSymbol:inputValue.name
            shortName:inputValue.name role:HWXObjectBindingRoleInput
            elementType:inputValue.type.elementType shape:inputValue.type.shape],
    ];
    NSData *image = [HWXObjectWriter
        buildObjectWithTaskDescriptor:program.data
                       constantRegion:constantRegion
                             bindings:objectBindings
              kernelRelocationOffsets:program.kernelRelocationOffsets
                  programRecordCount:program.programRecordCount
                   programFormatCode:program.programFormatCode
                                error:&error];
    if (!image) {
        emitError(diagnostics, @"h16g.encode.object",
                  error.localizedDescription ?: @"cannot construct HWX object",
                  result);
        return nil;
    }

    NSUInteger inputBytes = tensorBytes(inputValue.type);
    NSUInteger outputBytes = tensorBytes(result.outputType);
    NSUInteger inputWidth = inputValue.type.shape.lastObject.unsignedIntegerValue;
    NSUInteger inputHeight = inputValue.type.shape[2].unsignedIntegerValue;
    NSUInteger outputWidth = result.outputType.shape.lastObject.unsignedIntegerValue;
    NSUInteger outputHeight = result.outputType.shape[2].unsignedIntegerValue;
    NSMutableArray<ANEHWXBinding *> *runtimeBindings = [NSMutableArray array];
    [runtimeBindings addObject:[[ANEHWXBinding alloc]
        initWithIdentifier:inputValue.name role:ANESurfaceRoleInput
        logicalByteLength:inputBytes allocationByteLength:alignUp(inputBytes,0x4000)
        ioSurfaceIndex:0 rowStrideBytes:inputWidth*elementBytes(inputValue.type.elementType)
        planeStrideBytes:inputWidth*inputHeight*elementBytes(inputValue.type.elementType)
        batchStrideBytes:inputBytes]];
    for (ANEOperationNode *weight in weights)
        [runtimeBindings addObject:[[ANEHWXBinding alloc]
            initWithIdentifier:weight.identifier role:ANESurfaceRoleWeight
            logicalByteLength:64*64 allocationByteLength:64*64 ioSurfaceIndex:-1]];
    [runtimeBindings addObject:[[ANEHWXBinding alloc]
        initWithIdentifier:result.identifier role:ANESurfaceRoleOutput
        logicalByteLength:outputBytes allocationByteLength:alignUp(outputBytes,0x4000)
        ioSurfaceIndex:1 rowStrideBytes:outputWidth*elementBytes(result.outputType.elementType)
        planeStrideBytes:outputWidth*outputHeight*elementBytes(result.outputType.elementType)
        batchStrideBytes:outputBytes]];
    return [[ANEHWXArtifact alloc] initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeLayoutConvChain(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    NSURL *modelRoot, ANEDiagnosticEngine *diagnostics) {
    NSError *error = nil;
    if (![H16GDecodedFormValidator validateLayoutConvChainGraph:graph
        scheduled:scheduled error:&error]) {
        emitError(diagnostics, @"h16g.encode.illegal-layout-conv-form",
                  error.localizedDescription, graph.nodes.firstObject);
        return nil;
    }
    ANEOperationNode *s2d = nodeNamed(
        graph, scheduled.tasks[0].sourceNodeIdentifiers.firstObject);
    ANEOperationNode *conv = nodeNamed(
        graph, scheduled.tasks[1].sourceNodeIdentifiers.firstObject);
    ANEOperationNode *result = nodeNamed(
        graph, scheduled.tasks[2].sourceNodeIdentifiers.firstObject);
    ANEOperationNode *weight = convWeight(conv, NO);
    ANEGraphValue *input = graph.sourceFunction.inputs.firstObject;
    NSUInteger naturalChannels = input.type.shape[1].unsignedIntegerValue;
    NSUInteger spatial = input.type.shape[3].unsignedIntegerValue;
    NSUInteger packedChannels = naturalChannels * 16;
    NSUInteger weightBytes = packedChannels * packedChannels * 2;
    NSData *rawWeights = [ANEBlobResolver
        loadConstantForOperation:weight.sourceOperation
        expectedBytes:weightBytes modelRoot:modelRoot diagnostics:diagnostics];
    NSData *packedWeights = rawWeights ? [H16GConstantPacker
        packConv1x1Weights:rawWeights inputChannels:packedChannels
        outputChannels:packedChannels bytesPerWeight:2
        packingFormat:H16GConvWeightPackingFormatLayoutConv
        error:&error] : nil;
    H16GEncodedTDProgram *program = packedWeights ?
        [H16GLayoutConvChainEncoder encodeNaturalChannels:naturalChannels
            spatial:spatial blockSize:constantOperand(s2d, @"block_size")
            error:&error] : nil;
    if (!program) {
        if (rawWeights) emitError(diagnostics,
            @"h16g.encode.layout-conv-fields",
            error.localizedDescription ?: @"cannot encode fused layout-compute TDs",
            conv);
        return nil;
    }

    NSArray<HWXObjectBinding *> *objectBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:input.name
            shortName:input.name role:HWXObjectBindingRoleInput
            elementType:input.type.elementType shape:input.type.shape],
        [[HWXObjectBinding alloc]
            initWithSymbol:[result.identifier stringByAppendingString:@"@output"]
            shortName:result.identifier role:HWXObjectBindingRoleOutput
            elementType:result.outputType.elementType shape:result.outputType.shape],
    ];
    NSUInteger naturalBytes = tensorBytes(input.type);
    HWXObjectProgramInfo *programInfo = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:7 recordCount:program.programRecordCount
        formatCode:program.programFormatCode
        scratchByteLength:program.scratchByteLength
        scratchAllocationByteLength:naturalBytes * 36
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSData *image = [HWXObjectWriter buildObjectWithTaskDescriptor:program.data
        constantRegion:packedWeights bindings:objectBindings
        kernelRelocationOffsets:program.kernelRelocationOffsets
        programInfo:programInfo error:&error];
    if (!image) {
        emitError(diagnostics, @"h16g.encode.layout-conv-object",
                  error.localizedDescription ?: @"cannot construct fused layout HWX object",
                  result);
        return nil;
    }
    NSUInteger inputWidth = input.type.shape[3].unsignedIntegerValue;
    NSUInteger inputHeight = input.type.shape[2].unsignedIntegerValue;
    NSUInteger outputBytes = tensorBytes(result.outputType);
    NSUInteger outputWidth = result.outputType.shape[3].unsignedIntegerValue;
    NSUInteger outputHeight = result.outputType.shape[2].unsignedIntegerValue;
    NSArray<ANEHWXBinding *> *runtimeBindings = @[
        [[ANEHWXBinding alloc] initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:naturalBytes
            allocationByteLength:alignUp(naturalBytes,0x4000) ioSurfaceIndex:0
            rowStrideBytes:inputWidth*2
            planeStrideBytes:inputWidth*inputHeight*2
            batchStrideBytes:naturalBytes],
        [[ANEHWXBinding alloc] initWithIdentifier:weight.identifier
            role:ANESurfaceRoleWeight logicalByteLength:packedWeights.length
            allocationByteLength:packedWeights.length ioSurfaceIndex:-1],
        [[ANEHWXBinding alloc] initWithIdentifier:result.identifier
            role:ANESurfaceRoleOutput logicalByteLength:outputBytes
            allocationByteLength:alignUp(outputBytes,0x4000) ioSurfaceIndex:1
            rowStrideBytes:outputWidth*2
            planeStrideBytes:outputWidth*outputHeight*2
            batchStrideBytes:outputBytes],
    ];
    return [[ANEHWXArtifact alloc] initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeMixedTaskProgram(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    ANEDiagnosticEngine *diagnostics) {
    NSError *error = nil;
    H16GMixedTaskEncoding *encoding = [H16GMixedTaskEncoder
        encodeGraph:graph scheduled:scheduled error:&error];
    if (!encoding) {
        emitError(diagnostics, @"h16g.encode.unsupported-task-sequence",
                  error.localizedDescription ?: @"unsupported mixed task stream",
                  graph.nodes.firstObject);
        return nil;
    }
    ANEGraphValue *input = graph.sourceFunction.inputs.firstObject;
    ANEOperationNode *output = nodeNamed(graph, graph.outputValueNames.firstObject);
    NSArray<HWXObjectBinding *> *objectBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:input.name
            shortName:input.name role:HWXObjectBindingRoleInput
            elementType:input.type.elementType shape:input.type.shape],
        [[HWXObjectBinding alloc]
            initWithSymbol:[output.identifier stringByAppendingString:@"@output"]
            shortName:output.identifier role:HWXObjectBindingRoleOutput
            elementType:output.outputType.elementType shape:output.outputType.shape],
    ];
    H16GEncodedTDProgram *program = encoding.tdProgram;
    HWXObjectProgramInfo *programInfo = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:scheduled.tasks.count
        recordCount:program.programRecordCount formatCode:program.programFormatCode
        scratchByteLength:encoding.scratchByteLength
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSData *image = [HWXObjectWriter buildObjectWithTaskDescriptor:program.data
        constantRegion:encoding.constantRegion bindings:objectBindings
        kernelRelocationOffsets:program.kernelRelocationOffsets
        programInfo:programInfo error:&error];
    if (!image) {
        emitError(diagnostics, @"h16g.encode.object",
                  error.localizedDescription ?: @"cannot construct mixed HWX object",
                  output);
        return nil;
    }
    NSUInteger inputBytes = tensorBytes(input.type);
    NSUInteger outputBytes = tensorBytes(output.outputType);
    NSUInteger inputWidth = input.type.shape.lastObject.unsignedIntegerValue;
    NSUInteger inputHeight = input.type.shape[input.type.shape.count-2].unsignedIntegerValue;
    NSUInteger outputWidth = output.outputType.shape.lastObject.unsignedIntegerValue;
    NSUInteger outputHeight = output.outputType.shape[output.outputType.shape.count-2].unsignedIntegerValue;
    NSArray<ANEHWXBinding *> *bindings = @[
        [[ANEHWXBinding alloc] initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:inputBytes
            allocationByteLength:alignUp(inputBytes,0x4000) ioSurfaceIndex:0
            rowStrideBytes:inputWidth*2 planeStrideBytes:inputWidth*inputHeight*2
            batchStrideBytes:inputBytes],
        [[ANEHWXBinding alloc] initWithIdentifier:output.identifier
            role:ANESurfaceRoleOutput logicalByteLength:outputBytes
            allocationByteLength:alignUp(outputBytes,0x4000) ioSurfaceIndex:1
            rowStrideBytes:outputWidth*2 planeStrideBytes:outputWidth*outputHeight*2
            batchStrideBytes:outputBytes],
    ];
    return [[ANEHWXArtifact alloc] initWithImage:image bindings:bindings];
}

static ANEHWXArtifact *encodeLayoutProgram(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    ANEDiagnosticEngine *diagnostics) {
    ANEScheduledTask *task = scheduled.tasks.firstObject;
    ANEOperationNode *layout = layoutSourceForTask(graph, task);
    ANEGraphValue *input = layout.sourceOperation.operands[@"x"].value;
    ANEOperationNode *output = nodeNamed(graph, graph.outputValueNames.firstObject);
    if (!layout || !input || input.producer || !output ||
        task.tilePlan.inputShape.count != 4 ||
        task.tilePlan.outputShape.count != 4) {
        emitError(diagnostics, @"h16g.encode.layout-boundary",
                  @"standalone layout program requires one external NCHW input and output",
                  layout ?: graph.nodes.firstObject);
        return nil;
    }
    NSError *error = nil;
    H16GEncodedTDProgram *program = [H16GLayoutEncoder
        encodeOperationName:layout.operationName
        inputShape:task.tilePlan.inputShape
        outputShape:task.tilePlan.outputShape
        blockSize:constantOperand(layout, @"block_size")
        strategy:task.tilePlan.strategy error:&error];
    if (!program) {
        emitError(diagnostics, @"h16g.encode.layout-fields",
                  error.localizedDescription ?: @"cannot encode layout TDs",
                  layout);
        return nil;
    }
    H16GNCHWSurfaceLayout inputLayout=nchwSurfaceLayout(input.type);
    H16GNCHWSurfaceLayout outputLayout=nchwSurfaceLayout(output.outputType);
    NSArray<HWXObjectBinding *> *objectBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:input.name
            shortName:input.name role:HWXObjectBindingRoleInput
            elementType:input.type.elementType shape:input.type.shape
            rowStrideBytes:inputLayout.rowBytes
            planeStrideBytes:inputLayout.planeBytes
            batchStrideBytes:inputLayout.batchBytes
            storageByteLength:inputLayout.storageBytes],
        [[HWXObjectBinding alloc]
            initWithSymbol:[output.identifier stringByAppendingString:@"@output"]
            shortName:output.identifier role:HWXObjectBindingRoleOutput
            elementType:output.outputType.elementType shape:output.outputType.shape
            rowStrideBytes:outputLayout.rowBytes
            planeStrideBytes:outputLayout.planeBytes
            batchStrideBytes:outputLayout.batchBytes
            storageByteLength:outputLayout.storageBytes],
    ];
    HWXObjectProgramInfo *programInfo = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:task.tilePlan.descriptorCount
        recordCount:program.programRecordCount
        formatCode:program.programFormatCode
        scratchByteLength:program.scratchByteLength
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSData *image = [HWXObjectWriter buildObjectWithTaskDescriptor:program.data
        constantRegion:[NSData data] bindings:objectBindings
        kernelRelocationOffsets:program.kernelRelocationOffsets
        programInfo:programInfo error:&error];
    if (!image) {
        emitError(diagnostics, @"h16g.encode.layout-object",
                  error.localizedDescription ?: @"cannot construct layout HWX object",
                  layout);
        return nil;
    }
    NSUInteger inputBytes = tensorBytes(input.type);
    NSUInteger outputBytes = tensorBytes(output.outputType);
    NSArray<ANEHWXBinding *> *runtimeBindings = @[
        [[ANEHWXBinding alloc] initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:inputBytes
            allocationByteLength:alignUp(inputLayout.storageBytes,0x4000)
            ioSurfaceIndex:0 rowStrideBytes:inputLayout.rowBytes
            planeStrideBytes:inputLayout.planeBytes
            batchStrideBytes:inputLayout.batchBytes],
        [[ANEHWXBinding alloc] initWithIdentifier:output.identifier
            role:ANESurfaceRoleOutput logicalByteLength:outputBytes
            allocationByteLength:alignUp(outputLayout.storageBytes,0x4000)
            ioSurfaceIndex:1 rowStrideBytes:outputLayout.rowBytes
            planeStrideBytes:outputLayout.planeBytes
            batchStrideBytes:outputLayout.batchBytes],
    ];
    return [[ANEHWXArtifact alloc] initWithImage:image bindings:runtimeBindings];
}

@implementation H16GProgramEncoder
+ (ANEHWXArtifact *)encodeGraph:(ANEOperationGraph *)graph
                        scheduled:(ANEScheduledGraph *)scheduled
                        modelRoot:(NSURL *)modelRoot
                      diagnostics:(ANEDiagnosticEngine *)diagnostics {
    for (ANEOperationNode *node in graph.nodes)
        if ([node.operationName isEqualToString:@"conv"] &&
            node.outputType.shape.count==3 &&
            graph.sourceFunction.inputs.firstObject.type.shape.count==3 &&
            graph.sourceFunction.inputs.firstObject.type.shape[1].unsignedIntegerValue==80)
            return encodeWhisperStem1Program(graph,modelRoot,diagnostics);
    for (ANEOperationNode *node in graph.nodes)
        if ([node.operationName isEqualToString:@"linear"])
            return encodeTransformerLinearProgram(graph,scheduled,modelRoot,
                                                  diagnostics);
    if (hasLayoutConvChainSignature(graph, scheduled))
        return encodeLayoutConvChain(graph, scheduled, modelRoot, diagnostics);
    BOOL allConv = scheduled.tasks.count >= 3;
    for (ANEScheduledTask *task in scheduled.tasks)
        allConv = allConv && task.operationKind == ANEOperationKindConv;
    if (allConv)
        return encodeQuantizedConvChain(graph, scheduled, modelRoot, diagnostics);
    if (scheduled.tasks.count == 1 &&
        scheduled.tasks[0].operationKind == ANEOperationKindLayout &&
        layoutSourceForTask(graph, scheduled.tasks[0]))
        return encodeLayoutProgram(graph, scheduled, diagnostics);
    if (scheduled.tasks.count == 1 &&
        scheduled.tasks[0].operationKind == ANEOperationKindMatmul)
        return encodeSquareMatmulProgram(graph,scheduled,diagnostics);
    if (scheduled.tasks.count == 1 &&
        scheduled.tasks[0].operationKind == ANEOperationKindReduce)
        return encodeReductionProgram(graph,scheduled,diagnostics);
    if (scheduled.tasks.count == 1) {
        ANEOperationNode *node=nodeNamed(graph,
            scheduled.tasks[0].sourceNodeIdentifiers.firstObject);
        ANEGraphValue *input=node.sourceOperation.operands[@"x"].value;
        if (input && !input.producer &&
            input.type.shape.count >= 1 && input.type.shape.count <= 4 &&
            [H16GLUTEncoder squareViewSizeForOperationName:node.operationName
                elementCount:H16GElementCount(input.type.shape)] != NSNotFound)
            return encodeUnaryPointwiseProgram(graph,scheduled,diagnostics);
    }
    if (scheduled.tasks.count == 1 &&
        scheduled.tasks[0].operationKind == ANEOperationKindALU)
        return encodeBinaryALUProgram(graph,scheduled,diagnostics);
    if (scheduled.tasks.count != 1 ||
        scheduled.tasks[0].operationKind != ANEOperationKindConv)
        return encodeMixedTaskProgram(graph, scheduled, diagnostics);
    ANEScheduledTask *task = scheduled.tasks[0];
    ANEOperationNode *candidateConv = convSourceForTask(graph,task);
    ANEOperationNode *candidateWeight = convWeight(candidateConv,NO);
    NSArray<NSNumber *> *candidateWeightShape = candidateWeight.outputType.shape;
    if (candidateWeightShape.count == 4) {
        NSUInteger kernelHeight = candidateWeightShape[2].unsignedIntegerValue;
        NSUInteger kernelWidth = candidateWeightShape[3].unsignedIntegerValue;
        if (kernelHeight == 3 && kernelWidth == 3 &&
            candidateWeightShape[1].unsignedIntegerValue == 1)
            return encodeDepthwiseProgram(graph,scheduled,modelRoot,diagnostics);
        if ((kernelHeight == 3 || kernelHeight == 5) &&
            kernelWidth == kernelHeight &&
            candidateWeightShape[1].unsignedIntegerValue ==
                candidateWeightShape[0].unsignedIntegerValue)
            return encodeRegularConvProgram(graph,scheduled,modelRoot,diagnostics);
    }
    NSError *legalityError = nil;
    if (![H16GDecodedFormValidator validateFP16ConvGraph:graph
        scheduled:scheduled error:&legalityError]) {
        emitError(diagnostics,@"h16g.encode.illegal-fp16-conv-form",
                  legalityError.localizedDescription,graph.nodes.firstObject);
        return nil;
    }
    ANEOperationNode *conv = nil;
    BOOL reluEpilogue = NO;
    for (NSString *identifier in task.sourceNodeIdentifiers) {
        ANEOperationNode *node = nodeNamed(graph, identifier);
        if (node.kind == ANEOperationKindConv) conv = node;
        if ([node.operationName isEqualToString:@"relu"]) reluEpilogue = YES;
    }
    if (!conv || conv.outputType.shape.count != 4) {
        emitError(diagnostics, @"h16g.encode.invalid-conv",
                  @"scheduled Conv task has no four-dimensional Conv source",
                  conv ?: graph.nodes.firstObject);
        return nil;
    }
    ANEOperationNode *weight = nil;
    weight = convWeight(conv, NO);
    if (!weight || !weight.sourceOperation) {
        emitError(diagnostics, @"h16g.encode.missing-conv-weight",
                  @"Conv task has no 1x1 constant weight operand", conv);
        return nil;
    }
    NSArray<NSNumber *> *weightShape = weight.outputType.shape;
    NSUInteger outputChannels = weightShape[0].unsignedIntegerValue;
    NSUInteger inputChannels = weightShape[1].unsignedIntegerValue;
    NSArray<NSNumber *> *outputShape = conv.outputType.shape;
    NSUInteger height = outputShape[2].unsignedIntegerValue;
    NSUInteger width = outputShape[3].unsignedIntegerValue;
    NSUInteger bytesPerWeight = elementBytes(weight.outputType.elementType);
    if (height != width || outputShape[1].unsignedIntegerValue != outputChannels) {
        emitError(diagnostics, @"h16g.encode.unsupported-conv-shape",
                  @"decoded Conv encoder requires square NCHW output geometry",
                  conv);
        return nil;
    }
    NSUInteger weightBytes = inputChannels * outputChannels * bytesPerWeight;
    NSData *rawWeights = [ANEBlobResolver
        loadConstantForOperation:weight.sourceOperation
        expectedBytes:weightBytes modelRoot:modelRoot diagnostics:diagnostics];
    if (!rawWeights) return nil;
    NSError *error = nil;
    NSData *packed = [H16GConstantPacker packConv1x1Weights:rawWeights
        inputChannels:inputChannels outputChannels:outputChannels
        bytesPerWeight:bytesPerWeight error:&error];
    NSData *td = packed ? [H16GConvEncoder
        encodeConv1x1WithInputChannels:inputChannels
        outputChannels:outputChannels spatial:width
        bytesPerWeight:bytesPerWeight numericMode:task.numericMode
        reluEpilogue:reluEpilogue error:&error] : nil;
    if (!td) {
        emitError(diagnostics, @"h16g.encode.conv-fields",
                  error.localizedDescription ?: @"cannot encode Conv TD", conv);
        return nil;
    }
    ANEGraphValue *inputValue = graph.sourceFunction.inputs.firstObject;
    ANEOperationNode *result = nodeNamed(graph, task.nodeIdentifier);
    if (!inputValue || !result) {
        emitError(diagnostics, @"h16g.encode.missing-boundary",
                  @"scheduled graph has no external input or result", conv);
        return nil;
    }
    NSString *outputSymbol = [result.identifier stringByAppendingString:@"@output"];
    NSArray<HWXObjectBinding *> *objectBindings = @[
        [[HWXObjectBinding alloc] initWithSymbol:inputValue.name
            shortName:inputValue.name role:HWXObjectBindingRoleInput
            elementType:inputValue.type.elementType shape:inputValue.type.shape],
        [[HWXObjectBinding alloc] initWithSymbol:outputSymbol
            shortName:result.identifier role:HWXObjectBindingRoleOutput
            elementType:result.outputType.elementType shape:result.outputType.shape],
    ];
    NSData *image = [HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:packed bindings:objectBindings error:&error];
    if (!image) {
        emitError(diagnostics, @"h16g.encode.object",
                  error.localizedDescription ?: @"cannot construct HWX object", conv);
        return nil;
    }
    NSUInteger inputBytes = tensorBytes(inputValue.type);
    NSUInteger outputBytes = tensorBytes(result.outputType);
    NSUInteger inputWidth = inputValue.type.shape.lastObject.unsignedIntegerValue;
    NSUInteger inputHeight = inputValue.type.shape.count >= 2
        ? inputValue.type.shape[inputValue.type.shape.count - 2].unsignedIntegerValue : 1;
    NSUInteger outputWidth = result.outputType.shape.lastObject.unsignedIntegerValue;
    NSUInteger outputHeight = result.outputType.shape.count >= 2
        ? result.outputType.shape[result.outputType.shape.count - 2].unsignedIntegerValue : 1;
    NSUInteger inputElementBytes = elementBytes(inputValue.type.elementType);
    NSUInteger outputElementBytes = elementBytes(result.outputType.elementType);
    NSArray<ANEHWXBinding *> *runtimeBindings = @[
        [[ANEHWXBinding alloc] initWithIdentifier:inputValue.name
            role:ANESurfaceRoleInput logicalByteLength:inputBytes
            allocationByteLength:alignUp(inputBytes, 0x4000) ioSurfaceIndex:0
            rowStrideBytes:inputWidth * inputElementBytes
            planeStrideBytes:inputWidth * inputHeight * inputElementBytes
            batchStrideBytes:inputBytes],
        [[ANEHWXBinding alloc] initWithIdentifier:weight.identifier
            role:ANESurfaceRoleWeight logicalByteLength:packed.length
            allocationByteLength:packed.length ioSurfaceIndex:-1],
        [[ANEHWXBinding alloc] initWithIdentifier:result.identifier
            role:ANESurfaceRoleOutput logicalByteLength:outputBytes
            allocationByteLength:alignUp(outputBytes, 0x4000) ioSurfaceIndex:1
            rowStrideBytes:outputWidth * outputElementBytes
            planeStrideBytes:outputWidth * outputHeight * outputElementBytes
            batchStrideBytes:outputBytes],
    ];
    return [[ANEHWXArtifact alloc] initWithImage:image bindings:runtimeBindings];
}
@end
