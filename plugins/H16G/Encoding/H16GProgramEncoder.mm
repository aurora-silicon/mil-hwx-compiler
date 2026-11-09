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

static NSArray<NSString *> *listLiterals(ANEGraphArgument *argument) {
    while (argument.kind==ANEGraphArgumentKindCall &&
           argument.callArguments.count==1)
        argument=argument.callArguments.firstObject.value;
    if (argument.kind!=ANEGraphArgumentKindList) return nil;
    NSMutableArray<NSString *> *result=[NSMutableArray array];
    for (ANEGraphArgument *element in argument.elements) {
        NSString *literal=scalarLiteral(element);
        if (!literal) return nil;
        [result addObject:literal];
    }
    return result;
}

static BOOL constantListOperandIs(ANEOperationGraph *graph,
                                  ANEOperationNode *operation,
                                  NSString *name,
                                  NSArray<NSString *> *expected) {
    ANEOperationNode *constant=producerForOperand(graph,operation,name);
    return constant.sourceOperation &&
        [listLiterals(constant.sourceOperation.attributes[@"val"])
            isEqualToArray:expected];
}

static BOOL constantScalarOperandIs(ANEOperationGraph *graph,
                                    ANEOperationNode *operation,
                                    NSString *name, NSString *expected) {
    ANEOperationNode *constant=producerForOperand(graph,operation,name);
    return constant.sourceOperation &&
        [scalarLiteral(constant.sourceOperation.attributes[@"val"])
            isEqualToString:expected];
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

static NSString *operationLabel(ANEOperationNode *node) {
    return node.sourceOperation
        ? scalarLiteral(node.sourceOperation.attributes[@"name"]) : nil;
}

static ANEOperationNode *nodeWithLabel(ANEOperationGraph *graph,
                                       NSString *label) {
    for (ANEOperationNode *node in graph.nodes)
        if ([[operationLabel(node) stringByTrimmingCharactersInSet:
                [NSCharacterSet characterSetWithCharactersInString:@"\""]]
                isEqualToString:label]) return node;
    return nil;
}

static ANEHWXArtifact *encodeResNet50Program(
    ANEOperationGraph *graph, NSURL *modelRoot,
    ANEDiagnosticEngine *diagnostics) {
    struct ConvSpec { __unsafe_unretained NSString *name; uint32_t cin,cout,spatial,kernel,stride; };
    static const ConvSpec specs[] = {
        {@"conv_0",3,64,224,7,2}, {@"conv_1",64,64,56,1,1},
        {@"conv_2",64,64,56,3,1}, {@"conv_3",64,256,56,1,1},
        {@"conv_4",64,256,56,1,1}, {@"conv_5",256,64,56,1,1},
        {@"conv_6",64,64,56,3,1}, {@"conv_7",64,256,56,1,1},
        {@"conv_8",256,64,56,1,1}, {@"conv_9",64,64,56,3,1},
        {@"conv_10",64,256,56,1,1}, {@"conv_14",256,512,56,1,2},
        {@"conv_11",256,128,56,1,1}, {@"conv_12",128,128,56,3,2},
        {@"conv_13",128,512,28,1,1}, {@"conv_15",512,128,28,1,1},
        {@"conv_16",128,128,28,3,1}, {@"conv_17",128,512,28,1,1},
        {@"conv_18",512,128,28,1,1}, {@"conv_19",128,128,28,3,1},
        {@"conv_20",128,512,28,1,1}, {@"conv_21",512,128,28,1,1},
        {@"conv_22",128,128,28,3,1}, {@"conv_23",128,512,28,1,1},
        {@"conv_24",512,256,28,1,1}, {@"conv_25",256,256,28,3,2},
        {@"conv_27",512,1024,28,1,2}, {@"conv_26",256,1024,14,1,1},
        {@"conv_28",1024,256,14,1,1}, {@"conv_29",256,256,14,3,1},
        {@"conv_30",256,1024,14,1,1}, {@"conv_31",1024,256,14,1,1},
        {@"conv_32",256,256,14,3,1}, {@"conv_33",256,1024,14,1,1},
        {@"conv_34",1024,256,14,1,1}, {@"conv_35",256,256,14,3,1},
        {@"conv_36",256,1024,14,1,1}, {@"conv_37",1024,256,14,1,1},
        {@"conv_38",256,256,14,3,1}, {@"conv_39",256,1024,14,1,1},
        {@"conv_40",1024,256,14,1,1}, {@"conv_41",256,256,14,3,1},
        {@"conv_42",256,1024,14,1,1}, {@"conv_46",1024,2048,14,1,2},
        {@"conv_43",1024,512,14,1,1}, {@"conv_44",512,512,14,3,2},
        {@"conv_45",512,2048,7,1,1}, {@"conv_47",2048,512,7,1,1},
        {@"conv_48",512,512,7,3,1}, {@"conv_49",512,2048,7,1,1},
        {@"conv_50",2048,512,7,1,1}, {@"conv_51",512,512,7,3,1},
        {@"conv_52",512,2048,7,1,1},
    };
    if (graph.sourceFunction.inputs.count!=1 || graph.outputValueNames.count!=1 ||
        ![graph.sourceFunction.inputs.firstObject.type.shape
            isEqualToArray:@[@1,@3,@224,@224]]) return nil;
    NSUInteger convCount=0,addCount=0,reluCount=0,poolCount=0,meanCount=0,linearCount=0;
    for (ANEOperationNode *node in graph.nodes) {
        convCount += [node.operationName isEqualToString:@"conv"];
        addCount += [node.operationName isEqualToString:@"add"];
        reluCount += [node.operationName isEqualToString:@"relu"];
        poolCount += [node.operationName isEqualToString:@"max_pool"];
        meanCount += [node.operationName isEqualToString:@"reduce_mean"];
        linearCount += [node.operationName isEqualToString:@"linear"];
    }
    if (convCount!=53 || addCount!=16 || reluCount!=49 || poolCount!=1 ||
        meanCount!=1 || linearCount!=1) return nil;

    NSMutableArray<NSData *> *storage=[NSMutableArray array];
    std::vector<ane::h16::ResNet50ConvWeights> convolutions;
    convolutions.reserve(53);
    for (const auto &spec:specs) {
        ANEOperationNode *conv=nodeWithLabel(graph,spec.name);
        if (!conv || ![conv.operationName isEqualToString:@"conv"])
            return nil;
        ANEOperationNode *weight=producerForOperand(graph,conv,@"weight");
        ANEOperationNode *bias=producerForOperand(graph,conv,@"bias");
        if (!weight || !bias) return nil;
        NSArray<NSNumber *> *inputShape=conv.sourceOperation.operands[@"x"].value.type.shape;
        NSArray<NSNumber *> *outputShape=conv.outputType.shape;
        NSArray<NSNumber *> *weightShape=weight.outputType.shape;
        if (![inputShape isEqualToArray:@[@1,@(spec.cin),@(spec.spatial),@(spec.spatial)]] ||
            ![outputShape isEqualToArray:@[@1,@(spec.cout),
                @((spec.spatial+spec.stride-1)/spec.stride),
                @((spec.spatial+spec.stride-1)/spec.stride)]] ||
            ![weightShape isEqualToArray:@[@(spec.cout),@(spec.cin),
                                           @(spec.kernel),@(spec.kernel)]] ||
            ![bias.outputType.shape isEqualToArray:@[@(spec.cout)]] ||
            !constantListOperandIs(graph,conv,@"strides",
                                   @[[NSString stringWithFormat:@"%u",spec.stride],
                                     [NSString stringWithFormat:@"%u",spec.stride]]) ||
            !constantScalarOperandIs(graph,conv,@"groups",@"1")) return nil;
        const NSUInteger weightBytes=(NSUInteger)spec.cout*spec.cin*spec.kernel*spec.kernel*2;
        NSData *rawWeight=[ANEBlobResolver loadConstantForOperation:weight.sourceOperation
            expectedBytes:weightBytes modelRoot:modelRoot diagnostics:diagnostics];
        NSData *rawBias=[ANEBlobResolver loadConstantForOperation:bias.sourceOperation
            expectedBytes:spec.cout*2 modelRoot:modelRoot diagnostics:diagnostics];
        if (!rawWeight || !rawBias) return nil;
        [storage addObject:rawWeight]; [storage addObject:rawBias];
        ane::h16::ConvShape shape;
        shape.input={spec.cin,spec.spatial,spec.spatial};
        const uint32_t out=(spec.spatial+spec.stride-1)/spec.stride;
        shape.output={spec.cout,out,out}; shape.kernel=shape.kernelWidth=spec.kernel;
        shape.stride=spec.stride; shape.groups=1; shape.bias=true;
        convolutions.push_back({shape,
            static_cast<const uint8_t *>(rawWeight.bytes),rawWeight.length,
            static_cast<const uint8_t *>(rawBias.bytes),rawBias.length});
    }
    ANEOperationNode *linear=nodeWithLabel(graph,@"linear_0");
    if (!linear || ![linear.operationName isEqualToString:@"linear"])
        return nil;
    ANEOperationNode *headWeight=producerForOperand(graph,linear,@"weight");
    ANEOperationNode *headBias=producerForOperand(graph,linear,@"bias");
    if (!headWeight || !headBias) return nil;
    if (![linear.outputType.shape isEqualToArray:@[@1,@1000]] ||
        ![headWeight.outputType.shape isEqualToArray:@[@1000,@2048]] ||
        ![headBias.outputType.shape isEqualToArray:@[@1000]]) return nil;
    NSData *rawHeadWeight=[ANEBlobResolver loadConstantForOperation:headWeight.sourceOperation
        expectedBytes:1000*2048*2 modelRoot:modelRoot diagnostics:diagnostics];
    NSData *rawHeadBias=[ANEBlobResolver loadConstantForOperation:headBias.sourceOperation
        expectedBytes:1000*2 modelRoot:modelRoot diagnostics:diagnostics];
    if (!rawHeadWeight || !rawHeadBias) return nil;
    [storage addObject:rawHeadWeight]; [storage addObject:rawHeadBias];

    ane::h16::ResNet50Program encoded;
    try {
        encoded=ane::h16::encodeResNet50(convolutions,
            static_cast<const uint8_t *>(rawHeadWeight.bytes),rawHeadWeight.length,
            static_cast<const uint8_t *>(rawHeadBias.bytes),rawHeadBias.length);
    } catch (const std::exception &exception) {
        emitError(diagnostics,@"h16g.encode.resnet50-fields",
            [NSString stringWithUTF8String:exception.what()],linear);
        return nil;
    }
    NSData *td=[NSData dataWithBytes:encoded.program.task.data()
        length:encoded.program.task.size()];
    NSData *constants=[NSData dataWithBytes:encoded.program.constants.data()
        length:encoded.program.constants.size()];
    NSMutableArray<NSNumber *> *relocations=[NSMutableArray array];
    for (std::size_t offset:encoded.program.kernelRelocations)
        [relocations addObject:@(offset)];
    ANEGraphValue *input=graph.sourceFunction.inputs.firstObject;
    NSArray<NSNumber *> *outputObjectShape=@[@1,@1,@1,@1000];
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:input.name shortName:input.name
            role:HWXObjectBindingRoleInput elementType:input.type.elementType
            shape:input.type.shape rowStrideBytes:448 planeStrideBytes:100352
            batchStrideBytes:301056 storageByteLength:301056],
        [[HWXObjectBinding alloc]
            initWithSymbol:[linear.identifier stringByAppendingString:@"@output"]
            shortName:linear.identifier role:HWXObjectBindingRoleOutput
            elementType:linear.outputType.elementType shape:outputObjectShape
            rowStrideBytes:2048 planeStrideBytes:2048
            batchStrideBytes:2048 storageByteLength:2048],
    ];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:82 recordCount:encoded.programRecordCount
        formatCode:encoded.programFormatCode scratchByteLength:encoded.scratchBytes
        scratchAllocationByteLength:encoded.scratchBytes
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    info.kernelRegionConstants=YES;
    NSError *error=nil;
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:constants bindings:objectBindings
        kernelRelocationOffsets:relocations programInfo:info error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.resnet50-object",
            error.localizedDescription ?: @"cannot construct ResNet-50 HWX",linear);
        return nil;
    }
    NSArray<ANEHWXBinding *> *runtimeBindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:input.name role:ANESurfaceRoleInput
            logicalByteLength:301056 allocationByteLength:alignUp(301056,0x4000)
            ioSurfaceIndex:0 rowStrideBytes:448 planeStrideBytes:100352
            batchStrideBytes:301056],
        [[ANEHWXBinding alloc]initWithIdentifier:@"resnet50@weights"
            role:ANESurfaceRoleWeight logicalByteLength:constants.length
            allocationByteLength:constants.length ioSurfaceIndex:-1],
        [[ANEHWXBinding alloc]initWithIdentifier:linear.identifier
            role:ANESurfaceRoleOutput logicalByteLength:2000
            allocationByteLength:alignUp(2048,0x4000) ioSurfaceIndex:1
            rowStrideBytes:2048 planeStrideBytes:2048 batchStrideBytes:2048],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:runtimeBindings];
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

static BOOL fp16VectorIs(NSData *data,uint16_t expected) {
    if(data.length%sizeof(uint16_t)) return NO;
    const uint16_t *values=static_cast<const uint16_t *>(data.bytes);
    for(NSUInteger index=0;index<data.length/sizeof(uint16_t);++index)
        if(values[index]!=expected) return NO;
    return YES;
}

static BOOL standaloneLayerNormHardwareQualified(void) { return NO; }
static BOOL fusedAttentionHardwareQualified(void) { return NO; }

static ANEHWXArtifact *encodeTransformerLayerNormProgram(
    ANEOperationGraph *graph,NSURL *modelRoot,
    ANEDiagnosticEngine *diagnostics) {
    if(graph.sourceFunction.inputs.count!=1 || graph.outputValueNames.count!=1)
        return nil;
    ANEOperationNode *normalization=nil;
    NSUInteger active=0;
    for(ANEOperationNode *node in graph.nodes) {
        if(node.kind==ANEOperationKindConstant) continue;
        ++active;
        if([node.operationName isEqualToString:@"layer_norm"] ||
           [node.sourceOperation.operationName isEqualToString:@"layer_norm"])
            normalization=node;
    }
    BOOL decomposed=active==7 &&
        [normalization.sourceOperation.operationName isEqualToString:@"layer_norm"];
    if((active!=1 && !decomposed) || !normalization) return nil;
    ANEGraphValue *input=graph.sourceFunction.inputs.firstObject;
    NSArray<NSNumber *> *shape=input.type.shape;
    if(input.type.elementType!=ANEElementTypeFP16 || shape.count!=3 ||
       shape[0].unsignedIntegerValue!=1 ||
       ![normalization.outputType.shape isEqualToArray:shape]) {
        emitError(diagnostics,@"h16g.encode.transformer-layer-norm-shape",
            @"last-axis layer_norm requires equal fp16 [1,S,C] input and output",
            normalization);
        return nil;
    }
    if(!constantListOperandIs(graph,normalization,@"axes",@[@"-1"])) {
        emitError(diagnostics,@"h16g.encode.transformer-layer-norm-axis",
            @"transformer layer_norm requires axes=[-1]",normalization);
        return nil;
    }
    uint16_t epsilon=0;
    ANEGraphArgument *epsilonArgument=
        normalization.sourceOperation.arguments[@"epsilon"] ?:
        normalization.sourceOperation.attributes[@"epsilon"];
    if(!fp16ScalarFromArgument(graph,epsilonArgument,&epsilon) ||
       epsilon!=0x00a8) {
        emitError(diagnostics,@"h16g.encode.transformer-layer-norm-epsilon",
            @"transformer layer_norm requires epsilon=fp16(1e-5)",normalization);
        return nil;
    }
    if(!standaloneLayerNormHardwareQualified()) {
        emitError(diagnostics,
            @"h16g.encode.transformer-layer-norm-hardware-unproved",
            @"standalone H16G layer_norm remains disabled: the corrected L2 dataflow and serialized schedule have not completed numerical J616s qualification",
            normalization);
        return nil;
    }
    // Retain the decoded construction below as research evidence.  It must not
    // become compiler output until a standalone form passes numerical device
    // qualification without quarantining the session.
    const NSUInteger channels=shape[2].unsignedIntegerValue;
    for(NSString *name in @[@"gamma",@"beta"]) {
        ANEOperationNode *constant=producerForOperand(graph,normalization,name);
        if(!constant) continue;
        if(!constant.sourceOperation ||
           ![constant.outputType.shape isEqualToArray:@[@(channels)]]) {
            emitError(diagnostics,@"h16g.encode.transformer-layer-norm-affine",
                @"gamma and beta must be fp16 vectors with C elements",normalization);
            return nil;
        }
        NSData *raw=[ANEBlobResolver loadConstantForOperation:constant.sourceOperation
            expectedBytes:channels*2 modelRoot:modelRoot diagnostics:diagnostics];
        const uint16_t expected=[name isEqualToString:@"gamma"]?0x3c00:0;
        if(!raw || !fp16VectorIs(raw,expected)) {
            emitError(diagnostics,@"h16g.encode.transformer-layer-norm-affine-unproved",
                @"the supplied H16G ANE corpus proves only identity gamma and zero beta",
                normalization);
            return nil;
        }
    }
    ane::h16::TransformerLayerNormProgram encoded;
    try {
        encoded=ane::h16::encodeTransformerLayerNorm(
            (uint32_t)shape[1].unsignedIntegerValue,(uint32_t)channels);
    } catch(const std::exception &exception) {
        emitError(diagnostics,@"h16g.encode.transformer-layer-norm-fields",
            [NSString stringWithUTF8String:exception.what()],normalization);
        return nil;
    }
    NSData *td=[NSData dataWithBytes:encoded.program.task.data()
        length:encoded.program.task.size()];
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:input.name shortName:input.name
            role:HWXObjectBindingRoleInput elementType:input.type.elementType
            shape:shape],
        [[HWXObjectBinding alloc]
            initWithSymbol:[normalization.identifier stringByAppendingString:@"@output"]
            shortName:normalization.identifier role:HWXObjectBindingRoleOutput
            elementType:normalization.outputType.elementType shape:shape],
    ];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:encoded.program.taskCount
        recordCount:encoded.programRecordCount
        formatCode:encoded.programFormatCode
        scratchByteLength:encoded.scratchBytes
        scratchAllocationByteLength:encoded.scratchBytes
        descriptorLayout:encoded.scratchBytes
            ? HWXProgramDescriptorLayoutScratchBackedMixed
            : HWXProgramDescriptorLayoutLinear];
    info.kernelRegionConstants=YES;
    NSError *error=nil;
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:[NSData dataWithBytes:encoded.program.constants.data()
            length:encoded.program.constants.size()] bindings:objectBindings
        kernelRelocationOffsets:@[] programInfo:info error:&error];
    if(!image) {
        emitError(diagnostics,@"h16g.encode.transformer-layer-norm-object",
            error.localizedDescription ?: @"cannot construct transformer layer_norm HWX",
            normalization);
        return nil;
    }
    const NSUInteger bytes=tensorBytes(input.type);
    const NSUInteger row=channels*2;
    const NSUInteger plane=row*shape[1].unsignedIntegerValue;
    NSArray<ANEHWXBinding *> *runtimeBindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:bytes
            allocationByteLength:alignUp(bytes,0x4000) ioSurfaceIndex:0
            rowStrideBytes:row planeStrideBytes:plane batchStrideBytes:bytes],
        [[ANEHWXBinding alloc]initWithIdentifier:normalization.identifier
            role:ANESurfaceRoleOutput logicalByteLength:bytes
            allocationByteLength:alignUp(bytes,0x4000) ioSurfaceIndex:1
            rowStrideBytes:row planeStrideBytes:plane batchStrideBytes:bytes],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeWhisperStemProgram(
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
        out[0].unsignedIntegerValue!=1)
        return nil;
    NSUInteger inputChannels=in[1].unsignedIntegerValue;
    NSUInteger inputTokens=in[2].unsignedIntegerValue;
    NSUInteger channels=out[1].unsignedIntegerValue;
    NSUInteger outputTokens=out[2].unsignedIntegerValue;
    BOOL stem1=inputChannels==80 && outputTokens==inputTokens &&
        (channels==384 || channels==512);
    BOOL stem2=inputChannels==channels && inputTokens==3000 &&
        outputTokens==1500 && (channels==384 || channels==512);
    if (!stem1 && !stem2) return nil;
    if (!constantScalarOperandIs(graph,conv,@"pad_type",@"same") ||
        !constantListOperandIs(graph,conv,@"strides",
            stem2 ? @[@"2"] : @[@"1"]) ||
        !constantListOperandIs(graph,conv,@"pad",@[@"0",@"0"]) ||
        !constantListOperandIs(graph,conv,@"dilations",@[@"1"]) ||
        !constantScalarOperandIs(graph,conv,@"groups",@"1")) return nil;
    ANEOperationNode *weight=producerForOperand(graph,conv,@"weight");
    ANEOperationNode *bias=producerForOperand(graph,conv,@"bias");
    if (!weight.sourceOperation || !bias.sourceOperation ||
        ![weight.outputType.shape isEqualToArray:
            @[@(channels),@(inputChannels),@3]] ||
        ![bias.outputType.shape isEqualToArray:@[@(channels)]]) return nil;
    ANEOperationNode *result=conv;
    ANEOperationNode *post=conv.users.count==1 ? conv.users.firstObject : nil;
    BOOL gelu=[post.operationName isEqualToString:@"gelu"];
    if (post && !gelu) return nil;
    if (post) result=post;
    if (![graph.outputValueNames.firstObject isEqualToString:result.identifier])
        return nil;
    NSData *rawWeights=[ANEBlobResolver loadConstantForOperation:weight.sourceOperation
        expectedBytes:channels*inputChannels*3*2 modelRoot:modelRoot diagnostics:diagnostics];
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
        encoded=stem1 ? ane::h16::encodeWhisperStem1((uint32_t)inputTokens,
                (uint32_t)channels,
                static_cast<const uint8_t *>(rawWeights.bytes),rawWeights.length,
                static_cast<const uint8_t *>(rawBias.bytes),rawBias.length,options)
            : ane::h16::encodeWhisperStem2((uint32_t)channels,
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
    NSUInteger inputRow=alignUp(inputTokens*2,64);
    NSUInteger outputRow=alignUp(outputTokens*2,64);
    NSUInteger inputStorage=inputChannels*inputRow;
    NSUInteger outputStorage=channels*outputRow;
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:input.name shortName:input.name
            role:HWXObjectBindingRoleInput elementType:input.type.elementType
            shape:in rowStrideBytes:inputRow planeStrideBytes:inputStorage
            batchStrideBytes:inputStorage storageByteLength:inputStorage],
        [[HWXObjectBinding alloc]
            initWithSymbol:[result.identifier stringByAppendingString:@"@output"]
            shortName:result.identifier role:HWXObjectBindingRoleOutput
            elementType:result.outputType.elementType shape:result.outputType.shape
            rowStrideBytes:outputRow planeStrideBytes:outputStorage
            batchStrideBytes:outputStorage storageByteLength:outputStorage],
    ];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:encoded.program.taskCount
        recordCount:encoded.programRecordCount
        formatCode:encoded.programFormatCode scratchByteLength:0
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
            allocationByteLength:alignUp(inputStorage,0x4000) ioSurfaceIndex:0
            rowStrideBytes:inputRow planeStrideBytes:inputStorage
            batchStrideBytes:inputStorage],
        [[ANEHWXBinding alloc]initWithIdentifier:weight.identifier
            role:ANESurfaceRoleWeight logicalByteLength:constants.length
            allocationByteLength:constants.length ioSurfaceIndex:-1],
        [[ANEHWXBinding alloc]initWithIdentifier:result.identifier
            role:ANESurfaceRoleOutput logicalByteLength:outputBytes
            allocationByteLength:alignUp(outputStorage,0x4000) ioSurfaceIndex:1
            rowStrideBytes:outputRow planeStrideBytes:outputStorage
            batchStrideBytes:outputStorage],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeWhisperTransposeProgram(
    ANEOperationGraph *graph, ANEDiagnosticEngine *diagnostics) {
    if (graph.sourceFunction.inputs.count!=1 || graph.outputValueNames.count!=1)
        return nil;
    ANEGraphValue *input=graph.sourceFunction.inputs.firstObject;
    ANEOperationNode *transpose=nil;
    for (ANEOperationNode *node in graph.nodes)
        if ([node.operationName isEqualToString:@"transpose"])
            transpose=node;
    if (!transpose) return nil;
    NSArray<NSNumber *> *in=input.type.shape,*out=transpose.outputType.shape;
    if (in.count!=3 || out.count!=3 || in[0].unsignedIntegerValue!=1 ||
        out[0].unsignedIntegerValue!=1 || in[2].unsignedIntegerValue!=1500 ||
        out[1].unsignedIntegerValue!=1500 ||
        out[2].unsignedIntegerValue!=in[1].unsignedIntegerValue ||
        ![graph.outputValueNames.firstObject isEqualToString:transpose.identifier] ||
        !constantListOperandIs(graph,transpose,@"perm",@[@"0",@"2",@"1"]))
        return nil;
    NSUInteger channels=in[1].unsignedIntegerValue;
    ane::h16::TransformerLinearProgram encoded;
    try {
        encoded=ane::h16::encodeWhisperTranspose((uint32_t)channels);
    } catch (const std::exception &exception) {
        emitError(diagnostics,@"h16g.encode.whisper-transpose-fields",
            [NSString stringWithUTF8String:exception.what()],transpose);
        return nil;
    }
    NSData *td=[NSData dataWithBytes:encoded.program.task.data()
        length:encoded.program.task.size()];
    const NSUInteger sourceRow=alignUp(1500*2,64);
    const NSUInteger sourceStorage=channels*sourceRow;
    const NSUInteger resultRow=channels*2;
    const NSUInteger resultStorage=1500*resultRow;
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:input.name shortName:input.name
            role:HWXObjectBindingRoleInput elementType:input.type.elementType shape:in
            rowStrideBytes:sourceRow planeStrideBytes:sourceStorage
            batchStrideBytes:sourceStorage storageByteLength:sourceStorage],
        [[HWXObjectBinding alloc]
            initWithSymbol:[transpose.identifier stringByAppendingString:@"@output"]
            shortName:transpose.identifier role:HWXObjectBindingRoleOutput
            elementType:transpose.outputType.elementType shape:out
            rowStrideBytes:resultRow planeStrideBytes:resultStorage
            batchStrideBytes:resultStorage storageByteLength:resultStorage],
    ];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:1 recordCount:encoded.programRecordCount
        formatCode:0 scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    NSError *error=nil;
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:[NSData data] bindings:objectBindings
        kernelRelocationOffsets:@[] programInfo:info error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.whisper-transpose-object",
            error.localizedDescription ?: @"cannot construct Whisper transpose HWX",
            transpose);
        return nil;
    }
    NSUInteger bytes=tensorBytes(input.type);
    NSArray<ANEHWXBinding *> *runtimeBindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:bytes
            allocationByteLength:alignUp(sourceStorage,0x4000) ioSurfaceIndex:0
            rowStrideBytes:sourceRow planeStrideBytes:sourceStorage
            batchStrideBytes:sourceStorage],
        [[ANEHWXBinding alloc]initWithIdentifier:transpose.identifier
            role:ANESurfaceRoleOutput logicalByteLength:bytes
            allocationByteLength:alignUp(resultStorage,0x4000) ioSurfaceIndex:1
            rowStrideBytes:resultRow planeStrideBytes:resultStorage
            batchStrideBytes:resultStorage],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeWhisperPositionalAddProgram(
    ANEOperationGraph *graph, NSURL *modelRoot,
    ANEDiagnosticEngine *diagnostics) {
    if (graph.sourceFunction.inputs.count!=1 || graph.outputValueNames.count!=1)
        return nil;
    ANEGraphValue *input=graph.sourceFunction.inputs.firstObject;
    ANEOperationNode *add=nil;
    for (ANEOperationNode *node in graph.nodes)
        if ([node.operationName isEqualToString:@"add"]) add=node;
    if (!add || ![input.type.shape isEqualToArray:add.outputType.shape] ||
        input.type.shape.count!=3 || input.type.shape[0].unsignedIntegerValue!=1 ||
        input.type.shape[1].unsignedIntegerValue!=1500 ||
        (input.type.shape[2].unsignedIntegerValue!=384 &&
         input.type.shape[2].unsignedIntegerValue!=512) ||
        ![graph.outputValueNames.firstObject isEqualToString:add.identifier])
        return nil;
    ANEGraphValue *left=add.sourceOperation.operands[@"x"].value;
    ANEGraphValue *right=add.sourceOperation.operands[@"y"].value;
    ANEOperationNode *leftNode=producerForOperand(graph,add,@"x");
    ANEOperationNode *rightNode=producerForOperand(graph,add,@"y");
    ANEOperationNode *constant=nil;
    if ([left.name isEqualToString:input.name] && rightNode.sourceOperation)
        constant=rightNode;
    else if ([right.name isEqualToString:input.name] && leftNode.sourceOperation)
        constant=leftNode;
    if (!constant || ![constant.operationName isEqualToString:@"const"] ||
        ![constant.outputType.shape isEqualToArray:input.type.shape]) return nil;
    NSUInteger logicalBytes=tensorBytes(input.type);
    NSData *raw=[ANEBlobResolver loadConstantForOperation:constant.sourceOperation
        expectedBytes:logicalBytes modelRoot:modelRoot diagnostics:diagnostics];
    if (!raw) return nil;
    const NSUInteger size=1024,storageBytes=size*size*2;
    NSError *error=nil;
    H16GEncodedTDProgram *program=[H16GALUEncoder
        encodeOperationName:@"add" squareSize:size error:&error];
    if (!program) {
        emitError(diagnostics,@"h16g.encode.whisper-positional-add-fields",
            error.localizedDescription ?: @"cannot encode positional add",add);
        return nil;
    }
    NSMutableData *constants=[NSMutableData dataWithLength:storageBytes];
    [constants replaceBytesInRange:NSMakeRange(0,raw.length) withBytes:raw.bytes];
    NSArray<NSNumber *> *view=@[@1,@1,@1024,@1024];
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:input.name shortName:input.name
            role:HWXObjectBindingRoleInput elementType:input.type.elementType shape:view],
        [[HWXObjectBinding alloc]
            initWithSymbol:[add.identifier stringByAppendingString:@"@output"]
            shortName:add.identifier role:HWXObjectBindingRoleOutput
            elementType:add.outputType.elementType shape:view],
    ];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:1 recordCount:program.programRecordCount
        formatCode:program.programFormatCode scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    info.kernelRegionConstants=YES;
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:program.data
        constantRegion:constants bindings:objectBindings kernelRelocationOffsets:@[]
        programInfo:info error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.whisper-positional-add-object",
            error.localizedDescription ?: @"cannot construct positional add HWX",add);
        return nil;
    }
    NSArray<ANEHWXBinding *> *runtimeBindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:logicalBytes
            allocationByteLength:storageBytes ioSurfaceIndex:0
            rowStrideBytes:size*2 planeStrideBytes:storageBytes
            batchStrideBytes:storageBytes],
        [[ANEHWXBinding alloc]initWithIdentifier:constant.identifier
            role:ANESurfaceRoleWeight logicalByteLength:storageBytes
            allocationByteLength:storageBytes ioSurfaceIndex:-1],
        [[ANEHWXBinding alloc]initWithIdentifier:add.identifier
            role:ANESurfaceRoleOutput logicalByteLength:logicalBytes
            allocationByteLength:storageBytes ioSurfaceIndex:1
            rowStrideBytes:size*2 planeStrideBytes:storageBytes
            batchStrideBytes:storageBytes],
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

static ANEHWXArtifact *encodeTransformerLayoutProgram(
    ANEOperationGraph *graph, ANEDiagnosticEngine *diagnostics) {
    if (graph.sourceFunction.inputs.count!=1 ||
        graph.outputValueNames.count!=1) return nil;
    ANEGraphValue *input=graph.sourceFunction.inputs.firstObject;
    ANEOperationNode *output=nodeNamed(graph,graph.outputValueNames.firstObject);
    if (!output) return nil;
    NSArray<NSNumber *> *in=input.type.shape,*out=output.outputType.shape;
    if (input.type.elementType!=ANEElementTypeFP16 ||
        output.outputType.elementType!=ANEElementTypeFP16) return nil;
    NSUInteger active=0;
    BOOL reshape=NO,transpose=NO;
    for (ANEOperationNode *node in graph.nodes) {
        if (node.kind==ANEOperationKindConstant) continue;
        ++active;
        reshape|=[node.operationName isEqualToString:@"reshape"];
        transpose|=[node.operationName isEqualToString:@"transpose"];
    }
    if (active!=2 || !reshape || !transpose) return nil;
    ane::h16::AttentionLayout operation;
    NSUInteger tokens=0,heads=0,headDimension=0;
    if (in.count==3 && out.count==4 &&
        in[0].unsignedIntegerValue==1 && out[0].unsignedIntegerValue==1) {
        tokens=in[1].unsignedIntegerValue;
        heads=out[1].unsignedIntegerValue;
        if (out[2].unsignedIntegerValue==tokens) {
            operation=ane::h16::AttentionLayout::Split;
            headDimension=out[3].unsignedIntegerValue;
        } else if (out[3].unsignedIntegerValue==tokens) {
            operation=ane::h16::AttentionLayout::SplitTranspose;
            headDimension=out[2].unsignedIntegerValue;
        } else return nil;
        if (in[2].unsignedIntegerValue!=heads*headDimension) return nil;
    } else if (in.count==4 && out.count==3 &&
               in[0].unsignedIntegerValue==1 && out[0].unsignedIntegerValue==1) {
        operation=ane::h16::AttentionLayout::Merge;
        heads=in[1].unsignedIntegerValue;
        tokens=in[2].unsignedIntegerValue;
        headDimension=in[3].unsignedIntegerValue;
        if (out[1].unsignedIntegerValue!=tokens ||
            out[2].unsignedIntegerValue!=heads*headDimension) return nil;
    } else return nil;
    ane::h16::TransformerAttentionProgram encoded;
    try {
        encoded=ane::h16::encodeAttentionLayout((uint32_t)tokens,
            (uint32_t)heads,(uint32_t)headDimension,operation);
    } catch (const std::exception &exception) {
        emitError(diagnostics,@"h16g.encode.transformer-layout-fields",
            [NSString stringWithUTF8String:exception.what()],output);
        return nil;
    }
    const NSUInteger width=heads*headDimension;
    const NSUInteger widthBytes=2*width;
    const NSUInteger headRowBytes=2*headDimension;
    const NSUInteger naturalBytes=tokens*widthBytes;
    const NSUInteger inputRow=operation==ane::h16::AttentionLayout::Merge
        ? headRowBytes : widthBytes;
    const NSUInteger inputPlane=operation==ane::h16::AttentionLayout::Merge
        ? tokens*headRowBytes : naturalBytes;
    const NSUInteger outputRow=operation==ane::h16::AttentionLayout::Split
        ? headRowBytes
        : operation==ane::h16::AttentionLayout::SplitTranspose
            ? alignUp(2*tokens,64) : widthBytes;
    const NSUInteger outputPlane=operation==ane::h16::AttentionLayout::Split
        ? tokens*headRowBytes
        : operation==ane::h16::AttentionLayout::SplitTranspose
            ? alignUp(2*tokens,64)*headDimension : naturalBytes;
    const NSUInteger inputStorage=operation==ane::h16::AttentionLayout::Merge
        ? naturalBytes : naturalBytes;
    const NSUInteger outputStorage=operation==ane::h16::AttentionLayout::SplitTranspose
        ? heads*outputPlane : naturalBytes;
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:input.name shortName:input.name
            role:HWXObjectBindingRoleInput elementType:input.type.elementType
            shape:in rowStrideBytes:inputRow planeStrideBytes:inputPlane
            batchStrideBytes:inputStorage storageByteLength:inputStorage],
        [[HWXObjectBinding alloc]
            initWithSymbol:[output.identifier stringByAppendingString:@"@output"]
            shortName:output.identifier role:HWXObjectBindingRoleOutput
            elementType:output.outputType.elementType shape:out
            rowStrideBytes:outputRow planeStrideBytes:outputPlane
            batchStrideBytes:outputStorage storageByteLength:outputStorage],
    ];
    NSData *td=[NSData dataWithBytes:encoded.program.task.data()
        length:encoded.program.task.size()];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:1 recordCount:encoded.programRecordCount
        formatCode:encoded.programFormatCode scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    NSError *error=nil;
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:[NSData data] bindings:objectBindings
        kernelRelocationOffsets:@[] programInfo:info error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.transformer-layout-object",
            error.localizedDescription ?: @"cannot construct transformer layout HWX",output);
        return nil;
    }
    NSArray<ANEHWXBinding *> *runtimeBindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:input.name
            role:ANESurfaceRoleInput logicalByteLength:naturalBytes
            allocationByteLength:alignUp(inputStorage,0x4000) ioSurfaceIndex:0
            rowStrideBytes:inputRow planeStrideBytes:inputPlane
            batchStrideBytes:inputStorage],
        [[ANEHWXBinding alloc]initWithIdentifier:output.identifier
            role:ANESurfaceRoleOutput logicalByteLength:naturalBytes
            allocationByteLength:alignUp(outputStorage,0x4000) ioSurfaceIndex:1
            rowStrideBytes:outputRow planeStrideBytes:outputPlane
            batchStrideBytes:outputStorage],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:runtimeBindings];
}

static ANEHWXArtifact *encodeTransformerQKProgram(
    ANEOperationGraph *graph, ANEDiagnosticEngine *diagnostics) {
    if (graph.sourceFunction.inputs.count!=2 ||
        graph.outputValueNames.count!=1) return nil;
    ANEOperationNode *matmul=nil;
    NSUInteger active=0;
    for (ANEOperationNode *node in graph.nodes) {
        if (node.kind==ANEOperationKindConstant) continue;
        ++active;
        if ([node.operationName isEqualToString:@"matmul"]) matmul=node;
    }
    if (active!=1 || !matmul ||
        ![graph.outputValueNames.firstObject isEqualToString:matmul.identifier])
        return nil;
    ANEGraphValue *left=matmul.sourceOperation.operands[@"x"].value;
    ANEGraphValue *right=matmul.sourceOperation.operands[@"y"].value;
    NSArray<NSNumber *> *ls=left.type.shape,*rs=right.type.shape;
    NSArray<NSNumber *> *os=matmul.outputType.shape;
    if (left.type.elementType!=ANEElementTypeFP16 ||
        right.type.elementType!=ANEElementTypeFP16 ||
        matmul.outputType.elementType!=ANEElementTypeFP16 ||
        ls.count!=4 || rs.count!=4 || os.count!=4 ||
        ls[0].unsignedIntegerValue!=1 || rs[0].unsignedIntegerValue!=1 ||
        os[0].unsignedIntegerValue!=1) return nil;
    NSUInteger heads=ls[1].unsignedIntegerValue;
    NSUInteger tokens=ls[2].unsignedIntegerValue;
    NSUInteger headDimension=ls[3].unsignedIntegerValue;
    ane::h16::AttentionMatmul operation=ane::h16::AttentionMatmul::QK;
    BOOL pv=NO;
    if (ls[3].unsignedIntegerValue==tokens &&
        rs[1].unsignedIntegerValue==heads &&
        rs[2].unsignedIntegerValue==tokens &&
        os[1].unsignedIntegerValue==heads &&
        os[2].unsignedIntegerValue==tokens &&
        [rs isEqualToArray:os]) {
        pv=YES;operation=ane::h16::AttentionMatmul::PV;
        headDimension=rs[3].unsignedIntegerValue;
    }
    NSArray<NSNumber *> *expectedRight=@[@1,@(heads),@(headDimension),@(tokens)];
    NSArray<NSNumber *> *expectedRightTranspose=@[@1,@(heads),@(tokens),@(headDimension)];
    NSArray<NSNumber *> *expectedOutput=@[@1,@(heads),@(tokens),@(tokens)];
    BOOL transposeY=[rs isEqualToArray:expectedRightTranspose];
    if (!pv && ((!transposeY && ![rs isEqualToArray:expectedRight]) ||
        ![os isEqualToArray:expectedOutput]))
        return nil;
    if (transposeY && !pv) operation=ane::h16::AttentionMatmul::QKTransposeY;
    ane::h16::TransformerAttentionProgram encoded;
    try {
        encoded=ane::h16::encodeAttentionMatmul((uint32_t)tokens,
            (uint32_t)heads,(uint32_t)headDimension,
            operation);
    } catch (const std::exception &exception) {
        emitError(diagnostics,@"h16g.encode.transformer-qk-fields",
            [NSString stringWithUTF8String:exception.what()],matmul);
        return nil;
    }
    const NSUInteger headRow=2*headDimension;
    const NSUInteger scoreRow=alignUp(2*tokens,64);
    const NSUInteger leftPlane=pv ? tokens*scoreRow : tokens*headRow;
    const NSUInteger leftStorage=heads*leftPlane;
    const NSUInteger rightRow=(transposeY || pv) ? headRow : scoreRow;
    const NSUInteger rightPlane=pv ? tokens*headRow
        : transposeY ? tokens*headRow : scoreRow*headDimension;
    const NSUInteger rightStorage=heads*rightPlane;
    const NSUInteger outputRow=pv ? headRow : scoreRow;
    const NSUInteger outputPlane=pv ? tokens*headRow : tokens*scoreRow;
    const NSUInteger outputStorage=heads*outputPlane;
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:left.name shortName:left.name
            role:HWXObjectBindingRoleInput elementType:left.type.elementType
            shape:ls rowStrideBytes:(pv ? scoreRow : headRow) planeStrideBytes:leftPlane
            batchStrideBytes:leftStorage storageByteLength:leftStorage],
        [[HWXObjectBinding alloc]initWithSymbol:right.name shortName:right.name
            role:HWXObjectBindingRoleInput elementType:right.type.elementType
            shape:rs rowStrideBytes:rightRow planeStrideBytes:rightPlane
            batchStrideBytes:rightStorage storageByteLength:rightStorage],
        [[HWXObjectBinding alloc]
            initWithSymbol:[matmul.identifier stringByAppendingString:@"@output"]
            shortName:matmul.identifier role:HWXObjectBindingRoleOutput
            elementType:matmul.outputType.elementType shape:os
            rowStrideBytes:outputRow planeStrideBytes:outputPlane
            batchStrideBytes:outputStorage storageByteLength:outputStorage],
    ];
    NSData *td=[NSData dataWithBytes:encoded.program.task.data()
        length:encoded.program.task.size()];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:encoded.program.taskCount
        recordCount:encoded.programRecordCount formatCode:encoded.programFormatCode
        scratchByteLength:encoded.scratchBytes
        scratchAllocationByteLength:alignUp(encoded.program.scratchAllocationBytes,0x4000)
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSError *error=nil;
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:[NSData data] bindings:objectBindings
        kernelRelocationOffsets:@[] programInfo:info error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.transformer-qk-object",
            error.localizedDescription ?: @"cannot construct transformer QK HWX",matmul);
        return nil;
    }
    const NSUInteger leftLogical=tensorBytes(left.type);
    const NSUInteger rightLogical=tensorBytes(right.type);
    const NSUInteger outputLogical=tensorBytes(matmul.outputType);
    NSArray<ANEHWXBinding *> *bindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:left.name role:ANESurfaceRoleInput
            logicalByteLength:leftLogical allocationByteLength:alignUp(leftStorage,0x4000)
            ioSurfaceIndex:0 rowStrideBytes:(pv ? scoreRow : headRow) planeStrideBytes:leftPlane
            batchStrideBytes:leftStorage],
        [[ANEHWXBinding alloc]initWithIdentifier:right.name role:ANESurfaceRoleInput
            logicalByteLength:rightLogical allocationByteLength:alignUp(rightStorage,0x4000)
            ioSurfaceIndex:1 rowStrideBytes:rightRow planeStrideBytes:rightPlane
            batchStrideBytes:rightStorage],
        [[ANEHWXBinding alloc]initWithIdentifier:matmul.identifier
            role:ANESurfaceRoleOutput logicalByteLength:outputLogical
            allocationByteLength:alignUp(outputStorage,0x4000) ioSurfaceIndex:2
            rowStrideBytes:outputRow planeStrideBytes:outputPlane
            batchStrideBytes:outputStorage],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:bindings];
}

static ANEHWXArtifact *encodeTransformerSoftmaxProgram(
    ANEOperationGraph *graph, ANEDiagnosticEngine *diagnostics) {
    if (graph.sourceFunction.inputs.count!=1 || graph.outputValueNames.count!=1)
        return nil;
    ANEGraphValue *input=graph.sourceFunction.inputs.firstObject;
    ANEOperationNode *output=nodeNamed(graph,graph.outputValueNames.firstObject);
    if (!output || ![output.sourceOperation.operationName isEqualToString:@"softmax"] ||
        input.type.elementType!=ANEElementTypeFP16 ||
        output.outputType.elementType!=ANEElementTypeFP16 ||
        input.type.shape.count!=4 ||
        ![input.type.shape isEqualToArray:output.outputType.shape]) return nil;
    NSArray<NSNumber *> *shape=input.type.shape;
    NSUInteger heads=shape[1].unsignedIntegerValue;
    NSUInteger tokens=shape[2].unsignedIntegerValue;
    if (shape[0].unsignedIntegerValue!=1 ||
        shape[3].unsignedIntegerValue!=tokens) return nil;
    if (tokens!=1500 || heads!=8) {
        emitError(diagnostics,
            @"h16g.encode.transformer-softmax-hardware-unproved",
            @"standalone H16G softmax is enabled only for the measured S=1500, H=8 form; H=6 and other predictions are unsafe",
            output);
        return nil;
    }
    ane::h16::TransformerAttentionProgram encoded;
    try {
        encoded=ane::h16::encodeAttentionSoftmax((uint32_t)tokens,(uint32_t)heads);
    } catch (const std::exception &exception) {
        emitError(diagnostics,@"h16g.encode.transformer-softmax-fields",
            [NSString stringWithUTF8String:exception.what()],output);return nil;
    }
    NSError *error=nil;
    NSData *constants=[H16GLUTEncoder constantRegionForOperationName:@"exp" error:&error];
    if (!constants || constants.length!=128) {
        emitError(diagnostics,@"h16g.encode.transformer-softmax-table",
            error.localizedDescription ?: @"softmax exponential table is unavailable",output);
        return nil;
    }
    const NSUInteger row=alignUp(2*tokens,64),plane=tokens*row;
    const NSUInteger storage=heads*plane,logical=tensorBytes(input.type);
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:input.name shortName:input.name
            role:HWXObjectBindingRoleInput elementType:input.type.elementType
            shape:shape rowStrideBytes:row planeStrideBytes:plane
            batchStrideBytes:storage storageByteLength:storage],
        [[HWXObjectBinding alloc]
            initWithSymbol:[output.identifier stringByAppendingString:@"@output"]
            shortName:output.identifier role:HWXObjectBindingRoleOutput
            elementType:output.outputType.elementType shape:shape
            rowStrideBytes:row planeStrideBytes:plane
            batchStrideBytes:storage storageByteLength:storage],
    ];
    NSData *td=[NSData dataWithBytes:encoded.program.task.data()
        length:encoded.program.task.size()];
    NSMutableArray<NSNumber *> *relocations=[NSMutableArray array];
    for(std::size_t offset:encoded.program.kernelRelocations)
        [relocations addObject:@(offset)];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:encoded.program.taskCount
        recordCount:encoded.programRecordCount formatCode:encoded.programFormatCode
        scratchByteLength:encoded.scratchBytes
        scratchAllocationByteLength:alignUp(encoded.program.scratchAllocationBytes,0x4000)
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:constants bindings:objectBindings
        kernelRelocationOffsets:relocations programInfo:info error:&error];
    if (!image) {
        emitError(diagnostics,@"h16g.encode.transformer-softmax-object",
            error.localizedDescription ?: @"cannot construct transformer softmax HWX",output);
        return nil;
    }
    NSArray<ANEHWXBinding *> *bindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:input.name role:ANESurfaceRoleInput
            logicalByteLength:logical allocationByteLength:alignUp(storage,0x4000)
            ioSurfaceIndex:0 rowStrideBytes:row planeStrideBytes:plane
            batchStrideBytes:storage],
        [[ANEHWXBinding alloc]initWithIdentifier:output.identifier
            role:ANESurfaceRoleOutput logicalByteLength:logical
            allocationByteLength:alignUp(storage,0x4000) ioSurfaceIndex:1
            rowStrideBytes:row planeStrideBytes:plane batchStrideBytes:storage],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:bindings];
}

static ANEHWXArtifact *encodeTransformerFusedAttentionProgram(
    ANEOperationGraph *graph, ANEDiagnosticEngine *diagnostics) {
    if (graph.sourceFunction.inputs.count!=3 || graph.outputValueNames.count!=1)
        return nil;
    ANEGraphValue *q=graph.sourceFunction.inputs[0];
    ANEGraphValue *k=graph.sourceFunction.inputs[1];
    ANEGraphValue *v=graph.sourceFunction.inputs[2];
    ANEOperationNode *output=nodeNamed(graph,graph.outputValueNames.firstObject);
    NSArray<NSNumber *> *qs=q.type.shape,*ks=k.type.shape,*vs=v.type.shape;
    if (!output || q.type.elementType!=ANEElementTypeFP16 ||
        k.type.elementType!=ANEElementTypeFP16 || v.type.elementType!=ANEElementTypeFP16 ||
        output.outputType.elementType!=ANEElementTypeFP16 || qs.count!=4 ||
        ks.count!=4 || vs.count!=4 || ![qs isEqualToArray:vs] ||
        ![output.outputType.shape isEqualToArray:qs]) return nil;
    NSUInteger heads=qs[1].unsignedIntegerValue;
    NSUInteger tokens=qs[2].unsignedIntegerValue;
    NSUInteger headDimension=qs[3].unsignedIntegerValue;
    NSArray<NSNumber *> *expectedK=@[@1,@(heads),@(headDimension),@(tokens)];
    if (![ks isEqualToArray:expectedK]) return nil;
    NSUInteger matmuls=0,softmaxNodes=0;
    for(ANEOperationNode *node in graph.nodes) {
        matmuls += [node.operationName isEqualToString:@"matmul"];
        softmaxNodes += [node.sourceOperation.operationName isEqualToString:@"softmax"];
    }
    if (matmuls!=2 || !softmaxNodes) return nil;
    if(!fusedAttentionHardwareQualified()) {
        emitError(diagnostics,
            @"h16g.encode.fused-attention-accuracy-unproved",
            @"fused H16G attention remains disabled: the corrected resident P*128 PV stage has not completed numerical J616s qualification",
            output);
        return nil;
    }
    // Keep the decoded stream available to the parity tests, but never expose
    // it through the compiler until the resident PV scale is device-proven.
    ane::h16::TransformerAttentionProgram encoded;
    try { encoded=ane::h16::encodeFusedAttention((uint32_t)tokens,
        (uint32_t)heads,(uint32_t)headDimension); }
    catch(const std::exception &exception) {
        emitError(diagnostics,@"h16g.encode.fused-attention-fields",
            [NSString stringWithUTF8String:exception.what()],output);return nil;
    }
    NSError *error=nil;
    NSData *constants=[H16GLUTEncoder constantRegionForOperationName:@"exp" error:&error];
    if(!constants || constants.length!=128) {
        emitError(diagnostics,@"h16g.encode.fused-attention-table",
            error.localizedDescription ?: @"attention exponential table is unavailable",output);
        return nil;
    }
    const NSUInteger headRow=2*headDimension,qPlane=tokens*headRow;
    const NSUInteger qStorage=heads*qPlane,kRow=alignUp(2*tokens,64);
    const NSUInteger kPlane=kRow*headDimension,kStorage=heads*kPlane;
    NSArray<HWXObjectBinding *> *objectBindings=@[
        [[HWXObjectBinding alloc]initWithSymbol:q.name shortName:q.name
            role:HWXObjectBindingRoleInput elementType:q.type.elementType shape:qs
            rowStrideBytes:headRow planeStrideBytes:qPlane
            batchStrideBytes:qStorage storageByteLength:qStorage],
        [[HWXObjectBinding alloc]initWithSymbol:k.name shortName:k.name
            role:HWXObjectBindingRoleInput elementType:k.type.elementType shape:ks
            rowStrideBytes:kRow planeStrideBytes:kPlane
            batchStrideBytes:kStorage storageByteLength:kStorage],
        [[HWXObjectBinding alloc]initWithSymbol:v.name shortName:v.name
            role:HWXObjectBindingRoleInput elementType:v.type.elementType shape:vs
            rowStrideBytes:headRow planeStrideBytes:qPlane
            batchStrideBytes:qStorage storageByteLength:qStorage],
        [[HWXObjectBinding alloc]
            initWithSymbol:[output.identifier stringByAppendingString:@"@output"]
            shortName:output.identifier role:HWXObjectBindingRoleOutput
            elementType:output.outputType.elementType shape:qs
            rowStrideBytes:headRow planeStrideBytes:qPlane
            batchStrideBytes:qStorage storageByteLength:qStorage],
    ];
    NSData *td=[NSData dataWithBytes:encoded.program.task.data()
        length:encoded.program.task.size()];
    NSMutableArray<NSNumber *> *relocations=[NSMutableArray array];
    for(std::size_t offset:encoded.program.kernelRelocations)
        [relocations addObject:@(offset)];
    HWXObjectProgramInfo *info=[[HWXObjectProgramInfo alloc]
        initWithTaskCount:encoded.program.taskCount recordCount:encoded.programRecordCount
        formatCode:encoded.programFormatCode
        scratchByteLength:encoded.scratchBytes
        scratchAllocationByteLength:alignUp(encoded.program.scratchAllocationBytes,0x4000)
        descriptorLayout:HWXProgramDescriptorLayoutScratchBackedMixed];
    NSData *image=[HWXObjectWriter buildObjectWithTaskDescriptor:td
        constantRegion:constants bindings:objectBindings
        kernelRelocationOffsets:relocations programInfo:info error:&error];
    if(!image) {
        emitError(diagnostics,@"h16g.encode.fused-attention-object",
            error.localizedDescription ?: @"cannot construct fused attention HWX",output);
        return nil;
    }
    const NSUInteger logical=tensorBytes(q.type);
    NSArray<ANEHWXBinding *> *bindings=@[
        [[ANEHWXBinding alloc]initWithIdentifier:q.name role:ANESurfaceRoleInput
            logicalByteLength:logical allocationByteLength:alignUp(qStorage,0x4000)
            ioSurfaceIndex:0 rowStrideBytes:headRow planeStrideBytes:qPlane
            batchStrideBytes:qStorage],
        [[ANEHWXBinding alloc]initWithIdentifier:k.name role:ANESurfaceRoleInput
            logicalByteLength:tensorBytes(k.type) allocationByteLength:alignUp(kStorage,0x4000)
            ioSurfaceIndex:1 rowStrideBytes:kRow planeStrideBytes:kPlane
            batchStrideBytes:kStorage],
        [[ANEHWXBinding alloc]initWithIdentifier:v.name role:ANESurfaceRoleInput
            logicalByteLength:logical allocationByteLength:alignUp(qStorage,0x4000)
            ioSurfaceIndex:2 rowStrideBytes:headRow planeStrideBytes:qPlane
            batchStrideBytes:qStorage],
        [[ANEHWXBinding alloc]initWithIdentifier:output.identifier
            role:ANESurfaceRoleOutput logicalByteLength:logical
            allocationByteLength:alignUp(qStorage,0x4000) ioSurfaceIndex:3
            rowStrideBytes:headRow planeStrideBytes:qPlane batchStrideBytes:qStorage],
    ];
    return [[ANEHWXArtifact alloc]initWithImage:image bindings:bindings];
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
+ (ANEHWXArtifact *)encodeExactResNetGraph:(ANEOperationGraph *)graph
                                  modelRoot:(NSURL *)modelRoot
                                diagnostics:(ANEDiagnosticEngine *)diagnostics {
    return encodeResNet50Program(graph,modelRoot,diagnostics);
}
+ (ANEHWXArtifact *)encodeGraph:(ANEOperationGraph *)graph
                        scheduled:(ANEScheduledGraph * _Nullable)scheduled
                        modelRoot:(NSURL *)modelRoot
                      diagnostics:(ANEDiagnosticEngine *)diagnostics {
    if (ANEHWXArtifact *resnet=encodeResNet50Program(graph,modelRoot,diagnostics))
        return resnet;
    for(ANEOperationNode *node in graph.nodes)
        if([node.operationName isEqualToString:@"layer_norm"] ||
           [node.sourceOperation.operationName isEqualToString:@"layer_norm"])
            return encodeTransformerLayerNormProgram(graph,modelRoot,diagnostics);
    if (ANEHWXArtifact *attention=encodeTransformerFusedAttentionProgram(graph,diagnostics))
        return attention;
    if (ANEHWXArtifact *softmax=encodeTransformerSoftmaxProgram(graph,diagnostics))
        return softmax;
    if (ANEHWXArtifact *qk=encodeTransformerQKProgram(graph,diagnostics))
        return qk;
    if (ANEHWXArtifact *layout=encodeTransformerLayoutProgram(graph,diagnostics))
        return layout;
    for (ANEOperationNode *node in graph.nodes)
        if ([node.operationName isEqualToString:@"add"] &&
            graph.sourceFunction.inputs.count==1)
            return encodeWhisperPositionalAddProgram(graph,modelRoot,diagnostics);
    for (ANEOperationNode *node in graph.nodes)
        if ([node.operationName isEqualToString:@"transpose"] &&
            graph.sourceFunction.inputs.firstObject.type.shape.count==3)
            return encodeWhisperTransposeProgram(graph,diagnostics);
    for (ANEOperationNode *node in graph.nodes)
        if ([node.operationName isEqualToString:@"conv"] &&
            node.outputType.shape.count==3 &&
            graph.sourceFunction.inputs.firstObject.type.shape.count==3 &&
            (graph.sourceFunction.inputs.firstObject.type.shape[1].unsignedIntegerValue==80 ||
             graph.sourceFunction.inputs.firstObject.type.shape[1].unsignedIntegerValue==384 ||
             graph.sourceFunction.inputs.firstObject.type.shape[1].unsignedIntegerValue==512))
            return encodeWhisperStemProgram(graph,modelRoot,diagnostics);
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
