#import "ANEH14Compiler.h"
#import "ANEH13Compiler.h"

#import "ANEBlobResolver.h"
#import "ANEGraphVerifier.h"
#import "MILLexer.h"
#import "MILGraphImporter.h"
#import "MILParser.h"
#import "HWXObjectWriter.h"
#include "H14Program.h"

#include <cmath>
#include <cstdlib>
#include <exception>

static BOOL reject(ANEDiagnosticEngine *diagnostics, NSString *message,
                   ANEGraphOperation *operation = nil,
                   NSString *code = @"h14.unsupported-program") {
    ANESourceLocation start = ANESourceLocationMake(0, 1, 1);
    [diagnostics emitSeverity:ANEDiagnosticSeverityError
        code:code message:message
        range:operation ? operation.range : ANESourceRangeMake(start, start)];
    return NO;
}

/// One decoded-oracle parity target. H14 covers every family this compiler
/// encodes. H17 and H18 have templates only for the elementwise, unary and
/// scalar-constant families, and emit HWX only: no runtime reads their ANEC.
struct ParityTarget {
    ane::h14::Generation generation;
    HWXObjectArchitecture architecture;
    NSString *name;
    NSString *prefix;
};

static BOOL parityTarget(NSString *name, ParityTarget *target) {
    if ([name isEqualToString:@"H14"])
        *target = {ane::h14::Generation::H14, HWXObjectArchitectureH14, @"H14", @"h14"};
    else if ([name isEqualToString:@"H17"])
        *target = {ane::h14::Generation::H17, HWXObjectArchitectureH17, @"H17", @"h17"};
    else if ([name isEqualToString:@"H18"])
        *target = {ane::h14::Generation::H18, HWXObjectArchitectureH18, @"H18", @"h18"};
    else return NO;
    return YES;
}

static NSString *code(const ParityTarget &target, NSString *suffix) {
    return [NSString stringWithFormat:@"%@.%@", target.prefix, suffix];
}

static BOOL fp16Tensor(ANEGraphValue *value) {
    return value.type.kind == ANEValueTypeKindTensor &&
        value.type.elementType == ANEElementTypeFP16;
}

static BOOL tensor(ANEGraphValue *value, NSArray<NSNumber *> *shape) {
    return fp16Tensor(value) && [value.type.shape isEqualToArray:shape];
}

static BOOL tensorElementCount(ANEGraphValue *value, NSUInteger *count) {
    if (!fp16Tensor(value)) return NO;
    NSUInteger elements = 1;
    for (NSNumber *number in value.type.shape) {
        NSUInteger dimension = number.unsignedIntegerValue;
        if (!dimension || elements > NSUIntegerMax / dimension) return NO;
        elements *= dimension;
    }
    *count = elements;
    return YES;
}

static BOOL elementwiseShape(ANEGraphValue *value,
                             ane::h14::ElementwiseShape *shape) {
    if (!fp16Tensor(value) || value.type.shape.count != 4 ||
        value.type.shape[0].unsignedIntegerValue != 1) return NO;
    const uint64_t channels = value.type.shape[1].unsignedLongLongValue;
    const uint64_t height = value.type.shape[2].unsignedLongLongValue;
    const uint64_t width = value.type.shape[3].unsignedLongLongValue;
    if (!channels || !height || !width || channels > UINT32_MAX ||
        height > UINT32_MAX || width > UINT32_MAX) return NO;
    *shape = {static_cast<std::uint32_t>(channels),
              static_cast<std::uint32_t>(height),
              static_cast<std::uint32_t>(width)};
    return YES;
}

static BOOL binaryEncoding(NSString *name, ane::h14::BinaryOperation *encoding) {
    if ([name isEqualToString:@"add"]) *encoding = ane::h14::BinaryOperation::Add;
    else if ([name isEqualToString:@"mul"]) *encoding = ane::h14::BinaryOperation::Multiply;
    else if ([name isEqualToString:@"maximum"]) *encoding = ane::h14::BinaryOperation::Maximum;
    else if ([name isEqualToString:@"minimum"]) *encoding = ane::h14::BinaryOperation::Minimum;
    else if ([name isEqualToString:@"sub"]) *encoding = ane::h14::BinaryOperation::Subtract;
    else if ([name isEqualToString:@"real_div"]) *encoding = ane::h14::BinaryOperation::RealDivide;
    else return NO;
    return YES;
}

static BOOL unaryEncoding(NSString *name, ane::h14::UnaryOperation *encoding) {
    if ([name isEqualToString:@"abs"]) *encoding = ane::h14::UnaryOperation::Absolute;
    else if ([name isEqualToString:@"exp"]) *encoding = ane::h14::UnaryOperation::Exponential;
    else if ([name isEqualToString:@"gelu"]) *encoding = ane::h14::UnaryOperation::Gelu;
    else if ([name isEqualToString:@"leaky_relu"]) *encoding = ane::h14::UnaryOperation::LeakyRelu;
    else if ([name isEqualToString:@"relu"]) *encoding = ane::h14::UnaryOperation::Relu;
    else if ([name isEqualToString:@"rsqrt"]) *encoding = ane::h14::UnaryOperation::ReciprocalSquareRoot;
    else if ([name isEqualToString:@"sigmoid"]) *encoding = ane::h14::UnaryOperation::Sigmoid;
    else if ([name isEqualToString:@"silu"]) *encoding = ane::h14::UnaryOperation::Silu;
    else if ([name isEqualToString:@"sqrt"]) *encoding = ane::h14::UnaryOperation::SquareRoot;
    else if ([name isEqualToString:@"tanh"]) *encoding = ane::h14::UnaryOperation::Tanh;
    else return NO;
    return YES;
}

static BOOL normEncoding(NSString *name, ane::h14::NormOperation *encoding) {
    if ([name isEqualToString:@"softmax"]) *encoding = ane::h14::NormOperation::Softmax;
    else if ([name isEqualToString:@"layer_norm"]) *encoding = ane::h14::NormOperation::LayerNorm;
    else if ([name isEqualToString:@"reduce_sum"]) *encoding = ane::h14::NormOperation::ReduceSum;
    else if ([name isEqualToString:@"reduce_max"]) *encoding = ane::h14::NormOperation::ReduceMax;
    else if ([name isEqualToString:@"reduce_mean"]) *encoding = ane::h14::NormOperation::ReduceMean;
    else return NO;
    return YES;
}

/// The CHW surface H14 lays a logical MIL shape out as: leading unit
/// dimensions collapse into the batch, and a rank below three pads on the
/// left. Every decoded normalization and reduction surface follows this,
/// including the rank-reduced `keep_dims = false` results.
static BOOL normSurface(NSArray<NSNumber *> *shape,
                        ane::h14::ElementwiseShape *surface,
                        NSInteger *axisShift) {
    NSMutableArray<NSNumber *> *dimensions = [shape mutableCopy];
    NSInteger shift = 0;
    while (dimensions.count > 3 && dimensions[0].unsignedIntegerValue == 1) {
        [dimensions removeObjectAtIndex:0];
        --shift;
    }
    while (dimensions.count && dimensions.count < 3) {
        [dimensions insertObject:@1 atIndex:0];
        ++shift;
    }
    if (dimensions.count != 3) return NO;
    uint64_t extents[3];
    for (NSUInteger index = 0; index < 3; ++index) {
        extents[index] = dimensions[index].unsignedLongLongValue;
        if (!extents[index] || extents[index] > UINT32_MAX) return NO;
    }
    *surface = {static_cast<std::uint32_t>(extents[0]),
                static_cast<std::uint32_t>(extents[1]),
                static_cast<std::uint32_t>(extents[2])};
    if (axisShift) *axisShift = shift;
    return YES;
}

static BOOL int32Literal(ANEGraphArgument *argument, long long *value) {
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"int32"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    if (argument.kind != ANEGraphArgumentKindInteger) return NO;
    const char *text = argument.text.UTF8String;
    if (!text) return NO;
    char *end = nullptr;
    long long parsed = std::strtoll(text, &end, 10);
    if (end == text || *end) return NO;
    *value = parsed;
    return YES;
}

/// Resolves a constant axis operand — softmax's `int32` scalar or a rank-1
/// `int32` axes tensor — into the NCHW mask of the physical surface, with
/// `axisShift` mapping logical axes onto the canonical CHW the encoder keys
/// its templates on. Bit `i` of the mask is NCHW axis `i`.
static BOOL constantAxisMask(ANEGraphValue *value, NSUInteger rank,
                             NSInteger axisShift, std::uint32_t *mask) {
    if (!value || ![value.producer.operationName isEqualToString:@"const"] ||
        value.producer.arguments.count || !rank || rank > 4) return NO;
    ANEGraphArgument *literal = value.producer.attributes[@"val"];
    if (!literal || value.type.elementType != ANEElementTypeInt32) return NO;
    NSArray<ANEGraphArgument *> *axes = nil;
    if (value.type.kind == ANEValueTypeKindScalar) {
        axes = @[literal];
    } else {
        if (value.type.kind != ANEValueTypeKindTensor ||
            value.type.shape.count != 1 ||
            literal.kind != ANEGraphArgumentKindCall ||
            ![literal.calleeValueType isEqualToValueType:value.type] ||
            literal.callArguments.count != 1) return NO;
        ANEGraphArgument *payload = literal.callArguments[0].value;
        if (payload.kind != ANEGraphArgumentKindList ||
            payload.elements.count != value.type.shape[0].unsignedIntegerValue)
            return NO;
        axes = payload.elements;
    }
    if (!axes.count) return NO;
    std::uint32_t resolved = 0;
    for (ANEGraphArgument *element in axes) {
        long long axis = 0;
        if (!int32Literal(element, &axis)) return NO;
        if (axis < 0) axis += (long long)rank;
        if (axis < 0 || axis >= (long long)rank) return NO;
        // Canonical CHW index 0 is NCHW axis 1, so the surface bit is shifted.
        const long long surfaceAxis = axis + axisShift + 1;
        if (surfaceAxis < 1 || surfaceAxis > 3) return NO;
        const std::uint32_t bit = 1u << surfaceAxis;
        if (resolved & bit) return NO;
        resolved |= bit;
    }
    *mask = resolved;
    return YES;
}

static uint64_t roundToNearestEven(double value) {
    double lower = std::floor(value);
    double fraction = value - lower;
    return (uint64_t)lower +
        (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0.0));
}

static uint16_t fp16Bits(double value) {
    uint16_t sign = std::signbit(value) ? 0x8000u : 0;
    double magnitude = std::fabs(value);
    if (magnitude == 0.0) return sign;
    if (magnitude < std::ldexp(1.0, -14))
        return sign | (uint16_t)roundToNearestEven(std::ldexp(magnitude, 24));
    int exponent = 0;
    double fraction = std::frexp(magnitude, &exponent);
    int unbiasedExponent = exponent - 1;
    uint64_t significand = roundToNearestEven(std::ldexp(fraction, 11));
    if (significand == 2048) {
        significand = 1024;
        ++unbiasedExponent;
    }
    if (unbiasedExponent > 15) return sign | 0x7c00u;
    return sign | (uint16_t)((unbiasedExponent + 15) << 10) |
        (uint16_t)(significand - 1024);
}

static BOOL floatingText(ANEGraphArgument *argument, double *value) {
    if (argument.kind != ANEGraphArgumentKindFloatingPoint &&
        argument.kind != ANEGraphArgumentKindInteger) return NO;
    const char *text = argument.text.UTF8String;
    if (!text) return NO;
    char *end = nullptr;
    *value = std::strtod(text, &end);
    return end != text && !*end && std::isfinite(*value);
}

static BOOL fp16Scalar(ANEGraphArgument *argument, uint16_t *bits) {
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"fp16"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    double value = 0.0;
    if (!floatingText(argument, &value)) return NO;
    *bits = fp16Bits(value);
    return (*bits & 0x7c00u) != 0x7c00u;
}

static BOOL fp32Attribute(ANEGraphArgument *argument, float *valueOut) {
    if (argument.kind != ANEGraphArgumentKindCall ||
        ![argument.calleeName isEqualToString:@"fp32"] ||
        argument.callArguments.count != 1) return NO;
    double parsed = 0.0;
    if (!floatingText(argument.callArguments[0].value, &parsed)) return NO;
    *valueOut = static_cast<float>(parsed);
    return YES;
}

static NSString *stringArgument(ANEGraphArgument *argument) {
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"string"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    return argument.kind == ANEGraphArgumentKindString ? argument.text : nil;
}

static BOOL constantValue(ANEGraphValue *value) {
    return [value.producer.operationName isEqualToString:@"const"];
}

static NSUInteger logicalBytes(ANEGraphValue *value) {
    NSUInteger elements = 1;
    for (NSNumber *dimension in value.type.shape)
        elements *= dimension.unsignedIntegerValue;
    return elements * 2;
}

static NSDictionary *binding(ANEGraphValue *value,
                             const ane::h14::TensorLayout &layout) {
    NSMutableArray *physical = [NSMutableArray arrayWithCapacity:6];
    for (std::uint64_t dimension : layout.nchw) [physical addObject:@(dimension)];
    return @{@"name": value.name, @"dtype": @"float16",
        @"shape": value.type.shape, @"logicalBytes": @(logicalBytes(value)),
        @"index": @(layout.index), @"nchw": physical,
        @"allocationBytes": @(layout.allocationBytes)};
}

static HWXObjectBinding *objectBinding(const ane::h14::TensorLayout &layout,
                                       HWXObjectBindingRole role,
                                       NSUInteger ordinal) {
    NSArray<NSNumber *> *shape = @[@(layout.nchw[0]), @(layout.nchw[1]),
        @(layout.nchw[2]), @(layout.nchw[3])];
    NSUInteger batchStride = (NSUInteger)(layout.nchw[1] * layout.nchw[4]);
    NSUInteger storageBytes = (NSUInteger)(layout.nchw[0] * batchStride);
    NSString *name = [NSString stringWithFormat:
        role == HWXObjectBindingRoleInput ? @"input%lu" : @"output%lu",
        (unsigned long)ordinal];
    return [[HWXObjectBinding alloc] initWithSymbol:name shortName:name role:role
        elementType:ANEElementTypeFP16 shape:shape
        rowStrideBytes:(NSUInteger)layout.nchw[5]
        planeStrideBytes:(NSUInteger)layout.nchw[4]
        batchStrideBytes:batchStride storageByteLength:storageBytes];
}

/// `surfaceNames` holds the MIL names of the inputs then the output. Apple's
/// h17 objects lay their surfaces out in ascending name order (six naming
/// probes, receipts/2026-10-03-h17-target), which moves the descriptor's
/// resource slots when a broadcast operand is smaller than the result, so
/// H17 and H18 sort by name. H14 keeps input-then-output order.
static NSData *encodeHWX(const ane::h14::Program &program,
                         HWXObjectArchitecture architecture,
                         NSArray<NSString *> *surfaceNames, NSError **error) {
    NSMutableArray<HWXObjectBinding *> *bindings = [NSMutableArray array];
    for (NSUInteger index = 0; index < program.inputs.size(); ++index)
        [bindings addObject:objectBinding(program.inputs[index],
            HWXObjectBindingRoleInput, index)];
    [bindings addObject:objectBinding(program.output,
        HWXObjectBindingRoleOutput, 0)];
    if (architecture != HWXObjectArchitectureH14) {
        if (surfaceNames.count != bindings.count) {
            if (error) *error = [NSError errorWithDomain:@"dev.maderix.H14"
                code:3 userInfo:@{NSLocalizedDescriptionKey:
                    @"every surface needs a MIL name to be laid out"}];
            return nil;
        }
        NSMutableArray<NSNumber *> *order = [NSMutableArray array];
        for (NSUInteger index = 0; index < bindings.count; ++index)
            [order addObject:@(index)];
        [order sortUsingComparator:^NSComparisonResult(NSNumber *left,
                                                       NSNumber *right) {
            return [surfaceNames[left.unsignedIntegerValue]
                compare:surfaceNames[right.unsignedIntegerValue]];
        }];
        NSMutableArray<HWXObjectBinding *> *sorted = [NSMutableArray array];
        for (NSNumber *index in order)
            [sorted addObject:bindings[index.unsignedIntegerValue]];
        bindings = sorted;
    }
    NSData *task = [NSData dataWithBytes:program.taskStream.data()
                                  length:program.taskStream.size()];
    NSData *constants = [NSData dataWithBytes:program.constants.data()
                                       length:program.constants.size()];
    // firstTaskByteLength stays zero: the descriptor records the whole text
    // word count at 0x824, not the first task's size. H17 and H18 keep the
    // per-program unresolved word at 0x890, the slot the 0x8a0-byte layout
    // writes as its format code; H14 keeps it at 0x880.
    const BOOL h14 = architecture == HWXObjectArchitectureH14;
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:program.taskCount
        recordCount:program.programRecordCount
        formatCode:h14 ? 0 : program.unresolvedDescriptorWord
        scratchByteLength:program.scratchAllocationBytes descriptorLayout:HWXProgramDescriptorLayoutLinear];
    info.unresolvedDescriptorWord = program.unresolvedDescriptorWord;
    info.h14ScratchDescriptorWord = program.scratchDescriptorWord;
    if (!program.kernelTable.empty())
        info.kernelTable = [NSData dataWithBytes:program.kernelTable.data()
                                          length:program.kernelTable.size()];
    return [HWXObjectWriter buildObjectForArchitecture:architecture
        taskDescriptor:task constantRegion:constants bindings:bindings
        kernelRelocationOffsets:@[] programInfo:info error:error];
}

/// One fp16 scalar operand of a mul or add: a constant of type fp16 scalar whose literal has exactly `bits`.
static BOOL scalarOperation(ANEGraphOperation *operation, NSString *name,
                            ANEGraphValue *x, uint16_t bits) {
    ANEGraphValue *y = operation.operands[@"y"].value;
    uint16_t literal = 0;
    return [operation.operationName isEqualToString:name] &&
           operation.arguments.count == 2 && operation.operands[@"x"].value == x &&
           y && constantValue(y) && y.type.kind == ANEValueTypeKindScalar &&
           y.type.elementType == ANEElementTypeFP16 && !y.producer.arguments.count &&
           fp16Scalar(y.producer.attributes[@"val"], &literal) && literal == bits;
}

/// Matches the straight-line scaled exp chain, which Apple compiles to one three-task program:
/// e = exp(x), c = mul(x, 16 K), g = add(c, 16), p = mul(e, g), y = mul(p, 0.0625), with 16 K = fp16 0x1b18, over one fp16 [1, C, 1, 1] input.
/// Every other order, constant or surface is outside the decoded envelope.
static BOOL expScaledChainPlan(ANEGraphFunction *function,
                               NSArray<ANEGraphOperation *> *operations,
                               std::uint32_t *channels) {
    if (operations.count != 5 || function.inputs.count != 1) return NO;
    ANEGraphValue *x = function.inputs[0];
    ane::h14::ElementwiseShape shape{};
    if (!elementwiseShape(x, &shape) || shape.height != 1 || shape.width != 1)
        return NO;
    for (ANEGraphOperation *operation in operations)
        if (!tensor(operation.results[0], x.type.shape)) return NO;
    ANEGraphOperation *exponential = operations[0], *scaled = operations[1],
                      *shifted = operations[2], *product = operations[3],
                      *result = operations[4];
    ANEGraphValue *e = exponential.results[0], *c = scaled.results[0],
                  *g = shifted.results[0], *p = product.results[0];
    if (![exponential.operationName isEqualToString:@"exp"] ||
        exponential.arguments.count != 1 || exponential.operands[@"x"].value != x ||
        !scalarOperation(scaled, @"mul", x, 0x1b18) ||
        !scalarOperation(shifted, @"add", c, 0x4c00) ||
        ![product.operationName isEqualToString:@"mul"] ||
        product.arguments.count != 2 || product.operands[@"x"].value != e ||
        product.operands[@"y"].value != g ||
        !scalarOperation(result, @"mul", p, 0x2c00) ||
        !ane::h14::supportsExpScaledChainParity(shape.channels))
        return NO;
    *channels = shape.channels;
    return YES;
}

struct H14ParityPlan {
    BOOL unary;
    BOOL scalarConstant;
    ane::h14::UnaryOperation unaryOperation;
    ane::h14::BinaryOperation binaryOperation;
    uint16_t scalarBits;
    ane::h14::ElementwiseShape shape;
    ane::h14::ElementwiseShape operand;
};

/// Apple emits a different program per gelu mode, so each approximation is its
/// own encoder operation.
static BOOL geluEncoding(NSString *mode, ane::h14::UnaryOperation *encoding) {
    if ([mode isEqualToString:@"EXACT"]) *encoding = ane::h14::UnaryOperation::Gelu;
    else if ([mode isEqualToString:@"SIGMOID_APPROXIMATION"])
        *encoding = ane::h14::UnaryOperation::GeluSigmoidApproximation;
    else if ([mode isEqualToString:@"TANH_APPROXIMATION"])
        *encoding = ane::h14::UnaryOperation::GeluTanhApproximation;
    else return NO;
    return YES;
}

/// The decoded shapes a tensor may lower through: its literal NCHW geometry
/// when spatial, and its channel-flattened form, which shares the physical
/// 64-byte row layout the pipeline packs logical tensors into.
static NSUInteger parityShapes(ANEGraphValue *value,
                               ane::h14::ElementwiseShape shapes[2]) {
    NSUInteger elements = 0;
    if (!tensorElementCount(value, &elements) || elements > UINT32_MAX) return 0;
    NSUInteger count = 0;
    ane::h14::ElementwiseShape literal{};
    if (elementwiseShape(value, &literal) &&
        (literal.height != 1 || literal.width != 1))
        shapes[count++] = literal;
    shapes[count++] = {static_cast<std::uint32_t>(elements), 1, 1};
    return count;
}

/// Matches the operations whose `generation` task streams are decoded
/// word-for-word from Apple oracles, so they encode as one whole-tensor
/// program.
static BOOL parityPlan(ANEGraphOperation *operation,
                       ane::h14::Generation generation, H14ParityPlan *plan) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    NSString *name = operation.operationName;
    H14ParityPlan candidate{};
    ane::h14::ElementwiseShape shapes[2];
    NSUInteger shapeCount = parityShapes(operation.results[0], shapes);
    if (!shapeCount || !tensor(x, operation.results[0].type.shape)) return NO;
    BOOL leaky = [name isEqualToString:@"leaky_relu"];
    BOOL gelu = [name isEqualToString:@"gelu"];
    BOOL rsqrt = [name isEqualToString:@"rsqrt"];
    if (unaryEncoding(name, &candidate.unaryOperation)) {
        uint16_t alphaBits = 0;
        float epsilon = 0.0f, alpha = 0.0f;
        if (operation.arguments.count != (leaky || gelu || rsqrt ? 2u : 1u) ||
            (leaky && (!fp32Attribute(operation.arguments[@"alpha"], &alpha) ||
                       (alphaBits = fp16Bits(alpha)) != 0x3000)) ||
            (gelu && !geluEncoding(stringArgument(operation.arguments[@"mode"]),
                                   &candidate.unaryOperation)) ||
            (rsqrt && (!fp32Attribute(operation.arguments[@"epsilon"], &epsilon) ||
                       epsilon != 1e-6f)))
            return NO;
        for (NSUInteger index = 0; index < shapeCount; ++index) {
            if (!ane::h14::supportsElementwise(generation,
                    candidate.unaryOperation, shapes[index])) continue;
            candidate.shape = shapes[index];
            candidate.unary = YES;
            *plan = candidate;
            return YES;
        }
        return NO;
    }
    if (!binaryEncoding(name, &candidate.binaryOperation) ||
        operation.arguments.count != 2 || !y || constantValue(x)) return NO;
    BOOL runtime = !constantValue(y);
    // A runtime operand may be any decoded broadcast surface, not just the
    // result's own; a constant operand is the decoded fp16 0.5 scalar.
    ane::h14::ElementwiseShape operands[2];
    NSUInteger operandCount = 0;
    if (runtime) {
        operandCount = parityShapes(y, operands);
        if (!operandCount) return NO;
    } else if (y.type.kind != ANEValueTypeKindScalar ||
               y.type.elementType != ANEElementTypeFP16 ||
               y.producer.arguments.count ||
               !fp16Scalar(y.producer.attributes[@"val"], &candidate.scalarBits) ||
               candidate.scalarBits != 0x3800) {
        return NO;
    }
    for (NSUInteger index = 0; index < shapeCount; ++index)
        for (NSUInteger operandIndex = 0;
             operandIndex < (runtime ? operandCount : 1u); ++operandIndex) {
            const ane::h14::ElementwiseShape operand =
                runtime ? operands[operandIndex] : shapes[index];
            if (!ane::h14::supportsElementwise(generation,
                    candidate.binaryOperation, shapes[index], operand, !runtime))
                continue;
            candidate.shape = shapes[index];
            candidate.operand = operand;
            candidate.scalarConstant = !runtime;
            *plan = candidate;
            return YES;
        }
    return NO;
}

static BOOL boolean(ANEGraphArgument *argument, BOOL expected) {
    if (argument.kind == ANEGraphArgumentKindValue) {
        ANEGraphOperation *producer = argument.value.producer;
        if (![producer.operationName isEqualToString:@"const"] ||
            producer.results[0].type.kind != ANEValueTypeKindScalar ||
            producer.results[0].type.elementType != ANEElementTypeBool) return NO;
        argument = producer.attributes[@"val"];
    }
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"bool"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    return argument.kind == ANEGraphArgumentKindBoolean &&
        [argument.text isEqualToString:expected ? @"true" : @"false"];
}

/// Matches the softmax, layer_norm, and reduction programs whose whole H14
/// task streams are decoded from Apple oracles. `epsilon` must be the decoded
/// 1e-5 and layer_norm must carry no gamma or beta: Apple's compiler in this
/// harness rejects every affine form on H14 exactly as it does on H13, so no
/// oracle covers one (17 rejected `h14norm_*_affine` cases).
static BOOL normParityPlan(ANEGraphOperation *operation,
                           ane::h14::NormOperation *operationOut,
                           ane::h14::NormShape *shapeOut) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    NSString *name = operation.operationName;
    ane::h14::NormOperation encoding{};
    ane::h14::NormShape shape{};
    if (!normEncoding(name, &encoding) || !x) return NO;
    const BOOL softmax = encoding == ane::h14::NormOperation::Softmax;
    const BOOL layerNorm = encoding == ane::h14::NormOperation::LayerNorm;
    NSInteger inputShift = 0;
    if (!fp16Tensor(x) || !fp16Tensor(operation.results[0]) ||
        !normSurface(x.type.shape, &shape.input, &inputShift) ||
        !normSurface(operation.results[0].type.shape, &shape.output, nullptr))
        return NO;
    if (!constantAxisMask(operation.operands[softmax ? @"axis" : @"axes"].value,
                          x.type.shape.count, inputShift, &shape.axisMask))
        return NO;
    float epsilon = 0.0f;
    if (softmax) {
        if (operation.arguments.count != 2) return NO;
    } else if (layerNorm) {
        if (operation.arguments.count != 3 ||
            !fp32Attribute(operation.arguments[@"epsilon"], &epsilon) ||
            epsilon != 1e-5f) return NO;
    } else {
        if (operation.arguments.count != 3) return NO;
    }
    shape.keepDims = softmax || layerNorm ||
        boolean(operation.arguments[@"keep_dims"], YES);
    if (!shape.keepDims && !boolean(operation.arguments[@"keep_dims"], NO))
        return NO;
    if (!ane::h14::supportsNormParity(encoding, shape)) return NO;
    *operationOut = encoding;
    *shapeOut = shape;
    return YES;
}

/// The logical geometry of a rank-two or higher fp16 matmul. Every leading
/// dimension collapses into `rows`, because Apple's x surface is one dense
/// [1, 1, rows, reduction] plane however the MIL shape spells the batch.
struct H14ConvPlan {
    ane::h14::ConvShape shape;
    ANEGraphValue *weight;
    ANEGraphValue *bias;
};

/// Resolves a rank-1 `int32` constant into its literal values.
static BOOL int32Vector(ANEGraphValue *value, NSUInteger count,
                        long long *values) {
    if (!value || !constantValue(value) || value.producer.arguments.count ||
        value.type.kind != ANEValueTypeKindTensor ||
        value.type.elementType != ANEElementTypeInt32 ||
        value.type.shape.count != 1 ||
        value.type.shape[0].unsignedIntegerValue != count) return NO;
    ANEGraphArgument *literal = value.producer.attributes[@"val"];
    if (literal.kind != ANEGraphArgumentKindCall ||
        ![literal.calleeValueType isEqualToValueType:value.type] ||
        literal.callArguments.count != 1) return NO;
    ANEGraphArgument *payload = literal.callArguments[0].value;
    if (payload.kind != ANEGraphArgumentKindList ||
        payload.elements.count != count) return NO;
    for (NSUInteger index = 0; index < count; ++index)
        if (!int32Literal(payload.elements[index], &values[index])) return NO;
    return YES;
}

static BOOL int32Scalar(ANEGraphValue *value, long long *result) {
    if (!value || !constantValue(value) || value.producer.arguments.count ||
        value.type.kind != ANEValueTypeKindScalar ||
        value.type.elementType != ANEElementTypeInt32) return NO;
    return int32Literal(value.producer.attributes[@"val"], result);
}

static BOOL constantString(ANEGraphValue *value, NSString **text) {
    if (!value || !constantValue(value) || value.producer.arguments.count ||
        value.type.elementType != ANEElementTypeString) return NO;
    NSString *resolved = stringArgument(value.producer.attributes[@"val"]);
    if (!resolved) return NO;
    *text = resolved;
    return YES;
}

/// The CHW surface a convolution operand covers: a rank-4 NCHW tensor with a
/// batch of one, which is the only form the decoded corpus carries.
static BOOL convSurface(ANEGraphValue *value, ane::h14::ElementwiseShape *shape) {
    return elementwiseShape(value, shape);
}

/// Matches the convolutions whose whole H14 task stream and constant section
/// are decoded from Apple oracles.
///
/// coremltools 9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4 defines `conv` with a
/// `[Cout, Cin / groups, kh, kw]` weight, square `strides` and `dilations`
/// vectors, and either an explicit `pad` with `pad_type="custom"` or the
/// `same`, `same_lower` and `valid` spellings. The decoded corpus covers
/// square kernels, unit dilations, `same` and `valid`, and the explicit `pad`
/// vector only when it is all zeroes, which is what those two spellings emit.
static BOOL convParityPlan(ANEGraphOperation *operation, H14ConvPlan *plan,
                           BOOL generic = NO) {
    if (![operation.operationName isEqualToString:@"conv"]) return NO;
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *weight = operation.operands[@"weight"].value;
    ANEGraphValue *bias = operation.operands[@"bias"].value;
    H14ConvPlan candidate{};
    candidate.weight = weight;
    candidate.bias = bias;
    if (!x || !weight || !fp16Tensor(weight) || weight.type.shape.count != 4 ||
        !convSurface(x, &candidate.shape.input) ||
        !convSurface(operation.results[0], &candidate.shape.output)) return NO;
    if (operation.arguments.count != (bias ? 8u : 7u)) return NO;
    NSString *padType = nil;
    long long strides[2] = {0, 0}, dilations[2] = {0, 0}, padding[4] = {0, 0, 0, 0};
    long long groups = 1;
    if (!constantString(operation.operands[@"pad_type"].value, &padType) ||
        !int32Vector(operation.operands[@"strides"].value, 2, strides) ||
        !int32Vector(operation.operands[@"dilations"].value, 2, dilations) ||
        !int32Vector(operation.operands[@"pad"].value, 4, padding) ||
        !int32Scalar(operation.operands[@"groups"].value, &groups)) return NO;
    if (![padType isEqualToString:@"same"] && ![padType isEqualToString:@"valid"] &&
        !(generic && [padType isEqualToString:@"custom"]))
        return NO;
    if (strides[0] != strides[1] || dilations[0] != 1 || dilations[1] != 1 ||
        strides[0] < 1 || groups < 1) return NO;
    for (NSUInteger index = 0; index < 4; ++index)
        if (padding[index] && !generic) return NO;
    const std::uint32_t kernel = weight.type.shape[2].unsignedIntValue;
    if (weight.type.shape[3].unsignedIntValue != kernel || !kernel) return NO;
    if (weight.type.shape[0].unsignedIntValue != candidate.shape.output.channels)
        return NO;
    if (!candidate.shape.input.channels ||
        candidate.shape.input.channels % groups ||
        weight.type.shape[1].unsignedIntValue !=
            candidate.shape.input.channels / groups) return NO;
    // The result surface must be what the pad type asks for: `same` covers
    // ceil(D / stride) and `valid` covers floor((D - kernel) / stride) + 1.
    const std::uint32_t extents[2] = {candidate.shape.input.height,
                                      candidate.shape.input.width};
    const std::uint32_t results[2] = {candidate.shape.output.height,
                                      candidate.shape.output.width};
    for (NSUInteger axis = 0; axis < 2; ++axis) {
        const std::uint32_t stride = static_cast<std::uint32_t>(strides[0]);
        if (generic && [padType isEqualToString:@"custom"]) {
            const auto p = (kernel - 1) / 2;
            for (unsigned i = 0; i < 4; ++i)
                if (padding[i] != p) return NO;
            if (results[axis] != (extents[axis] + 2 * p - kernel) / stride + 1) return NO;
        } else if ([padType isEqualToString:@"same"]) {
            if (results[axis] != (extents[axis] + stride - 1) / stride) return NO;
        } else {
            if (extents[axis] < kernel ||
                results[axis] != (extents[axis] - kernel) / stride + 1) return NO;
        }
    }
    candidate.shape.kernel = kernel;
    candidate.shape.kernelWidth = kernel;
    candidate.shape.stride = static_cast<std::uint32_t>(strides[0]);
    candidate.shape.groups = static_cast<std::uint32_t>(groups);
    candidate.shape.bias = bias != nil;
    if (bias && (!fp16Tensor(bias) || bias.type.shape.count != 1 ||
                 bias.type.shape[0].unsignedIntValue !=
                     candidate.shape.output.channels)) return NO;
    if (generic ? !ane::h14::convGenericRefusal(candidate.shape).empty()
                : !ane::h14::supportsConvParity(candidate.shape)) return NO;
    *plan = candidate;
    return YES;
}

static BOOL matvecGeometry(ANEGraphValue *x, ANEGraphValue *result,
                           NSUInteger *rows, NSUInteger *reduction,
                           NSUInteger *columns) {
    if (!fp16Tensor(x) || !fp16Tensor(result)) return NO;
    NSArray<NSNumber *> *inputShape = x.type.shape;
    NSArray<NSNumber *> *outputShape = result.type.shape;
    NSUInteger rank = inputShape.count;
    if (rank < 2 || outputShape.count != rank) return NO;
    NSUInteger rowCount = 1;
    for (NSUInteger index = 0; index + 1 < rank; ++index) {
        NSUInteger dimension = inputShape[index].unsignedIntegerValue;
        if (!dimension || outputShape[index].unsignedIntegerValue != dimension ||
            rowCount > NSUIntegerMax / dimension) return NO;
        rowCount *= dimension;
    }
    NSUInteger inner = inputShape[rank - 1].unsignedIntegerValue;
    NSUInteger outer = outputShape[rank - 1].unsignedIntegerValue;
    if (!inner || !outer || inner > UINT32_MAX || outer > UINT32_MAX ||
        rowCount > UINT32_MAX || inner > NSUIntegerMax / 2 / outer) return NO;
    *rows = rowCount;
    *reduction = inner;
    *columns = outer;
    return YES;
}

/// Matches the matmul and linear operations whose two-task H14 streams are
/// decoded word-for-word from Apple oracles, and resolves their weight to the
/// [columns, reduction] row-major fp16 form the encoder packs.
static BOOL matvecPlan(ANEGraphOperation *operation, NSURL *modelRoot,
                       ANEDiagnosticEngine *diagnostics,
                       ane::h14::MatvecShape *shape,
                       NSData *__autoreleasing *weightsOut) {
    BOOL linear = [operation.operationName isEqualToString:@"linear"];
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = linear ? operation.operands[@"weight"].value
                              : operation.operands[@"y"].value;
    if (linear && operation.operands[@"bias"])
        return reject(diagnostics,
            @"H14 emits one program per artifact, so a linear bias has no encoder",
            operation, @"h14.linear-bias-unsupported");
    if (!linear && (operation.arguments.count != 4 ||
                    !boolean(operation.arguments[@"transpose_x"], NO)))
        return reject(diagnostics,
            @"H14 matmul requires an explicit transpose_x = false, an explicit transpose_y, x, and y",
            operation, @"h14.transpose-x-unsupported");
    BOOL transposeY = linear || boolean(operation.arguments[@"transpose_y"], YES);
    if (!linear && !transposeY &&
        !boolean(operation.arguments[@"transpose_y"], NO))
        return reject(diagnostics, @"H14 matmul requires an explicit transpose_y flag",
                      operation);
    NSUInteger rows = 0, reduction = 0, columns = 0;
    if (!x || !y || constantValue(x) || !constantValue(y) ||
        (linear && operation.arguments.count != 2) ||
        !matvecGeometry(x, operation.results[0], &rows, &reduction, &columns) ||
        !tensor(y, transposeY ? @[@(columns), @(reduction)]
                              : @[@(reduction), @(columns)]))
        return reject(diagnostics,
            @"H14 matmul requires a runtime fp16 x, a constant rank-2 weight matching its transpose flag, and a matching output shape",
            operation);
    ane::h14::MatvecShape candidate{static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(reduction),
        static_cast<std::uint32_t>(columns)};
    if (!ane::h14::supportsMatvecParity(candidate))
        return reject(diagnostics,
            @"H14 matmul supports only the decoded geometries: rows 1, 2, 8, or 64 with reduction and columns each 256, 512, or 1024; also rows 64 with (reduction, columns) (2048, 2048) or (4096, 2048)",
            operation, @"h14.outside-parity-envelope");
    NSUInteger count = reduction * columns;
    NSData *weights = [ANEBlobResolver loadConstantForOperation:y.producer
        expectedBytes:count * 2 modelRoot:modelRoot diagnostics:diagnostics];
    if (!weights) return NO;
    if (!transposeY) {
        // Apple rejects transpose_y=false, so the exact host transpose of the
        // [reduction, columns] weight feeds the same decoded encoder.
        NSMutableData *rowMajor = [NSMutableData dataWithLength:count * 2];
        const uint16_t *source = static_cast<const uint16_t *>(weights.bytes);
        uint16_t *destination = static_cast<uint16_t *>(rowMajor.mutableBytes);
        for (NSUInteger column = 0; column < columns; ++column)
            for (NSUInteger index = 0; index < reduction; ++index)
                destination[column * reduction + index] =
                    source[index * columns + column];
        weights = rowMajor;
    }
    *shape = candidate;
    *weightsOut = weights;
    return YES;
}

#include "H14ConvCompiler.inc"

@implementation ANEH14Compiler
+ (BOOL)compileMILData:(NSData *)milData
             modelRoot:(NSURL *)modelRoot
                target:(NSString *)targetName
                format:(NSString *)format
       outputDirectory:(NSURL *)directory
              schedule:(NSString *)schedule
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error {
    ParityTarget target{};
    if (!parityTarget(targetName, &target))
        return reject(diagnostics, [NSString stringWithFormat:
            @"parity target must be H14, H17, or H18, not '%@'", targetName],
            nil, @"parity.unsupported-target");
    const BOOL h14 = target.generation == ane::h14::Generation::H14;
    NSString *unsupported = code(target, @"unsupported-program");
    BOOL hwx = [format isEqualToString:@"hwx"];
    if (!hwx && (!h14 || ![format isEqualToString:@"anec"])) {
        NSString *message = h14
            ? @"H14 artifact format must be 'anec' or 'hwx'"
            : [NSString stringWithFormat:@"%@ artifact format must be 'hwx'",
                  target.name];
        reject(diagnostics, message, nil, code(target, @"unsupported-format"));
        if (error) *error = [NSError errorWithDomain:
            [@"dev.maderix." stringByAppendingString:target.name] code:2
            userInfo:@{NSLocalizedDescriptionKey: message}];
        return NO;
    }
    // H17 convolution graphs take their schedule to the formula backend below.
    if (target.generation != ane::h14::Generation::H17 &&
        ![schedule isEqualToString:@"per-op"] &&
        (!h14 || (![schedule isEqualToString:@"chain"] &&
                  ![schedule isEqualToString:@"linked"])))
        return reject(diagnostics, h14
            ? @"H14 schedule must be 'per-op', 'chain', or 'linked'"
            : [NSString stringWithFormat:@"%@ schedule must be 'per-op'",
                  target.name], nil, code(target, @"unsupported-schedule"));
    MILLexer *lexer = [[MILLexer alloc] initWithData:milData
                                         diagnostics:diagnostics];
    MILParser *parser = [[MILParser alloc] initWithTokens:lexer.lexAllTokens
        diagnostics:diagnostics];
    if (diagnostics.errorCount) return NO;
    MILProgramSyntax *syntax = parser.parseProgram;
    ANEGraphModule *module = syntax
        ? [MILGraphImporter importProgram:syntax diagnostics:diagnostics] : nil;
    if (!module || diagnostics.errorCount ||
        ![ANEGraphVerifier verifyModule:module diagnostics:diagnostics])
        return NO;
    if (module.functions.count != 1)
        return reject(diagnostics, [NSString stringWithFormat:
            @"%@ requires exactly one function", target.name], nil, unsupported);
    ANEGraphFunction *function = module.functions[0];
    if (target.generation == ane::h14::Generation::H17)
        for (ANEGraphOperation *operation in function.operations)
            if ([operation.operationName isEqualToString:@"conv"])
                return [ANEH13Compiler compileConvMILData:milData modelRoot:modelRoot
                    target:targetName format:format outputDirectory:directory
                    schedule:schedule diagnostics:diagnostics error:error];
    if (target.generation == ane::h14::Generation::H17 &&
        ![schedule isEqualToString:@"per-op"])
        return reject(diagnostics, @"H17 schedule must be 'per-op' outside convolution graphs",
                      nil, code(target, @"unsupported-schedule"));

    for (ANEGraphOperation *candidate in function.operations)
        if (candidate.results.count != 1)
            return reject(diagnostics, [NSString stringWithFormat:
                @"%@ does not lower multi-result operations", target.name],
                candidate, code(target, @"unsupported-multi-result-operation"));
    NSMutableArray<ANEGraphOperation *> *sourceOperations = [NSMutableArray array];
    for (ANEGraphOperation *candidate in function.operations)
        if (![candidate.operationName isEqualToString:@"const"])
            [sourceOperations addObject:candidate];
    if (!sourceOperations.count)
        return reject(diagnostics, [NSString stringWithFormat:
            @"%@ requires at least one operation", target.name], nil, unsupported);
    BOOL allConv = h14, hasConv = NO, hasRelu = NO, unknownConv = NO;
    for (ANEGraphOperation *op in sourceOperations) {
        const BOOL conv = [op.operationName isEqualToString:@"conv"];
        const BOOL relu = [op.operationName isEqualToString:@"relu"];
        allConv = allConv && (conv || relu);
        hasConv = hasConv || conv; hasRelu = hasRelu || relu;
        H14ConvPlan plan{};
        if (conv && !convParityPlan(op, &plan)) unknownConv = YES;
    }
    const BOOL linkedConv = [schedule isEqualToString:@"linked"];
    const char *forceGeneric = std::getenv("MILHWXC_H14_CONV_GENERIC");
    if (allConv && hasConv && (linkedConv || hasRelu || unknownConv ||
        (forceGeneric && std::string(forceGeneric) != "0")))
        return emitConvGraph(function, sourceOperations, modelRoot, directory,
                             format, linkedConv, diagnostics, error);
    if (linkedConv)
        return reject(diagnostics, @"H14 linked scheduling requires a straight convolution chain",
                      nil, @"h14.linked-outside-envelope");
    BOOL chain = sourceOperations.count > 1;
    BOOL chainSchedule = [schedule isEqualToString:@"chain"];
    ANEGraphOperation *lastOperation = sourceOperations.lastObject;
    if (function.returnValues.count != 1 ||
        function.returnValues[0] != lastOperation.results[0])
        return reject(diagnostics, [NSString stringWithFormat:chain
                ? @"%@ chains must return only the last operation result"
                : @"%@ requires one operation with its result returned",
                target.name],
            lastOperation, chain ? code(target, @"unsupported-chain")
                                 : unsupported);
    NSMutableSet<NSString *> *inputNames = [NSMutableSet set];
    for (ANEGraphValue *input in function.inputs) {
        [inputNames addObject:input.name];
        BOOL used = NO;
        for (ANEGraphOperation *candidate in sourceOperations)
            for (ANEGraphArgument *operand in candidate.operands.allValues)
                used = used || operand.value == input;
        if (!used)
            return reject(diagnostics, [NSString stringWithFormat:
                @"%@ function inputs must all be used", target.name],
                sourceOperations[0], unsupported);
    }
    for (NSUInteger index = 0; index + 1 < sourceOperations.count; ++index) {
        ANEGraphValue *value = sourceOperations[index].results[0];
        BOOL used = NO;
        for (NSUInteger consumer = index + 1; consumer < sourceOperations.count;
             ++consumer)
            for (ANEGraphArgument *operand in
                    sourceOperations[consumer].operands.allValues)
                used = used || operand.value == value;
        if (!used)
            return reject(diagnostics, [NSString stringWithFormat:
                @"%@ operation results not returned must be consumed later",
                target.name],
                sourceOperations[index], code(target, @"unsupported-chain"));
    }
    std::uint32_t expChainChannels = 0;
    const BOOL expChain = h14 && expScaledChainPlan(function, sourceOperations,
                                                    &expChainChannels);
    if (expChain) chainSchedule = NO;
    if (chainSchedule) {
        if (!chain || function.inputs.count > 2)
            return reject(diagnostics,
                @"H14 composed scheduling needs at least two operations and at most two boundary inputs",
                sourceOperations[0], @"h14.chain-outside-envelope");
        ANEGraphOperation *first = sourceOperations[0];
        if (sourceOperations.count != 2)
            return reject(diagnostics,
                @"h14.chain-unrepresentable-edge: composed scheduling supports exactly one producer followed by relu",
                sourceOperations[0], @"h14.chain-unrepresentable-edge");
        if (![lastOperation.operationName isEqualToString:@"relu"])
            return reject(diagnostics,
                @"h14.chain-unrepresentable-edge: only a final relu has a decoded single-kernel fusion",
                lastOperation, @"h14.chain-unrepresentable-edge");
        for (ANEGraphValue *input in function.inputs) {
            BOOL firstUse = NO;
            for (ANEGraphArgument *operand in first.operands.allValues)
                firstUse = firstUse || operand.value == input;
            if (!firstUse)
                return reject(diagnostics,
                    @"H14 composed scheduling needs every boundary input resident in the first operation",
                    first, @"h14.chain-outside-envelope");
        }
        for (NSUInteger index = 0; index + 1 < sourceOperations.count; ++index) {
            ANEGraphValue *value = sourceOperations[index].results[0];
            BOOL nextUse = NO, laterUse = NO;
            for (ANEGraphArgument *operand in
                    sourceOperations[index + 1].operands.allValues)
                nextUse = nextUse || operand.value == value;
            for (NSUInteger later = index + 2; later < sourceOperations.count;
                 ++later)
                for (ANEGraphArgument *operand in
                        sourceOperations[later].operands.allValues)
                    laterUse = laterUse || operand.value == value;
            if (!nextUse || laterUse)
                return reject(diagnostics,
                    @"H14 composed scheduling needs a straight-line chain with one live intermediate",
                    sourceOperations[index], @"h14.chain-outside-envelope");
        }
    }

    NSMutableArray<NSDictionary *> *programRecords = [NSMutableArray array];
    NSMutableArray<NSData *> *payloads = [NSMutableArray array];
    NSMutableArray<NSNumber *> *dispatchPlan = [NSMutableArray array];
    NSMutableArray<NSString *> *intermediates = [NSMutableArray array];
    NSMutableDictionary<NSString *, NSDictionary *> *tensors =
        [NSMutableDictionary dictionary];
    for (ANEGraphValue *input in function.inputs)
        tensors[input.name] = @{@"shape": input.type.shape,
            @"logicalBytes": @(logicalBytes(input)), @"role": @"input"};
    std::vector<ane::h14::Program> chainPrograms;
    NSMutableArray<NSDictionary *> *chainOpRecords = [NSMutableArray array];

    if (expChain) {
        ane::h14::Program program;
        NSData *payload = nil;
        try {
            program = ane::h14::encodeExpScaledChainParity(expChainChannels);
            if (hwx) {
                payload = encodeHWX(program, target.architecture,
                    @[function.inputs[0].name, lastOperation.results[0].name],
                    error);
                if (!payload) return NO;
            } else {
                std::vector<std::uint8_t> anec = ane::h14::encodeANEC(program);
                payload = [NSData dataWithBytes:anec.data() length:anec.size()];
            }
        } catch (const std::exception &exception) {
            return reject(diagnostics,
                [NSString stringWithUTF8String:exception.what()], lastOperation,
                unsupported);
        }
        tensors[lastOperation.results[0].name] = @{
            @"shape": lastOperation.results[0].type.shape,
            @"logicalBytes": @(logicalBytes(lastOperation.results[0])),
            @"role": @"output"};
        [programRecords addObject:@{
            @"file": [@"program-0." stringByAppendingString:format],
            @"bytes": @(payload.length), @"taskDescriptors": @(program.taskCount),
            @"encoder": @"apple-parity-expchain", @"operation": @"exp_scaled_chain",
            @"inputs": @[binding(function.inputs[0], program.inputs[0])],
            @"constantInputs": @{},
            @"outputs": @[binding(lastOperation.results[0], program.output)],
            @"constantOffset": @(program.constantOffsetBytes),
            @"constantBytes": @(program.constants.size()),
            @"firstTaskBytes": @(program.firstTaskBytes),
        }];
        [payloads addObject:payload];
        [dispatchPlan addObject:@0];
    }


    for (ANEGraphOperation *operation in expChain ? @[] : sourceOperations) {
        NSString *name = operation.operationName;
        if (chainSchedule && operation == lastOperation) {
            ANEGraphValue *value = operation.operands[@"x"].value;
            if (operation.arguments.count != 1 || !value)
                return reject(diagnostics,
                    @"h14.chain-unrepresentable-edge: final relu must have exactly one x argument",
                    operation, @"h14.chain-unrepresentable-edge");
            [chainOpRecords addObject:@{
                @"result": operation.results[0].name,
                @"operation": operation.operationName,
                @"inputs": @[@{@"name": value.name}],
            }];
            tensors[operation.results[0].name] = @{
                @"shape": operation.results[0].type.shape,
                @"logicalBytes": @(logicalBytes(operation.results[0])),
                @"role": @"output",
            };
            continue;
        }
        BOOL matvec = [name isEqualToString:@"matmul"] ||
            [name isEqualToString:@"linear"];
        H14ParityPlan plan{};
        ane::h14::MatvecShape matvecShape{};
        ane::h14::NormOperation normOperation{};
        ane::h14::NormShape normShape{};
        NSData *matvecWeights = nil;
        NSData *convWeights = nil;
        NSData *convBias = nil;
        H14ConvPlan convolution{};
        BOOL normalization = NO;
        if (!h14 && (matvec || [name isEqualToString:@"conv"] ||
                     normEncoding(name, &normOperation)))
            return reject(diagnostics, [NSString stringWithFormat:
                @"%@ has decoded templates only for elementwise, unary, and scalar-constant operations, not '%@'",
                target.name, name], operation,
                code(target, @"outside-parity-envelope"));
        if (matvec) {
            if (!matvecPlan(operation, modelRoot, diagnostics, &matvecShape,
                            &matvecWeights)) return NO;
        } else if (convParityPlan(operation, &convolution)) {
            const std::uint32_t reduction = convolution.shape.input.channels /
                convolution.shape.groups * convolution.shape.kernel *
                convolution.shape.kernelWidth;
            convWeights = [ANEBlobResolver
                loadConstantForOperation:convolution.weight.producer
                expectedBytes:convolution.shape.output.channels * reduction * 2
                modelRoot:modelRoot diagnostics:diagnostics];
            if (!convWeights) return NO;
            if (convolution.bias) {
                convBias = [ANEBlobResolver
                    loadConstantForOperation:convolution.bias.producer
                    expectedBytes:convolution.shape.output.channels * 2
                    modelRoot:modelRoot diagnostics:diagnostics];
                if (!convBias) return NO;
            }
        } else if ([name isEqualToString:@"conv"]) {
            return reject(diagnostics,
                @"H14 conv needs a decoded geometry: an fp16 rank-4 input and result with a batch of one, a constant weight, unit dilations, zero explicit padding, supported padding, and a shape inside the oracle parity envelope",
                operation, @"h14.conv-outside-envelope");
        } else if (normParityPlan(operation, &normOperation, &normShape)) {
            normalization = YES;
        } else if (normEncoding(name, &normOperation)) {
            return reject(diagnostics, [NSString stringWithFormat:
                @"H14 '%@' needs a decoded geometry inside the normalization parity envelope",
                name], operation, @"h14.norm-outside-envelope");
        } else if (!parityPlan(operation, target.generation, &plan)) {
            return reject(diagnostics, [NSString stringWithFormat:
                @"%@ supports only the decoded fp16 elementwise, scalar-constant, and unary parity envelope",
                target.name], operation, code(target, @"outside-parity-envelope"));
        }

        const BOOL convolutionProgram = convWeights != nil;
        NSUInteger inputCount = matvec || normalization || convolutionProgram ||
            plan.unary || plan.scalarConstant ? 1 : 2;
        ane::h14::Program program;
        try {
            program = convolutionProgram
                ? ane::h14::encodeConvParity(convolution.shape,
                    static_cast<const std::uint8_t *>(convWeights.bytes),
                    convWeights.length,
                    static_cast<const std::uint8_t *>(convBias.bytes),
                    convBias.length)
                : matvec
                ? ane::h14::encodeMatvecParity(matvecShape,
                    static_cast<const std::uint8_t *>(matvecWeights.bytes),
                    matvecWeights.length)
                : (normalization
                    ? ane::h14::encodeNormParity(normOperation, normShape)
                    : (plan.unary
                        ? ane::h14::encodeElementwise(target.generation,
                            plan.unaryOperation, plan.shape)
                        : ane::h14::encodeElementwise(target.generation,
                            plan.binaryOperation, plan.shape, plan.operand,
                            plan.scalarConstant, plan.scalarBits)));
        } catch (const std::exception &exception) {
            return reject(diagnostics,
                [NSString stringWithUTF8String:exception.what()], operation,
                unsupported);
        }
        if (program.inputs.size() != inputCount)
            return reject(diagnostics, [NSString stringWithFormat:
                @"%@ encoder returned unexpected inputs", target.name],
                operation, unsupported);
        NSData *payload = nil;
        try {
            if (hwx) {
                NSMutableArray<NSString *> *names = [NSMutableArray array];
                [names addObject:operation.operands[@"x"].value.name];
                if (inputCount == 2)
                    [names addObject:operation.operands[@"y"].value.name];
                [names addObject:operation.results[0].name];
                payload = encodeHWX(program, target.architecture, names, error);
                if (!payload) return NO;
            } else {
                std::vector<std::uint8_t> anec = ane::h14::encodeANEC(program);
                payload = [NSData dataWithBytes:anec.data() length:anec.size()];
            }
        } catch (const std::exception &exception) {
            return reject(diagnostics,
                [NSString stringWithUTF8String:exception.what()], operation,
                unsupported);
        }

        NSMutableArray<ANEGraphValue *> *inputValues = [NSMutableArray array];
        [inputValues addObject:operation.operands[@"x"].value];
        if (inputCount == 2)
            [inputValues addObject:operation.operands[@"y"].value];
        NSMutableArray<NSDictionary *> *inputRecords = [NSMutableArray array];
        for (NSUInteger index = 0; index < inputCount; ++index) {
            ANEGraphValue *value = inputValues[index];
            NSMutableDictionary *item = [binding(value, program.inputs[index])
                mutableCopy];
            if (![inputNames containsObject:value.name])
                item[@"role"] = @"intermediate";
            [inputRecords addObject:item];
        }
        BOOL final = operation == lastOperation;
        NSMutableDictionary *outputRecord =
            [binding(operation.results[0], program.output) mutableCopy];
        NSString *outputRole = final ? @"output" : @"intermediate";
        if (!final) {
            outputRecord[@"role"] = @"intermediate";
            [intermediates addObject:operation.results[0].name];
        }
        tensors[operation.results[0].name] = @{@"shape": operation.results[0].type.shape,
            @"logicalBytes": @(logicalBytes(operation.results[0])), @"role": outputRole};
        NSUInteger programIndex = programRecords.count;
        NSDictionary *record = @{
            @"file": [NSString stringWithFormat:@"program-%lu.%@",
                (unsigned long)programIndex, format],
            @"bytes": @(payload.length), @"taskDescriptors": @(program.taskCount),
            @"encoder": convolutionProgram ? @"apple-parity-conv"
                : (matvec ? @"apple-parity-matvec"
                : (normalization ? @"apple-parity-norm"
                : [target.prefix stringByAppendingString:@"-oracle-parity"])),
            @"operation": operation.operationName, @"inputs": inputRecords,
            @"constantInputs": @{}, @"outputs": @[outputRecord],
            @"constantOffset": @(program.constantOffsetBytes),
            @"constantBytes": @(program.constants.size()),
            @"firstTaskBytes": @(program.firstTaskBytes),
        };
        [programRecords addObject:record];
        [payloads addObject:payload];
        [dispatchPlan addObject:@(programIndex)];
        if (chainSchedule) {
            chainPrograms.push_back(std::move(program));
            [chainOpRecords addObject:@{
                @"result": operation.results[0].name,
                @"operation": operation.operationName,
                @"encoder": record[@"encoder"],
                @"inputs": record[@"inputs"],
                @"outputs": record[@"outputs"],
            }];
        }
    }

    NSMutableArray<NSDictionary *> *taskRecords = nil;
    NSDictionary *scratch = nil;
    if (chainSchedule) {
        NSDictionary *firstRecord = programRecords.firstObject;
        NSArray *boundaryInputs = firstRecord[@"inputs"];
        NSMutableSet<NSString *> *boundaryNames = [NSMutableSet set];
        for (NSDictionary *item in boundaryInputs)
            [boundaryNames addObject:item[@"name"]];
        if (boundaryInputs.count != function.inputs.count ||
            ![boundaryNames isEqualToSet:inputNames])
            return reject(diagnostics,
                @"H14 first operation bindings differ from the function inputs",
                sourceOperations[0], @"h14.chain-outside-envelope");
        NSMutableArray<NSString *> *fusedConsumers = [NSMutableArray array];
        ane::h14::Program combined;
        NSData *combinedPayload = nil;
        try {
            for (NSUInteger index = 1; index < chainOpRecords.count; ++index) {
                NSDictionary *consumer = chainOpRecords[index];
                NSDictionary *producer = chainOpRecords[index - 1];
                NSMutableArray<NSString *> *runtimeInputs = [NSMutableArray array];
                for (NSDictionary *item in consumer[@"inputs"])
                    if (!item[@"binding"]) [runtimeInputs addObject:item[@"name"]];
                if (runtimeInputs.count != 1 ||
                    ![runtimeInputs.firstObject isEqualToString:producer[@"result"]])
                    return reject(diagnostics,
                        @"h14.chain-unrepresentable-edge: composed scheduling covers a linear producer-consumer path, and this operation does not consume the previous operation's result",
                        lastOperation, @"h14.chain-unrepresentable-edge");
                if (![consumer[@"operation"] isEqualToString:@"relu"])
                    return reject(diagnostics,
                        @"h14.chain-unrepresentable-edge: no decoded fusion folds this operation into its producer; Apple's own chains keep such intermediates L2-resident through chaining words with no resolved formula",
                        lastOperation, @"h14.chain-unrepresentable-edge");
                const NSString *encoder = producer[@"encoder"];
                if ([encoder isEqualToString:@"apple-parity-matvec"])
                    ane::h14::fuseMatmulPostOperation(chainPrograms[0],
                        ane::h14::PostOperation::Relu);
                else if ([encoder isEqualToString:@"h14-oracle-parity"])
                    ane::h14::fuseElementwisePostOperation(chainPrograms[0],
                        ane::h14::PostOperation::Relu);
                else
                    return reject(diagnostics,
                        @"h14.chain-unrepresentable-edge: this producer encoding carries no decoded post-operation field",
                        lastOperation, @"h14.chain-unrepresentable-edge");
                [fusedConsumers addObject:consumer[@"result"]];
            }
            combined = ane::h14::composePrograms(chainPrograms);
            if (hwx) {
                combinedPayload = encodeHWX(combined, target.architecture, @[],
                                            error);
                if (!combinedPayload) return NO;
            } else {
                std::vector<std::uint8_t> anec = ane::h14::encodeANEC(combined);
                combinedPayload = [NSData dataWithBytes:anec.data()
                                                 length:anec.size()];
            }
        } catch (const std::exception &exception) {
            return reject(diagnostics,
                [NSString stringWithUTF8String:exception.what()], lastOperation,
                @"h14.chain-outside-envelope");
        }
        taskRecords = [NSMutableArray arrayWithCapacity:combined.taskCount];
        NSString *producerOperation = chainOpRecords.firstObject[@"operation"];
        for (NSUInteger index = 0; index < combined.taskCount; ++index)
            [taskRecords addObject:@{@"index": @(index),
                @"operation": producerOperation}];
        scratch = @{@"bytes": @0, @"regions": @{}};
        for (NSUInteger index = 0; index + 1 < chainOpRecords.count; ++index) {
            NSString *name = chainOpRecords[index][@"result"];
            [tensors removeObjectForKey:name];
            [intermediates removeObject:name];
        }
        NSMutableDictionary *fusedOutput =
            [chainOpRecords.firstObject[@"outputs"][0] mutableCopy];
        fusedOutput[@"name"] = chainOpRecords.lastObject[@"result"];
        [fusedOutput removeObjectForKey:@"role"];
        NSMutableDictionary *combinedRecord = [@{
            @"file": [@"program-0." stringByAppendingString:format],
            @"bytes": @(combinedPayload.length),
            @"taskDescriptors": @(combined.taskCount),
            @"encoder": @"composed-chain", @"operation": @"chain",
            @"inputs": boundaryInputs, @"constantInputs": @{},
            @"outputs": @[fusedOutput],
            @"constantOffset": @(combined.constantOffsetBytes),
            @"constantBytes": @(combined.constants.size()),
            @"firstTaskBytes": @(combined.firstTaskBytes),
        } mutableCopy];
        if (fusedConsumers.count) combinedRecord[@"fused"] = fusedConsumers;
        [programRecords removeAllObjects];
        [programRecords addObject:combinedRecord];
        [payloads removeAllObjects];
        [payloads addObject:combinedPayload];
        [dispatchPlan removeAllObjects];
        [dispatchPlan addObject:@0];
    }
    NSMutableDictionary *manifest = [@{
        @"schema": [NSString stringWithFormat:@"mil-hwxc.%@-anec-package.v1",
            target.prefix],
        @"target": target.name, @"artifactFormat": format,
        @"programs": programRecords, @"dispatchPlan": dispatchPlan,
        @"intermediates": intermediates, @"tensors": tensors,
    } mutableCopy];
    if (chainSchedule)
        [manifest addEntriesFromDictionary:@{@"schedule": @"chain",
            @"tasks": taskRecords, @"scratch": scratch}];
    if (programRecords.count == 1)
        [manifest addEntriesFromDictionary:programRecords[0]];
    NSData *metadata = [NSJSONSerialization dataWithJSONObject:manifest
        options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:error];
    if (!metadata) return NO;
    NSFileManager *manager = NSFileManager.defaultManager;
    if ([manager fileExistsAtPath:directory.path]) {
        NSArray *existing = [manager contentsOfDirectoryAtPath:directory.path
                                                         error:error];
        if (!existing) return NO;
        if (existing.count != 0) {
            if (error) *error = [NSError errorWithDomain:
                [@"dev.maderix." stringByAppendingString:target.name]
                code:1 userInfo:@{NSLocalizedDescriptionKey:
                    [NSString stringWithFormat:@"%@ output directory must be empty",
                        target.name]}];
            return NO;
        }
    } else if (![manager createDirectoryAtURL:directory
        withIntermediateDirectories:YES attributes:nil error:error]) return NO;
    for (NSUInteger index = 0; index < payloads.count; ++index)
        if (![payloads[index] writeToURL:[directory URLByAppendingPathComponent:
                programRecords[index][@"file"]] options:NSDataWritingAtomic
                error:error])
            return NO;
    return [metadata writeToURL:
        [directory URLByAppendingPathComponent:@"manifest.json"]
        options:NSDataWritingAtomic error:error];
}
@end
