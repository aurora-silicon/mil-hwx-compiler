#import "ANEH13Compiler.h"

#import "ANEBlobResolver.h"
#import "ANEGraphVerifier.h"
#import "ANEReluRegions.h"
#import "MILLexer.h"
#import "MILGraphImporter.h"
#import "MILParser.h"
#import "HWXObjectWriter.h"
#include "H13Program.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
static BOOL reject(ANEDiagnosticEngine *diagnostics, NSString *message,
                   ANEGraphOperation *operation = nil,
                   NSString *code = @"h13.unsupported-program") {
    ANESourceLocation start = ANESourceLocationMake(0, 1, 1);
    [diagnostics emitSeverity:ANEDiagnosticSeverityError
        code:code message:message
        range:operation ? operation.range : ANESourceRangeMake(start, start)];
    return NO;
}

static BOOL fp16Tensor(ANEGraphValue *value) {
    return value.type.kind == ANEValueTypeKindTensor &&
        value.type.elementType == ANEElementTypeFP16;
}

static BOOL boolTensor(ANEGraphValue *value) {
    return value.type.kind == ANEValueTypeKindTensor &&
        value.type.elementType == ANEElementTypeBool;
}

static BOOL tensor(ANEGraphValue *value, NSArray<NSNumber *> *shape) {
    return fp16Tensor(value) && [value.type.shape isEqualToArray:shape];
}

static BOOL tensorElementCount(ANEGraphValue *value, NSUInteger *count) {
    if (!fp16Tensor(value) && !boolTensor(value)) return NO;
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
                             ane::h13::ElementwiseShape *shape) {
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


static BOOL binaryEncoding(NSString *name, ane::h13::BinaryOperation *encoding) {
    if ([name isEqualToString:@"add"]) *encoding = ane::h13::BinaryOperation::Add;
    else if ([name isEqualToString:@"mul"]) *encoding = ane::h13::BinaryOperation::Multiply;
    else if ([name isEqualToString:@"maximum"]) *encoding = ane::h13::BinaryOperation::Maximum;
    else if ([name isEqualToString:@"minimum"]) *encoding = ane::h13::BinaryOperation::Minimum;
    else if ([name isEqualToString:@"sub"]) *encoding = ane::h13::BinaryOperation::Subtract;
    else if ([name isEqualToString:@"real_div"]) *encoding = ane::h13::BinaryOperation::RealDivide;
    else return NO;
    return YES;
}

static BOOL unaryEncoding(NSString *name, ane::h13::UnaryOperation *encoding) {
    if ([name isEqualToString:@"abs"]) *encoding = ane::h13::UnaryOperation::Absolute;
    else if ([name isEqualToString:@"exp"]) *encoding = ane::h13::UnaryOperation::Exponential;
    else if ([name isEqualToString:@"gelu"]) *encoding = ane::h13::UnaryOperation::Gelu;
    else if ([name isEqualToString:@"leaky_relu"]) *encoding = ane::h13::UnaryOperation::LeakyRelu;
    else if ([name isEqualToString:@"relu"]) *encoding = ane::h13::UnaryOperation::Relu;
    else if ([name isEqualToString:@"rsqrt"]) *encoding = ane::h13::UnaryOperation::ReciprocalSquareRoot;
    else if ([name isEqualToString:@"sigmoid"]) *encoding = ane::h13::UnaryOperation::Sigmoid;
    else if ([name isEqualToString:@"silu"]) *encoding = ane::h13::UnaryOperation::Silu;
    else if ([name isEqualToString:@"sqrt"]) *encoding = ane::h13::UnaryOperation::SquareRoot;
    else if ([name isEqualToString:@"tanh"]) *encoding = ane::h13::UnaryOperation::Tanh;
    else return NO;
    return YES;
}

static BOOL normEncoding(NSString *name, ane::h13::NormOperation *encoding) {
    if ([name isEqualToString:@"softmax"]) *encoding = ane::h13::NormOperation::Softmax;
    else if ([name isEqualToString:@"layer_norm"]) *encoding = ane::h13::NormOperation::LayerNorm;
    else if ([name isEqualToString:@"reduce_sum"]) *encoding = ane::h13::NormOperation::ReduceSum;
    else if ([name isEqualToString:@"reduce_max"]) *encoding = ane::h13::NormOperation::ReduceMax;
    else if ([name isEqualToString:@"reduce_mean"]) *encoding = ane::h13::NormOperation::ReduceMean;
    else return NO;
    return YES;
}

/// The CHW surface H13 lays a logical MIL shape out as: leading unit
/// dimensions collapse into the batch, and a rank below three pads on the
/// left. Every decoded normalization and reduction surface follows this,
/// including the rank-reduced `keep_dims = false` results.
static BOOL normSurface(NSArray<NSNumber *> *shape,
                        ane::h13::ElementwiseShape *surface,
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

static BOOL int32TensorScalar(ANEGraphValue *value, long long *result) {
    if (!value || ![value.producer.operationName isEqualToString:@"const"] ||
        value.producer.arguments.count ||
        value.type.kind != ANEValueTypeKindTensor || value.type.shape.count ||
        value.type.elementType != ANEElementTypeInt32) return NO;
    ANEGraphArgument *literal = value.producer.attributes[@"val"];
    if (literal.kind != ANEGraphArgumentKindCall ||
        ![literal.calleeValueType isEqualToValueType:value.type] ||
        literal.callArguments.count != 1) return NO;
    return int32Literal(literal.callArguments[0].value, result);
}

static NSArray<NSNumber *> *int32TensorElements(ANEGraphValue *value) {
    if (!value || ![value.producer.operationName isEqualToString:@"const"] ||
        value.producer.arguments.count ||
        value.type.kind != ANEValueTypeKindTensor ||
        value.type.shape.count != 1 ||
        value.type.elementType != ANEElementTypeInt32) return nil;
    const NSUInteger count = value.type.shape[0].unsignedIntegerValue;
    if (!count || count > 64) return nil;
    ANEGraphArgument *literal = value.producer.attributes[@"val"];
    if (literal.kind != ANEGraphArgumentKindCall ||
        ![literal.calleeValueType isEqualToValueType:value.type] ||
        literal.callArguments.count != 1) return nil;
    ANEGraphArgument *payload = literal.callArguments[0].value;
    if (payload.kind != ANEGraphArgumentKindList ||
        payload.elements.count != count) return nil;
    NSMutableArray<NSNumber *> *elements =
        [NSMutableArray arrayWithCapacity:count];
    for (ANEGraphArgument *element in payload.elements) {
        long long parsed = 0;
        if (!int32Literal(element, &parsed)) return nil;
        [elements addObject:@(parsed)];
    }
    return elements;
}

static NSArray<NSNumber *> *boolTensorElements(ANEGraphValue *value) {
    if (!value || ![value.producer.operationName isEqualToString:@"const"] ||
        value.producer.arguments.count ||
        value.type.kind != ANEValueTypeKindTensor ||
        value.type.shape.count != 1 ||
        value.type.elementType != ANEElementTypeBool) return nil;
    NSUInteger count = value.type.shape[0].unsignedIntegerValue;
    if (!count || count > 64) return nil;
    ANEGraphArgument *literal = value.producer.attributes[@"val"];
    if (literal.kind != ANEGraphArgumentKindCall ||
        ![literal.calleeValueType isEqualToValueType:value.type] ||
        literal.callArguments.count != 1) return nil;
    ANEGraphArgument *payload = literal.callArguments[0].value;
    if (payload.kind != ANEGraphArgumentKindList ||
        payload.elements.count != count) return nil;
    NSMutableArray<NSNumber *> *elements =
        [NSMutableArray arrayWithCapacity:count];
    for (ANEGraphArgument *element in payload.elements) {
        if (element.kind != ANEGraphArgumentKindBoolean ||
            (![element.text isEqualToString:@"true"] &&
             ![element.text isEqualToString:@"false"])) return nil;
        [elements addObject:@([element.text isEqualToString:@"true"])];
    }
    return elements;
}

static BOOL viewOperation(NSString *name) {
    return [name isEqualToString:@"reshape"] ||
        [name isEqualToString:@"squeeze"] ||
        [name isEqualToString:@"expand_dims"] ||
        [name isEqualToString:@"split"] ||
        [name isEqualToString:@"slice_by_index"] ||
        [name isEqualToString:@"pad"] ||
        [name isEqualToString:@"tile"] ||
        [name isEqualToString:@"transpose"];
}

/// The view family that never reorders elements: each result is the input's
/// row-major sequence under a shape relabel, so transpose composed with one
/// of them is a pure view exactly when the transpose alone is.
static BOOL reshapeFamily(NSString *name) {
    return [name isEqualToString:@"reshape"] ||
        [name isEqualToString:@"squeeze"] ||
        [name isEqualToString:@"expand_dims"];
}

static ANEGraphValue *storageOrigin(ANEGraphValue *value) {
    while (value && viewOperation(value.producer.operationName)) {
        ANEGraphValue *source = value.producer.operands[@"x"].value;
        if (!source || source == value) return nil;
        value = source;
    }
    return value;
}

static BOOL constantBackedView(ANEGraphValue *value) {
    ANEGraphValue *origin = storageOrigin(value);
    return [origin.producer.operationName isEqualToString:@"const"];
}
struct H13SplitAliasPlan {
    NSUInteger resultElements;
};

static BOOL splitAliasPlan(ANEGraphOperation *operation,
                           ANEDiagnosticEngine *diagnostics,
                           H13SplitAliasPlan *plan) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    long long axis = 0, count = 0;
    if (operation.arguments.count != 3 || !x ||
        !int32TensorScalar(operation.operands[@"axis"].value, &axis) ||
        !int32TensorScalar(operation.operands[@"num_splits"].value, &count))
        return reject(diagnostics,
            @"H13 split requires x and exact rank-zero tensor<int32,[]> axis and num_splits constants",
            operation, @"h13.invalid-split-parameters");
    if (constantBackedView(x))
        return reject(diagnostics,
            @"H13 cannot lower split views backed by constant storage",
            operation, @"h13.unsupported-constant-split-source");
    if (count != 2 || operation.results.count != 2)
        return reject(diagnostics, @"H13 split supports exactly two results",
            operation, @"h13.unsupported-split-count");
    if (!fp16Tensor(x) || !x.type.shape.count)
        return reject(diagnostics,
            @"H13 split requires a positive-rank static fp16 input",
            operation, @"h13.invalid-split-shape");
    if (axis < 0) axis += (long long)x.type.shape.count;
    if (axis < 0 || axis >= (long long)x.type.shape.count)
        return reject(diagnostics, @"H13 split axis is out of range",
            operation, @"h13.unsupported-split-axis");
    for (NSUInteger index = 0; index < (NSUInteger)axis; ++index)
        if (x.type.shape[index].unsignedIntegerValue != 1)
            return reject(diagnostics,
                @"H13 split requires unit dimensions before its axis because one binding slice cannot represent interleaved output chunks",
                operation, @"h13.noncontiguous-split-axis");
    NSUInteger inputElements = 0, resultElements = 0;
    if (!tensorElementCount(x, &inputElements) ||
        !tensorElementCount(operation.results[0], &resultElements) ||
        resultElements > NSUIntegerMax / 2 || inputElements != resultElements * 2)
        return reject(diagnostics,
            @"H13 split requires two equal positive static fp16 result shapes",
            operation, @"h13.invalid-split-shape");
    for (ANEGraphValue *result in operation.results) {
        NSUInteger elements = 0;
        if (!tensorElementCount(result, &elements) || elements != resultElements)
            return reject(diagnostics,
                @"H13 split requires two equal positive static fp16 result shapes",
                operation, @"h13.invalid-split-shape");
    }
    plan->resultElements = resultElements;
    return YES;
}

static BOOL argumentUsesValue(ANEGraphArgument *argument, ANEGraphValue *value) {
    if (!argument) return NO;
    if (argument.kind == ANEGraphArgumentKindValue)
        return argument.value == value;
    if (argument.kind == ANEGraphArgumentKindTuple ||
        argument.kind == ANEGraphArgumentKindList) {
        for (ANEGraphArgument *element in argument.elements)
            if (argumentUsesValue(element, value)) return YES;
    }
    return NO;
}

static BOOL operationUsesValue(ANEGraphOperation *operation, ANEGraphValue *value) {
    for (ANEGraphArgument *argument in operation.arguments.allValues)
        if (argumentUsesValue(argument, value)) return YES;
    return NO;
}

static NSArray<ANEGraphValue *> *concatSources(ANEGraphOperation *operation) {
    ANEGraphArgument *values = operation.arguments[@"values"];
    if (!values || (values.kind != ANEGraphArgumentKindTuple &&
                    values.kind != ANEGraphArgumentKindList) ||
        !values.elements.count)
        return nil;
    NSMutableArray<ANEGraphValue *> *sources =
        [NSMutableArray arrayWithCapacity:values.elements.count];
    for (ANEGraphArgument *element in values.elements) {
        if (element.kind != ANEGraphArgumentKindValue || !element.value)
            return nil;
        [sources addObject:element.value];
    }
    return sources;
}

/// Concat of independent sources is a named ISA hole: the decoded corpus
/// has no concat or copy encoder, and one binding slice cannot join N
/// buffers. Validate the encoder-shaped form, then refuse it.
static BOOL concatPlan(ANEGraphOperation *operation,
                       ANEDiagnosticEngine *diagnostics) {
    NSArray<ANEGraphValue *> *sources = concatSources(operation);
    long long axis = 0;
    if (!sources || sources.count < 2 || operation.results.count != 1 ||
        !int32TensorScalar(operation.operands[@"axis"].value, &axis))
        return reject(diagnostics,
            @"H13 concat requires a tuple of at least two values and an exact rank-zero tensor<int32,[]> axis",
            operation, @"h13.invalid-concat-parameters");
    ANEGraphValue *result = operation.results[0];
    ANEGraphValue *first = sources[0];
    if (!fp16Tensor(first) || !fp16Tensor(result) || !first.type.shape.count ||
        result.type.shape.count != first.type.shape.count)
        return reject(diagnostics,
            @"H13 concat requires static fp16 sources and result of equal rank",
            operation, @"h13.invalid-concat-shape");
    const long long rank = (long long)first.type.shape.count;
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        return reject(diagnostics, @"H13 concat axis is out of range",
            operation, @"h13.unsupported-concat-axis");
    NSUInteger concatExtent = 0;
    for (ANEGraphValue *source in sources) {
        if (!fp16Tensor(source) ||
            source.type.shape.count != first.type.shape.count)
            return reject(diagnostics,
                @"H13 concat requires static fp16 sources of equal rank",
                operation, @"h13.invalid-concat-shape");
        for (NSUInteger index = 0; index < (NSUInteger)rank; ++index) {
            NSUInteger extent = source.type.shape[index].unsignedIntegerValue;
            if (index == (NSUInteger)axis) {
                if (!extent || concatExtent > NSUIntegerMax - extent)
                    return reject(diagnostics,
                        @"H13 concat requires positive static axis extents",
                        operation, @"h13.invalid-concat-shape");
                concatExtent += extent;
                continue;
            }
            if (extent != first.type.shape[index].unsignedIntegerValue)
                return reject(diagnostics,
                    @"H13 concat requires matching extents off the concat axis",
                    operation, @"h13.invalid-concat-shape");
        }
    }
    if (result.type.shape[(NSUInteger)axis].unsignedIntegerValue != concatExtent)
        return reject(diagnostics,
            @"H13 concat result extent on the axis must equal the sum of the sources",
            operation, @"h13.invalid-concat-shape");
    for (NSUInteger index = 0; index < (NSUInteger)rank; ++index) {
        if (index == (NSUInteger)axis) continue;
        if (result.type.shape[index].unsignedIntegerValue !=
            first.type.shape[index].unsignedIntegerValue)
            return reject(diagnostics,
                @"H13 concat result extents off the axis must match the sources",
                operation, @"h13.invalid-concat-shape");
    }
    return reject(diagnostics,
        @"H13 has no concat encoder: every decoded H13 program writes one output surface from one input or a broadcast pair, and the binding ABI carries one contiguous slice per binding, so joining independent sources into one buffer needs a data-movement program the decoded corpus does not hold",
        operation, @"h13.unsupported-concat");
}

/// A MIL transpose resolves to one of three host-side forms: a free alias
/// when the permutation preserves the row-major element order (only unit
/// dimensions move), a fold into a consuming matmul's transpose flag when
/// the permutation swaps the operand's trailing two dimensions, or an exact
/// rejection because the binding ABI carries one contiguous slice per
/// operand and the decoded corpus holds no data-movement encoder.
///
/// `fastAxisSwapped` records whether the view's fastest (row) axis maps to
/// a non-fast storage axis: the NCHW descriptor requires row elements
/// contiguous, so such a permutation is inexpressible as a surface
/// interpretation at any size, while a fast-axis-stable permutation only
/// swaps the plane and row strides, which the descriptor can carry but no
/// decoded task stream reads.
struct H13TransposeViewPlan {
    BOOL layoutPreserving;
    BOOL tailSwap;
    BOOL fastAxisSwapped;
};

static BOOL transposeViewPlan(ANEGraphOperation *operation,
                              ANEDiagnosticEngine *diagnostics,
                              H13TransposeViewPlan *plan) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    NSArray<NSNumber *> *perm =
        int32TensorElements(operation.operands[@"perm"].value);
    if (operation.results.count != 1 || operation.arguments.count != 2 ||
        !x || perm.count != x.type.shape.count ||
        (!fp16Tensor(x) && !boolTensor(x)) || !x.type.shape.count)
        return reject(diagnostics,
            @"H13 transpose requires x and an exact rank-matching tensor<int32,[rank]> perm constant over a positive-rank static fp16 or bool input",
            operation, @"h13.invalid-transpose-parameters");
    const NSUInteger rank = x.type.shape.count;
    std::vector<NSUInteger> source(rank);
    for (NSUInteger index = 0; index < rank; ++index) {
        long long axis = perm[index].longLongValue;
        if (axis < 0) axis += (long long)rank;
        if (axis < 0 || axis >= (long long)rank)
            return reject(diagnostics,
                @"H13 transpose perm is out of range",
                operation, @"h13.invalid-transpose-parameters");
        source[index] = (NSUInteger)axis;
    }
    for (NSUInteger outer = 0; outer < rank; ++outer)
        for (NSUInteger inner = outer + 1; inner < rank; ++inner)
            if (source[outer] == source[inner])
                return reject(diagnostics,
                    @"H13 transpose perm must list every axis exactly once",
                    operation, @"h13.invalid-transpose-parameters");
    NSMutableArray<NSNumber *> *expected =
        [NSMutableArray arrayWithCapacity:rank];
    for (NSUInteger index = 0; index < rank; ++index)
        [expected addObject:x.type.shape[source[index]]];
    if (operation.results[0].type.kind != ANEValueTypeKindTensor ||
        operation.results[0].type.elementType != x.type.elementType ||
        ![operation.results[0].type.shape isEqualToArray:expected])
        return reject(diagnostics,
            @"H13 transpose result shape must be the permuted input shape",
            operation, @"h13.invalid-transpose-parameters");
    // The row-major element order survives a permutation exactly when every
    // free output dimension reads its input dimension with the same stride:
    // for each output axis whose moved-in extent is non-unit (unit extents
    // are always the constant index zero), the product of the non-unit
    // extents after it must equal its input dimension's input stride.
    BOOL preserving = YES;
    for (NSUInteger index = 0; index < rank && preserving; ++index) {
        const NSUInteger movedIn =
            x.type.shape[source[index]].unsignedIntegerValue;
        if (movedIn == 1) continue;
        unsigned long long outSuffix = 1, inSuffix = 1;
        for (NSUInteger after = index + 1; after < rank; ++after) {
            const NSUInteger extent =
                x.type.shape[source[after]].unsignedIntegerValue;
            if (extent != 1) {
                if (outSuffix > ULLONG_MAX / extent) {
                    preserving = NO;
                    break;
                }
                outSuffix *= extent;
            }
        }
        if (!preserving) break;
        for (NSUInteger after = source[index] + 1; after < rank; ++after) {
            const NSUInteger extent =
                x.type.shape[after].unsignedIntegerValue;
            if (inSuffix > ULLONG_MAX / extent) {
                preserving = NO;
                break;
            }
            inSuffix *= extent;
        }
        if (preserving) preserving = outSuffix == inSuffix;
    }
    BOOL tail = rank >= 2;
    for (NSUInteger index = 0; index + 2 < rank; ++index)
        if (source[index] != index) tail = NO;
    if (source[rank - 2] != rank - 1 || source[rank - 1] != rank - 2) tail = NO;
    plan->layoutPreserving = preserving;
    plan->tailSwap = tail;
    plan->fastAxisSwapped = rank >= 1 && source[rank - 1] != rank - 1;
    return YES;
}

/// Whether the decoded corpus lowers this transpose as one 1-task program:
/// fp16, perm exactly [0,2,1] over a rank-3 [1, A, B] surface or [0,2,1,3]
/// over a rank-4 [1, C, H, W] surface, with the Apple-normalized elementwise
/// triples of both sides covered by a table row.
static BOOL transposeParityShapes(ANEGraphOperation *operation,
                                  ANEGraphValue *input, ANEGraphValue *result,
                                  ane::h13::ElementwiseShape *inputShape,
                                  ane::h13::ElementwiseShape *outputShape) {
    if (!input || !result || !fp16Tensor(input) ||
        input.type.shape.count < 3 || input.type.shape.count > 4 ||
        result.type.shape.count != input.type.shape.count)
        return NO;
    NSArray<NSNumber *> *perm =
        int32TensorElements(operation.operands[@"perm"].value);
    if (perm.count != input.type.shape.count) return NO;
    const NSUInteger rank = input.type.shape.count;
    const NSUInteger expectedPerm[] = {0, 2, 1, 3};
    for (NSUInteger index = 0; index < rank; ++index)
        if (perm[index].unsignedIntegerValue != expectedPerm[index]) return NO;
    if ([input.type.shape[0] unsignedIntegerValue] != 1) return NO;
    if (rank == 3) {
        inputShape->channels = 1;
        inputShape->height = [input.type.shape[1] unsignedIntegerValue];
        inputShape->width = [input.type.shape[2] unsignedIntegerValue];
        outputShape->channels = 1;
        outputShape->height = inputShape->width;
        outputShape->width = inputShape->height;
    } else {
        inputShape->channels = [input.type.shape[1] unsignedIntegerValue];
        inputShape->height = [input.type.shape[2] unsignedIntegerValue];
        inputShape->width = [input.type.shape[3] unsignedIntegerValue];
        outputShape->channels = inputShape->height;
        outputShape->height = inputShape->channels;
        outputShape->width = inputShape->width;
    }
    if (!inputShape->channels || !inputShape->height || !inputShape->width)
        return NO;
    NSArray<NSNumber *> *expectedShape = rank == 3
        ? @[@1, @(outputShape->height), @(outputShape->width)]
        : @[@1, @(outputShape->channels), @(outputShape->height),
            @(outputShape->width)];
    if (![result.type.shape isEqualToArray:expectedShape]) return NO;
    return ane::h13::supportsTransposeParity(*inputShape, *outputShape);
}

/// Whether the decoded corpus lowers this slice_by_index as one 1-task
/// program: a rank-4 [1, C, H, W] fp16 input whose only sliced axis is the
/// last one, from element 0, with every earlier axis full, unit strides, and
/// the Apple-normalized elementwise triples of both sides covered by a table
/// row — the rel-pos windowing spell.
static BOOL sliceParityShapes(ANEGraphOperation *operation,
                              ANEGraphValue *input, ANEGraphValue *result,
                              ane::h13::ElementwiseShape *inputShape,
                              ane::h13::ElementwiseShape *outputShape) {
    if (!input || !result || !fp16Tensor(input) ||
        input.type.shape.count != 4 ||
        result.type.shape.count != 4 ||
        [input.type.shape[0] unsignedIntegerValue] != 1 ||
        [result.type.shape[0] unsignedIntegerValue] != 1)
        return NO;
    NSArray<NSNumber *> *begin =
        int32TensorElements(operation.operands[@"begin"].value);
    NSArray<NSNumber *> *end =
        int32TensorElements(operation.operands[@"end"].value);
    if (!begin || !end || begin.count != 4 || end.count != 4)
        return NO;
    for (NSString *key in @[@"stride", @"mask", @"squeeze_mask"])
        if (operation.operands[key].value) return NO;
    NSArray<NSNumber *> *endMask =
        boolTensorElements(operation.operands[@"end_mask"].value);
    NSArray<NSNumber *> *beginMask =
        boolTensorElements(operation.operands[@"begin_mask"].value);
    if (beginMask) return NO;
    for (NSUInteger index = 0; index < 4; ++index) {
        const long long extent = input.type.shape[index].longLongValue;
        long long lower = begin[index].longLongValue;
        long long upper = endMask && endMask[index].boolValue
            ? extent : end[index].longLongValue;
        if (lower < 0) lower += extent;
        if (upper < 0) upper += extent;
        if (lower != 0) return NO;
        if (upper == extent) continue;
        if (index != 3) return NO;
    }
    inputShape->channels = [input.type.shape[1] unsignedIntegerValue];
    inputShape->height = [input.type.shape[2] unsignedIntegerValue];
    inputShape->width = [input.type.shape[3] unsignedIntegerValue];
    outputShape->channels = [result.type.shape[1] unsignedIntegerValue];
    outputShape->height = [result.type.shape[2] unsignedIntegerValue];
    outputShape->width = [result.type.shape[3] unsignedIntegerValue];
    if (outputShape->channels != inputShape->channels ||
        outputShape->height != inputShape->height)
        return NO;
    return ane::h13::supportsSliceParity(*inputShape, *outputShape);
}

/// A contiguous slice_by_index lowers as one offset view of its storage:
/// unit strides, at most one sliced dimension, unit dimensions before it,
/// and end_mask/begin_mask/negative bounds resolved to literal ranges.
struct H13SliceViewPlan {
    NSUInteger offset;
    NSUInteger resultElements;
    BOOL identity;
};

static BOOL sliceByIndexViewPlan(ANEGraphOperation *operation,
                                 ANEDiagnosticEngine *diagnostics,
                                 H13SliceViewPlan *plan) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    NSArray<NSNumber *> *begin =
        int32TensorElements(operation.operands[@"begin"].value);
    NSArray<NSNumber *> *end =
        int32TensorElements(operation.operands[@"end"].value);
    NSArray<NSNumber *> *endMask =
        boolTensorElements(operation.operands[@"end_mask"].value);
    NSArray<NSNumber *> *beginMask =
        boolTensorElements(operation.operands[@"begin_mask"].value);
    NSArray<NSNumber *> *stride =
        int32TensorElements(operation.operands[@"stride"].value);
    NSArray<NSNumber *> *mask =
        boolTensorElements(operation.operands[@"mask"].value);
    NSArray<NSNumber *> *squeezeMask =
        boolTensorElements(operation.operands[@"squeeze_mask"].value);
    if (operation.results.count != 1 || !x ||
        !fp16Tensor(x) || !x.type.shape.count ||
        x.type.shape.count > 64 || !begin || !end ||
        begin.count != x.type.shape.count ||
        end.count != x.type.shape.count)
        return reject(diagnostics,
            @"H13 slice_by_index requires x with exact rank-matching tensor<int32,[rank]> begin and end constants over a positive-rank static fp16 input",
            operation, @"h13.invalid-slice-parameters");
    const NSUInteger rank = x.type.shape.count;
    if ((endMask && endMask.count != rank) ||
        (beginMask && beginMask.count != rank) ||
        (stride && stride.count != rank) ||
        (mask && mask.count != rank) ||
        (squeezeMask && squeezeMask.count != rank))
        return reject(diagnostics,
            @"H13 slice_by_index mask and stride constants must match the input rank",
            operation, @"h13.invalid-slice-parameters");
    for (NSUInteger index = 0; index < rank; ++index) {
        if (mask && !mask[index].boolValue)
            return reject(diagnostics,
                @"H13 slice_by_index cannot lower a false mask entry because the dropped axis leaves the one-slice binding ABI",
                operation, @"h13.noncontiguous-slice");
        if (squeezeMask && squeezeMask[index].boolValue)
            return reject(diagnostics,
                @"H13 slice_by_index cannot lower a squeeze_mask entry because the dropped axis leaves the one-slice binding ABI",
                operation, @"h13.noncontiguous-slice");
        if (stride && stride[index].longLongValue != 1)
            return reject(diagnostics,
                @"H13 slice_by_index requires unit strides because a strided view is not one contiguous binding slice",
                operation, @"h13.noncontiguous-slice");
    }
    NSInteger slicedAxis = -1;
    long long sliceBegin = 0, sliceUpper = 0;
    for (NSUInteger index = 0; index < rank; ++index) {
        const NSUInteger extent =
            x.type.shape[index].unsignedIntegerValue;
        long long lower = begin[index].longLongValue;
        long long upper = endMask && endMask[index].boolValue
            ? (long long)extent : end[index].longLongValue;
        if (beginMask && beginMask[index].boolValue) lower = 0;
        if (lower < 0) lower += (long long)extent;
        if (upper < 0) upper += (long long)extent;
        if (lower < 0 || lower > (long long)extent ||
            upper < lower || upper > (long long)extent)
            return reject(diagnostics,
                @"H13 slice_by_index ranges must resolve inside [0, extent] with end not below begin",
                operation, @"h13.invalid-slice-range");
        if (lower == 0 && upper == (long long)extent) continue;
        if (slicedAxis >= 0)
            return reject(diagnostics,
                @"H13 slice_by_index can slice at most one dimension because interleaved output chunks exceed the one-slice binding ABI",
                operation, @"h13.noncontiguous-slice");
        slicedAxis = (NSInteger)index;
        sliceBegin = lower;
        sliceUpper = upper;
    }
    NSUInteger trailing = 1;
    for (NSUInteger index = slicedAxis == -1 ? rank : (NSUInteger)slicedAxis + 1;
         index < rank; ++index)
        trailing *= x.type.shape[index].unsignedIntegerValue;
    NSUInteger headChunks = 1;
    for (NSUInteger index = 0;
         slicedAxis >= 0 && index < (NSUInteger)slicedAxis; ++index)
        headChunks *= x.type.shape[index].unsignedIntegerValue;
    for (NSUInteger index = 0;
         slicedAxis >= 0 && index < (NSUInteger)slicedAxis; ++index)
        if (x.type.shape[index].unsignedIntegerValue != 1) {
            const NSUInteger extent =
                x.type.shape[(NSUInteger)slicedAxis].unsignedIntegerValue;
            const NSUInteger kept =
                (NSUInteger)(sliceUpper - sliceBegin);
            return reject(diagnostics,
                [NSString stringWithFormat:
                    @"H13 slice_by_index over non-unit head dimensions reads %lu chunks of %lu elements spaced %lu apart, and one binding slice cannot represent interleaved chunks: a chunked consumer decomposition has no MIL-expressible direct reference to byte-prove against, because no view op exposes mid-range head slices",
                    (unsigned long)headChunks, (unsigned long)(kept * trailing),
                    (unsigned long)(extent * trailing)],
                operation, @"h13.noncontiguous-slice");
        }
    NSUInteger inputElements = 0, resultElements = 0;
    if (!tensorElementCount(x, &inputElements) ||
        !tensorElementCount(operation.results[0], &resultElements))
        return reject(diagnostics,
            @"H13 slice_by_index requires static fp16 input and result shapes",
            operation, @"h13.invalid-slice-shape");
    NSUInteger offset = 0;
    if (slicedAxis >= 0) {
        // A contiguous [begin, end) range on the sliced axis with unit head
        // dimensions is one offset view over that range's trailing
        // elements — the same valueBaseOffsets mechanism the split lowering
        // established, so the end may fall anywhere inside the extent.
        const NSUInteger count =
            (NSUInteger)(sliceUpper - sliceBegin);
        const NSUInteger sliceOffset = (NSUInteger)sliceBegin;
        if (count > NSUIntegerMax / trailing || resultElements > NSUIntegerMax / 2 ||
            count * trailing != resultElements || sliceOffset > NSUIntegerMax / trailing)
            return reject(diagnostics,
                @"H13 slice_by_index result shape must equal the sliced extents",
                operation, @"h13.invalid-slice-shape");
        offset = sliceOffset * trailing;
    } else if (resultElements != inputElements) {
        return reject(diagnostics,
            @"H13 slice_by_index result shape must equal the sliced extents",
            operation, @"h13.invalid-slice-shape");
    }
    plan->offset = offset;
    plan->resultElements = resultElements;
    plan->identity = offset == 0 && resultElements == inputElements;
    return YES;
}

/// pad and tile only lower as identity views: any replicated or padded
/// element needs a producer program, and the decoded corpus holds none.
static BOOL identityExtentViewPlan(ANEGraphOperation *operation,
                                   NSString *parameter,
                                   long long identityExtent,
                                   ANEDiagnosticEngine *diagnostics,
                                   NSString *unsupportedCode) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    NSArray<NSNumber *> *extents =
        int32TensorElements(operation.operands[parameter].value);
    if (operation.results.count != 1 || !x || !fp16Tensor(x) ||
        !x.type.shape.count)
        return reject(diagnostics,
            [NSString stringWithFormat:
                @"H13 %@ requires x and an exact tensor<int32,[rank]> %@ constant over a positive-rank static fp16 input",
                operation.operationName, parameter],
            operation, @"h13.invalid-view-parameters");
    for (NSNumber *extent in extents)
        if (extent.longLongValue != identityExtent) {
            NSMutableArray<NSString *> *texts =
                [NSMutableArray arrayWithCapacity:extents.count];
            for (NSNumber *value in extents)
                [texts addObject:value.stringValue];
            return reject(diagnostics,
                [NSString stringWithFormat:
                    @"H13 %@ with %@ [%@] needs a data-movement program: every decoded H13 encoder writes an output surface of its input's shape and the binding ABI carries one contiguous slice, so no host-side form can materialize the grown or replicated region — and Apple's own tool rejects pad in every form including the identity view (6/6 callback_status=1), so pad elimination belongs to the frontend, not to a new encoder",
                    operation.operationName, parameter,
                    [texts componentsJoinedByString:@","]],
                operation, unsupportedCode);
        }
    NSUInteger inputElements = 0, resultElements = 0;
    if (!tensorElementCount(x, &inputElements) ||
        !tensorElementCount(operation.results[0], &resultElements) ||
        inputElements != resultElements ||
        ![operation.results[0].type.shape isEqualToArray:x.type.shape])
        return reject(diagnostics,
            [NSString stringWithFormat:
                @"H13 %@ identity view requires result shapes equal to x",
                operation.operationName],
            operation, @"h13.invalid-view-shape");
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

static BOOL fp16Scalar(ANEGraphArgument *argument, uint16_t *bits) {
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"fp16"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    if (argument.kind != ANEGraphArgumentKindFloatingPoint &&
        argument.kind != ANEGraphArgumentKindInteger) return NO;
    const char *text = argument.text.UTF8String;
    if (!text) return NO;
    char *end = nullptr;
    double value = std::strtod(text, &end);
    if (end == text || *end || !std::isfinite(value)) return NO;
    *bits = fp16Bits(value);
    return (*bits & 0x7c00u) != 0x7c00u;
}

static BOOL exactFP16Attribute(ANEGraphArgument *argument, uint16_t *bits,
                               double *valueOut) {
    if (argument.kind != ANEGraphArgumentKindCall ||
        ![argument.calleeName isEqualToString:@"fp32"] ||
        argument.callArguments.count != 1) return NO;
    argument = argument.callArguments[0].value;
    if (argument.kind != ANEGraphArgumentKindFloatingPoint &&
        argument.kind != ANEGraphArgumentKindInteger) return NO;
    const char *text = argument.text.UTF8String;
    if (!text) return NO;
    char *end = nullptr;
    double parsed = std::strtod(text, &end);
    float value = static_cast<float>(parsed);
    if (end == text || *end || !std::isfinite(value)) return NO;
    *bits = fp16Bits(value);
    uint16_t exponent = (*bits >> 10) & 0x1fu;
    uint16_t fraction = *bits & 0x03ffu;
    double magnitude = exponent
        ? std::ldexp(1024.0 + fraction, exponent - 25)
        : std::ldexp(static_cast<double>(fraction), -24);
    double decoded = (*bits & 0x8000u) ? -magnitude : magnitude;
    if (static_cast<double>(value) != decoded) return NO;
    *valueOut = value;
    return YES;
}

static BOOL fp32Attribute(ANEGraphArgument *argument, float *valueOut) {
    if (argument.kind != ANEGraphArgumentKindCall ||
        ![argument.calleeName isEqualToString:@"fp32"] ||
        argument.callArguments.count != 1) return NO;
    argument = argument.callArguments[0].value;
    if (argument.kind != ANEGraphArgumentKindFloatingPoint &&
        argument.kind != ANEGraphArgumentKindInteger) return NO;
    const char *text = argument.text.UTF8String;
    if (!text) return NO;
    char *end = nullptr;
    double parsed = std::strtod(text, &end);
    if (end == text || *end || !std::isfinite(parsed)) return NO;
    *valueOut = static_cast<float>(parsed);
    return YES;
}

static NSData *splatFP16(uint16_t bits) {
    NSMutableData *data = [NSMutableData dataWithLength:128];
    uint16_t *words = static_cast<uint16_t *>(data.mutableBytes);
    for (NSUInteger index = 0; index < 64; ++index) words[index] = bits;
    return data;
}

static ANEGraphArgument *valueArgument(ANEGraphValue *value,
                                       ANESourceRange range) {
    return [[ANEGraphArgument alloc] initWithKind:ANEGraphArgumentKindValue
        text:nil value:value calleeName:nil calleeValueType:nil
        callArguments:@[] elements:@[] range:range];
}

static ANEGraphArgument *booleanArgument(BOOL value, ANESourceRange range) {
    return [[ANEGraphArgument alloc] initWithKind:ANEGraphArgumentKindBoolean
        text:value ? @"true" : @"false" value:nil calleeName:nil
        calleeValueType:nil callArguments:@[] elements:@[] range:range];
}

static ANEGraphOperation *binaryOperation(NSString *name, ANEGraphValue *x,
                                          ANEGraphValue *y,
                                          ANEGraphValue *result,
                                          ANESourceRange range) {
    return [[ANEGraphOperation alloc] initWithOperationName:name results:@[result]
        arguments:@{@"x": valueArgument(x, range), @"y": valueArgument(y, range)}
        attributes:@{} range:range];
}

static NSString *hexData(NSData *data) {
    const uint8_t *bytes = static_cast<const uint8_t *>(data.bytes);
    NSMutableString *hex = [NSMutableString stringWithCapacity:data.length * 2];
    for (NSUInteger index = 0; index < data.length; ++index)
        [hex appendFormat:@"%02x", bytes[index]];
    return hex;
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

static BOOL matmulGeometry(ANEGraphValue *x, ANEGraphValue *result,
                           BOOL transposeX, NSUInteger *reduction,
                           NSUInteger *rows, NSUInteger *columns) {
    if (!fp16Tensor(x) || !fp16Tensor(result)) return NO;
    NSArray<NSNumber *> *inputShape = x.type.shape;
    NSArray<NSNumber *> *outputShape = result.type.shape;
    NSUInteger rank = inputShape.count;
    if (outputShape.count != rank || (!transposeX && rank == 0) ||
        (transposeX && rank < 2)) return NO;

    NSUInteger leading = rank - (transposeX ? 2 : 1);
    NSUInteger rowCount = 1;
    for (NSUInteger index = 0; index < leading; ++index) {
        NSUInteger dimension = inputShape[index].unsignedIntegerValue;
        if (!dimension || outputShape[index].unsignedIntegerValue != dimension ||
            rowCount > NSUIntegerMax / dimension) return NO;
        rowCount *= dimension;
    }
    NSUInteger candidate = inputShape[rank - (transposeX ? 2 : 1)].unsignedIntegerValue;
    NSUInteger outputColumns = outputShape[rank - 1].unsignedIntegerValue;
    if (!candidate || !outputColumns) return NO;
    if (transposeX) {
        NSUInteger rowDimension = inputShape[rank - 1].unsignedIntegerValue;
        if (!rowDimension || outputShape[rank - 2].unsignedIntegerValue != rowDimension ||
            rowCount > NSUIntegerMax / rowDimension) return NO;
        rowCount *= rowDimension;
    }
    if (rowCount > NSUIntegerMax / candidate ||
        rowCount > NSUIntegerMax / outputColumns) return NO;
    *reduction = candidate;
    *rows = rowCount;
    *columns = outputColumns;
    return YES;
}

/// The NCHW surface a rank-4 elementwise operand lays out as, batch included.
static BOOL batchedShape(ANEGraphValue *value, ane::h13::BatchedShape *shape) {
    // The encoder-geometry oracles record the rank-3 spelling Apple's own
    // frontend produces: [1, A, B] compiles as the [1, 1, A, B] surface.
    if (!fp16Tensor(value) || value.type.shape.count < 3 ||
        value.type.shape.count > 4) return NO;
    uint64_t extents[4] = {1, 1, 1, 1};
    NSUInteger base = 4 - value.type.shape.count;
    for (NSUInteger index = 0; index < value.type.shape.count; ++index) {
        extents[base + index] =
            value.type.shape[index].unsignedLongLongValue;
        if (!extents[base + index] || extents[base + index] > UINT32_MAX)
            return NO;
    }
    *shape = {static_cast<std::uint32_t>(extents[0]),
              static_cast<std::uint32_t>(extents[1]),
              static_cast<std::uint32_t>(extents[2]),
              static_cast<std::uint32_t>(extents[3])};
    return YES;
}

/// Matches the matmul geometries whose whole H13 task streams are decoded
/// byte-for-byte from Apple oracles, so they encode as one whole-tensor
/// program with every row and column in place. The decoded stream depends on
/// both transpose flags and on whether the second operand is a runtime
/// surface, so all three are part of the key.
static BOOL matmulParityShape(ANEGraphValue *x, ANEGraphValue *result,
                              BOOL transposeX, BOOL transposeY,
                              BOOL runtimeWeight, BOOL preferNative,
                              ane::h13::MatmulShape *shape) {
    NSUInteger reduction = 0, rows = 0, columns = 0;
    if (!fp16Tensor(x) ||
        !matmulGeometry(x, result, transposeX, &reduction, &rows, &columns) ||
        rows > UINT32_MAX || reduction > UINT32_MAX || columns > UINT32_MAX)
        return NO;
    if (preferNative && rows == 1 && !runtimeWeight) return NO;
    const ane::h13::MatmulShape candidate{static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(reduction),
        static_cast<std::uint32_t>(columns), transposeX == YES,
        transposeY == YES, runtimeWeight == YES};
    if (!ane::h13::supportsMatmulParity(candidate)) return NO;
    if (shape) *shape = candidate;
    return YES;
}

/// True when the second matmul operand is a runtime surface of the shape the
/// transpose flag implies, with rank matching x and every batch axis at 1.
static BOOL runtimeMatmulOperand(ANEGraphValue *y, NSUInteger reduction,
                                 NSUInteger columns, BOOL transposeY,
                                 NSUInteger rank) {
    if (!fp16Tensor(y) || y.type.shape.count != rank || rank < 2) return NO;
    for (NSUInteger index = 0; index + 2 < rank; ++index)
        if (y.type.shape[index].unsignedIntegerValue != 1) return NO;
    const NSUInteger rows = transposeY ? columns : reduction;
    const NSUInteger width = transposeY ? reduction : columns;
    return y.type.shape[rank - 2].unsignedIntegerValue == rows &&
        y.type.shape[rank - 1].unsignedIntegerValue == width;
}

/// True when Apple's decoded corpus covers this geometry as one program, so
/// the planner must not slice its reduction, rows, or columns.
static BOOL matmulParityCovered(NSUInteger rows, NSUInteger reduction,
                                NSUInteger columns, BOOL transposeX,
                                BOOL transposeY, BOOL runtimeWeight, BOOL preferNative) {
    if (preferNative && rows == 1 && !runtimeWeight) return NO;
    if (rows > UINT32_MAX || reduction > UINT32_MAX || columns > UINT32_MAX)
        return NO;
    return ane::h13::supportsMatmulParity({static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(reduction),
        static_cast<std::uint32_t>(columns), transposeX == YES,
        transposeY == YES, runtimeWeight == YES});
}

static NSDictionary *binding(ANEGraphValue *value,
                             NSArray<NSNumber *> *logicalShape,
                             const ane::h13::TensorLayout &layout) {
    NSMutableArray *physical = [NSMutableArray arrayWithCapacity:6];
    for (std::uint64_t dimension : layout.nchw) [physical addObject:@(dimension)];
    NSUInteger elements = 1;
    for (NSNumber *dimension in logicalShape)
        elements *= dimension.unsignedIntegerValue;
    const NSUInteger elementSize = layout.elementSize == 1 ? 1 : 2;
    return @{@"name": value.name,
        @"dtype": elementSize == 1 ? @"bool" : @"float16",
        @"shape": logicalShape, @"logicalBytes": @(elements * elementSize),
        @"index": @(layout.index), @"nchw": physical,
        @"allocationBytes": @(layout.allocationBytes)};
}

static HWXObjectBinding *objectBinding(const ane::h13::TensorLayout &layout,
                                       HWXObjectBindingRole role,
                                       NSUInteger ordinal) {
    NSArray<NSNumber *> *shape = @[@(layout.nchw[0]), @(layout.nchw[1]),
        @(layout.nchw[2]), @(layout.nchw[3])];
    NSUInteger batchStride = (NSUInteger)(layout.nchw[1] * layout.nchw[4]);
    NSString *name = [NSString stringWithFormat:
        role == HWXObjectBindingRoleInput ? @"input%lu" : @"output%lu",
        (unsigned long)ordinal];
    // Apple's descriptor declares one batch element's span as the surface
    // size and spaces the surfaces by the whole allocation.
    HWXObjectBinding *binding = [[HWXObjectBinding alloc]
        initWithSymbol:name shortName:name role:role
        elementType:layout.elementSize == 1 ? ANEElementTypeBool
                                            : ANEElementTypeFP16
        shape:shape
        rowStrideBytes:(NSUInteger)layout.nchw[5]
        planeStrideBytes:(NSUInteger)layout.nchw[4]
        batchStrideBytes:batchStride storageByteLength:batchStride];
    binding.allocationByteLength = (NSUInteger)layout.allocationBytes;
    return binding;
}

static NSArray<NSNumber *> *relocationOffsets(const ane::h13::Program &program) {
    NSMutableArray<NSNumber *> *offsets =
        [NSMutableArray arrayWithCapacity:program.kernelRelocations.size()];
    for (std::size_t offset : program.kernelRelocations)
        [offsets addObject:@(offset)];
    return offsets;
}

static NSData *encodeHWX(const ane::h13::Program &program,
                         const std::vector<std::uint8_t> &anec,
                         NSError **error) {
    // The writer walks this array to assign surface addresses. Apple's matmul
    // objects place the output surface below both operands, a broadcast puts
    // it between them, and every other decoded object places the inputs first.
    NSMutableArray<HWXObjectBinding *> *bindings = [NSMutableArray array];
    HWXObjectBinding *output = objectBinding(program.output,
        HWXObjectBindingRoleOutput, 0);
    const std::size_t outputIndex =
        std::min(program.outputBindingIndex, program.inputs.size());
    for (NSUInteger index = 0; index < program.inputs.size(); ++index) {
        if (index == outputIndex) [bindings addObject:output];
        [bindings addObject:objectBinding(program.inputs[index],
            HWXObjectBindingRoleInput, index)];
    }
    if (outputIndex >= program.inputs.size()) [bindings addObject:output];
    NSData *task = [NSData dataWithBytes:anec.data() + ane::h13::anecHeaderBytes
                                  length:program.task.size()];
    NSData *constants = [NSData dataWithBytes:program.constants.data()
                                       length:program.constants.size()];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:program.taskCount
        firstTaskByteLength:program.firstTaskBytes recordCount:1
        formatCode:0 scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    info.scratchAllocationByteLength =
        static_cast<NSUInteger>(program.scratchAllocationBytes);
    return [HWXObjectWriter buildObjectForArchitecture:HWXObjectArchitectureH13
        taskDescriptor:task constantRegion:constants bindings:bindings
        kernelRelocationOffsets:relocationOffsets(program) programInfo:info
        error:error];
}

static void recordTensor(NSMutableDictionary<NSString *, NSDictionary *> *tensors,
                         ANEGraphValue *value, NSArray<NSNumber *> *shape,
                         NSString *role) {
    if (tensors[value.name]) return;
    NSUInteger elements = 1;
    for (NSNumber *dimension in shape) elements *= dimension.unsignedIntegerValue;
    // Schema evolution stays backward compatible: fp16 records keep their
    // exact prior shape; only bool surfaces add the dtype field.
    const BOOL boolElements =
        value.type.kind == ANEValueTypeKindTensor &&
        value.type.elementType == ANEElementTypeBool;
    tensors[value.name] = boolElements
        ? @{@"shape": shape, @"logicalBytes": @(elements),
            @"dtype": @"bool", @"role": role}
        : @{@"shape": shape, @"logicalBytes": @(elements * 2),
            @"role": role};
}

static void addSlice(NSMutableDictionary *record, ANEGraphValue *value,
                     NSUInteger offset, NSUInteger count, NSUInteger physical) {
    record[@"slice"] = @{@"tensor": value.name,
        @"elementOffset": @(offset), @"elementCount": @(count),
        @"physicalElements": @(physical)};
}

static BOOL constantValue(ANEGraphValue *value) {
    return [value.producer.operationName isEqualToString:@"const"];
}

static NSString *stringArgument(ANEGraphArgument *argument) {
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"string"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    return argument.kind == ANEGraphArgumentKindString ? argument.text : nil;
}

struct H13ParityPlan {
    BOOL unary;
    BOOL scalarConstant;
    BOOL constantBlob;
    ane::h13::UnaryOperation unaryOperation;
    ane::h13::BinaryOperation binaryOperation;
    uint16_t scalarBits;
    ane::h13::ElementwiseShape shape;
    NSUInteger elements;
    NSUInteger inputCount;
};

/// The decoded shapes a tensor may lower through: its literal NCHW geometry
/// when spatial, and its channel-flattened form, which shares the physical
/// 64-byte row layout the pipeline already packs logical tensors into.
static NSUInteger parityShapes(ANEGraphValue *value,
                               ane::h13::ElementwiseShape shapes[3]) {
    NSUInteger elements = 0;
    if (!tensorElementCount(value, &elements) || elements > UINT32_MAX) return 0;
    NSUInteger count = 0;
    ane::h13::ElementwiseShape literal{};
    if (elementwiseShape(value, &literal) &&
        (literal.height != 1 || literal.width != 1))
        shapes[count++] = literal;
    // The encoder's rank-3 spell: Apple normalizes [1, A, B] to the
    // [1, 1, A, B] surface (encoder-geometry oracles), which is neither the
    // literal rank-4 form nor the flattened channel form.
    if (value.type.kind == ANEValueTypeKindTensor &&
        value.type.elementType == ANEElementTypeFP16 &&
        value.type.shape.count == 3 &&
        value.type.shape[0].unsignedIntegerValue == 1)
        shapes[count++] = {1,
                           static_cast<std::uint32_t>(
                               value.type.shape[1].unsignedIntegerValue),
                           static_cast<std::uint32_t>(
                               value.type.shape[2].unsignedIntegerValue)};
    shapes[count++] = {static_cast<std::uint32_t>(elements), 1, 1};
    return count;
}

static BOOL nativeBinaryPlan(ANEGraphOperation *operation) {
    NSString *name = operation.operationName;
    BOOL multiply = [name isEqualToString:@"mul"];
    if (operation.arguments.count != 2 ||
        (!multiply && ![name isEqualToString:@"add"] &&
         ![name isEqualToString:@"maximum"] &&
         ![name isEqualToString:@"minimum"])) return NO;
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    if (!tensor(x, operation.results[0].type.shape)) return NO;
    return tensor(y, x.type.shape) ||
        (multiply && constantValue(y) &&
         y.type.kind == ANEValueTypeKindScalar &&
         y.type.elementType == ANEElementTypeFP16);
}

/// Matches the operations whose H13 task streams are decoded byte-for-byte
/// from Apple oracles, so they encode as one whole-tensor program.
static BOOL parityPlan(ANEGraphOperation *operation,
                       NSDictionary<NSString *, NSData *> *synthesizedConstants,
                       BOOL preferNative, H13ParityPlan *plan) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    NSString *name = operation.operationName;
    H13ParityPlan candidate{};
    ane::h13::ElementwiseShape shapes[2];
    NSUInteger shapeCount = parityShapes(operation.results[0], shapes);
    if (!shapeCount || !tensor(x, operation.results[0].type.shape)) return NO;
    if (preferNative && nativeBinaryPlan(operation) &&
        shapes[shapeCount - 1].channels <= 64) return NO;
    BOOL leaky = [name isEqualToString:@"leaky_relu"];
    BOOL gelu = [name isEqualToString:@"gelu"];
    BOOL rsqrt = [name isEqualToString:@"rsqrt"];
    if (unaryEncoding(name, &candidate.unaryOperation)) {
        uint16_t alphaBits = 0;
        double alpha = 0.0;
        float epsilon = 0.0f;
        if (operation.arguments.count != (leaky || gelu || rsqrt ? 2u : 1u) ||
            (leaky && (!exactFP16Attribute(operation.arguments[@"alpha"],
                                           &alphaBits, &alpha) ||
                       alphaBits != 0x3000)) ||
            (gelu && ![stringArgument(operation.arguments[@"mode"])
                          isEqualToString:@"EXACT"]) ||
            (rsqrt && (!fp32Attribute(operation.arguments[@"epsilon"], &epsilon) ||
                       epsilon != 1e-6f)))
            return NO;
        for (NSUInteger index = 0; index < shapeCount; ++index) {
            if (!ane::h13::supportsElementwise(candidate.unaryOperation, shapes[index]))
                continue;
            candidate.shape = shapes[index];
            candidate.elements = (NSUInteger)shapes[index].channels *
                shapes[index].height * shapes[index].width;
            candidate.unary = YES;
            candidate.inputCount = 1;
            *plan = candidate;
            return YES;
        }
        return NO;
    }
    if (!binaryEncoding(name, &candidate.binaryOperation) ||
        operation.arguments.count != 2 || !y) return NO;
    if (synthesizedConstants[x.name] || constantValue(x)) return NO;
    BOOL runtime = !synthesizedConstants[y.name] && !constantValue(y);
    if (runtime) {
        if (!tensor(y, operation.results[0].type.shape)) return NO;
    } else if (y.type.kind == ANEValueTypeKindTensor &&
               tensor(y, operation.results[0].type.shape)) {
        // The decoded constant-blob twin: the whole constant rides the
        // program's constant section, so no runtime surface binds it.
        candidate.constantBlob = YES;
    } else if (synthesizedConstants[y.name] ||
               y.type.kind != ANEValueTypeKindScalar ||
               y.producer.arguments.count ||
               !fp16Scalar(y.producer.attributes[@"val"], &candidate.scalarBits) ||
               candidate.scalarBits != 0x3800) {
        return NO;
    }
    for (NSUInteger index = 0; index < shapeCount; ++index) {
        if (candidate.constantBlob
                ? !ane::h13::supportsElementwiseConstant(candidate.binaryOperation,
                                                         shapes[index])
                : !ane::h13::supportsElementwise(candidate.binaryOperation,
                                                 shapes[index], !runtime))
            continue;
        candidate.shape = shapes[index];
        candidate.elements = (NSUInteger)shapes[index].channels *
            shapes[index].height * shapes[index].width;
        candidate.scalarConstant = !runtime && !candidate.constantBlob;
        candidate.inputCount = runtime ? 2 : 1;
        *plan = candidate;
        return YES;
    }
    return NO;
}

struct H13BroadcastPlan {
    ane::h13::BinaryOperation operation;
    ane::h13::BroadcastOperand operand;
    ane::h13::BroadcastShape shape;
    ane::h13::BatchedShape result;
    uint16_t scalarBits;
    NSUInteger inputCount;
};

static BOOL sameBatchedShape(ane::h13::BatchedShape left,
                             ane::h13::BatchedShape right) {
    return left.batch == right.batch && left.channels == right.channels &&
        left.height == right.height && left.width == right.width;
}

/// Matches the broadcast forms whose whole H13 task streams are decoded from
/// Apple oracles: an NCHW runtime tensor against a second runtime tensor of
/// any broadcastable NCHW shape, an inline fp16 scalar, or a per-channel
/// constant Apple folds into the constant section's bias and scale blocks.
static BOOL broadcastPlan(ANEGraphOperation *operation,
                          NSDictionary<NSString *, NSData *> *synthesizedConstants,
                          BOOL preferNative, H13BroadcastPlan *plan,
                          ANEGraphValue *__autoreleasing *constantOut) {
    if (preferNative && nativeBinaryPlan(operation)) {
        // Native keeps every shape it already served. Where the decoded
        // table covers the exact geometry as one whole-tensor program, the
        // broadcast beats the thousands-of-programs 64-lane split, so fall
        // through only for covered shapes.
        ANEGraphValue *x = operation.operands[@"x"].value;
        ANEGraphValue *y = operation.operands[@"y"].value;
        H13BroadcastPlan probe{};
        if (!x || !y || !batchedShape(x, &probe.shape.x) ||
            !batchedShape(operation.results[0], &probe.result) ||
            !batchedShape(y, &probe.shape.y) ||
            probe.shape.y.batch != probe.shape.x.batch ||
            !binaryEncoding(operation.operationName, &probe.operation) ||
            !ane::h13::supportsBroadcast(probe.operation,
                                         ane::h13::BroadcastOperand::Runtime,
                                         probe.shape))
            return NO;
    }
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    H13BroadcastPlan candidate{};
    if (!binaryEncoding(operation.operationName, &candidate.operation) ||
        operation.arguments.count != 2 || !x || !y) return NO;
    if (synthesizedConstants[x.name] || constantValue(x) ||
        synthesizedConstants[y.name]) return NO;
    if (!batchedShape(x, &candidate.shape.x) ||
        !batchedShape(operation.results[0], &candidate.result)) return NO;
    ANEGraphValue *constant = nil;
    if (constantValue(y)) {
        candidate.inputCount = 1;
        if (y.producer.arguments.count) return NO;
        if (y.type.kind == ANEValueTypeKindScalar &&
            y.type.elementType == ANEElementTypeFP16) {
            if (!fp16Scalar(y.producer.attributes[@"val"], &candidate.scalarBits))
                return NO;
            candidate.operand = ane::h13::BroadcastOperand::Scalar;
        } else if (batchedShape(y, &candidate.shape.y) &&
                   candidate.shape.y.batch == 1 &&
                   (candidate.shape.y.channels == 1) +
                           (candidate.shape.y.height == 1) +
                           (candidate.shape.y.width == 1) ==
                       2) {
            // A per-channel vector aligned to one broadcastable axis (C for
            // the classic form, H or W for the rank-3 encoder forms); the
            // decoded table decides which alignments Apple lowers.
            candidate.operand = ane::h13::BroadcastOperand::Constant;
            constant = y;
        } else {
            return NO;
        }
    } else if (batchedShape(y, &candidate.shape.y)) {
        candidate.operand = ane::h13::BroadcastOperand::Runtime;
        candidate.inputCount = 2;
    } else {
        return NO;
    }
    ane::h13::BatchedShape broadcast = candidate.shape.x;
    if (candidate.operand == ane::h13::BroadcastOperand::Runtime)
        broadcast = {std::max(candidate.shape.x.batch, candidate.shape.y.batch),
                     std::max(candidate.shape.x.channels, candidate.shape.y.channels),
                     std::max(candidate.shape.x.height, candidate.shape.y.height),
                     std::max(candidate.shape.x.width, candidate.shape.y.width)};
    if (!sameBatchedShape(broadcast, candidate.result)) return NO;
    if (!ane::h13::supportsBroadcast(candidate.operation, candidate.operand,
                                     candidate.shape)) return NO;
    *plan = candidate;
    *constantOut = constant;
    return YES;
}

/// The fp16 values of a per-channel `[1, C, 1, 1]` constant operand, from an
/// inline typed list or a BLOBFILE payload.
static NSData *perChannelConstantData(ANEGraphValue *value, NSUInteger channels,
                                      NSURL *modelRoot,
                                      ANEDiagnosticEngine *diagnostics,
                                      NSMutableDictionary<NSString *, NSData *> *resolved) {
    NSData *data = resolved[value.name];
    if (data) return data;
    ANEGraphOperation *producer = value.producer;
    ANEGraphArgument *literal = producer.attributes[@"val"];
    if (literal.kind != ANEGraphArgumentKindCall ||
        ![literal.calleeValueType isEqualToValueType:value.type] ||
        literal.callArguments.count != 1) {
        reject(diagnostics,
            @"H13 per-channel constants require a matching typed inline list or BLOBFILE payload",
            producer, @"h13.invalid-constant-payload");
        return nil;
    }
    ANEGraphArgument *payload = literal.callArguments[0].value;
    if (payload.kind == ANEGraphArgumentKindList &&
        payload.elements.count == channels) {
        NSMutableData *dense = [NSMutableData dataWithLength:channels * 2];
        uint16_t *words = static_cast<uint16_t *>(dense.mutableBytes);
        for (NSUInteger index = 0; index < channels; ++index)
            if (!fp16Scalar(payload.elements[index], &words[index])) {
                reject(diagnostics,
                    @"H13 per-channel constants require finite fp16 elements",
                    producer, @"h13.invalid-constant-payload");
                return nil;
            }
        data = dense;
    } else if (payload.kind == ANEGraphArgumentKindCall &&
               [payload.calleeName isEqualToString:@"BLOBFILE"]) {
        data = [ANEBlobResolver loadConstantForOperation:producer
            expectedBytes:channels * 2 modelRoot:modelRoot diagnostics:diagnostics];
    }
    if (!data) {
        if (!diagnostics.errorCount)
            reject(diagnostics,
                @"H13 per-channel constants require C finite fp16 elements",
                producer, @"h13.invalid-constant-payload");
        return nil;
    }
    resolved[value.name] = data;
    return data;
}

struct H13NormPlan {
    ane::h13::NormOperation operation;
    ane::h13::NormShape shape;
    NSUInteger inputElements;
    NSUInteger outputElements;
};

/// Matches the softmax, layer_norm, and reduction programs whose whole H13
/// task streams are decoded from Apple oracles. `epsilon` must be the decoded
/// 1e-5 and layer_norm must carry no gamma or beta: Apple's compiler in this
/// harness rejects every affine form, so no oracle covers one.
static BOOL normParityPlan(ANEGraphOperation *operation, H13NormPlan *plan) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    NSString *name = operation.operationName;
    H13NormPlan candidate{};
    if (!normEncoding(name, &candidate.operation) || !x) return NO;
    const BOOL softmax = candidate.operation == ane::h13::NormOperation::Softmax;
    const BOOL layerNorm = candidate.operation == ane::h13::NormOperation::LayerNorm;
    NSInteger inputShift = 0;
    if (!fp16Tensor(x) || !fp16Tensor(operation.results[0]) ||
        !normSurface(x.type.shape, &candidate.shape.input, &inputShift) ||
        !normSurface(operation.results[0].type.shape, &candidate.shape.output, nullptr))
        return NO;
    if (!constantAxisMask(operation.operands[softmax ? @"axis" : @"axes"].value,
                          x.type.shape.count, inputShift,
                          &candidate.shape.axisMask))
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
    candidate.shape.keepDims = softmax || layerNorm ||
        boolean(operation.arguments[@"keep_dims"], YES);
    if (!candidate.shape.keepDims &&
        !boolean(operation.arguments[@"keep_dims"], NO)) return NO;
    if (!tensorElementCount(x, &candidate.inputElements) ||
        !tensorElementCount(operation.results[0], &candidate.outputElements))
        return NO;
    if (!ane::h13::supportsNormParity(candidate.operation, candidate.shape))
        return NO;
    *plan = candidate;
    return YES;
}

struct H13ConvPlan {
    ane::h13::ConvShape shape;
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
    if (payload.kind == ANEGraphArgumentKindList) {
        if (payload.elements.count != count) return NO;
        for (NSUInteger index = 0; index < count; ++index)
            if (!int32Literal(payload.elements[index], &values[index])) return NO;
        return YES;
    }
    // The rank-3 conv spell writes one-element vectors as bare scalars:
    // `tensor<int32, [1]>(1)` carries an int literal, not a list.
    if (count == 1 && int32Literal(payload, &values[0])) return YES;
    return NO;
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
static BOOL convSurface(ANEGraphValue *value, ane::h13::ElementwiseShape *shape) {
    return elementwiseShape(value, shape);
}

/// Matches the convolutions whose whole H13 task stream and constant section
/// are decoded from Apple oracles.
///
/// coremltools 9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4 defines `conv` with a
/// `[Cout, Cin / groups, kh, kw]` weight, square `strides` and `dilations`
/// vectors, and either an explicit `pad` with `pad_type="custom"` or the
/// `same`, `same_lower` and `valid` spellings. The decoded corpus covers
/// unit dilations, `same` and `valid` with zero explicit padding,
/// rectangular kernels, the encoder's rank-4 custom-pad padconv (a declared
/// asymmetric pad lowers when a template row covers the surface pair), and
/// the encoder's rank-3 spell: a `[1, C, T]` input with a
/// `[Cout, Cin / groups, k]` weight binds as the W-major `[1, C, 1, T]`
/// surface every decoded rank-3 program records, so the respell inserts the
/// unit axis before T and moves the kernel extent to W; the rank-3 custom
/// spelling still lowers only where its pads equal a `same`/`valid` spell.
/// The H-major respell has no surface the rank-3 tensor binds as and never
/// matches.
static BOOL convParityPlan(ANEGraphOperation *operation, H13ConvPlan *plan) {
    if (![operation.operationName isEqualToString:@"conv"]) return NO;
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *weight = operation.operands[@"weight"].value;
    ANEGraphValue *bias = operation.operands[@"bias"].value;
    H13ConvPlan candidate{};
    candidate.weight = weight;
    candidate.bias = bias;
    if (!x || !weight || !fp16Tensor(weight)) return NO;
    const BOOL rank3 = x.type.shape.count == 3;
    if (rank3) {
        // The respell onto the W-major surface the rank-3 tensor binds as:
        // [1, C, T] becomes the [1, C, 1, T] surface and the kernel extent
        // moves to W, so the weight spell [Cout, Cin / groups, k] reads as
        // kh = 1, kw = k.
        if (weight.type.shape.count != 3 ||
            operation.results[0].type.shape.count != 3 ||
            !fp16Tensor(x) || !fp16Tensor(operation.results[0]) ||
            x.type.shape[0].unsignedIntegerValue != 1 ||
            operation.results[0].type.shape[0].unsignedIntegerValue != 1)
            return NO;
        candidate.shape.input = {x.type.shape[1].unsignedIntValue, 1,
                                 x.type.shape[2].unsignedIntValue};
        candidate.shape.output = {
            operation.results[0].type.shape[1].unsignedIntValue, 1,
            operation.results[0].type.shape[2].unsignedIntValue};
        if (!candidate.shape.input.channels || !candidate.shape.input.width ||
            !candidate.shape.output.channels ||
            !candidate.shape.output.width) {
        return NO;
    }
    } else if (x.type.shape.count == 4) {
        if (weight.type.shape.count != 4 ||
            operation.results[0].type.shape.count != 4 ||
            !convSurface(x, &candidate.shape.input) ||
            !convSurface(operation.results[0], &candidate.shape.output))
            return NO;
    } else return NO;
    const NSUInteger strideCount = rank3 ? 1 : 2;
    const NSUInteger padCount = rank3 ? 2 : 4;
    if (operation.arguments.count != (bias ? 8u : 7u)) return NO;
    NSString *padType = nil;
    long long strides[2] = {0, 0}, dilations[2] = {0, 0}, padding[4] = {0, 0, 0, 0};
    long long groups = 1;
    if (!constantString(operation.operands[@"pad_type"].value, &padType) ||
        !int32Vector(operation.operands[@"strides"].value, strideCount, strides) ||
        !int32Vector(operation.operands[@"dilations"].value, strideCount, dilations) ||
        !int32Vector(operation.operands[@"pad"].value, padCount, padding) ||
        !int32Scalar(operation.operands[@"groups"].value, &groups)) return NO;
    const std::uint32_t kernelExtent = weight.type.shape[2].unsignedIntValue;
    const std::uint32_t kernel = rank3 ? 1 : kernelExtent;
    const std::uint32_t kernelWidth = rank3 ? kernelExtent
        : weight.type.shape[3].unsignedIntValue;
    if (!kernel || !kernelWidth) return NO;
    BOOL normalizedCustom = NO;
    BOOL declaredPads = NO;
    if (![padType isEqualToString:@"same"] && ![padType isEqualToString:@"valid"]) {
        if (rank3) {
            // The rank-3 conv spelling refuses on Apple's own tool, so the
            // decoder carries only the same/valid respells. Accept the
            // encoder's rank-3 custom spelling when its pads are exactly
            // what one of those two spells: zeros for `valid`, or the
            // symmetric (kernel - 1) / 2 of a unit-stride odd kernel for
            // `same`.
            if (strides[0] != 1 || kernelExtent % 2 == 0) {
                return NO;
            }
            const long long half = (kernelExtent - 1) / 2;
            if (padding[0] == 0 && padding[1] == 0) {
                padType = @"valid";
            } else if (padding[0] == half && padding[1] == half) {
                padType = @"same";
            } else {
                return NO;
            }
            normalizedCustom = YES;
        } else {
            // Rank 4 carries a decoded custom-pad form: the encoder's
            // W-padded rel-pos padconv (pad [0, 0, 1, 0], k1x1 grouped) is
            // captured as a two-task program that reads the unpadded input
            // directly
            // (encoder_conv_pad_c8_n8_k1x1_s1_g8_bias0_p0010_f4). The
            // declared pads only have to reproduce the declared output;
            // the template key pins the exact surfaces.
            declaredPads = YES;
        }
    }
    if (rank3) {
        // One spatial axis: both respelled axes run the same stride and
        // unit dilation.
        strides[1] = strides[0];
        dilations[1] = dilations[0];
    }
    if (strides[0] != strides[1] || dilations[0] != 1 || dilations[1] != 1 ||
        strides[0] < 1 || groups < 1) return NO;
    if (!normalizedCustom && !declaredPads)
        for (NSUInteger index = 0; index < padCount; ++index)
            if (padding[index]) return NO;
    if (weight.type.shape[0].unsignedIntValue != candidate.shape.output.channels)
        return NO;
    if (!candidate.shape.input.channels ||
        candidate.shape.input.channels % groups ||
        weight.type.shape[1].unsignedIntValue !=
            candidate.shape.input.channels / groups) return NO;
    // The result surface must be what the pad type asks for: `same` covers
    // ceil(D / stride), `valid` covers floor((D - k_axis) / stride) + 1,
    // and a declared custom pad covers
    // floor((D + pad_lo + pad_hi - k_axis) / stride) + 1, each axis against
    // its own kernel extent.
    const std::uint32_t extents[2] = {candidate.shape.input.height,
                                      candidate.shape.input.width};
    const std::uint32_t results[2] = {candidate.shape.output.height,
                                      candidate.shape.output.width};
    const std::uint32_t kernels[2] = {kernel, kernelWidth};
    for (NSUInteger axis = 0; axis < 2; ++axis) {
        const std::uint32_t stride = static_cast<std::uint32_t>(strides[0]);
        if ([padType isEqualToString:@"same"]) {
            if (results[axis] != (extents[axis] + stride - 1) / stride) return NO;
        } else if (declaredPads) {
            const std::uint32_t low = static_cast<std::uint32_t>(
                axis == 0 ? padding[0] : padding[2]);
            const std::uint32_t high = static_cast<std::uint32_t>(
                axis == 0 ? padding[1] : padding[3]);
            if (low > extents[axis] || high > extents[axis])
                return NO;
            const std::uint32_t padded = extents[axis] + low + high;
            if (padded < kernels[axis] ||
                results[axis] != (padded - kernels[axis]) / stride + 1)
                return NO;
        } else {
            if (extents[axis] < kernels[axis] ||
                results[axis] != (extents[axis] - kernels[axis]) / stride + 1)
                return NO;
        }
    }
    candidate.shape.kernel = kernel;
    candidate.shape.kernelWidth = kernelWidth;
    candidate.shape.stride = static_cast<std::uint32_t>(strides[0]);
    candidate.shape.groups = static_cast<std::uint32_t>(groups);
    candidate.shape.bias = bias != nil;
    if (bias && (!fp16Tensor(bias) || bias.type.shape.count != 1 ||
                 bias.type.shape[0].unsignedIntValue !=
                     candidate.shape.output.channels)) return NO;
    if (!ane::h13::supportsConvParity(candidate.shape)) {
        return NO;
    }
    *plan = candidate;
    return YES;
}

static NSData *linearBiasData(ANEGraphValue *bias, NSUInteger columns,
                                NSURL *modelRoot,
                                ANEDiagnosticEngine *diagnostics,
                                NSMutableDictionary<NSString *, NSData *> *resolved) {
    if (!tensor(bias, @[@(columns)])) {
        reject(diagnostics,
            @"H13 linear bias must be a constant fp16 vector of N elements",
            bias.producer, @"h13.invalid-linear-bias");
        return nil;
    }
    ANEGraphOperation *producer = bias.producer;
    ANEGraphArgument *literal = producer.attributes[@"val"];
    if (producer.arguments.count || literal.kind != ANEGraphArgumentKindCall ||
        ![literal.calleeValueType isEqualToValueType:bias.type] ||
        literal.callArguments.count != 1) {
        reject(diagnostics,
            @"H13 linear bias requires a typed inline list or BLOBFILE payload",
            producer, @"h13.invalid-linear-bias");
        return nil;
    }
    ANEGraphArgument *payload = literal.callArguments[0].value;
    NSData *data = resolved[bias.name];
    if (!data && payload.kind == ANEGraphArgumentKindList &&
        payload.elements.count == columns) {
        NSMutableData *dense = [NSMutableData dataWithLength:columns * 2];
        uint16_t *words = static_cast<uint16_t *>(dense.mutableBytes);
        for (NSUInteger index = 0; index < columns; ++index) {
            if (!fp16Scalar(payload.elements[index], &words[index])) {
                reject(diagnostics, @"H13 linear bias requires finite fp16 elements",
                    producer, @"h13.invalid-linear-bias");
                return nil;
            }
        }
        data = dense;
    } else if (!data && payload.kind == ANEGraphArgumentKindCall &&
               [payload.calleeName isEqualToString:@"BLOBFILE"]) {
        data = [ANEBlobResolver loadConstantForOperation:producer
            expectedBytes:columns * 2 modelRoot:modelRoot diagnostics:diagnostics];
    }
    if (!data) {
        if (!diagnostics.errorCount)
            reject(diagnostics, @"H13 linear bias requires N finite fp16 elements",
                producer, @"h13.invalid-linear-bias");
        return nil;
    }
    resolved[bias.name] = data;
    return data;
}

/// Parses a batched runtime-or-constant matmul: x of rank 3 [B,rows,K] or
/// rank 4 [1,B,rows,K] against a same-rank y of [.., B, K, N] with both
/// transpose flags false, exactly as the decoded batched corpus carries.
/// Returns NotBatched when the operands do not form a batched pair, so the
/// existing single-batch paths (and their rejections) still apply.
enum H13BatchedParse { H13BatchedParseNo, H13BatchedParseYes, H13BatchedParseShape };

static H13BatchedParse batchedMatmulParse(ANEGraphOperation *operation,
                                          ane::h13::BatchedMatmulShape *shape) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    if (operation.arguments.count != 4 || !x || !y ||
        !fp16Tensor(x) || !fp16Tensor(y) || x.type.shape.count < 3 ||
        x.type.shape.count > 4 || y.type.shape.count != x.type.shape.count)
        return H13BatchedParseNo;
    const BOOL transposeX =
        boolean(operation.arguments[@"transpose_x"], YES);
    const BOOL transposeY =
        boolean(operation.arguments[@"transpose_y"], YES);
    if ((transposeX && !boolean(operation.arguments[@"transpose_x"], YES)) ||
        (transposeY && !boolean(operation.arguments[@"transpose_y"], YES)))
        return H13BatchedParseNo;
    NSArray<NSNumber *> *xs = x.type.shape, *ys = y.type.shape,
        *os = operation.results[0].type.shape;
    if (xs.count == 4 && xs[0].unsignedIntegerValue != 1) return H13BatchedParseNo;
    const NSUInteger rows = transposeX
        ? xs[xs.count - 1].unsignedIntegerValue
        : xs[xs.count - 2].unsignedIntegerValue;
    const NSUInteger reduction = transposeX
        ? xs[xs.count - 2].unsignedIntegerValue
        : xs[xs.count - 1].unsignedIntegerValue;
    const NSUInteger columns = transposeY
        ? ys[ys.count - 2].unsignedIntegerValue
        : ys[ys.count - 1].unsignedIntegerValue;
    NSUInteger batch = 1;
    for (NSUInteger index = 0; index + 2 < xs.count; ++index)
        batch *= xs[index].unsignedIntegerValue;
    if (!rows || !reduction || !columns || batch <= 1) return H13BatchedParseNo;
    NSUInteger yBatch = 1;
    for (NSUInteger index = 0; index + 2 < ys.count; ++index)
        yBatch *= ys[index].unsignedIntegerValue;
    NSUInteger oBatch = 1;
    for (NSUInteger index = 0; os.count == xs.count && index + 2 < os.count; ++index)
        oBatch *= os[index].unsignedIntegerValue;
    const NSUInteger yReduction = transposeY
        ? ys[ys.count - 1].unsignedIntegerValue
        : ys[ys.count - 2].unsignedIntegerValue;
    if (yBatch != batch || oBatch != batch || yReduction != reduction ||
        os.count != xs.count ||
        os[os.count - 2].unsignedIntegerValue != rows ||
        os[os.count - 1].unsignedIntegerValue != columns)
        return H13BatchedParseNo;
    shape->batch = static_cast<std::uint32_t>(batch);
    shape->rows = static_cast<std::uint32_t>(rows);
    shape->reduction = static_cast<std::uint32_t>(reduction);
    shape->columns = static_cast<std::uint32_t>(columns);
    shape->transposeX = transposeX;
    shape->transposeY = transposeY;
    shape->runtimeWeight = !constantValue(y);
    return H13BatchedParseYes;
}

/// True when the constant producer's payload is a BLOBFILE reference, so a
/// byte-exact load cannot fail with a diagnostic.
static BOOL blobBackedConstant(ANEGraphValue *value) {
    if (!value || !constantValue(value)) return NO;
    ANEGraphArgument *literal = value.producer.attributes[@"val"];
    if (literal.kind != ANEGraphArgumentKindCall ||
        literal.callArguments.count != 1) return NO;
    ANEGraphArgument *payload = literal.callArguments[0].value;
    return payload.kind == ANEGraphArgumentKindCall &&
        [payload.calleeName isEqualToString:@"BLOBFILE"];
}

/// The only operation consuming `value`, or nil when the value has zero or
/// several consumers.
static ANEGraphOperation *singleConsumerOf(
    NSArray<ANEGraphOperation *> *operations, ANEGraphValue *value) {
    ANEGraphOperation *consumer = nil;
    for (ANEGraphOperation *operation in operations) {
        if (!operationUsesValue(operation, value)) continue;
        if (consumer) return nil;
        consumer = operation;
    }
    return consumer;
}

/// Matches the captured five-operation FFN chain — matmul → bias add → silu
/// → matmul → bias add at the d1024 s375 geometry with constant weights and
/// biases, each intermediate feeding exactly one consumer — and returns the
/// whole-chain operation to emit in its place, or nil.
static ANEGraphOperation *ffnChainOperation(
    NSArray<ANEGraphOperation *> *operations, ANEGraphOperation *first,
    ANESourceRange range) {
    if (![first.operationName isEqualToString:@"matmul"]) return nil;
    ANEGraphValue *x = first.operands[@"x"].value;
    ANEGraphValue *weight1 = first.operands[@"y"].value;
    if (!fp16Tensor(x) || x.type.shape.count != 3 ||
        !boolean(first.arguments[@"transpose_x"], NO) ||
        !boolean(first.arguments[@"transpose_y"], YES) ||
        !weight1 || !constantValue(weight1) || weight1.type.shape.count != 2)
        return nil;
    const NSUInteger inner = x.type.shape[2].unsignedIntegerValue;
    const NSUInteger middle = weight1.type.shape[0].unsignedIntegerValue;
    if (weight1.type.shape[1].unsignedIntegerValue != inner) return nil;
    ANEGraphValue *product1 = first.results[0];
    if (!fp16Tensor(product1) || product1.type.shape.count != 3 ||
        product1.type.shape[2].unsignedIntegerValue != middle) return nil;
    ANEGraphOperation *add1 = singleConsumerOf(operations, product1);
    if (!add1 || ![add1.operationName isEqualToString:@"add"] ||
        add1.operands[@"x"].value != product1) return nil;
    ANEGraphValue *bias1 = add1.operands[@"y"].value;
    if (!bias1 || !constantValue(bias1) || bias1.type.shape.count != 1 ||
        bias1.type.shape[0].unsignedIntegerValue != middle) return nil;
    ANEGraphValue *sum1 = add1.results[0];
    ANEGraphOperation *silu = singleConsumerOf(operations, sum1);
    if (!silu || ![silu.operationName isEqualToString:@"silu"] ||
        silu.operands[@"x"].value != sum1) return nil;
    ANEGraphValue *activated = silu.results[0];
    ANEGraphOperation *second = singleConsumerOf(operations, activated);
    if (!second || ![second.operationName isEqualToString:@"matmul"] ||
        second.operands[@"x"].value != activated ||
        !boolean(second.arguments[@"transpose_x"], NO) ||
        !boolean(second.arguments[@"transpose_y"], YES)) return nil;
    ANEGraphValue *weight2 = second.operands[@"y"].value;
    if (!weight2 || !constantValue(weight2) || weight2.type.shape.count != 2 ||
        weight2.type.shape[1].unsignedIntegerValue != middle)
        return nil;
    const NSUInteger columns =
        weight2.type.shape[0].unsignedIntegerValue;
    ANEGraphValue *product2 = second.results[0];
    if (!fp16Tensor(product2) || product2.type.shape.count != 3 ||
        product2.type.shape[1].unsignedIntegerValue !=
            x.type.shape[1].unsignedIntegerValue ||
        product2.type.shape[2].unsignedIntegerValue != columns) return nil;
    ANEGraphOperation *add2 = singleConsumerOf(operations, product2);
    if (!add2 || ![add2.operationName isEqualToString:@"add"] ||
        add2.operands[@"x"].value != product2) return nil;
    ANEGraphValue *bias2 = add2.operands[@"y"].value;
    if (!bias2 || !constantValue(bias2) || bias2.type.shape.count != 1 ||
        bias2.type.shape[0].unsignedIntegerValue != columns) return nil;
    if (!ane::h13::supportsFFNChain(
            static_cast<std::uint32_t>(x.type.shape[1].unsignedIntegerValue),
            static_cast<std::uint32_t>(inner),
            static_cast<std::uint32_t>(middle),
            static_cast<std::uint32_t>(middle),
            static_cast<std::uint32_t>(columns))) return nil;
    // The chain owns the final result: a fresh value under the same name,
    // because the skipped add still defines the original.
    ANEGraphValue *result = [[ANEGraphValue alloc]
        initWithName:add2.results[0].name type:add2.results[0].type];
    return [[ANEGraphOperation alloc] initWithOperationName:@"ffn-chain"
        results:@[result]
        arguments:@{@"x": valueArgument(x, range),
                    @"weight1": valueArgument(weight1, range),
                    @"bias1": valueArgument(bias1, range),
                    @"weight2": valueArgument(weight2, range),
                    @"bias2": valueArgument(bias2, range)}
        attributes:@{} range:range];
}

static BOOL lowerOperation(ANEGraphOperation *operation, NSURL *modelRoot,
                           ANEDiagnosticEngine *diagnostics, BOOL preferNative,
                           NSDictionary<NSString *, NSData *> *synthesizedConstants,
                           NSMutableDictionary<NSString *, NSData *> *resolvedConstants,
                           NSUInteger elementOffset, NSUInteger inputElementCount,
                           NSUInteger outputElementOffset, ane::h13::Program &program,
                           NSArray<ANEGraphValue *> *__autoreleasing *inputsOut,
                           ANEGraphValue *__autoreleasing *constantInputOut,
                           NSData *__autoreleasing *constantDataOut,
                           NSString *__autoreleasing *manifestOperationOut) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    NSString *name = operation.operationName;
    NSArray<NSString *> *binaryNames =
        @[@"add", @"mul", @"maximum", @"minimum", @"sub", @"real_div"];
    NSUInteger binaryIndex = [binaryNames indexOfObject:name];
    NSArray<ANEGraphValue *> *inputs = nil;
    ANEGraphValue *constantInput = nil;
    NSData *constantData = nil;
    NSString *manifestOperation = name;

    H13BroadcastPlan broadcast{};
    ANEGraphValue *broadcastConstant = nil;
    if (broadcastPlan(operation, synthesizedConstants, preferNative, &broadcast,
                      &broadcastConstant)) {
        NSData *values = nil;
        if (broadcast.operand == ane::h13::BroadcastOperand::Constant) {
            values = perChannelConstantData(broadcastConstant,
                (NSUInteger)broadcast.shape.y.channels *
                    broadcast.shape.y.height * broadcast.shape.y.width,
                modelRoot, diagnostics, resolvedConstants);
            if (!values) return NO;
        }
        program = ane::h13::encodeBroadcast(broadcast.operation,
            broadcast.operand, broadcast.shape,
            static_cast<const std::uint8_t *>(values.bytes), values.length,
            broadcast.scalarBits);
        *inputsOut = broadcast.inputCount == 2 ? @[x, y] : @[x];
        *constantInputOut = nil;
        *constantDataOut = nil;
        *manifestOperationOut = name;
        return YES;
    }


    H13ParityPlan plan{};
    if (parityPlan(operation, synthesizedConstants, preferNative, &plan)) {
        if (plan.constantBlob) {
            NSData *whole = resolvedConstants[y.name];
            if (!whole) {
                whole = synthesizedConstants[y.name];
                if (!whole) {
                    whole = [ANEBlobResolver loadConstantForOperation:y.producer
                        expectedBytes:plan.elements * 2 modelRoot:modelRoot
                        diagnostics:diagnostics];
                    if (!whole) return NO;
                }
                resolvedConstants[y.name] = whole;
            }
            if (whole.length != plan.elements * 2)
                return reject(diagnostics,
                    @"H13 constant-blob payload has the wrong size",
                    operation, @"h13.invalid-constant-payload");
            program = ane::h13::encodeElementwiseConstant(plan.binaryOperation,
                plan.shape, static_cast<const std::uint8_t *>(whole.bytes),
                whole.length);
            *inputsOut = @[x];
            *constantInputOut = nil;
            *constantDataOut = nil;
            *manifestOperationOut = name;
            return YES;
        }
        program = plan.unary
            ? ane::h13::encodeElementwise(plan.unaryOperation, plan.shape)
            : ane::h13::encodeElementwise(plan.binaryOperation, plan.shape,
                                          plan.scalarConstant, plan.scalarBits);
        *inputsOut = plan.inputCount == 2 ? @[x, y] : @[x];
        *constantInputOut = nil;
        *constantDataOut = nil;
        *manifestOperationOut = name;
        return YES;
    }

    if ([name isEqualToString:@"transpose"]) {
        ane::h13::ElementwiseShape transposeIn{}, transposeOut{};
        if (!transposeParityShapes(operation, x, operation.results[0],
                                   &transposeIn, &transposeOut))
            return reject(diagnostics,
                @"H13 transpose has no decoded 1-task program for this surface pair",
                operation, @"h13.nonfoldable-transpose");
        program = ane::h13::encodeTransposeParity(transposeIn, transposeOut);
        *inputsOut = @[x];
        *constantInputOut = nil;
        *constantDataOut = nil;
        *manifestOperationOut = name;
        return YES;
    }

    if ([name isEqualToString:@"slice_by_index"]) {
        ane::h13::ElementwiseShape sliceIn{}, sliceOut{};
        if (!sliceParityShapes(operation, x, operation.results[0],
                               &sliceIn, &sliceOut))
            return reject(diagnostics,
                @"H13 slice_by_index has no decoded 1-task program for this surface pair",
                operation, @"h13.noncontiguous-slice");
        program = ane::h13::encodeSliceParity(sliceIn, sliceOut);
        *inputsOut = @[x];
        *constantInputOut = nil;
        *constantDataOut = nil;
        *manifestOperationOut = name;
        return YES;
    }

    H13NormPlan normalization{};
    if (normParityPlan(operation, &normalization)) {
        program = ane::h13::encodeNormParity(normalization.operation,
                                             normalization.shape);
        *inputsOut = @[x];
        *constantInputOut = nil;
        *constantDataOut = nil;
        *manifestOperationOut = name;
        return YES;
    }

    H13ConvPlan convolution{};
    if (convParityPlan(operation, &convolution)) {
        const std::uint32_t reduction = convolution.shape.input.channels /
            convolution.shape.groups * convolution.shape.kernel *
            convolution.shape.kernelWidth;
        NSData *weights = resolvedConstants[convolution.weight.name];
        if (!weights) {
            weights = [ANEBlobResolver loadConstantForOperation:convolution.weight.producer
                expectedBytes:convolution.shape.output.channels * reduction * 2
                modelRoot:modelRoot diagnostics:diagnostics];
            if (!weights) return NO;
            resolvedConstants[convolution.weight.name] = weights;
        }
        NSData *biasData = nil;
        if (convolution.bias) {
            biasData = perChannelConstantData(convolution.bias,
                convolution.shape.output.channels, modelRoot, diagnostics,
                resolvedConstants);
            if (!biasData) return NO;
        }
        // The MIL weight is already `[Cout, Cin / groups, kh, kw]` row-major,
        // which is the order the packing consumes, so no host transform runs.
        program = ane::h13::encodeConvParity(convolution.shape,
            static_cast<const std::uint8_t *>(weights.bytes), weights.length,
            static_cast<const std::uint8_t *>(biasData.bytes), biasData.length);
        *inputsOut = @[x];
        *constantInputOut = nil;
        *constantDataOut = nil;
        *manifestOperationOut = name;
        return YES;
    }

    if (binaryIndex != NSNotFound) {
        if (operation.arguments.count != 2 || !x || !y)
            return reject(diagnostics,
                @"H13 binary operations require x and y value operands", operation);
        BOOL xIsConstant = synthesizedConstants[x.name] || constantValue(x);
        BOOL yIsConstant = synthesizedConstants[y.name] || constantValue(y);
        NSUInteger elements = 0;
        if (!xIsConstant && !yIsConstant) {
            if (binaryIndex >= 4)
                return reject(diagnostics, [NSString stringWithFormat:
                    @"H13 '%@' with two runtime inputs cannot lower exactly through the verified binary modes", name],
                    operation, @"h13.nonfoldable-binary");
            if (!tensorElementCount(x, &elements) || !tensor(y, x.type.shape) ||
                !tensor(operation.results[0], x.type.shape))
                return reject(diagnostics,
                    @"H13 binary operations require fp16 inputs with the same positive static shape",
                    operation);
            inputs = @[x, y];
        } else {
            if (xIsConstant == yIsConstant || (binaryIndex >= 4 && xIsConstant)) {
                NSString *code = binaryIndex >= 4
                    ? @"h13.nonfoldable-binary" : @"h13.invalid-constant-input";
                return reject(diagnostics,
                    @"H13 binary folding requires one runtime fp16 tensor and one eligible const operand",
                    operation, code);
            }
            ANEGraphValue *runtimeInput = xIsConstant ? y : x;
            constantInput = xIsConstant ? x : y;
            if (!tensorElementCount(runtimeInput, &elements) ||
                !tensor(operation.results[0], runtimeInput.type.shape))
                return reject(diagnostics,
                    @"H13 folded binary operations require one fp16 input and output with the same positive static shape",
                    operation, @"h13.invalid-constant-input");
            if (elementOffset >= elements)
                return reject(diagnostics, @"H13 elementwise slice exceeds its tensor",
                    operation, @"h13.invalid-slice");
            NSUInteger sliceElements = MIN((NSUInteger)64, elements - elementOffset);

            NSData *wholeData = synthesizedConstants[constantInput.name];
            BOOL repeatedConstant = wholeData.length == 128;
            if (wholeData) {
                if (!tensor(constantInput, runtimeInput.type.shape) ||
                    (!repeatedConstant && wholeData.length != elements * 2))
                    return reject(diagnostics,
                        @"H13 synthesized constants must match the runtime fp16 tensor shape",
                        operation, @"h13.invalid-constant-input");
            } else {
                ANEGraphOperation *producer = constantInput.producer;
                BOOL scalar = constantInput.type.kind == ANEValueTypeKindScalar &&
                    constantInput.type.elementType == ANEElementTypeFP16;
                if ((!tensor(constantInput, runtimeInput.type.shape) && !scalar) ||
                    (scalar && binaryIndex != 0 && binaryIndex != 1 &&
                     binaryIndex != 4) || producer.arguments.count)
                    return reject(diagnostics,
                        @"H13 constants must be a matching fp16 tensor; only add, mul, and runtime-minus-constant sub accept an inline fp16 scalar broadcast",
                        producer, @"h13.invalid-constant-input");

                ANEGraphArgument *literal = producer.attributes[@"val"];
                if (scalar) {
                    uint16_t bits = 0;
                    if (!fp16Scalar(literal, &bits))
                        return reject(diagnostics,
                            @"H13 scalar binary folding requires one finite inline fp16 value",
                            producer, @"h13.invalid-constant-payload");
                    wholeData = splatFP16(bits);
                    repeatedConstant = YES;
                } else {
                    wholeData = resolvedConstants[constantInput.name];
                    if (!wholeData) {
                        if (literal.kind != ANEGraphArgumentKindCall ||
                            ![literal.calleeValueType isEqualToValueType:constantInput.type] ||
                            literal.callArguments.count != 1)
                            return reject(diagnostics,
                                @"H13 tensor constants require a matching typed inline list or BLOBFILE payload",
                                producer, @"h13.invalid-constant-payload");
                        ANEGraphArgument *payload = literal.callArguments[0].value;
                        if (payload.kind == ANEGraphArgumentKindList &&
                            payload.elements.count == elements) {
                            NSMutableData *dense = [NSMutableData dataWithLength:elements * 2];
                            uint16_t *words = static_cast<uint16_t *>(dense.mutableBytes);
                            for (NSUInteger index = 0; index < elements; ++index)
                                if (!fp16Scalar(payload.elements[index], &words[index]))
                                    return reject(diagnostics,
                                        @"H13 inline tensor constants require finite fp16 elements matching the tensor shape",
                                        producer, @"h13.invalid-constant-payload");
                            wholeData = dense;
                        } else if (payload.kind == ANEGraphArgumentKindCall &&
                                   [payload.calleeName isEqualToString:@"BLOBFILE"]) {
                            wholeData = [ANEBlobResolver loadConstantForOperation:producer
                                expectedBytes:elements * 2 modelRoot:modelRoot
                                diagnostics:diagnostics];
                            if (!wholeData)
                                return reject(diagnostics,
                                    @"H13 could not load the constant BLOBFILE payload",
                                    producer, @"h13.invalid-constant-payload");
                        } else {
                            return reject(diagnostics,
                                @"H13 inline tensor constants require finite fp16 elements matching the tensor shape",
                                producer, @"h13.invalid-constant-payload");
                        }
                        resolvedConstants[constantInput.name] = wholeData;
                    }
                }
            }

            if (!repeatedConstant && wholeData.length != elements * 2)
                return reject(diagnostics, @"H13 constant payload has the wrong size",
                    operation, @"h13.invalid-constant-payload");
            NSMutableData *padded = [NSMutableData dataWithLength:128];
            NSRange sourceRange = NSMakeRange(
                repeatedConstant ? 0 : elementOffset * 2, sliceElements * 2);
            [wholeData getBytes:padded.mutableBytes range:sourceRange];
            constantData = padded;
            NSMutableData *folded = [constantData mutableCopy];
            uint16_t *words = static_cast<uint16_t *>(folded.mutableBytes);
            if (binaryIndex == 4) {
                for (NSUInteger index = 0; index < sliceElements; ++index) {
                    if ((words[index] & 0x7c00u) == 0x7c00u)
                        return reject(diagnostics,
                            @"H13 sub constants must be finite for exact add lowering",
                            operation, @"h13.nonfinite-constant");
                    words[index] ^= 0x8000u;
                }
                manifestOperation = @"add";
            } else if (binaryIndex == 5) {
                for (NSUInteger index = 0; index < sliceElements; ++index) {
                    uint16_t exponent = (words[index] >> 10) & 0x1fu;
                    if (exponent == 0 || exponent == 0x1fu || (words[index] & 0x03ffu))
                        return reject(diagnostics,
                            @"H13 real_div constants require finite nonzero powers of two whose reciprocals are exactly representable in fp16",
                            operation, @"h13.inexact-reciprocal");
                    uint16_t sign = words[index] & 0x8000u;
                    words[index] = exponent == 30 ? sign | 0x0200u
                                                 : sign | ((30 - exponent) << 10);
                }
                manifestOperation = @"mul";
            }
            constantData = folded;
            inputs = @[runtimeInput, constantInput];
        }
        const ane::h13::BinaryOperation operations[] = {
            ane::h13::BinaryOperation::Add, ane::h13::BinaryOperation::Multiply,
            ane::h13::BinaryOperation::Maximum, ane::h13::BinaryOperation::Minimum,
            ane::h13::BinaryOperation::Add, ane::h13::BinaryOperation::Multiply};
        program = ane::h13::encodeBinary(operations[binaryIndex]);
    } else if ([name isEqualToString:@"matmul"]) {
        NSUInteger reduction = 0, rows = 0, columns = 0;
        BOOL transposeX = boolean(operation.arguments[@"transpose_x"], YES);
        BOOL transposeY = boolean(operation.arguments[@"transpose_y"], YES);
        BOOL geometry = matmulGeometry(x, operation.results[0], transposeX,
                                       &reduction, &rows, &columns);
        if (operation.arguments.count != 4 || constantValue(x) ||
            (!transposeX && !boolean(operation.arguments[@"transpose_x"], NO)) ||
            (!transposeY && !boolean(operation.arguments[@"transpose_y"], NO)) ||
            !geometry || !y)
            return reject(diagnostics,
                @"H13 matmul requires positive fp16 x rows, matching explicit transpose flags, and a matching positive output shape",
                operation);
        ane::h13::BatchedMatmulShape batched{};
        if (batchedMatmulParse(operation, &batched) == H13BatchedParseYes) {
            std::vector<std::uint8_t> packed;
            if (!batched.runtimeWeight) {
                ANEGraphValue *weight = operation.operands[@"y"].value;
                const NSUInteger packRows = batched.transposeY
                    ? batched.columns : batched.reduction;
                const NSUInteger packCols = batched.transposeY
                    ? batched.reduction : batched.columns;
                NSData *dense = resolvedConstants[weight.name];
                if (!dense) {
                    dense = [ANEBlobResolver loadConstantForOperation:weight.producer
                        expectedBytes:(NSUInteger)batched.batch * packRows * packCols * 2
                        modelRoot:modelRoot diagnostics:diagnostics];
                    if (!dense) return NO;
                    resolvedConstants[weight.name] = dense;
                }
                try {
                    packed = ane::h13::packBatchedWeights(batched,
                        static_cast<const std::uint8_t *>(dense.bytes),
                        dense.length);
                } catch (const std::exception &exception) {
                    return reject(diagnostics,
                        [NSString stringWithUTF8String:exception.what()],
                        operation, @"h13.matmul-outside-envelope");
                }
            }
            if (!ane::h13::supportsBatchedMatmul(batched))
                return reject(diagnostics,
                    [NSString stringWithFormat:
                        @"H13 batched matmul (B=%lu, rows=%lu, reduction=%lu, columns=%lu, tx=%d, ty=%d, %@ y) is outside the decoded batched envelope, which covers B in {2,4,8,16} at the attention geometries (375,128,749), (375,375,128), and (375,128,375) with their captured flag forms, plus the B=8 head projection (375,1024,128) at transpose_y with identity packing",
                        (unsigned long)batched.batch, (unsigned long)batched.rows,
                        (unsigned long)batched.reduction,
                        (unsigned long)batched.columns,
                        batched.transposeX, batched.transposeY,
                        batched.runtimeWeight ? @"runtime" : @"constant"],
                    operation, @"h13.matmul-outside-envelope");
            program = ane::h13::encodeBatchedMatmul(batched,
                packed.empty() ? nullptr : packed.data(), packed.size());
            *inputsOut = batched.runtimeWeight
                ? @[operation.operands[@"y"].value, x] : @[x];
            *constantInputOut = nil;
            *constantDataOut = nil;
            *manifestOperationOut = name;
            return YES;
        }
        const BOOL runtimeWeight =
            !constantValue(y) && !synthesizedConstants[y.name];
        ane::h13::MatmulShape parityShape{};
        if (runtimeWeight) {
            // Both operands runtime: attention's QK^T and PV. Apple stages the
            // second operand as its own surface, so there is nothing to pack
            // and nothing to slice.
            BOOL batchedOperand = NO;
            NSUInteger batchExtent = 1;
            for (NSUInteger index = 0; index + 2 < y.type.shape.count; ++index) {
                const NSUInteger extent =
                    y.type.shape[index].unsignedIntegerValue;
                if (extent != 1) {
                    batchedOperand = YES;
                    batchExtent *= extent;
                }
            }
            if (!runtimeMatmulOperand(y, reduction, columns, transposeY,
                                      x.type.shape.count) && batchedOperand)
                return reject(diagnostics,
                    [NSString stringWithFormat:
                        @"H13 batched runtime-operand matmul outside the decoded batched envelope: the batched primitive covers rank-3 [B,rows,K] and rank-4 [1,B,rows,K] operands with both transpose flags false at the attention geometries, while this form's flags or geometry have no decoded capture; flattening the %@ leading dimensions into rows would multiply every batch against one shared y instead of the per-batch y (B=%lu) this graph carries",
                        x.type.shape.count > 2 ?
                            [NSString stringWithFormat:@"%lu leading",
                                (unsigned long)(x.type.shape.count - 2)] : @"its",
                        (unsigned long)batchExtent],
                    operation, @"h13.matmul-outside-envelope");
            if (!runtimeMatmulOperand(y, reduction, columns, transposeY,
                                      x.type.shape.count) ||
                !matmulParityShape(x, operation.results[0], transposeX, transposeY,
                                   YES, preferNative, &parityShape) ||
                inputElementCount != rows * reduction)
                return reject(diagnostics,
                    @"H13 matmul with two runtime operands needs a decoded geometry: fp16 static shapes of matching rank, the second operand shaped by transpose_y, and rows, reduction, and columns inside the oracle parity envelope",
                    operation, @"h13.matmul-outside-envelope");
            program = ane::h13::encodeMatmulParity(parityShape, nullptr, 0);
            // Apple declares the second operand first and lays the output out
            // below both, so the package's channel 5 carries y and 6 carries x.
            *inputsOut = @[y, x];
            *constantInputOut = nil;
            *constantDataOut = nil;
            *manifestOperationOut = name;
            return YES;
        }
        if (!tensor(y, transposeY ? @[@(columns), @(reduction)]
                                  : @[@(reduction), @(columns)])) {
            BOOL batchedWeight = NO;
            NSUInteger batchExtent = 1;
            for (NSUInteger index = 0; index + 2 < y.type.shape.count; ++index) {
                const NSUInteger extent =
                    y.type.shape[index].unsignedIntegerValue;
                if (extent != 1) {
                    batchedWeight = YES;
                    batchExtent *= extent;
                }
            }
            if (batchedWeight)
                return reject(diagnostics,
                    [NSString stringWithFormat:
                        @"H13 batched constant weight needs a per-batch decomposition the decoded corpus cannot serve: the matvec program model carries one constant section per program, so B=%lu per-batch weights would need B programs over the batch-major contiguous x and output slices, and no decoded geometry covers their (rows, reduction, columns) — the same batched matvec primitive the runtime-operand path needs",
                        (unsigned long)batchExtent],
                    operation, @"h13.matmul-outside-envelope");
            return reject(diagnostics,
                @"H13 matmul requires a constant rank-2 W shaped by transpose_y",
                operation);
        }
        NSData *synthesizedWeight = synthesizedConstants[y.name];
        if (synthesizedWeight) {
            // A channel-plane chunk weight synthesized by the composite
            // linear lowering: bytes already row-major [columns, reduction].
            if (synthesizedWeight.length != columns * reduction * 2)
                return reject(diagnostics,
                    @"H13 synthesized matmul weight has the wrong size",
                    operation, @"h13.invalid-constant-payload");
        } else {
            ANEGraphArgument *value = y.producer.attributes[@"val"];
            if (y.producer.arguments.count || value.kind != ANEGraphArgumentKindCall ||
                ![value.calleeValueType isEqualToValueType:y.type] ||
                value.callArguments.count != 1 ||
                ![value.callArguments[0].value.calleeName isEqualToString:@"BLOBFILE"])
                return reject(diagnostics,
                    @"H13 weights require a matching tensor value with one BLOBFILE payload", y.producer);
            if (columns > NSUIntegerMax / reduction ||
                columns * reduction > NSUIntegerMax / 2)
                return reject(diagnostics, @"H13 matmul weight size overflows",
                    operation, @"h13.invalid-constant-payload");
        }
        NSUInteger count = columns * reduction;
        NSData *weights = synthesizedWeight ?: resolvedConstants[y.name];
        if (!weights) {
            weights = [ANEBlobResolver loadConstantForOperation:y.producer
                expectedBytes:count * 2 modelRoot:modelRoot diagnostics:diagnostics];
            if (!weights) return NO;
            resolvedConstants[y.name] = weights;
        }
        if (transposeX && rows > 1 &&
            !matmulParityShape(x, operation.results[0], YES, YES, NO,
                               preferNative, nullptr))
            return reject(diagnostics,
                @"H13 transpose_x=true matmul supports exactly one logical row outside the decoded parity envelope",
                operation, @"h13.transpose-x-multirow");
        // Apple refuses a transpose_y=false constant weight, so the host
        // transpose below feeds the transpose_y=true program it accepts.
        if (matmulParityShape(x, operation.results[0], transposeX, YES, NO,
                              preferNative, &parityShape) &&
            inputElementCount == rows * reduction) {
            // transpose_y=false weights are [K, N]; Apple rejects that form, so
            // the exact host transpose feeds the same encoder.
            NSData *rowMajor = weights;
            if (!transposeY) {
                NSMutableData *transposed =
                    [NSMutableData dataWithLength:count * 2];
                const uint16_t *source =
                    static_cast<const uint16_t *>(weights.bytes);
                uint16_t *destination =
                    static_cast<uint16_t *>(transposed.mutableBytes);
                for (NSUInteger column = 0; column < columns; ++column)
                    for (NSUInteger index = 0; index < reduction; ++index)
                        destination[column * reduction + index] =
                            source[index * columns + column];
                rowMajor = transposed;
            }
            program = ane::h13::encodeMatmulParity(parityShape,
                static_cast<const std::uint8_t *>(rowMajor.bytes),
                rowMajor.length);
            *inputsOut = @[x];
            *constantInputOut = nil;
            *constantDataOut = nil;
            *manifestOperationOut = name;
            return YES;
        }
        if (transposeX && rows > 1)
            return reject(diagnostics,
                @"H13 transpose_x=true matmul supports exactly one logical row",
                operation, @"h13.transpose-x-multirow");
        NSUInteger reductionStart = elementOffset % reduction;
        NSUInteger chunkReduction = inputElementCount;
        if (!chunkReduction || chunkReduction > 512 ||
            reductionStart > reduction - chunkReduction)
            return reject(diagnostics, @"H13 matmul reduction slice is invalid",
                operation, @"h13.invalid-slice");
        NSUInteger physicalReduction = chunkReduction <= 256 ? 256 : 512;
        NSUInteger outputColumn = outputElementOffset % columns;
        NSUInteger outputElements = MIN((NSUInteger)512, columns - outputColumn);
        NSMutableData *paddedWeights =
            [NSMutableData dataWithLength:512 * physicalReduction * 2];
        const uint8_t *source = static_cast<const uint8_t *>(weights.bytes);
        uint8_t *destination = static_cast<uint8_t *>(paddedWeights.mutableBytes);
        if (transposeY) {
            for (NSUInteger row = 0; row < outputElements; ++row)
                std::memcpy(destination + row * physicalReduction * 2,
                    source + ((outputColumn + row) * reduction + reductionStart) * 2,
                    chunkReduction * 2);
        } else {
            for (NSUInteger row = 0; row < chunkReduction; ++row)
                std::memcpy(destination + row * 512 * 2,
                    source + ((reductionStart + row) * columns + outputColumn) * 2,
                    outputElements * 2);
        }
        program = ane::h13::encodeMatvec(
            static_cast<std::uint32_t>(physicalReduction),
            static_cast<const std::uint8_t *>(paddedWeights.bytes),
            paddedWeights.length, transposeY);
        inputs = @[x];
    } else if ([name isEqualToString:@"linear"]) {
        ANEGraphValue *weight = operation.operands[@"weight"].value;
        ANEGraphValue *bias = operation.operands[@"bias"].value;
        NSUInteger reduction = 0, rows = 0, columns = 0;
        if (!fp16Tensor(x) ||
            !matmulGeometry(x, operation.results[0], NO,
                            &reduction, &rows, &columns))
            return reject(diagnostics,
                @"H13 linear requires positive fp16 x rows and a matching output shape",
                operation, @"h13.invalid-linear");
        if (!weight || !constantValue(weight))
            return reject(diagnostics,
                @"H13 linear requires a constant weight tensor", operation,
                @"h13.linear-nonconstant-weight");
        NSData *weights = resolvedConstants[weight.name];
        if (!weights) {
            weights = [ANEBlobResolver loadConstantForOperation:weight.producer
                expectedBytes:columns * reduction * 2
                modelRoot:modelRoot diagnostics:diagnostics];
            if (!weights) return NO;
            resolvedConstants[weight.name] = weights;
        }
        ane::h13::LinearBiasMode biasMode = ane::h13::LinearBiasMode::None;
        NSData *biasData = nil;
        if (bias) {
            biasData = resolvedConstants[bias.name];
            if (!biasData) {
                biasData = [ANEBlobResolver
                    loadConstantForOperation:bias.producer
                    expectedBytes:columns * 2
                    modelRoot:modelRoot diagnostics:diagnostics];
                if (!biasData) return NO;
                resolvedConstants[bias.name] = biasData;
            }
            biasMode = ane::h13::LinearBiasMode::Uniform;
            const uint16_t *halves =
                static_cast<const uint16_t *>(biasData.bytes);
            for (NSUInteger index = 1; index < columns; ++index)
                if (halves[index] != halves[0]) {
                    biasMode = ane::h13::LinearBiasMode::Block;
                    break;
                }
        }
        if (!ane::h13::supportsLinearParity(
                static_cast<std::uint32_t>(rows),
                static_cast<std::uint32_t>(reduction),
                static_cast<std::uint32_t>(columns), biasMode))
            return reject(diagnostics,
                [NSString stringWithFormat:
                    @"H13 rank-3 linear (%lu rows, %lu reduction, %lu columns) with a %@ bias is outside the decoded parity envelope, which covers the captured m375 encoder geometries (k1024 at n128/640/1024/4096 and k4096 at n1024) with absent, uniform, or per-column bias blocks",
                    (unsigned long)rows, (unsigned long)reduction,
                    (unsigned long)columns,
                    biasMode == ane::h13::LinearBiasMode::None ? @"absent"
                        : biasMode == ane::h13::LinearBiasMode::Uniform
                        ? @"uniform" : @"per-column"],
                operation, @"h13.linear-outside-envelope");
        program = ane::h13::encodeLinearParity(
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(reduction),
            static_cast<std::uint32_t>(columns), biasMode,
            static_cast<const uint8_t *>(weights.bytes), weights.length,
            biasMode == ane::h13::LinearBiasMode::Block
                ? static_cast<const uint8_t *>(biasData.bytes) : nullptr,
            biasMode == ane::h13::LinearBiasMode::Block ? biasData.length : 0,
            biasMode == ane::h13::LinearBiasMode::Uniform
                ? static_cast<const uint16_t *>(biasData.bytes)[0] : 0x3401);
        *inputsOut = @[x];
        *manifestOperationOut = name;
        return YES;
    } else if ([name isEqualToString:@"ffn-chain"]) {
        ANEGraphValue *weight1 = operation.operands[@"weight1"].value;
        ANEGraphValue *bias1 = operation.operands[@"bias1"].value;
        ANEGraphValue *weight2 = operation.operands[@"weight2"].value;
        ANEGraphValue *bias2 = operation.operands[@"bias2"].value;
        NSUInteger reduction = 0, rows = 0, columns = 0;
        NSUInteger middle = 0;
        if (!fp16Tensor(x) || x.type.shape.count != 3 ||
            !matmulGeometry(x, operation.results[0], NO,
                            &reduction, &rows, &columns) ||
            !weight1 || weight1.type.shape.count != 2 ||
            !(middle = weight1.type.shape[0].unsignedIntegerValue) ||
            weight1.type.shape[1].unsignedIntegerValue != reduction ||
            !weight2 || weight2.type.shape.count != 2 ||
            weight2.type.shape[0].unsignedIntegerValue != columns ||
            weight2.type.shape[1].unsignedIntegerValue != middle)
            return reject(diagnostics,
                @"H13 FFN chain requires rank-3 x [1, rows, reduction], a [middle, reduction] weight1, and a [columns, middle] weight2",
                operation, @"h13.invalid-ffn-chain");
        NSData *weights1 = resolvedConstants[weight1.name];
        if (!weights1) {
            weights1 = [ANEBlobResolver loadConstantForOperation:weight1.producer
                expectedBytes:middle * reduction * 2
                modelRoot:modelRoot diagnostics:diagnostics];
            if (!weights1) return NO;
            resolvedConstants[weight1.name] = weights1;
        }
        NSData *biasData1 = resolvedConstants[bias1.name];
        if (!biasData1) {
            biasData1 = [ANEBlobResolver loadConstantForOperation:bias1.producer
                expectedBytes:middle * 2 modelRoot:modelRoot diagnostics:diagnostics];
            if (!biasData1) return NO;
            resolvedConstants[bias1.name] = biasData1;
        }
        NSData *weights2 = resolvedConstants[weight2.name];
        if (!weights2) {
            weights2 = [ANEBlobResolver loadConstantForOperation:weight2.producer
                expectedBytes:middle * columns * 2
                modelRoot:modelRoot diagnostics:diagnostics];
            if (!weights2) return NO;
            resolvedConstants[weight2.name] = weights2;
        }
        NSData *biasData2 = resolvedConstants[bias2.name];
        if (!biasData2) {
            biasData2 = [ANEBlobResolver loadConstantForOperation:bias2.producer
                expectedBytes:columns * 2 modelRoot:modelRoot diagnostics:diagnostics];
            if (!biasData2) return NO;
            resolvedConstants[bias2.name] = biasData2;
        }
        if (!ane::h13::supportsFFNChain(
                static_cast<std::uint32_t>(rows),
                static_cast<std::uint32_t>(reduction),
                static_cast<std::uint32_t>(middle),
                static_cast<std::uint32_t>(middle),
                static_cast<std::uint32_t>(columns)))
            return reject(diagnostics,
                @"H13 FFN chain is outside the decoded parity envelope, which covers the captured d1024 s375 form only",
                operation, @"h13.ffn-chain-outside-envelope");
        program = ane::h13::encodeFFNChain(
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(reduction),
            static_cast<std::uint32_t>(middle),
            static_cast<std::uint32_t>(middle),
            static_cast<std::uint32_t>(columns),
            static_cast<const uint8_t *>(weights1.bytes),
            static_cast<const uint8_t *>(biasData1.bytes),
            static_cast<const uint8_t *>(weights2.bytes),
            static_cast<const uint8_t *>(biasData2.bytes));
        *inputsOut = @[x];
        *manifestOperationOut = name;
        return YES;
    } else if ([name isEqualToString:@"less"] ||
               [name isEqualToString:@"floor"] ||
               [name isEqualToString:@"select"] ||
               [name isEqualToString:@"floor_div"]) {
        // The boolean registry ops lower through their decoded templates:
        // fixed task streams keyed by (family, CHW surface, const-operand
        // twin). less emits a bool result and select reads a bool cond —
        // the captured 1-byte surfaces — so the fp16-only result gate and
        // every operand check here is family-specific.
        ane::h13::H13BooleanShape shape{};
        if ([name isEqualToString:@"less"]) shape.kind = ane::h13::H13BooleanKind::Less;
        else if ([name isEqualToString:@"floor"]) shape.kind = ane::h13::H13BooleanKind::Floor;
        else if ([name isEqualToString:@"select"]) shape.kind = ane::h13::H13BooleanKind::Select;
        else shape.kind = ane::h13::H13BooleanKind::FloorDiv;
        NSMutableArray<ANEGraphValue *> *operands = [NSMutableArray array];
        if (shape.kind == ane::h13::H13BooleanKind::Select) {
            for (NSString *key in @[@"a", @"b", @"cond"]) {
                ANEGraphValue *operand = operation.operands[key].value;
                if (!operand && operation.arguments[key].kind ==
                    ANEGraphArgumentKindValue)
                    operand = operation.arguments[key].value;
                [operands addObject:operand];
            }
        } else {
            [operands addObject:operation.operands[@"x"].value];
            ANEGraphValue *second = operation.operands[@"y"].value;
            if (second) [operands addObject:second];
        }
        // The shape comes from the full-tensor operand: for select with a
        // scalar fill that is b, never the scalar.
        ANEGraphValue *primary = (shape.kind == ane::h13::H13BooleanKind::Select &&
                                  operands[0].type.shape.count == 0)
            ? operands[1] : operands[0];
        if (!primary || primary.type.shape.count > 4) goto boolean_reject;
        {
            // Flat tensors (rank 2 and below) key as (elements, 1, 1) —
            // the captured less shapes are flat vectors — while rank 3/4
            // shapes collapse leading unit dimensions into CHW.
            NSUInteger elements = 0;
            if (!tensorElementCount(primary, &elements)) goto boolean_reject;
            NSArray<NSNumber *> *shapeText = primary.type.shape;
            if (shapeText.count <= 2) {
                shapeText = @[@(elements), @1, @1];
            } else {
                while (shapeText.count > 3 &&
                       [shapeText[0] isEqualToNumber:@1])
                    shapeText = [shapeText subarrayWithRange:
                        NSMakeRange(1, shapeText.count - 1)];
                while (shapeText.count < 3)
                    shapeText = [@[@1] arrayByAddingObjectsFromArray:shapeText];
            }
            if (shapeText.count != 3) goto boolean_reject;
            shape.channels = (std::uint32_t)shapeText[0].unsignedIntegerValue;
            shape.height = (std::uint32_t)shapeText[1].unsignedIntegerValue;
            shape.width = (std::uint32_t)shapeText[2].unsignedIntegerValue;
        }
        // The captured dtypes are load-bearing: less emits a bool result
        // and select reads a bool cond; Apple's own tool rejects the fp16
        // forms outright, so they have no device encoding at all.
        if (shape.kind == ane::h13::H13BooleanKind::Less &&
            !(operation.results[0].type.kind == ANEValueTypeKindTensor &&
              operation.results[0].type.elementType == ANEElementTypeBool))
            goto boolean_reject;
        if (shape.kind == ane::h13::H13BooleanKind::Select) {
            ANEGraphValue *cond = operation.operands[@"cond"].value;
            if (!cond || cond.type.kind != ANEValueTypeKindTensor ||
                cond.type.elementType != ANEElementTypeBool)
                goto boolean_reject;
        }
        // The const-operand twins: floor over a BLOBFILE x, floor_div over
        // the captured scalar-2.0 y, select over the -inf blob a. The
        // select -inf twin's constant section is not yet derivable from a
        // uniform-value capture, so runtime-a stays the lowering form.
        shape.constInput =
            (shape.kind == ane::h13::H13BooleanKind::Floor &&
             constantValue(primary)) ||
            (shape.kind == ane::h13::H13BooleanKind::FloorDiv &&
             constantValue(operation.operands[@"y"].value)) ||
            (shape.kind == ane::h13::H13BooleanKind::Select &&
             constantValue(operation.operands[@"a"].value));
        if (shape.kind == ane::h13::H13BooleanKind::Select && shape.constInput)
            return reject(diagnostics,
                @"H13 select with a constant a belongs to the frontend, which materializes the fill as a runtime constant input and routes the +0.0-fill family through its exact mul rewrite: the captured constant-a form carries uniform -inf values whose retained section cannot discriminate the packing for arbitrary constants, so this path lowers only runtime-a forms with a bool cond",
                operation, @"h13.select-needs-decoded-encoder");
        if (!ane::h13::supportsBooleanOp(shape)) {
        boolean_reject:
            return reject(diagnostics,
                [NSString stringWithFormat:
                    @"H13 %@ is outside the decoded boolean envelope: the captured geometries are less at CHW (375,1,1)/(750,1,1)/(1500,1,1)/(64,1,1) with a bool result, floor at (1,1,1)/(64,1,1)/(512,1,1) runtime or blob x, select at (64,1,1)/(8,375,375) with runtime a and a bool cond, and floor_div at (1,1,1)/(64,1,1) with runtime y or the scalar 2.0 — fp16-result less and fp16-cond select are rejected by Apple's own tool, so no device form exists for them",
                    name],
                operation, @"h13.boolean-outside-envelope");
        }
        const uint8_t *scalarLane = nullptr;
        if (shape.kind == ane::h13::H13BooleanKind::Floor && shape.constInput) {
            ANEGraphValue *xValue = operation.operands[@"x"].value;
            NSData *payload = resolvedConstants[xValue.name];
            if (!payload) {
                payload = [ANEBlobResolver loadConstantForOperation:xValue.producer
                    expectedBytes:2 modelRoot:modelRoot
                    diagnostics:diagnostics];
                if (!payload) return NO;
                resolvedConstants[xValue.name] = payload;
            }
            if (payload.length < 2)
                return reject(diagnostics,
                    @"H13 floored constants must carry one fp16 lane",
                    operation, @"h13.invalid-constant-payload");
            scalarLane = static_cast<const uint8_t *>(payload.bytes);
        }
        if (shape.kind == ane::h13::H13BooleanKind::FloorDiv &&
            shape.constInput) {
            ANEGraphValue *divisor = operation.operands[@"y"].value;
            uint16_t divisorBits = 0;
            BOOL halves = divisor.type.kind == ANEValueTypeKindScalar &&
                divisor.type.elementType == ANEElementTypeFP16 &&
                constantValue(divisor) &&
                fp16Scalar(divisor.producer.attributes[@"val"], &divisorBits);
            if (!halves || divisorBits != 0x4000)
                return reject(diagnostics,
                    @"H13 floor_div lowers the captured scalar-2.0 divisor only",
                    operation, @"h13.invalid-constant-input");
        }
        program = ane::h13::encodeBooleanOp(shape, scalarLane,
                                            scalarLane ? 2 : 0);
        {
            NSMutableArray *inputs = [NSMutableArray array];
            for (ANEGraphValue *operand in operands) {
                BOOL folded =
                    (shape.constInput && operand == operands.lastObject &&
                     shape.kind == ane::h13::H13BooleanKind::FloorDiv) ||
                    (shape.constInput && operand == operands[0] &&
                     shape.kind == ane::h13::H13BooleanKind::Floor);
                if (!folded) [inputs addObject:operand];
            }
            *inputsOut = inputs;
        }
        *constantInputOut = nil;
        *constantDataOut = nil;
        *manifestOperationOut = name;
        return YES;
     } else {
        ane::h13::NormOperation normOperation{};
        if (normEncoding(name, &normOperation))
            return reject(diagnostics, [NSString stringWithFormat:
                @"H13 '%@' needs a decoded geometry: fp16 static shapes, "
                 "constant axes, no gamma or beta, epsilon 1e-5, and an input "
                 "and output surface inside the oracle parity envelope", name],
                operation, @"h13.norm-outside-envelope");
        if ([name isEqualToString:@"tile"]) {
            ANEGraphValue *reps = operation.operands[@"reps"].value;
            NSArray<NSNumber *> *repValues = int32TensorElements(reps);
            ane::h13::H13TileShape tile{};
            NSArray<NSNumber *> *inShape = x ? x.type.shape : nil;
            while (inShape.count > 3)
                inShape = [inShape subarrayWithRange:
                    NSMakeRange(1, inShape.count - 1)];
            while (inShape.count < 3)
                inShape = [@[@1] arrayByAddingObjectsFromArray:inShape];
            NSArray<NSNumber *> *repText = repValues;
            while (repText.count > 3)
                repText = [repText subarrayWithRange:
                    NSMakeRange(1, repText.count - 1)];
            while (repText.count < 3)
                repText = [@[@1] arrayByAddingObjectsFromArray:repText];
            BOOL tileParse = x && inShape.count == 3 && repText.count == 3;
            for (NSUInteger index = 0; index < 3 && tileParse; ++index) {
                long long rep = repText[index].longLongValue;
                if (rep < 1 || rep > UINT32_MAX) tileParse = NO;
            }
            if (tileParse) {
                tile.inChannels = inShape[0].unsignedIntegerValue;
                tile.inHeight = inShape[1].unsignedIntegerValue;
                tile.inWidth = inShape[2].unsignedIntegerValue;
                tile.repChannels = (std::uint32_t)repText[0].longLongValue;
                tile.repHeight = (std::uint32_t)repText[1].longLongValue;
                tile.repWidth = (std::uint32_t)repText[2].longLongValue;
                tile.runtimeInput = !constantValue(x);
                if (!ane::h13::supportsTileOp(tile))
                    return reject(diagnostics,
                        @"H13 tile is outside the decoded tile envelope, which covers the captured materialized forms",
                        operation, @"h13.unsupported-tile");
                if (!tile.runtimeInput) {
                    // The decoded blob template scatters its operand into
                    // constant-section lanes with no clean patch map (the
                    // capture's const section carries capture-specific
                    // scattered bytes, not the dense blob), so no
                    // byte-exact reproducer exists for a general blob
                    // operand; the runtime-operand templates are exact.
                    return reject(diagnostics,
                        @"H13 tile with a constant operand is outside the decoded tile envelope: the blob template's constant-section lane embedding has no byte-exact reproducer, only the runtime-operand forms lower",
                        operation, @"h13.unsupported-tile");
                }
                program = ane::h13::encodeTileOp(tile);
                *inputsOut = tile.runtimeInput ? @[x] : @[];
                *constantInputOut = nil;
                *constantDataOut = nil;
                *manifestOperationOut = name;
                return YES;
            }
            return reject(diagnostics,
                @"H13 tile is outside the decoded tile envelope, which covers the captured materialized forms",
                operation, @"h13.unsupported-tile");
        }
        if ([name isEqualToString:@"conv"])
            return reject(diagnostics,
                @"H13 conv needs a decoded geometry: an fp16 rank-4 input and "
                 "result with a batch of one (or the encoder's rank-3 spell, "
                 "which lowers only through the W-major surface it binds as), "
                 "a constant [Cout, Cin/groups, kh, kw] weight, unit dilations, "
                 "and a kernel, stride, group count and surface pair inside "
                 "the oracle parity envelope. Padding lowers as decoded: "
                 "pad_type 'same' or 'valid' with zero pads, the rank-3 "
                 "custom spell whose pads equal one of those two, and the "
                 "rank-4 custom spell whose declared pads reproduce the "
                 "declared output where a template row covers the surface "
                 "pair (the W-padded rel-pos padconv k1x1 g8, pad [0,0,1,0], "
                 "lowers as its own two-task capture). Refused on named gaps: "
                 "rank-3 spellings with no W-major row (the checked-in n2048 "
                 "capture is H-major, whose surface the rank-3 tensor does "
                 "not bind as), custom pads with no covering row, and the "
                 "stride-2 multi-tap forms whose sections carry transformed "
                 "halfwords (see receipts/2026-09-17-encoder-padconv-respell "
                 "and receipts/2026-09-17-encoder-conv-lowering)",
                operation, @"h13.conv-outside-envelope");
        ane::h13::UnaryOperation unaryOperation{};
        if (unaryEncoding(name, &unaryOperation))
            return reject(diagnostics, [NSString stringWithFormat:
                @"H13 %@ is outside the decoded unary envelope: the captured geometries are silu, sigmoid, exp, gelu, leaky_relu, relu, rsqrt, sqrt, and tanh at CHW (64,1,1)/(512,1,1), and abs at (64,1,1)/(128,1,1)/(256,1,1)/(512,1,1)/(1024,1,1)/(2048,1,1)/(4096,1,1) — LUT unaries have no 64-lane split",
                name],
                operation, @"h13.unary-outside-envelope");
        return reject(diagnostics, [NSString stringWithFormat:
            @"H13 has no source-qualified encoder for '%@'", name], operation);
    }

    *inputsOut = inputs;
    *constantInputOut = constantInput;
    *constantDataOut = constantData;
    *manifestOperationOut = manifestOperation;
    return YES;
}

@implementation ANEH13Compiler
+ (BOOL)compileMILData:(NSData *)milData
             modelRoot:(NSURL *)modelRoot
                format:(NSString *)format
       outputDirectory:(NSURL *)directory
              schedule:(NSString *)schedule
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error {
    BOOL hwx = [format isEqualToString:@"hwx"];
    if (!hwx && ![format isEqualToString:@"anec"]) {
        reject(diagnostics, @"H13 artifact format must be 'anec' or 'hwx'",
               nil, @"h13.unsupported-format");
        if (error) *error = [NSError errorWithDomain:@"dev.maderix.H13" code:2
            userInfo:@{NSLocalizedDescriptionKey:
                @"H13 artifact format must be 'anec' or 'hwx'"}];
        return NO;
    }
    MILLexer *lexer = [[MILLexer alloc] initWithData:milData
                                         diagnostics:diagnostics];
    if (![schedule isEqualToString:@"per-op"] &&
        ![schedule isEqualToString:@"chain"] &&
        ![schedule isEqualToString:@"regions"])
        return reject(diagnostics,
            @"H13 schedule must be 'per-op', 'chain', or 'regions'", nil,
            @"h13.unsupported-schedule");
    NSArray<MILToken *> *tokens = lexer.lexAllTokens;
    if (diagnostics.errorCount) return NO;
    MILParser *parser = [[MILParser alloc] initWithTokens:tokens
        diagnostics:diagnostics];
    MILProgramSyntax *syntax = parser.parseProgram;
    ANEGraphModule *module = syntax
        ? [MILGraphImporter importProgram:syntax diagnostics:diagnostics] : nil;
    if (!module || diagnostics.errorCount ||
        ![ANEGraphVerifier verifyModule:module diagnostics:diagnostics])
        return NO;
    if (module.functions.count != 1)
        return reject(diagnostics, @"H13 requires exactly one function");
    ANEGraphFunction *function = module.functions[0];
    for (ANEGraphOperation *candidate in function.operations) {
        if ([candidate.operationName isEqualToString:@"split"]) {
            H13SplitAliasPlan plan{};
            if (!splitAliasPlan(candidate, diagnostics, &plan)) return NO;
        } else if (candidate.results.count != 1) {
            return reject(diagnostics,
                @"H13 does not lower multi-result operations",
                candidate, @"h13.unsupported-multi-result-operation");
        }
    }
    NSMutableArray<ANEGraphOperation *> *sourceOperations = [NSMutableArray array];
    for (ANEGraphOperation *candidate in function.operations)
        if (![candidate.operationName isEqualToString:@"const"])
            [sourceOperations addObject:candidate];
    if (!sourceOperations.count)
        return reject(diagnostics, @"H13 requires at least one operation");

    BOOL chain = sourceOperations.count > 1;
    NSString *chainCode = @"h13.unsupported-chain";
    ANEGraphOperation *lastSourceOperation = sourceOperations.lastObject;
    if (!function.returnValues.count)
        return reject(diagnostics, @"H13 requires at least one function result",
            lastSourceOperation, chain ? chainCode : @"h13.unsupported-program");
    for (ANEGraphValue *returnedValue in function.returnValues)
        if (!fp16Tensor(returnedValue) && !boolTensor(returnedValue))
            return reject(diagnostics,
                @"H13 logical result conversions require explicit hardware or GPU coverage",
                returnedValue.producer ?: lastSourceOperation,
                @"h13.unsupported-logical-result-conversion");

    for (ANEGraphValue *input in function.inputs) {
        BOOL used = NO;
        for (ANEGraphOperation *candidate in sourceOperations)
            if (operationUsesValue(candidate, input)) used = YES;
        if (!used)
            return reject(diagnostics, @"H13 function inputs must all be used",
                sourceOperations[0], chain ? chainCode : @"h13.unsupported-program");
    }
    for (NSUInteger index = 0; index < sourceOperations.count; ++index) {
        BOOL used = NO;
        for (ANEGraphValue *value in sourceOperations[index].results) {
            if ([function.returnValues containsObject:value]) used = YES;
            for (NSUInteger consumer = index + 1;
                 consumer < sourceOperations.count; ++consumer)
                if (operationUsesValue(sourceOperations[consumer], value))
                    used = YES;
        }
        if (!used)
            return reject(diagnostics,
                @"H13 operation results not returned must be consumed by a later operation",
                sourceOperations[index], chainCode);
    }

    BOOL chainSchedule = [schedule isEqualToString:@"chain"];
    if (chainSchedule) {
        if (sourceOperations.count < 2 || function.inputs.count > 2)
            return reject(diagnostics,
                @"H13 composed scheduling needs at least two operations and at most two boundary inputs",
                sourceOperations[0], @"h13.chain-outside-envelope");
        ANEGraphOperation *first = sourceOperations[0];
        if (sourceOperations.count != 2)
            return reject(diagnostics,
                @"h13.chain-unrepresentable-edge: composed scheduling supports exactly one producer followed by relu",
                sourceOperations[0], @"h13.chain-unrepresentable-edge");
        if (![lastSourceOperation.operationName isEqualToString:@"relu"])
            return reject(diagnostics,
                @"h13.chain-unrepresentable-edge: only a final relu has a decoded single-kernel fusion",
                lastSourceOperation, @"h13.chain-unrepresentable-edge");
        for (ANEGraphValue *input in function.inputs) {
            BOOL firstUse = NO;
            for (ANEGraphArgument *operand in first.operands.allValues)
                firstUse = firstUse || operand.value == input;
            if (!firstUse)
                return reject(diagnostics,
                    @"H13 composed scheduling needs every boundary input resident in the first operation",
                    first, @"h13.chain-outside-envelope");
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
                    @"H13 composed scheduling needs a straight-line chain with one live intermediate",
                    sourceOperations[index], @"h13.chain-outside-envelope");
        }
    }

    NSMutableArray<ANEGraphOperation *> *operations = [NSMutableArray array];
    NSMutableArray<ANEGraphValue *> *manifestValues = [NSMutableArray array];
    NSMutableDictionary<NSString *, NSDictionary *> *aliases =
        [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSData *> *synthesizedConstants =
        [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSData *> *resolvedConstants =
        [NSMutableDictionary dictionary];
    NSMapTable<ANEGraphOperation *, NSNumber *> *reductionOffsets =
        [NSMapTable strongToStrongObjectsMapTable];
    NSMapTable<ANEGraphOperation *, NSNumber *> *reductionCounts =
        [NSMapTable strongToStrongObjectsMapTable];
    NSMapTable<ANEGraphOperation *, NSNumber *> *outputBaseOffsets =
        [NSMapTable strongToStrongObjectsMapTable];
    NSMutableSet<NSString *> *chunkedAccumulations = [NSMutableSet set];
    NSMapTable<ANEGraphValue *, ANEGraphValue *> *loweredValues =
        [NSMapTable strongToStrongObjectsMapTable];
    NSMapTable<ANEGraphValue *, NSNumber *> *valueBaseOffsets =
        [NSMapTable strongToStrongObjectsMapTable];
    NSMutableDictionary<NSString *, NSDictionary *> *transposeFolds =
        [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSDictionary *> *compositeFolds =
        [NSMutableDictionary dictionary];
    NSUInteger ffnChainSkip = 0;
    for (ANEGraphOperation *candidate in sourceOperations) {
        if (ffnChainSkip) {
            --ffnChainSkip;
            continue;
        }
        if ([candidate.operationName isEqualToString:@"matmul"]) {
            if (ANEGraphOperation *chain = ffnChainOperation(sourceOperations,
                                                             candidate,
                                                             candidate.range)) {
                [operations addObject:chain];
                [manifestValues addObject:chain.results[0]];
                ffnChainSkip = 4;
                continue;
            }
        }
        NSMutableDictionary<NSString *, ANEGraphArgument *> *arguments =
            [candidate.arguments mutableCopy];
        for (NSString *key in candidate.operands) {
            ANEGraphValue *value = candidate.operands[key].value;
            ANEGraphValue *lowered = [loweredValues objectForKey:value];
            if (lowered) arguments[key] = valueArgument(lowered, candidate.range);
        }
        NSString *name = candidate.operationName;
        ANEGraphValue *x = arguments[@"x"].value;
        if ([name isEqualToString:@"split"]) {
            H13SplitAliasPlan plan{};
            if (!splitAliasPlan(candidate, diagnostics, &plan)) return NO;
            NSUInteger baseOffset =
                [[valueBaseOffsets objectForKey:x] unsignedIntegerValue];
            for (NSUInteger index = 0; index < candidate.results.count; ++index) {
                if (index && plan.resultElements >
                    (NSUIntegerMax - baseOffset) / index)
                    return reject(diagnostics, @"H13 split alias offset overflows",
                        candidate, @"h13.invalid-split-shape");
                ANEGraphValue *sourceResult = candidate.results[index];
                ANEGraphValue *alias = [[ANEGraphValue alloc]
                    initWithName:x.name type:sourceResult.type];
                [loweredValues setObject:alias forKey:sourceResult];
                [valueBaseOffsets setObject:
                    @(baseOffset + index * plan.resultElements) forKey:alias];
            }
            continue;
        }
        if ([name isEqualToString:@"concat"])
            return concatPlan(candidate, diagnostics);

        if ([name isEqualToString:@"transpose"]) {
            H13TransposeViewPlan plan{};
            if (!transposeViewPlan(candidate, diagnostics, &plan)) return NO;
            if (plan.layoutPreserving) {
                if (constantBackedView(candidate.operands[@"x"].value))
                    return reject(diagnostics,
                        @"H13 cannot lower transposed views backed by constant storage",
                        candidate, @"h13.unsupported-constant-view-source");
                ANEGraphValue *alias = [[ANEGraphValue alloc]
                    initWithName:x.name type:candidate.results[0].type];
                NSNumber *baseOffset = [valueBaseOffsets objectForKey:x];
                if (baseOffset) {
                    [valueBaseOffsets setObject:baseOffset forKey:alias];
                } else {
                    aliases[candidate.results[0].name] = @{
                        @"aliasOf": x.name,
                        @"shape": candidate.results[0].type.shape};
                    [manifestValues addObject:candidate.results[0]];
                }
                [loweredValues setObject:alias forKey:candidate.results[0]];
                continue;
            }
            NSUInteger uses = 0;
            NSString *operandKey = nil;
            ANEGraphOperation *consumer = nil;
            for (ANEGraphOperation *other in sourceOperations) {
                if (other == candidate) continue;
                for (NSString *key in other.operands)
                    if (other.operands[key].value == candidate.results[0]) {
                        ++uses;
                        operandKey = key;
                        consumer = other;
                    }
            }
            for (ANEGraphValue *returnedValue in function.returnValues)
                if (returnedValue == candidate.results[0]) uses = 2;
            BOOL foldable = plan.tailSwap && uses == 1 && consumer &&
                [consumer.operationName isEqualToString:@"matmul"] &&
                ([operandKey isEqualToString:@"x"] ||
                 [operandKey isEqualToString:@"y"]);
            // A [0,2,1,3] transpose of [1,C,P,D] feeding a reshape that
            // merges the transposed pair into one reduction, consumed by one
            // linear, decomposes into C contiguous [P,D] channel-plane
            // matvecs the c-plane accumulation synthesizes — zero data
            // movement, byte-equal to the per-plane direct form.
            BOOL compositeFolded = NO;
            if (!foldable && uses == 1 && consumer &&
                [consumer.operationName isEqualToString:@"reshape"] &&
                x.type.shape.count == 4 &&
                x.type.shape[0].unsignedIntegerValue == 1) {
                const NSUInteger channels =
                    x.type.shape[1].unsignedIntegerValue;
                const NSUInteger planeRows =
                    x.type.shape[2].unsignedIntegerValue;
                const NSUInteger inner =
                    x.type.shape[3].unsignedIntegerValue;
                const NSUInteger rank = x.type.shape.count;
                std::vector<NSUInteger> source(rank);
                BOOL planePerm = YES;
                for (NSUInteger index = 0; index < rank; ++index) {
                    long long axis =
                        [int32TensorElements(candidate.operands[@"perm"].value)[index]
                            longLongValue];
                    if (axis < 0) axis += (long long)rank;
                    source[index] = (NSUInteger)axis;
                }
                if (source[0] != 0 || source[1] != 2 || source[2] != 1 ||
                    source[3] != 3)
                    planePerm = NO;
                NSArray<NSNumber *> *reshaped = consumer.results[0].type.shape;
                NSUInteger reshapeUses = 0;
                ANEGraphOperation *reshapeConsumer = nil;
                for (ANEGraphOperation *other in sourceOperations) {
                    if (other == consumer) continue;
                    for (NSString *key in other.operands)
                        if (other.operands[key].value == consumer.results[0]) {
                            ++reshapeUses;
                            reshapeConsumer = other;
                        }
                }
                BOOL returnedShape = NO;
                for (ANEGraphValue *returnedValue in function.returnValues)
                    if (returnedValue == consumer.results[0]) returnedShape = YES;
                if (planePerm && channels && planeRows && inner &&
                    reshaped.count == 3 &&
                    [reshaped[0] isEqualToNumber:@1] &&
                    [reshaped[1] unsignedIntegerValue] == planeRows &&
                    [reshaped[2] unsignedIntegerValue] == channels * inner &&
                    reshapeUses == 1 && !returnedShape && reshapeConsumer &&
                    [reshapeConsumer.operationName
                        isEqualToString:@"linear"]) {
                    compositeFolds[consumer.results[0].name] = @{
                        @"source": x,
                        @"channels": @(channels),
                        @"planeRows": @(planeRows),
                        @"inner": @(inner)};
                    compositeFolded = YES;
                } else if (planePerm && reshaped.count == 3 &&
                         [reshaped[1] unsignedIntegerValue] == planeRows &&
                         [reshaped[2] unsignedIntegerValue] == channels * inner)
                    return reject(diagnostics,
                        [NSString stringWithFormat:
                            @"H13 transpose→reshape composite over [1,%lu,%lu,%lu] lowers through the channel-plane decomposition only when exactly one linear with a constant weight consumes the merged reduction: this one has %lu consumer%@",
                            (unsigned long)channels, (unsigned long)planeRows,
                            (unsigned long)inner, (unsigned long)reshapeUses,
                            reshapeUses == 1 && reshapeConsumer ?
                                [NSString stringWithFormat:@" ('%@')", reshapeConsumer.operationName] : @""],
                        candidate, @"h13.nonfoldable-transpose");
            }
            if (!foldable && !compositeFolded) {
                // A decoded 1-task transpose program materializes the
                // permuted surface as one whole-op stream.
                ane::h13::ElementwiseShape transposeIn{}, transposeOut{};
                if (transposeParityShapes(candidate, x, candidate.results[0],
                                          &transposeIn, &transposeOut)) {
                    ANEGraphValue *result = [[ANEGraphValue alloc]
                        initWithName:candidate.results[0].name
                        type:candidate.results[0].type];
                    [operations addObject:[[ANEGraphOperation alloc]
                        initWithOperationName:name results:@[result]
                        arguments:arguments attributes:candidate.attributes
                        range:candidate.range]];
                    [manifestValues addObject:result];
                    [loweredValues setObject:result forKey:candidate.results[0]];
                    continue;
                }
                // Exact per-class blocker, so the encoder's transposes each
                // reject with the reason that actually stops them.
                NSString *message = nil;
                if (plan.fastAxisSwapped)
                    message = [NSString stringWithFormat:
                        @"H13 %@ transpose moves the storage-fastest axis, so its view reads rows whose elements sit a non-unit storage stride apart: the NCHW descriptor carries contiguous rows and no surface interpretation expresses the permutation at any size; a consuming matmul's transpose flag is the only decoded mechanism that absorbs one, and this view has %@",
                        plan.tailSwap ? @"a tail-swap" : @"a",
                        uses == 1 && consumer
                            ? [NSString stringWithFormat:
                                  @"a '%@' consumer with no such flag",
                                  consumer.operationName]
                            : @"several consumers or a returned value needing a materialized surface the decoded corpus cannot produce"];
                else if (uses != 1)
                    message = @"H13 transpose with several consumers or a returned value needs a materialized transposed surface, and the decoded corpus holds no data-movement encoder";
                else if (consumer && reshapeFamily(consumer.operationName))
                    message = @"H13 transpose feeding a shape view stays a pure view only when the permutation itself preserves row-major element order, because reshape, squeeze, and expand_dims never reorder elements: this one moves non-unit axes, so the composite is a genuine permutation and the consumer's contiguous-row reads would need chunked runs of the storage-fastest axis, which no single binding slice covers";
                else
                    message = @"H13 transpose that keeps the fastest axis in place but swaps non-unit middle axes could only fold into a consumer that reads its operand with permuted plane and row strides: the descriptor can express the swap, but every decoded add, conv, and broadcast task stream bakes unpermuted operand strides and no decoded encoder permutes surface strides";
                return reject(diagnostics, message,
                    candidate, @"h13.nonfoldable-transpose");
            }
            if (foldable)
                transposeFolds[candidate.results[0].name] = @{
                    @"value": x,
                    @"flag": [operandKey isEqualToString:@"x"]
                        ? @"transpose_x" : @"transpose_y"};
            continue;
        }
        if ([name isEqualToString:@"slice_by_index"]) {
            // A decoded 1-task slice program materializes the narrowed
            // surface as one whole-op stream; constant storage stays on the
            // view path, whose refusals name the exact blocker.
            ane::h13::ElementwiseShape sliceIn{}, sliceOut{};
            if (!constantBackedView(candidate.operands[@"x"].value) &&
                sliceParityShapes(candidate, x, candidate.results[0],
                                  &sliceIn, &sliceOut)) {
                ANEGraphValue *result = [[ANEGraphValue alloc]
                    initWithName:candidate.results[0].name
                    type:candidate.results[0].type];
                [operations addObject:[[ANEGraphOperation alloc]
                    initWithOperationName:name results:@[result]
                    arguments:arguments attributes:candidate.attributes
                    range:candidate.range]];
                [manifestValues addObject:result];
                [loweredValues setObject:result forKey:candidate.results[0]];
                continue;
            }
            H13SliceViewPlan plan{};
            if (!sliceByIndexViewPlan(candidate, diagnostics, &plan)) return NO;
            if (constantBackedView(candidate.operands[@"x"].value))
                return reject(diagnostics,
                    @"H13 cannot lower slice views backed by constant storage",
                    candidate, @"h13.unsupported-constant-view-source");
            NSNumber *baseOffset = [valueBaseOffsets objectForKey:x];
            if (plan.identity && !baseOffset) {
                aliases[candidate.results[0].name] = @{
                    @"aliasOf": x.name,
                    @"shape": candidate.results[0].type.shape};
                [manifestValues addObject:candidate.results[0]];
                ANEGraphValue *alias = [[ANEGraphValue alloc]
                    initWithName:x.name type:candidate.results[0].type];
                [loweredValues setObject:alias forKey:candidate.results[0]];
            } else {
                NSUInteger base = baseOffset.unsignedIntegerValue;
                if (plan.offset > NSUIntegerMax - base ||
                    plan.resultElements > NSUIntegerMax - base)
                    return reject(diagnostics,
                        @"H13 slice view offset overflows",
                        candidate, @"h13.invalid-slice-shape");
                ANEGraphValue *alias = [[ANEGraphValue alloc]
                    initWithName:x.name type:candidate.results[0].type];
                [loweredValues setObject:alias forKey:candidate.results[0]];
                [valueBaseOffsets setObject:@(base + plan.offset) forKey:alias];
            }
            continue;
        }
        if ([name isEqualToString:@"tile"]) {
            // Materialized tile needs a decoded form; the all-ones identity
            // stays the free alias handled below.
            ANEGraphValue *repsValue = candidate.operands[@"reps"].value;
            NSArray<NSNumber *> *repCheck =
                int32TensorElements(repsValue);
            BOOL identityReps = repCheck.count > 0;
            for (NSNumber *rep in repCheck)
                identityReps = identityReps && rep.longLongValue == 1;
            if (!identityReps) {
                ANEGraphValue *result = [[ANEGraphValue alloc]
                    initWithName:candidate.results[0].name
                    type:candidate.results[0].type];
                [operations addObject:[[ANEGraphOperation alloc]
                    initWithOperationName:name results:@[result] arguments:arguments
                    attributes:candidate.attributes range:candidate.range]];
                [manifestValues addObject:result];
                [loweredValues setObject:result forKey:candidate.results[0]];
                continue;
            }
        }
        if ([name isEqualToString:@"pad"] || [name isEqualToString:@"tile"]) {
            BOOL padding = [name isEqualToString:@"pad"];
            if (!identityExtentViewPlan(candidate, padding ? @"pad" : @"reps",
                                        padding ? 0 : 1, diagnostics,
                                        padding ? @"h13.unsupported-pad"
                                                : @"h13.unsupported-tile"))
                return NO;
            if (constantBackedView(candidate.operands[@"x"].value))
                return reject(diagnostics,
                    @"H13 cannot lower identity views backed by constant storage",
                    candidate, @"h13.unsupported-constant-view-source");
            NSNumber *baseOffset = [valueBaseOffsets objectForKey:x];
            if (baseOffset) {
                ANEGraphValue *alias = [[ANEGraphValue alloc]
                    initWithName:x.name type:candidate.results[0].type];
                [valueBaseOffsets setObject:baseOffset forKey:alias];
                [loweredValues setObject:alias forKey:candidate.results[0]];
            } else {
                aliases[candidate.results[0].name] = @{
                    @"aliasOf": x.name,
                    @"shape": candidate.results[0].type.shape};
                [manifestValues addObject:candidate.results[0]];
                ANEGraphValue *alias = [[ANEGraphValue alloc]
                    initWithName:x.name type:candidate.results[0].type];
                [loweredValues setObject:alias forKey:candidate.results[0]];
            }
            continue;
        }

        BOOL reshape = [name isEqualToString:@"reshape"];
        BOOL squeeze = [name isEqualToString:@"squeeze"];
        BOOL expand = [name isEqualToString:@"expand_dims"];
        if ((reshape || squeeze || expand) &&
            compositeFolds[candidate.results[0].name])
            continue;
        if (reshape || squeeze || expand) {
            NSString *parameterName = reshape ? @"shape" : @"axes";
            ANEGraphArgument *parameter = arguments[parameterName];
            BOOL validCount = squeeze ? (candidate.arguments.count == 1 ||
                                         candidate.arguments.count == 2)
                                      : candidate.arguments.count == 2;
            NSUInteger inputElements = 0, resultElements = 0;
            if (!validCount || !x || (!squeeze && !parameter) ||
                (parameter && (parameter.kind != ANEGraphArgumentKindValue ||
                               !constantValue(parameter.value))) ||
                !tensorElementCount(x, &inputElements) ||
                !tensorElementCount(candidate.results[0], &resultElements) ||
                inputElements != resultElements)
                return reject(diagnostics,
                    @"H13 shape aliases require static fp16 or bool input and result shapes with equal element counts and constant shape parameters",
                candidate, @"h13.invalid-shape-alias");
            if (constantBackedView(candidate.operands[@"x"].value))
                return reject(diagnostics,
                    @"H13 cannot lower shape views backed by constant storage",
                    candidate, @"h13.unsupported-constant-view-source");
            ANEGraphValue *alias = [[ANEGraphValue alloc]
                initWithName:x.name type:candidate.results[0].type];
            NSNumber *baseOffset = [valueBaseOffsets objectForKey:x];
            if (baseOffset) {
                [valueBaseOffsets setObject:baseOffset forKey:alias];
            } else {
                aliases[candidate.results[0].name] = @{
                    @"aliasOf": x.name,
                    @"shape": candidate.results[0].type.shape};
                [manifestValues addObject:candidate.results[0]];
            }
            [loweredValues setObject:alias forKey:candidate.results[0]];
            continue;
        }

        ANEGraphValue *result = [[ANEGraphValue alloc]
            initWithName:candidate.results[0].name type:candidate.results[0].type];
        if ([name isEqualToString:@"relu"]) {
            if (candidate.arguments.count != 1 || !x)
                return reject(diagnostics, @"H13 relu requires one x value operand",
                    candidate);
            ane::h13::ElementwiseShape shapes[2];
            NSUInteger shapeCount = parityShapes(result, shapes);
            BOOL nativeRelu = NO;
            for (NSUInteger index = 0; index < shapeCount; ++index)
                nativeRelu = nativeRelu ||
                    ane::h13::supportsElementwise(ane::h13::UnaryOperation::Relu,
                                                  shapes[index]);
            if (nativeRelu) {
                [operations addObject:[[ANEGraphOperation alloc]
                    initWithOperationName:name results:@[result] arguments:arguments
                    attributes:candidate.attributes range:candidate.range]];
            } else {
                NSString *zeroName = [NSString stringWithFormat:@"$h13.%@.zero",
                    result.name];
                ANEGraphValue *zero = [[ANEGraphValue alloc]
                    initWithName:zeroName type:x.type];
                synthesizedConstants[zeroName] = splatFP16(0);
                [operations addObject:binaryOperation(@"maximum", x, zero,
                    result, candidate.range)];
            }
            [manifestValues addObject:result];
        } else if ([name isEqualToString:@"clip"]) {
            if (candidate.arguments.count != 3 || !x)
                return reject(diagnostics,
                    @"H13 clip requires x, alpha, and beta arguments", candidate);
            uint16_t alphaBits = 0, betaBits = 0;
            double alpha = 0.0, beta = 0.0;
            if (!exactFP16Attribute(arguments[@"alpha"], &alphaBits, &alpha) ||
                !exactFP16Attribute(arguments[@"beta"], &betaBits, &beta))
                return reject(diagnostics,
                    @"H13 clip alpha and beta must be finite fp32 scalars exactly representable in fp16",
                    candidate, @"h13.inexact-constant");
            if (alpha > beta)
                return reject(diagnostics,
                    @"H13 clip requires alpha less than or equal to beta",
                    candidate, @"h13.invalid-clip-range");
            NSString *prefix = [NSString stringWithFormat:@"$h13.%@", result.name];
            NSString *alphaName = [prefix stringByAppendingString:@".alpha"];
            NSString *betaName = [prefix stringByAppendingString:@".beta"];
            NSString *lowName = [prefix stringByAppendingString:@".clipped-low"];
            ANEGraphValue *alphaValue = [[ANEGraphValue alloc]
                initWithName:alphaName type:x.type];
            ANEGraphValue *low = [[ANEGraphValue alloc]
                initWithName:lowName type:result.type];
            ANEGraphValue *betaValue = [[ANEGraphValue alloc]
                initWithName:betaName type:result.type];
            synthesizedConstants[alphaName] = splatFP16(alphaBits);
            synthesizedConstants[betaName] = splatFP16(betaBits);
            [operations addObject:binaryOperation(@"maximum", x, alphaValue,
                low, candidate.range)];
            [operations addObject:binaryOperation(@"minimum", low, betaValue,
                result, candidate.range)];
            [manifestValues addObject:low];
            [manifestValues addObject:result];
        } else if ([name isEqualToString:@"matmul"] ||
                   [name isEqualToString:@"linear"]) {
            for (NSString *operand in @[@"x", @"y"]) {
                NSDictionary *fold =
                    transposeFolds[candidate.operands[operand].value.name];
                if (!fold) continue;
                arguments[operand] = valueArgument(fold[@"value"], candidate.range);
                arguments[fold[@"flag"]] = booleanArgument(
                    !boolean(arguments[fold[@"flag"]], YES), candidate.range);
                if ([operand isEqualToString:@"x"])
                    x = arguments[@"x"].value;
            }
            BOOL linear = [name isEqualToString:@"linear"];
            ANEGraphValue *weight = linear ? arguments[@"weight"].value
                                           : arguments[@"y"].value;
            ANEGraphValue *bias = linear ? arguments[@"bias"].value : nil;
            if (linear && (!weight || !constantValue(weight)))
                return reject(diagnostics,
                    @"H13 linear requires a constant weight tensor", candidate,
                    @"h13.linear-nonconstant-weight");
            if (linear && bias && !constantValue(bias))
                return reject(diagnostics,
                    @"H13 linear requires a constant bias tensor", candidate,
                    @"h13.linear-nonconstant-bias");

            NSUInteger reduction = 0, rows = 0, columns = 0;
            BOOL transposeX = linear ? NO : boolean(arguments[@"transpose_x"], YES);
            BOOL geometry = matmulGeometry(x, result, transposeX,
                                           &reduction, &rows, &columns);
            if (linear && (candidate.arguments.count < 2 ||
                           candidate.arguments.count > 3 || !geometry ||
                           !tensor(weight, @[@(columns), @(reduction)])))
                return reject(diagnostics,
                    @"H13 linear requires positive fp16 x rows, constant [N,K] weight, optional constant [N] bias, and a matching output shape",
                    candidate, @"h13.invalid-linear");

            // The decoded encoder corpus covers these rank-3 linears as one
            // whole-tensor Apple program; pass the operation through instead
            // of rewriting it into the channel-plane decomposition.
            if (linear && geometry &&
                (!bias || blobBackedConstant(bias))) {
                ane::h13::LinearBiasMode biasMode = ane::h13::LinearBiasMode::None;
                if (bias) {
                    biasMode = ane::h13::LinearBiasMode::Uniform;
                    NSData *biasData = resolvedConstants[bias.name];
                    if (!biasData) {
                        biasData = [ANEBlobResolver loadConstantForOperation:
                            bias.producer expectedBytes:columns * 2
                            modelRoot:modelRoot diagnostics:diagnostics];
                        if (!biasData) return NO;
                        resolvedConstants[bias.name] = biasData;
                    }
                    const uint16_t *halves =
                        static_cast<const uint16_t *>(biasData.bytes);
                    for (NSUInteger index = 1; index < columns; ++index)
                        if (halves[index] != halves[0]) {
                            biasMode = ane::h13::LinearBiasMode::Block;
                            break;
                        }
                }
                if (ane::h13::supportsLinearParity(
                        static_cast<std::uint32_t>(rows),
                        static_cast<std::uint32_t>(reduction),
                        static_cast<std::uint32_t>(columns), biasMode)) {
                    [operations addObject:[[ANEGraphOperation alloc]
                        initWithOperationName:name results:@[result]
                        arguments:arguments attributes:candidate.attributes
                        range:candidate.range]];
                    [manifestValues addObject:result];
                    continue;
                }
            }

            // transpose→reshape→linear composite: lower as C contiguous
            // [1,P,D] channel-plane matvecs whose partials accumulate with
            // adds and the expanded bias, byte-equal to the per-plane direct
            // form (per-plane slices of a rank-4 [1,C,P,D] input are the
            // offset views the slice lowering already emits).
            NSDictionary *composite = compositeFolds[x.name];
            if (linear && composite) {
                const NSUInteger channels =
                    [composite[@"channels"] unsignedIntegerValue];
                const NSUInteger planeRows =
                    [composite[@"planeRows"] unsignedIntegerValue];
                const NSUInteger inner =
                    [composite[@"inner"] unsignedIntegerValue];
                ANEGraphValue *source = composite[@"source"];
                if (reduction != channels * inner || rows != planeRows ||
                    columns > NSUIntegerMax / reduction ||
                    rows > NSUIntegerMax / columns)
                    return reject(diagnostics,
                        @"H13 channel-plane linear requires the merged reduction to equal the transposed channel and inner extents",
                        candidate, @"h13.invalid-linear");
                NSData *weights = resolvedConstants[weight.name];
                if (!weights) {
                    weights = [ANEBlobResolver loadConstantForOperation:weight.producer
                        expectedBytes:columns * reduction * 2
                        modelRoot:modelRoot diagnostics:diagnostics];
                    if (!weights) return NO;
                    resolvedConstants[weight.name] = weights;
                }
                NSData *biasData = nil;
                if (bias) {
                    biasData = perChannelConstantData(bias, columns, modelRoot,
                        diagnostics, resolvedConstants);
                    if (!biasData) return NO;
                }
                const NSUInteger sourceBase =
                    [[valueBaseOffsets objectForKey:source]
                        unsignedIntegerValue];
                if (channels > NSUIntegerMax / (planeRows * inner) ||
                    sourceBase > NSUIntegerMax - channels * planeRows * inner)
                    return reject(diagnostics,
                        @"H13 channel-plane offsets overflow", candidate,
                        @"h13.invalid-linear");
                ANEValueType *planeType = [[ANEValueType alloc]
                    initWithKind:ANEValueTypeKindTensor
                    elementType:ANEElementTypeFP16
                    shape:@[@1, @(planeRows), @(inner)]];
                ANEValueType *chunkWeightType = [[ANEValueType alloc]
                    initWithKind:ANEValueTypeKindTensor
                    elementType:ANEElementTypeFP16
                    shape:@[@(columns), @(inner)]];
                const uint16_t *weightWords =
                    static_cast<const uint16_t *>(weights.bytes);
                NSMutableArray<ANEGraphValue *> *partials =
                    [NSMutableArray arrayWithCapacity:channels];
                for (NSUInteger channel = 0; channel < channels; ++channel) {
                    NSString *prefix = [NSString stringWithFormat:
                        @"$h13.%@.plane%lu", result.name,
                        (unsigned long)channel];
                    ANEGraphValue *plane = [[ANEGraphValue alloc]
                        initWithName:source.name type:planeType];
                    [valueBaseOffsets setObject:
                        @(sourceBase + channel * planeRows * inner)
                        forKey:plane];
                    NSString *chunkWeightName =
                        [prefix stringByAppendingString:@".weight"];
                    ANEGraphValue *chunkWeight = [[ANEGraphValue alloc]
                        initWithName:chunkWeightName type:chunkWeightType];
                    NSMutableData *chunk =
                        [NSMutableData dataWithLength:columns * inner * 2];
                    uint16_t *chunkWords =
                        static_cast<uint16_t *>(chunk.mutableBytes);
                    for (NSUInteger column = 0; column < columns; ++column)
                        for (NSUInteger element = 0; element < inner; ++element)
                            chunkWords[column * inner + element] =
                                weightWords[column * reduction +
                                    channel * inner + element];
                    synthesizedConstants[chunkWeightName] = chunk;
                    ANEGraphValue *partial =
                        (channels == 1 && !bias) ? result : [[ANEGraphValue alloc]
                            initWithName:[prefix stringByAppendingString:@".matmul"]
                                type:result.type];
                    NSDictionary *matmulArguments = @{
                        @"x": valueArgument(plane, candidate.range),
                        @"y": valueArgument(chunkWeight, candidate.range),
                        @"transpose_x": booleanArgument(NO, candidate.range),
                        @"transpose_y": booleanArgument(YES, candidate.range),
                    };
                    [operations addObject:[[ANEGraphOperation alloc]
                        initWithOperationName:@"matmul" results:@[partial]
                        arguments:matmulArguments attributes:@{}
                        range:candidate.range]];
                    [manifestValues addObject:partial];
                    [partials addObject:partial];
                }
                ANEGraphValue *accumulator = partials[0];
                for (NSUInteger channel = 1; channel < channels; ++channel) {
                    const BOOL finalAccumulation = channel + 1 == channels;
                    NSString *sumName = [NSString stringWithFormat:
                        @"$h13.%@.accum%lu", result.name,
                        (unsigned long)channel];
                    ANEGraphValue *sum = (finalAccumulation && !bias)
                        ? result : [[ANEGraphValue alloc]
                            initWithName:sumName type:result.type];
                    [operations addObject:binaryOperation(@"add", accumulator,
                        partials[channel], sum, candidate.range)];
                    [manifestValues addObject:sum];
                    accumulator = sum;
                }
                if (bias) {
                    NSString *biasName = [NSString stringWithFormat:
                        @"$h13.%@.bias", result.name];
                    ANEGraphValue *expandedBias = [[ANEGraphValue alloc]
                        initWithName:biasName type:result.type];
                    NSMutableData *expanded =
                        [NSMutableData dataWithLength:rows * columns * 2];
                    for (NSUInteger row = 0; row < rows; ++row)
                        std::memcpy(static_cast<uint8_t *>(expanded.mutableBytes) +
                            row * columns * 2, biasData.bytes, columns * 2);
                    synthesizedConstants[biasName] = expanded;
                    [operations addObject:binaryOperation(@"add", accumulator,
                        expandedBias, result, candidate.range)];
                }
                [manifestValues addObject:result];
                [loweredValues setObject:result forKey:candidate.results[0]];
                continue;
            }

            if (linear && bias && rows > 1) {
                NSData *biasData = linearBiasData(bias, columns, modelRoot,
                    diagnostics, resolvedConstants);
                if (!biasData) return NO;
                ANEValueType *rowInputType = [[ANEValueType alloc]
                    initWithKind:ANEValueTypeKindTensor
                    elementType:ANEElementTypeFP16 shape:@[@1, @(reduction)]];
                ANEValueType *rowOutputType = [[ANEValueType alloc]
                    initWithKind:ANEValueTypeKindTensor
                    elementType:ANEElementTypeFP16 shape:@[@1, @(columns)]];
                for (NSUInteger row = 0; row < rows; ++row) {
                    ANEGraphValue *rowInput = [[ANEGraphValue alloc]
                        initWithName:x.name type:rowInputType];
                    NSDictionary *matmulArguments = @{
                        @"x": valueArgument(rowInput, candidate.range),
                        @"y": valueArgument(weight, candidate.range),
                        @"transpose_x": booleanArgument(NO, candidate.range),
                        @"transpose_y": booleanArgument(YES, candidate.range),
                    };
                    NSString *prefix = [NSString stringWithFormat:@"%@.row%lu",
                        result.name, (unsigned long)row];
                    NSString *matrixName = [NSString stringWithFormat:@"$h13.%@.linear",
                        prefix];
                    ANEGraphValue *matrix = [[ANEGraphValue alloc]
                        initWithName:matrixName type:rowOutputType];
                    if (reduction > 512 &&
                        !matmulParityCovered(1, reduction, columns, NO, YES, NO, !chainSchedule)) {
                        NSUInteger chunks = (reduction - 1) / 512 + 1;
                        NSMutableArray<ANEGraphValue *> *partials =
                            [NSMutableArray arrayWithCapacity:chunks];
                        for (NSUInteger chunk = 0; chunk < chunks; ++chunk) {
                            NSString *partialName = [NSString stringWithFormat:
                                @"$h13.%@.partial%lu", prefix,
                                (unsigned long)chunk];
                            ANEGraphValue *partial = [[ANEGraphValue alloc]
                                initWithName:partialName type:rowOutputType];
                            ANEGraphOperation *partialOperation = [[ANEGraphOperation alloc]
                                initWithOperationName:@"matmul" results:@[partial]
                                arguments:matmulArguments attributes:@{}
                                range:candidate.range];
                            [reductionOffsets setObject:@(row * reduction + chunk * 512)
                                                 forKey:partialOperation];
                            [reductionCounts setObject:@(MIN((NSUInteger)512,
                                reduction - chunk * 512)) forKey:partialOperation];
                            [operations addObject:partialOperation];
                            [manifestValues addObject:partial];
                            [partials addObject:partial];
                        }
                        ANEGraphValue *accumulator = partials[0];
                        for (NSUInteger chunk = 1; chunk < chunks; ++chunk) {
                            BOOL final = chunk + 1 == chunks;
                            NSString *sumName = [NSString stringWithFormat:
                                @"$h13.%@.accum%lu", prefix,
                                (unsigned long)chunk];
                            ANEGraphValue *sum = final ? matrix : [[ANEGraphValue alloc]
                                initWithName:sumName type:rowOutputType];
                            [operations addObject:binaryOperation(@"add", accumulator,
                                partials[chunk], sum, candidate.range)];
                            [manifestValues addObject:sum];
                            accumulator = sum;
                        }
                        [chunkedAccumulations addObject:matrix.name];
                    } else {
                        ANEGraphOperation *matmulOperation = [[ANEGraphOperation alloc]
                            initWithOperationName:@"matmul" results:@[matrix]
                            arguments:matmulArguments attributes:@{}
                            range:candidate.range];
                        [reductionOffsets setObject:@(row * reduction)
                                             forKey:matmulOperation];
                        [operations addObject:matmulOperation];
                        [manifestValues addObject:matrix];
                    }
                    NSString *biasName = [NSString stringWithFormat:@"$h13.%@.bias",
                        prefix];
                    ANEGraphValue *expandedBias = [[ANEGraphValue alloc]
                        initWithName:biasName type:rowOutputType];
                    synthesizedConstants[biasName] = biasData;
                    ANEGraphValue *rowOutput = [[ANEGraphValue alloc]
                        initWithName:result.name type:rowOutputType];
                    ANEGraphOperation *add = binaryOperation(@"add", matrix,
                        expandedBias, rowOutput, candidate.range);
                    [outputBaseOffsets setObject:@(row * columns) forKey:add];
                    [operations addObject:add];
                }
                [manifestValues addObject:result];
                [loweredValues setObject:result forKey:candidate.results[0]];
                continue;
            }

            ANEGraphValue *matmulResult = result;
            if (linear && bias) {
                NSString *name = [NSString stringWithFormat:@"$h13.%@.linear",
                    result.name];
                matmulResult = [[ANEGraphValue alloc]
                    initWithName:name type:result.type];
            }
            NSDictionary<NSString *, ANEGraphArgument *> *matmulArguments = arguments;
            if (linear)
                matmulArguments = @{
                    @"x": valueArgument(x, candidate.range),
                    @"y": valueArgument(weight, candidate.range),
                    @"transpose_x": booleanArgument(NO, candidate.range),
                    @"transpose_y": booleanArgument(YES, candidate.range),
                };
            const BOOL runtimeWeight = !linear && weight &&
                !constantValue(weight) && !synthesizedConstants[weight.name];
            // A decoded batched geometry must reach the whole-op batched
            // encoder: its reduction is the batch's inner span, and slicing
            // it per 512 columns would dissolve the per-batch task groups.
            ane::h13::BatchedMatmulShape batchedShape{};
            const BOOL batchedCovered = !linear &&
                batchedMatmulParse(candidate, &batchedShape) ==
                    H13BatchedParseYes &&
                ane::h13::supportsBatchedMatmul(batchedShape);
            if (geometry && !runtimeWeight && !batchedCovered &&
                reduction > 512 &&
                !matmulParityCovered(rows, reduction, columns, transposeX, YES,
                                     NO, !chainSchedule)) {
                NSUInteger chunks = (reduction - 1) / 512 + 1;
                NSMutableArray<ANEGraphValue *> *partials =
                    [NSMutableArray arrayWithCapacity:chunks];
                for (NSUInteger chunk = 0; chunk < chunks; ++chunk) {
                    NSString *partialName = [NSString stringWithFormat:
                        @"$h13.%@.partial%lu", result.name, (unsigned long)chunk];
                    ANEGraphValue *partial = [[ANEGraphValue alloc]
                        initWithName:partialName type:matmulResult.type];
                    ANEGraphOperation *partialOperation = [[ANEGraphOperation alloc]
                        initWithOperationName:@"matmul" results:@[partial]
                        arguments:matmulArguments attributes:@{} range:candidate.range];
                    [reductionOffsets setObject:@(chunk * 512)
                                         forKey:partialOperation];
                    [reductionCounts setObject:@(MIN((NSUInteger)512,
                        reduction - chunk * 512)) forKey:partialOperation];
                    [operations addObject:partialOperation];
                    [manifestValues addObject:partial];
                    [partials addObject:partial];
                }
                ANEGraphValue *accumulator = partials[0];
                for (NSUInteger chunk = 1; chunk < chunks; ++chunk) {
                    BOOL final = chunk + 1 == chunks;
                    NSString *sumName = [NSString stringWithFormat:
                        @"$h13.%@.accum%lu", result.name, (unsigned long)chunk];
                    ANEGraphValue *sum = final ? matmulResult : [[ANEGraphValue alloc]
                        initWithName:sumName type:matmulResult.type];
                    [operations addObject:binaryOperation(@"add", accumulator,
                        partials[chunk], sum, candidate.range)];
                    [manifestValues addObject:sum];
                    accumulator = sum;
                }
                [chunkedAccumulations addObject:matmulResult.name];
            } else {
                ANEGraphOperation *matmulOperation = [[ANEGraphOperation alloc]
                    initWithOperationName:@"matmul" results:@[matmulResult]
                    arguments:matmulArguments attributes:@{} range:candidate.range];
                [operations addObject:matmulOperation];
                [manifestValues addObject:matmulResult];
            }

            if (linear && bias) {
                NSData *biasData = linearBiasData(bias, columns, modelRoot,
                    diagnostics, resolvedConstants);
                if (!biasData) return NO;
                if (rows * columns > NSUIntegerMax / 2)
                    return reject(diagnostics, @"H13 linear output size overflows",
                        candidate, @"h13.invalid-linear-bias");
                NSMutableData *expanded =
                    [NSMutableData dataWithLength:rows * columns * 2];
                for (NSUInteger row = 0; row < rows; ++row)
                    std::memcpy(static_cast<uint8_t *>(expanded.mutableBytes) +
                        row * columns * 2, biasData.bytes, columns * 2);
                NSString *biasName = [NSString stringWithFormat:@"$h13.%@.bias",
                    result.name];
                ANEGraphValue *expandedBias = [[ANEGraphValue alloc]
                    initWithName:biasName type:result.type];
                synthesizedConstants[biasName] = expanded;
                [operations addObject:binaryOperation(@"add", matmulResult,
                    expandedBias, result, candidate.range)];
                [manifestValues addObject:result];
            }
        } else {
            [operations addObject:[[ANEGraphOperation alloc]
                initWithOperationName:name results:@[result] arguments:arguments
                attributes:candidate.attributes range:candidate.range]];
            [manifestValues addObject:result];
        }
        [loweredValues setObject:result forKey:candidate.results[0]];
    }

    NSMutableSet<NSString *> *inputNames = [NSMutableSet set];
    for (ANEGraphValue *input in function.inputs) [inputNames addObject:input.name];
    if (!operations.count) {
        for (ANEGraphValue *logical in function.returnValues) {
            ANEGraphValue *returned = [loweredValues objectForKey:logical];
            if (returned && [inputNames containsObject:returned.name])
                return reject(diagnostics,
                    @"H13 cannot return an alias of a function input because no program produces it",
                    lastSourceOperation, @"h13.returned-input-alias");
        }
        return reject(diagnostics, @"H13 requires at least one encoded operation");
    }
    ANEGraphOperation *lastOperation = operations.lastObject;
    NSMutableSet<NSString *> *producedStorageNames = [NSMutableSet set];
    for (ANEGraphOperation *candidate in operations)
        for (ANEGraphValue *value in candidate.results)
            [producedStorageNames addObject:value.name];
    NSMutableArray<ANEGraphValue *> *returnedValues = [NSMutableArray array];
    NSMutableSet<NSString *> *returnedSourceNames = [NSMutableSet set];
    NSMutableSet<NSString *> *returnedStorageNameSet = [NSMutableSet set];
    for (ANEGraphValue *logical in function.returnValues) {
        ANEGraphValue *returned = [loweredValues objectForKey:logical] ?: logical;
        if ([inputNames containsObject:returned.name])
            return reject(diagnostics,
                @"H13 cannot return an alias of a function input because no program produces it",
                logical.producer ?: lastSourceOperation, @"h13.returned-input-alias");
        if (![producedStorageNames containsObject:returned.name])
            return reject(diagnostics,
                @"H13 logical result has no encoded physical output",
                logical.producer ?: lastSourceOperation,
                @"h13.unsupported-logical-result-storage");
        [returnedValues addObject:returned];
        [returnedSourceNames addObject:logical.name];
        [returnedStorageNameSet addObject:returned.name];
    }

    NSMutableOrderedSet<NSString *> *returnedStorageNames =
        [NSMutableOrderedSet orderedSet];
    for (ANEGraphOperation *candidate in operations)
        for (ANEGraphValue *value in candidate.results)
            if ([returnedStorageNameSet containsObject:value.name])
                [returnedStorageNames addObject:value.name];
    NSMutableSet<NSString *> *intermediateStorageNames = [NSMutableSet set];
    NSMutableArray<NSString *> *intermediateNames = [NSMutableArray array];
    for (ANEGraphValue *value in manifestValues) {
        BOOL alias = aliases[value.name] != nil;
        if ([returnedSourceNames containsObject:value.name] ||
            (!alias && [returnedStorageNameSet containsObject:value.name]))
            continue;
        [intermediateNames addObject:value.name];
        if (!alias) [intermediateStorageNames addObject:value.name];
    }
    NSMutableDictionary<NSString *, NSArray<NSNumber *> *> *outputShapes =
        [NSMutableDictionary dictionary];
    for (NSString *storageName in returnedStorageNames) {
        for (ANEGraphValue *value in manifestValues)
            if ([value.name isEqualToString:storageName] && !aliases[value.name]) {
                outputShapes[storageName] = value.type.shape;
                break;
            }
        if (!outputShapes[storageName])
            return reject(diagnostics, @"H13 returned value has no physical output storage",
                          lastSourceOperation,
                          @"h13.unsupported-logical-result-storage");
    }

    NSMutableArray<NSDictionary *> *programRecords = [NSMutableArray array];
    NSMutableArray<NSData *> *payloads = [NSMutableArray array];
    std::vector<ane::h13::Program> chainPrograms;
    NSMutableArray<NSNumber *> *dispatchPlan = [NSMutableArray array];
    NSMutableArray<NSDictionary *> *taskRecords = nil;
    NSDictionary *scratch = nil;
    NSMutableDictionary<NSString *, NSDictionary *> *tensors =
        [NSMutableDictionary dictionary];
    for (ANEGraphValue *input in function.inputs)
        recordTensor(tensors, input, input.type.shape, @"input");
    NSArray<NSString *> *binaryNames =
        @[@"add", @"mul", @"maximum", @"minimum", @"sub", @"real_div"];
    NSMutableSet<NSString *> *regionConsumers = [NSMutableSet set];
    NSMutableArray<NSDictionary *> *regions = [NSMutableArray array];
    ANEGraphOperation *operation = nil;
    try {
        for (operation in operations) {
            if ([regionConsumers containsObject:operation.results[0].name]) continue;
            NSUInteger sliceCount = 1, inputSliceElements = 0,
                inputPhysicalElements = 0, outputSliceElements = 0,
                outputPhysicalElements = 0, outputChunks = 1;
            BOOL matmul = [operation.operationName isEqualToString:@"matmul"];
            BOOL linearParity =
                [operation.operationName isEqualToString:@"linear"];
            BOOL ffnChain =
                [operation.operationName isEqualToString:@"ffn-chain"];
            ANEGraphValue *secondOperand = operation.operands[@"y"].value;
            BOOL runtimeWeight = matmul && secondOperand &&
                !constantValue(secondOperand) &&
                !synthesizedConstants[secondOperand.name];
            ane::h13::BatchedMatmulShape batchedShape{};
            BOOL batched = matmul &&
                batchedMatmulParse(operation, &batchedShape) == H13BatchedParseYes &&
                ane::h13::supportsBatchedMatmul(batchedShape);
            NSString *booleanOp = operation.operationName;
            BOOL booleanLowered = [booleanOp isEqualToString:@"less"] ||
                [booleanOp isEqualToString:@"floor"] ||
                [booleanOp isEqualToString:@"select"] ||
                [booleanOp isEqualToString:@"floor_div"];
            BOOL matvecParity = !batched && matmul &&
                matmulParityShape(operation.operands[@"x"].value, operation.results[0],
                    boolean(operation.arguments[@"transpose_x"], YES),
                    runtimeWeight
                        ? boolean(operation.arguments[@"transpose_y"], YES) : YES,
                    runtimeWeight, !chainSchedule, nullptr);
            H13ParityPlan operationPlan{};
            H13NormPlan normalizationPlan{};
            H13BroadcastPlan broadcastPlanned{};
            ANEGraphValue *broadcastConstant = nil;
            BOOL broadcast = broadcastPlan(operation, synthesizedConstants,
                                           !chainSchedule, &broadcastPlanned, &broadcastConstant);
            BOOL parity = !broadcast &&
                parityPlan(operation, synthesizedConstants, !chainSchedule, &operationPlan);
            H13ConvPlan convolutionPlan{};
            BOOL normalization = !parity && !broadcast &&
                normParityPlan(operation, &normalizationPlan);
            BOOL convolution = !parity && !broadcast && !normalization &&
                convParityPlan(operation, &convolutionPlan);
            BOOL tiled = [operation.operationName isEqualToString:@"tile"];
            ane::h13::ElementwiseShape transposeIn{}, transposeOut{};
            BOOL transposeParity =
                [operation.operationName isEqualToString:@"transpose"] &&
                transposeParityShapes(operation, operation.operands[@"x"].value,
                    operation.results[0], &transposeIn, &transposeOut);
            ane::h13::ElementwiseShape sliceIn{}, sliceOut{};
            BOOL sliceParity =
                [operation.operationName isEqualToString:@"slice_by_index"] &&
                sliceParityShapes(operation, operation.operands[@"x"].value,
                    operation.results[0], &sliceIn, &sliceOut);
            if (tiled) {
                // The tile program reads its whole input surface and writes
                // its whole output surface in one program, so both slice
                // spans are the whole-tensor element counts.
                NSUInteger xElements = 0;
                NSArray<NSNumber *> *xType = operation.operands[@"x"].value.type.shape;
                if (xType.count) {
                    xElements = 1;
                    for (NSNumber *dimension in xType)
                        xElements *= dimension.unsignedIntegerValue;
                }
                NSUInteger oElements = 0;
                NSArray<NSNumber *> *oType = operation.results[0].type.shape;
                if (oType.count) {
                    oElements = 1;
                    for (NSNumber *dimension in oType)
                        oElements *= dimension.unsignedIntegerValue;
                }
                if (xElements)
                    inputSliceElements = inputPhysicalElements = xElements;
                if (oElements)
                    outputSliceElements = outputPhysicalElements = oElements;
            } else if (booleanLowered) {
                // Whole-tensor shapes regardless of the result dtype:
                // bool compare results and select conds count elements,
                // not bytes.
                NSArray<NSNumber *> *resultShape = operation.results[0].type.shape;
                if (resultShape.count) {
                    NSUInteger elements = 1;
                    for (NSNumber *dimension in resultShape)
                        elements *= dimension.unsignedIntegerValue;
                    if (elements)
                        inputSliceElements = outputSliceElements =
                            inputPhysicalElements = outputPhysicalElements =
                                elements;
                }
            } else if (linearParity || ffnChain || transposeParity ||
                       sliceParity) {
                // Whole-tensor programs: one input slice and one output
                // slice covering the rank-3 surfaces.
                NSUInteger xElements = 1;
                NSArray<NSNumber *> *xShape =
                    operation.operands[@"x"].value.type.shape;
                for (NSNumber *dimension in xShape)
                    xElements *= dimension.unsignedIntegerValue;
                NSUInteger oElements = 1;
                NSArray<NSNumber *> *oShape = operation.results[0].type.shape;
                for (NSNumber *dimension in oShape)
                    oElements *= dimension.unsignedIntegerValue;
                inputSliceElements = inputPhysicalElements = xElements;
                outputSliceElements = outputPhysicalElements = oElements;
            } else if (parity) {
                inputSliceElements = outputSliceElements =
                    inputPhysicalElements = outputPhysicalElements =
                        operationPlan.elements;
            } else if (broadcast) {
                // Each operand covers its own whole tensor: a broadcast reads
                // fewer elements from y than it writes to the result.
                NSUInteger elements = 0;
                if (!tensorElementCount(operation.results[0], &elements))
                    return reject(diagnostics,
                        @"H13 broadcast result must have a positive static shape",
                        operation);
                outputSliceElements = outputPhysicalElements = elements;
                if (broadcastConstant)
                    recordTensor(tensors, broadcastConstant,
                                 broadcastConstant.type.shape, @"constant");
            } else if (normalization) {
                inputSliceElements = inputPhysicalElements =
                    normalizationPlan.inputElements;
                outputSliceElements = outputPhysicalElements =
                    normalizationPlan.outputElements;
            } else if (convolution) {
                NSUInteger elements = 0;
                if (!tensorElementCount(operation.operands[@"x"].value, &elements))
                    return reject(diagnostics,
                        @"H13 convolution input must have a positive static shape",
                        operation);
                inputSliceElements = inputPhysicalElements = elements;
                if (!tensorElementCount(operation.results[0], &elements))
                    return reject(diagnostics,
                        @"H13 convolution result must have a positive static shape",
                        operation);
                outputSliceElements = outputPhysicalElements = elements;
                recordTensor(tensors, convolutionPlan.weight,
                             convolutionPlan.weight.type.shape, @"constant");
                if (convolutionPlan.bias)
                    recordTensor(tensors, convolutionPlan.bias,
                                 convolutionPlan.bias.type.shape, @"constant");
            } else if ([binaryNames containsObject:operation.operationName]) {
                NSUInteger elements = 0;
                if (tensorElementCount(operation.results[0], &elements)) {
                    sliceCount = (elements - 1) / 64 + 1;
                    inputPhysicalElements = outputPhysicalElements = 64;
                }
            } else if (matmul) {
                NSUInteger reduction = 0, rows = 0, columns = 0;
                BOOL transposeX = boolean(operation.arguments[@"transpose_x"], YES);
                if (matmulGeometry(operation.operands[@"x"].value, operation.results[0],
                                   transposeX, &reduction, &rows, &columns)) {
                    if (batched) {
                        inputSliceElements = inputPhysicalElements =
                            (NSUInteger)batchedShape.batch *
                            batchedShape.rows * batchedShape.reduction;
                        outputSliceElements = outputPhysicalElements =
                            (NSUInteger)batchedShape.batch *
                            batchedShape.rows * batchedShape.columns;
                    } else if (matvecParity) {
                        inputSliceElements = inputPhysicalElements =
                            rows * reduction;
                        outputSliceElements = outputPhysicalElements =
                            rows * columns;
                    } else {
                        outputChunks = (columns - 1) / 512 + 1;
                        if (rows > NSUIntegerMax / outputChunks)
                            return reject(diagnostics,
                                @"H13 matmul program count overflows", operation);
                        sliceCount = rows * outputChunks;
                        NSNumber *count = [reductionCounts objectForKey:operation];
                        inputSliceElements =
                            count ? count.unsignedIntegerValue : reduction;
                        // The matvec task moves 512 input channels at both reductions, so the physical surface is 512 channels.
                        inputPhysicalElements = 512;
                        outputPhysicalElements = 512;
                    }
                }
                if (fp16Tensor(secondOperand) && !runtimeWeight)
                    recordTensor(tensors, secondOperand,
                                 secondOperand.type.shape, @"constant");
            } else if (linearParity) {
                if (operation.operands[@"weight"].value)
                    recordTensor(tensors, operation.operands[@"weight"].value,
                                 operation.operands[@"weight"].value.type.shape,
                                 @"constant");
                if (operation.operands[@"bias"].value)
                    recordTensor(tensors, operation.operands[@"bias"].value,
                                 operation.operands[@"bias"].value.type.shape,
                                 @"constant");
            } else if (ffnChain) {
                for (NSString *key in @[@"weight1", @"bias1", @"weight2",
                                        @"bias2"]) {
                    ANEGraphValue *constant = operation.operands[key].value;
                    if (constant)
                        recordTensor(tensors, constant, constant.type.shape,
                                     @"constant");
                }
            }

            for (NSUInteger sliceIndex = 0; sliceIndex < sliceCount; ++sliceIndex) {
                NSUInteger inputOffset = 0, outputOffset = 0;
                if (matmul && !batched) {
                    NSUInteger reduction = 0, rows = 0, geometryColumns = 0;
                    BOOL transposeX = boolean(operation.arguments[@"transpose_x"], YES);
                    matmulGeometry(operation.operands[@"x"].value, operation.results[0],
                                   transposeX, &reduction, &rows, &geometryColumns);
                    NSUInteger columns = operation.results[0].type.shape.lastObject.unsignedIntegerValue;
                    NSNumber *reductionOffset = [reductionOffsets objectForKey:operation];
                    if (matvecParity) {
                        inputOffset = reductionOffset.unsignedIntegerValue;
                    } else {
                        NSUInteger row = sliceIndex / outputChunks;
                        NSUInteger chunk = sliceIndex % outputChunks;
                        inputOffset =
                            row * reduction + reductionOffset.unsignedIntegerValue;
                        outputOffset = row * columns + chunk * 512;
                        outputSliceElements =
                            MIN((NSUInteger)512, columns - chunk * 512);
                    }
                } else if (batched) {
                    inputOffset = outputOffset = 0;
                } else if (parity || broadcast || normalization || convolution) {
                    outputOffset =
                        [[outputBaseOffsets objectForKey:operation] unsignedIntegerValue];
                } else if (!tiled && !booleanLowered && !linearParity &&
                       !ffnChain && !transposeParity && !sliceParity) {
                    // Tile, boolean, rank-3 linear, FFN-chain, and slice
                    // programs already sized whole-tensor spans. The 64-lane
                    // split must not clobber them.
                    inputOffset = sliceIndex * 64;
                    outputOffset = inputOffset +
                        [[outputBaseOffsets objectForKey:operation] unsignedIntegerValue];
                    NSUInteger elements = 0;
                    if (tensorElementCount(operation.results[0], &elements))
                        inputSliceElements = outputSliceElements =
                            MIN((NSUInteger)64, elements - inputOffset);
                }
                ane::h13::Program program;
                NSArray<ANEGraphValue *> *inputs = nil;
                ANEGraphValue *constantInput = nil;
                NSData *constantData = nil;
                NSString *manifestOperation = nil;
                if (!lowerOperation(operation, modelRoot, diagnostics, !chainSchedule,
                                    synthesizedConstants, resolvedConstants,
                                    inputOffset, inputSliceElements, outputOffset,
                                    program, &inputs, &constantInput, &constantData,
                                    &manifestOperation))
                    return NO;
                ANEGraphValue *outputValue = operation.results[0];
                ANEGraphOperation *reluConsumer = nil;
                if ([schedule isEqualToString:@"regions"] && sliceCount == 1 &&
                    outputOffset == 0 && ![outputBaseOffsets objectForKey:operation] &&
                    ((matvecParity && !runtimeWeight) || (parity && !operationPlan.unary))) {
                    ANEGraphOperation *candidate = ANEReluRegionConsumer(
                        operations, operation, returnedStorageNameSet);
                    if (candidate) {
                        auto fused = program;
                        try {
                            if (matvecParity)
                                ane::h13::fuseMatmulPostOperation(fused, ane::h13::PostOperation::Relu);
                            else
                                ane::h13::fuseElementwisePostOperation(fused, ane::h13::PostOperation::Relu);
                            program = std::move(fused);
                            reluConsumer = candidate;
                            outputValue = candidate.results[0];
                            [regionConsumers addObject:outputValue.name];
                            [intermediateNames removeObject:operation.results[0].name];
                            [intermediateStorageNames removeObject:operation.results[0].name];
                        } catch (const std::invalid_argument &) {
                            // This legal standalone encoding has no decoded epilogue.
                        }
                    }
                }
                std::vector<std::uint8_t> anec = ane::h13::encodeANEC(program);
                NSMutableArray *inputRecords =
                    [NSMutableArray arrayWithCapacity:inputs.count];
                NSMutableDictionary *constantInputs = [NSMutableDictionary dictionary];
                for (NSUInteger index = 0; index < inputs.count; ++index) {
                    ANEGraphValue *input = inputs[index];
                    NSArray<NSNumber *> *fullShape = input == constantInput
                        ? operation.results[0].type.shape : input.type.shape;
                    BOOL intermediate = input != constantInput &&
                        [intermediateStorageNames containsObject:input.name];
                    NSString *role = input == constantInput ? @"constant"
                        : (intermediate ? @"intermediate" : @"input");
                    recordTensor(tensors, input, fullShape, role);
                    NSUInteger fullElements = 1;
                    for (NSNumber *dimension in fullShape)
                        fullElements *= dimension.unsignedIntegerValue;
                    // A broadcast and a runtime-operand matmul read each
                    // operand whole, and the operands differ in size, so the
                    // program-wide slice counts do not apply to them.
                    const BOOL wholeOperand = broadcast || runtimeWeight;
                    NSUInteger sliceElements =
                        wholeOperand ? fullElements : inputSliceElements;
                    NSUInteger physicalElements =
                        wholeOperand ? fullElements : inputPhysicalElements;
                    NSArray<NSNumber *> *logicalShape =
                        inputOffset == 0 && sliceElements == fullElements
                            ? fullShape : @[@(sliceElements)];
                    NSMutableDictionary *record =
                        [binding(input, logicalShape, program.inputs.at(index)) mutableCopy];
                    BOOL aliasShape =
                        ![fullShape isEqualToArray:tensors[input.name][@"shape"]];
                    NSUInteger baseOffset =
                        [[valueBaseOffsets objectForKey:input] unsignedIntegerValue];
                    if (inputOffset > NSUIntegerMax - baseOffset)
                        return reject(diagnostics, @"H13 input alias slice overflows",
                            operation, @"h13.invalid-alias-slice");
                    NSUInteger bindingOffset = baseOffset + inputOffset;
                    if (bindingOffset || sliceElements != fullElements ||
                        physicalElements != sliceElements || aliasShape)
                        addSlice(record, input, bindingOffset, sliceElements,
                                 physicalElements);
                    if (input == constantInput) {
                        record[@"binding"] = @"constant";
                        constantInputs[input.name] = hexData(constantData);
                    } else if (intermediate) {
                        record[@"role"] = @"intermediate";
                    }
                    [inputRecords addObject:record];
                }
                BOOL intermediateOutput =
                    [intermediateStorageNames containsObject:outputValue.name];
                NSString *outputRole = intermediateOutput ? @"intermediate" : @"output";
                NSArray<NSNumber *> *fullOutputShape = outputShapes[outputValue.name]
                    ?: outputValue.type.shape;
                recordTensor(tensors, outputValue, fullOutputShape, outputRole);
                NSUInteger fullOutputElements = 1;
                for (NSNumber *dimension in fullOutputShape)
                    fullOutputElements *= dimension.unsignedIntegerValue;
                NSArray<NSNumber *> *outputShape =
                    outputOffset == 0 && outputSliceElements == fullOutputElements
                        ? fullOutputShape : @[@(outputSliceElements)];
                NSMutableDictionary *outputRecord =
                    [binding(outputValue, outputShape, program.output) mutableCopy];
                if (outputOffset || outputSliceElements != fullOutputElements ||
                    outputPhysicalElements != outputSliceElements)
                    addSlice(outputRecord, outputValue, outputOffset,
                             outputSliceElements, outputPhysicalElements);
                if (intermediateOutput) outputRecord[@"role"] = @"intermediate";
                NSUInteger programIndex = programRecords.count;
                NSData *payload = nil;
                if (hwx) {
                    payload = encodeHWX(program, anec, error);
                    if (!payload) return NO;
                } else {
                    payload = [NSData dataWithBytes:anec.data() length:anec.size()];
                }
                NSString *file = [NSString stringWithFormat:@"program-%lu.%@",
                    (unsigned long)programIndex, format];
                NSDictionary *record = @{
                    @"file": file, @"bytes": @(payload.length),
                    @"taskDescriptors": @(program.taskCount),
                    @"encoder": batched
                        ? (runtimeWeight ? @"apple-parity-batched-matmul"
                                         : @"apple-parity-batched-matvec")
                        : linearParity
                        ? @"apple-parity-linear"
                        : ffnChain
                        ? @"apple-parity-ffn-chain"
                        : transposeParity
                        ? @"apple-parity-transpose"
                        : sliceParity
                        ? @"apple-parity-slice"
                        : tiled
                        ? @"apple-parity-tile"
                        : booleanLowered
                        ? @"apple-parity-boolean"
                        : matvecParity
                        ? (runtimeWeight ? @"apple-parity-matmul"
                                         : @"apple-parity-matvec")
                        : (broadcast ? @"apple-parity-broadcast"
                        : (normalization ? @"apple-parity-norm"
                        : (convolution ? @"apple-parity-conv"
                        : (parity ? @"h13-oracle-parity" : @"h13-source-qualified")))),
                    @"operation": manifestOperation,
                    @"inputs": inputRecords, @"constantInputs": constantInputs,
                    @"outputs": @[outputRecord],
                    @"constantOffset": @(program.constantOffsetBytes),
                    @"constantBytes": @(program.constants.size()),
                    @"scratchBytes": @(program.scratchAllocationBytes),
                };
                if (reluConsumer)
                    [regions addObject:@{@"program": @(programIndex),
                        @"operations": @[operation.results[0].name, outputValue.name],
                        @"postOperation": @"relu"}];
                [programRecords addObject:record];
                [dispatchPlan addObject:@(programIndex)];
                [payloads addObject:payload];
                if (chainSchedule)
                    chainPrograms.push_back(std::move(program));
            }
        }
        if (chainSchedule) {
            if (chainPrograms.size() != programRecords.count ||
                chainPrograms.empty())
                throw std::logic_error(
                    "h13.chain-outside-envelope: chain program bookkeeping is inconsistent");
            NSDictionary *firstRecord = programRecords.firstObject;
            NSString *producerResult = sourceOperations[0].results[0].name;
            NSUInteger producerPrograms = 0;
            for (NSDictionary *record in programRecords)
                for (NSDictionary *item in record[@"outputs"])
                    producerPrograms += [item[@"name"] isEqualToString:producerResult];
            NSArray *firstOutputs = firstRecord[@"outputs"];
            if (producerPrograms != 1 || firstOutputs.count != 1 ||
                ![firstOutputs[0][@"name"] isEqualToString:producerResult])
                throw std::invalid_argument(
                    "h13.chain-outside-envelope: the producer must encode as one whole-tensor program");
            NSArray *boundaryInputs = firstRecord[@"inputs"];
            if (boundaryInputs.count != function.inputs.count)
                throw std::invalid_argument(
                    "h13.chain-outside-envelope: first operation bindings differ from the function inputs");
            NSMutableSet<NSString *> *boundaryNames = [NSMutableSet set];
            for (NSDictionary *item in boundaryInputs) {
                if (item[@"slice"] || item[@"binding"])
                    throw std::invalid_argument(
                        "h13.chain-outside-envelope: boundary inputs must be whole runtime tensors");
                [boundaryNames addObject:item[@"name"]];
            }
            if (![boundaryNames isEqualToSet:inputNames] || firstOutputs[0][@"slice"])
                throw std::invalid_argument(
                    "h13.chain-outside-envelope: producer bindings must be whole boundary tensors");

            const NSString *encoder = firstRecord[@"encoder"];
            if ([encoder isEqualToString:@"apple-parity-matvec"])
                ane::h13::fuseMatmulPostOperation(chainPrograms[0],
                    ane::h13::PostOperation::Relu);
            else if ([encoder isEqualToString:@"h13-oracle-parity"])
                ane::h13::fuseElementwisePostOperation(chainPrograms[0],
                    ane::h13::PostOperation::Relu);
            else
                throw std::invalid_argument(
                    "h13.chain-unrepresentable-edge: this producer encoding carries no decoded post-operation field");
            chainPrograms.resize(1);
            ane::h13::Program combined = ane::h13::composePrograms(chainPrograms);
            std::vector<std::uint8_t> bytes = ane::h13::encodeANEC(combined);
            NSData *combinedPayload = nil;
            if (hwx) {
                combinedPayload = encodeHWX(combined, bytes, error);
                if (!combinedPayload) return NO;
            } else {
                combinedPayload = [NSData dataWithBytes:bytes.data()
                                                 length:bytes.size()];
            }
            taskRecords = [NSMutableArray arrayWithCapacity:combined.taskCount];
            NSString *producerOperation = sourceOperations[0].operationName;
            for (NSUInteger index = 0; index < combined.taskCount; ++index)
                [taskRecords addObject:@{@"index": @(index),
                    @"operation": producerOperation}];
            scratch = @{@"bytes": @(combined.scratchAllocationBytes),
                        @"regions": @{}};
            [tensors removeObjectForKey:producerResult];
            [intermediateNames removeObject:producerResult];
            [intermediateStorageNames removeObject:producerResult];
            NSString *finalResult = lastSourceOperation.results[0].name;
            NSMutableDictionary *fusedOutput = [firstOutputs[0] mutableCopy];
            fusedOutput[@"name"] = finalResult;
            [fusedOutput removeObjectForKey:@"role"];
            [fusedOutput removeObjectForKey:@"slice"];
            NSString *file = [NSString stringWithFormat:@"program-0.%@", format];
            NSDictionary *combinedRecord = @{
                @"file": file, @"bytes": @(combinedPayload.length),
                @"taskDescriptors": @(combined.taskCount),
                @"encoder": @"composed-chain", @"operation": @"chain",
                @"inputs": boundaryInputs, @"constantInputs": @{},
                @"outputs": @[fusedOutput],
                @"constantOffset": @(combined.constantOffsetBytes),
                @"constantBytes": @(combined.constants.size()),
                @"scratchBytes": @(combined.scratchAllocationBytes),
                @"fused": @[finalResult],
            };
            [programRecords removeAllObjects];
            [programRecords addObject:combinedRecord];
            [payloads removeAllObjects];
            [payloads addObject:combinedPayload];
            [dispatchPlan removeAllObjects];
            [dispatchPlan addObject:@0];
        }
    } catch (const std::exception &exception) {
        NSString *message = [NSString stringWithUTF8String:exception.what()];
        NSString *code = chainSchedule ? @"h13.chain-outside-envelope"
                                       : @"h13.unsupported-program";
        NSRange codeEnd = [message rangeOfString:@": "];
        if ([message hasPrefix:@"h13."] && codeEnd.location != NSNotFound &&
            codeEnd.location <= 48)
            code = [message substringToIndex:codeEnd.location];
        return reject(diagnostics, message, operation, code);
    }
    for (NSString *name in aliases) {
        NSDictionary *alias = aliases[name];
        NSArray<NSNumber *> *shape = alias[@"shape"];
        NSUInteger elements = 1;
        for (NSNumber *dimension in shape)
            elements *= dimension.unsignedIntegerValue;
        NSString *role = [returnedSourceNames containsObject:name]
            ? @"output" : @"intermediate";
        BOOL boolAlias = [tensors[alias[@"aliasOf"]][@"dtype"] isEqualToString:@"bool"];
        tensors[name] = boolAlias
            ? @{@"shape": shape, @"logicalBytes": @(elements),
                @"role": role, @"aliasOf": alias[@"aliasOf"], @"dtype": @"bool"}
            : @{@"shape": shape, @"logicalBytes": @(elements * 2),
                @"role": role, @"aliasOf": alias[@"aliasOf"]};
    }
    for (NSString *name in intermediateStorageNames) {
        NSMutableArray<NSArray<NSNumber *> *> *produced = [NSMutableArray array];
        NSMutableArray<NSArray<NSNumber *> *> *consumed = [NSMutableArray array];
        for (NSDictionary *record in programRecords) {
            for (NSString *direction in @[@"outputs", @"inputs"]) {
                for (NSDictionary *item in record[direction]) {
                    if (![item[@"name"] isEqualToString:name]) continue;
                    NSDictionary *slice = item[@"slice"];
                    NSUInteger offset = [slice[@"elementOffset"] unsignedIntegerValue];
                    const NSUInteger elementSize =
                        [item[@"dtype"] isEqualToString:@"bool"] ? 1 : 2;
                    NSUInteger count = slice
                        ? [slice[@"elementCount"] unsignedIntegerValue]
                        : [item[@"logicalBytes"] unsignedIntegerValue] / elementSize;
                    NSUInteger physical = slice[@"physicalElements"]
                        ? [slice[@"physicalElements"] unsignedIntegerValue] : count;
                    [([direction isEqualToString:@"outputs"] ? produced : consumed)
                        addObject:@[@(offset), @(offset + physical)]];
                }
            }
        }
        [produced sortUsingComparator:^NSComparisonResult(
            NSArray<NSNumber *> *left, NSArray<NSNumber *> *right) {
            return [left[0] compare:right[0]];
        }];
        NSUInteger previousEnd = 0;
        for (NSArray<NSNumber *> *range in produced) {
            NSUInteger start = [range[0] unsignedIntegerValue];
            if (start < previousEnd)
                return reject(diagnostics,
                    @"H13 intermediate physical writes must not overlap",
                    lastOperation, chainCode);
            previousEnd = [range[1] unsignedIntegerValue];
        }
        for (NSArray<NSNumber *> *consumer in consumed) {
            NSUInteger cursor = [consumer[0] unsignedIntegerValue];
            NSUInteger end = [consumer[1] unsignedIntegerValue];
            for (NSArray<NSNumber *> *producer in produced) {
                NSUInteger producerStart = [producer[0] unsignedIntegerValue];
                NSUInteger producerEnd = [producer[1] unsignedIntegerValue];
                if (producerEnd <= cursor) continue;
                if (producerStart > cursor) break;
                cursor = MAX(cursor, producerEnd);
                if (cursor >= end) break;
            }
            // Programs read in 64-element lanes, so a consumer may reach
            // the end of the lane containing the last covered element —
            // the readable surface padding — but never beyond it.
            if (cursor < end && end > (cursor + 63) / 64 * 64)
                return reject(diagnostics,
                    @"H13 intermediate consumer physical range exceeds producer writes",
                    lastOperation, chainCode);
        }
    }

    for (NSString *name in chunkedAccumulations) {
        NSMutableDictionary *record = [tensors[name] mutableCopy];
        if (record) {
            record[@"accumulation"] = @"chunked-fp16";
            tensors[name] = record;
        }
    }

    NSMutableArray<NSDictionary *> *physicalOutputs = [NSMutableArray array];
    NSMutableDictionary<NSString *, NSNumber *> *physicalElementCounts =
        [NSMutableDictionary dictionary];
    for (NSString *storageName in returnedStorageNames) {
        NSDictionary *physicalTensor = tensors[storageName];
        NSArray<NSNumber *> *physicalShape = physicalTensor[@"shape"];
        NSUInteger physicalElements = 1;
        for (NSNumber *dimension in physicalShape)
            physicalElements *= dimension.unsignedIntegerValue;
        physicalElementCounts[storageName] = @(physicalElements);
        [physicalOutputs addObject:@{
            @"tensor": storageName,
            @"dtype": [physicalTensor[@"dtype"] isEqualToString:@"bool"]
                ? @"bool" : @"float16",
            @"shape": physicalShape,
            @"logicalBytes": physicalTensor[@"logicalBytes"],
        }];
    }
    NSMutableArray<NSDictionary *> *logicalResults = [NSMutableArray array];
    for (NSUInteger index = 0; index < function.returnValues.count; ++index) {
        ANEGraphValue *logical = function.returnValues[index];
        ANEGraphValue *returned = returnedValues[index];
        NSString *storageName = returned.name;
        NSUInteger physicalElements =
            physicalElementCounts[storageName].unsignedIntegerValue;
        NSUInteger resultOffset =
            [[valueBaseOffsets objectForKey:returned] unsignedIntegerValue];
        NSUInteger elements = 0;
        if (!tensorElementCount(logical, &elements) ||
            elements > physicalElements || resultOffset > physicalElements - elements)
            return reject(diagnostics,
                @"H13 logical result exceeds its physical output storage",
                logical.producer ?: lastSourceOperation,
                @"h13.unsupported-logical-result-storage");
        [logicalResults addObject:@{
            @"name": logical.name,
            @"dtype": boolTensor(logical) ? @"bool" : @"float16",
            @"shape": logical.type.shape, @"conversion": @"identity",
            @"physical": @{
                @"tensor": storageName,
                @"elementOffset": @(resultOffset),
                @"elementCount": @(elements),
            },
        }];
    }

    NSMutableDictionary *manifest = [@{
        @"schema": @"mil-hwxc.h13-anec-package.v2",
        @"target": @"H13", @"artifactFormat": format,
        @"programs": programRecords, @"dispatchPlan": dispatchPlan,
        @"intermediates": intermediateNames, @"tensors": tensors,
        @"physicalOutputs": physicalOutputs, @"logicalResults": logicalResults,
    } mutableCopy];
    if ([schedule isEqualToString:@"regions"]) {
        manifest[@"schedule"] = @"regions";
        manifest[@"regions"] = regions;
    }
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
        NSArray *existing = [manager contentsOfDirectoryAtPath:directory.path error:error];
        if (!existing) return NO;
        if (existing.count != 0) {
            if (error) *error = [NSError errorWithDomain:@"dev.maderix.H13"
                code:1 userInfo:@{NSLocalizedDescriptionKey:
                    @"H13 output directory must be empty"}];
            return NO;
        }
    } else if (![manager createDirectoryAtURL:directory withIntermediateDirectories:YES
        attributes:nil error:error]) return NO;
    for (NSUInteger index = 0; index < payloads.count; ++index)
        if (![payloads[index] writeToURL:
                [directory URLByAppendingPathComponent:programRecords[index][@"file"]]
                options:NSDataWritingAtomic error:error])
            return NO;
    return [metadata writeToURL:[directory URLByAppendingPathComponent:@"manifest.json"]
        options:NSDataWritingAtomic error:error];
}
@end
