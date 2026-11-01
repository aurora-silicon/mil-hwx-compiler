#pragma once

#import "ANEGraphIR.h"
#include "LayerOps.h"
#include <cmath>
#include <cstdlib>

namespace ane::layers {
struct Plan {
    Operation operation;
    Shape shape;
    NSArray<ANEGraphValue *> *inputs;
    bool placementOnly = false;
};

static inline ANEGraphArgument *literal(ANEGraphArgument *arg) {
    if (arg.kind == ANEGraphArgumentKindValue) {
        auto *producer = arg.value.producer;
        if (![producer.operationName isEqualToString:@"const"] || producer.arguments.count) return nil;
        return literal(producer.attributes[@"val"]);
    }
    if (arg.kind == ANEGraphArgumentKindCall && arg.callArguments.count == 1)
        return literal(arg.callArguments[0].value);
    return arg;
}

static inline bool number(ANEGraphArgument *arg, double expected) {
    arg = literal(arg);
    if (!arg || (arg.kind != ANEGraphArgumentKindInteger &&
                 arg.kind != ANEGraphArgumentKindFloatingPoint)) return false;
    char *end = nullptr;
    const double value = strtod(arg.text.UTF8String, &end);
    return end && !*end && std::isfinite(value) && value == expected;
}

static inline bool text(ANEGraphArgument *arg, NSString *expected) {
    arg = literal(arg);
    return arg && [arg.text isEqualToString:expected];
}

static inline bool scalarType(ANEGraphArgument *arg, ANEElementType type) {
    if (!arg) return false;
    ANEValueType *valueType=nil;
    if (arg.kind==ANEGraphArgumentKindValue) valueType=arg.value.type;
    else if (arg.kind==ANEGraphArgumentKindCall) {
        valueType=arg.calleeValueType;
        // Scalar constructors such as int32(1) carry a callee name; only
        // tensor<...>(...) constructors have a calleeValueType in the IR.
        if (!valueType && arg.callArguments.count==1) {
            NSString *name=type==ANEElementTypeInt32 ? @"int32" :
                type==ANEElementTypeFP16 ? @"fp16" :
                type==ANEElementTypeFP32 ? @"fp32" :
                type==ANEElementTypeBool ? @"bool" :
                type==ANEElementTypeString ? @"string" : nil;
            return name && [arg.calleeName isEqualToString:name];
        }
    }
    else return (type==ANEElementTypeInt32 && arg.kind==ANEGraphArgumentKindInteger) ||
        (type==ANEElementTypeBool && arg.kind==ANEGraphArgumentKindBoolean) ||
        (type==ANEElementTypeString && arg.kind==ANEGraphArgumentKindString);
    return valueType && valueType.elementType==type &&
        (valueType.kind==ANEValueTypeKindScalar || valueType.shape.count==0);
}

static inline bool surface(ANEGraphValue *value, Shape *shape) {
    if (!value || value.type.kind != ANEValueTypeKindTensor ||
        value.type.elementType != ANEElementTypeFP16 || value.type.shape.count != 4 ||
        value.type.shape[0].unsignedIntegerValue != 1) return false;
    for (NSUInteger i = 1; i != 4; ++i)
        if (!value.type.shape[i].unsignedLongLongValue ||
            value.type.shape[i].unsignedLongLongValue > UINT32_MAX) return false;
    *shape = {value.type.shape[1].unsignedIntValue, value.type.shape[2].unsignedIntValue,
              value.type.shape[3].unsignedIntValue, 0};
    return true;
}

static inline bool plan(ANEGraphOperation *op, bool h14, Plan *out, bool allowPlacement = false) {
    if (op.results.count != 1) return false;
    NSString *name = op.operationName;
    Shape input{}, output{};
    if (!surface(op.results[0], &output)) return false;
    Plan p{};
    auto *x = op.operands[@"x"].value;
    if ([name isEqualToString:@"concat"]) {
        auto *values = op.arguments[@"values"];
        if (op.arguments.count < 2 || op.arguments.count > 3 ||
            values.kind != ANEGraphArgumentKindTuple || values.elements.count != 2 ||
            !scalarType(op.arguments[@"axis"],ANEElementTypeInt32) ||
            (!number(op.arguments[@"axis"], 1) && !number(op.arguments[@"axis"], -3)) ||
            (op.arguments[@"interleave"] && (!scalarType(op.arguments[@"interleave"],ANEElementTypeBool) ||
                                             !text(op.arguments[@"interleave"], @"false")))) return false;
        x = values.elements[0].value;
        auto *z = values.elements[1].value;
        Shape rhs{};
        if (!surface(x, &input) || !surface(z, &rhs) || input.height != rhs.height ||
            input.width != rhs.width || input.channels > UINT32_MAX - rhs.channels) return false;
        input.otherChannels = rhs.channels;
        p.operation = Operation::Concat;
        p.inputs = @[x, z];
    } else {
        if (!surface(x, &input)) return false;
        p.inputs = @[x];
        if ([name isEqualToString:@"sigmoid"] || [name isEqualToString:@"hardswish"]) {
            if (op.arguments.count != 1) return false;
            p.operation = [name isEqualToString:@"sigmoid"] ? Operation::Sigmoid : Operation::HardSwish;
        } else {
            const bool up = [name isEqualToString:@"upsample_nearest_neighbor"] || [name isEqualToString:@"upsample_bilinear"];
            const bool bilinear = [name isEqualToString:@"resize_bilinear"] || [name isEqualToString:@"upsample_bilinear"];
            if (!up && ![name isEqualToString:@"resize_nearest_neighbor"] && ![name isEqualToString:@"resize_bilinear"]) return false;
            if (op.arguments.count != (bilinear ? 4u : 3u) || input.height > UINT32_MAX / 2 || input.width > UINT32_MAX / 2) return false;
            if (up) {
                for (NSString *key in @[@"scale_factor_height",@"scale_factor_width"])
                    if (!scalarType(op.arguments[key],ANEElementTypeInt32) && !scalarType(op.arguments[key],ANEElementTypeFP32)) return false;
                if (!number(op.arguments[@"scale_factor_height"], 2) || !number(op.arguments[@"scale_factor_width"], 2) ||
                    (bilinear && (!scalarType(op.arguments[@"align_corners"],ANEElementTypeBool) ||
                                  !text(op.arguments[@"align_corners"], @"false")))) return false;
            } else if (!scalarType(op.arguments[@"target_size_height"],ANEElementTypeInt32) ||
                       !scalarType(op.arguments[@"target_size_width"],ANEElementTypeInt32) ||
                       !number(op.arguments[@"target_size_height"], input.height * 2) ||
                       !number(op.arguments[@"target_size_width"], input.width * 2) ||
                       (bilinear && (!scalarType(op.arguments[@"sampling_mode"],ANEElementTypeString) ||
                                     !text(op.arguments[@"sampling_mode"], @"UNALIGN_CORNERS")))) return false;
            p.operation = bilinear ? Operation::BilinearHalf2x : Operation::Nearest2x;
        }
    }
    const unsigned scale = p.operation == Operation::Nearest2x || p.operation == Operation::BilinearHalf2x ? 2 : 1;
    if (output.channels != input.channels + input.otherChannels || output.height != input.height * scale ||
        output.width != input.width * scale) return false;
    if (!supports(h14, p.operation, input)) {
        // This is a view candidate, never a substitute copy kernel. The graph
        // allocator must prove exclusive adjacent producer placement before
        // the empty program can disappear from the linked task stream.
        if (!allowPlacement || h14 || p.operation != Operation::Concat ||
            input.channels%16 || input.otherChannels%16 || output.channels>4096 ||
            input.height>4096 || input.width>4096) return false;
        p.placementOnly = true;
    }
    for (ANEGraphValue *value in p.inputs)
        if ([value.producer.operationName isEqualToString:@"const"]) return false;
    p.shape = input;
    *out = p;
    return true;
}

static inline NSUInteger uses(ANEGraphArgument *arg, ANEGraphValue *value) {
    if (arg.value == value) return 1;
    NSUInteger count = 0;
    for (ANEGraphArgument *child in arg.elements) count += uses(child, value);
    for (ANEGraphNamedArgument *child in arg.callArguments) count += uses(child.value, value);
    return count;
}

static inline ANEGraphArgument *remap(ANEGraphArgument *arg,
    NSDictionary<NSString *, ANEGraphValue *> *values) {
    if (!arg) return nil;
    auto *value = arg.value ? values[arg.value.name] : nil;
    if (value)
        return [[ANEGraphArgument alloc] initWithKind:ANEGraphArgumentKindValue
            text:nil value:value calleeName:nil calleeValueType:nil
            callArguments:@[] elements:@[] range:arg.range];
    if (!arg.elements.count) return arg;
    auto *elements = [NSMutableArray<ANEGraphArgument *> array];
    for (ANEGraphArgument *child in arg.elements) [elements addObject:remap(child, values)];
    return [[ANEGraphArgument alloc] initWithKind:arg.kind text:arg.text
        value:arg.value calleeName:arg.calleeName calleeValueType:arg.calleeValueType
        callArguments:arg.callArguments elements:elements range:arg.range];
}

/// Canonical MIL hardswish: x * (relu6(x + 3) / 6). Intermediate results may
/// not escape: Apple's native program keeps its own intermediate precision.
static inline NSMutableArray<ANEGraphOperation *> *fuseHardSwish(
    NSArray<ANEGraphOperation *> *ops, NSArray<ANEGraphValue *> *__strong *returns) {
    auto *result = [NSMutableArray<ANEGraphOperation *> array];
    auto *renamed = [NSMutableDictionary<NSString *, ANEGraphValue *> dictionary];
    for (NSUInteger i = 0; i < ops.count; ++i) {
        bool match = i + 3 < ops.count;
        auto *a = ops[i];
        auto *b = match ? ops[i + 1] : nil;
        auto *h = match ? ops[i + 2] : nil;
        auto *y = match ? ops[i + 3] : nil;
        auto *x = a.operands[@"x"].value;
        match = match && [a.operationName isEqualToString:@"add"] && a.arguments.count == 2 && x &&
            scalarType(a.arguments[@"y"],ANEElementTypeFP16) &&
            number(a.arguments[@"y"], 3) && [b.operationName isEqualToString:@"relu6"] && b.arguments.count == 1 &&
            b.operands[@"x"].value == a.results[0] && [h.operationName isEqualToString:@"real_div"] &&
            h.arguments.count == 2 && h.operands[@"x"].value == b.results[0] &&
            scalarType(h.arguments[@"y"],ANEElementTypeFP16) && number(h.arguments[@"y"], 6) &&
            [y.operationName isEqualToString:@"mul"] && y.arguments.count == 2 &&
            ((y.operands[@"x"].value == x && y.operands[@"y"].value == h.results[0]) ||
             (y.operands[@"y"].value == x && y.operands[@"x"].value == h.results[0]));
        if (match) {
            for (ANEGraphOperation *op in @[a,b,h,y])
                if (op.results.count!=1 || ![op.results[0].type isEqualToValueType:x.type]) match=false;
            for (ANEGraphValue *intermediate in @[a.results[0], b.results[0], h.results[0]]) {
                NSUInteger count = 0;
                for (ANEGraphOperation *op in ops)
                    for (ANEGraphArgument *arg in op.arguments.allValues) count += uses(arg, intermediate);
                if (count != 1 || [*returns containsObject:intermediate]) match = false;
            }
        }
        if (match) {
            auto *output = [[ANEGraphValue alloc] initWithName:y.results[0].name type:y.results[0].type];
            [result addObject:[[ANEGraphOperation alloc] initWithOperationName:@"hardswish"
                results:@[output] arguments:@{@"x": remap(a.arguments[@"x"], renamed)} attributes:@{} range:a.range]];
            renamed[output.name] = output;
            i += 3;
        } else {
            auto *arguments = [NSMutableDictionary<NSString *, ANEGraphArgument *> dictionary];
            bool changed = false;
            for (NSString *key in a.arguments) {
                arguments[key] = remap(a.arguments[key], renamed);
                changed = changed || arguments[key] != a.arguments[key];
            }
            if (changed) {
                auto *outputs = [NSMutableArray<ANEGraphValue *> array];
                for (ANEGraphValue *value in a.results) {
                    auto *output = [[ANEGraphValue alloc] initWithName:value.name type:value.type];
                    renamed[value.name] = output;
                    [outputs addObject:output];
                }
                [result addObject:[[ANEGraphOperation alloc] initWithOperationName:a.operationName
                    results:outputs arguments:arguments attributes:a.attributes range:a.range]];
            } else [result addObject:a];
        }
    }
    auto *rewritten = [NSMutableArray<ANEGraphValue *> array];
    for (ANEGraphValue *value in *returns) [rewritten addObject:renamed[value.name] ?: value];
    *returns = rewritten;
    return result;
}
} // namespace ane::layers
