#import "ANESpatialCompiler.h"
#import "ANEBlobResolver.h"
#import "ANEGraphVerifier.h"
#import "MILLexer.h"
#import "MILParser.h"
#import "MILGraphImporter.h"
#include "H13Program.h"
#include "H14Program.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
// All arithmetic and movement below use existing kernels. Bindings stage only
// contiguous dense slices, just as the original package ABI does. No host
// resize, convolution, gather, transpose or concatenation operation is emitted.
constexpr NSUInteger limit = 1u << 29;
static void need(BOOL condition, const char *code) {
    if (!condition) throw std::invalid_argument(code);
}
static NSUInteger product(NSArray<NSNumber *> *shape) {
    NSUInteger n = 1;
    need(shape.count > 0 && shape.count <= 4, "spatial.invalid-shape");
    for (NSNumber *d in shape) {
        need(d.longLongValue > 0 && d.unsignedLongLongValue <= limit / n,
             "spatial.shape-overflow");
        n *= d.unsignedIntegerValue;
    }
    return n;
}
static NSUInteger elements(ANEGraphValue *v) {
    need(v && v.type.kind == ANEValueTypeKindTensor &&
         v.type.elementType == ANEElementTypeFP16, "spatial.requires-fp16-tensor");
    return product(v.type.shape);
}
static ANEGraphArgument *literal(ANEGraphArgument *a) {
    if (a.kind == ANEGraphArgumentKindValue) {
        need([a.value.producer.operationName isEqualToString:@"const"],
             "spatial.requires-constant-parameter");
        a = a.value.producer.attributes[@"val"];
    }
    return a;
}
static ANEGraphArgument *scalar(ANEGraphArgument *a) {
    a = literal(a);
    if (a.kind == ANEGraphArgumentKindCall && a.callArguments.count == 1)
        a = a.callArguments[0].value;
    if (a.kind == ANEGraphArgumentKindList && a.elements.count == 1)
        a = a.elements[0];
    return a;
}
static double number(ANEGraphArgument *a) {
    a = scalar(a);
    need(a && (a.kind == ANEGraphArgumentKindInteger ||
         a.kind == ANEGraphArgumentKindFloatingPoint), "spatial.invalid-number");
    char *end = nullptr;
    double x = std::strtod(a.text.UTF8String, &end);
    need(end && !*end && std::isfinite(x), "spatial.invalid-number");
    return x;
}
static long long integer(ANEGraphArgument *a) {
    double x = number(a);
    need(std::floor(x) == x && std::fabs(x) <= limit, "spatial.invalid-integer");
    return static_cast<long long>(x);
}
static NSString *string(ANEGraphArgument *a) {
    a = scalar(a);
    need(a && a.kind == ANEGraphArgumentKindString, "spatial.invalid-string");
    return a.text;
}
static BOOL boolean(ANEGraphArgument *a, BOOL fallback) {
    if (!a) return fallback;
    a = scalar(a);
    need(a && a.kind == ANEGraphArgumentKindBoolean, "spatial.invalid-bool");
    return [a.text isEqualToString:@"true"];
}
static NSArray<ANEGraphArgument *> *list(ANEGraphArgument *a) {
    a = literal(a);
    if (a.kind == ANEGraphArgumentKindCall && a.callArguments.count == 1)
        a = a.callArguments[0].value;
    need(a && (a.kind == ANEGraphArgumentKindList ||
         a.kind == ANEGraphArgumentKindTuple), "spatial.invalid-list");
    return a.elements;
}
static uint16_t half(double x) {
    // IEEE binary16, ties to even, independently of the host's _Float16 ABI.
    const uint16_t sign = std::signbit(x) ? 0x8000 : 0;
    double a = std::fabs(x);
    if (a == 0) return sign;
    auto roundEven = [](double v) -> uint64_t {
        double f = std::floor(v), r = v - f;
        return static_cast<uint64_t>(f) + (r > .5 || (r == .5 && std::fmod(f, 2)));
    };
    if (a < std::ldexp(1., -14))
        return sign | static_cast<uint16_t>(roundEven(std::ldexp(a, 24)));
    int e = 0;
    double m = std::frexp(a, &e);
    uint64_t bits = roundEven(std::ldexp(m, 11));
    if (bits == 2048) { bits = 1024; ++e; }
    need(e <= 16, "spatial.fp16-overflow");
    return sign | static_cast<uint16_t>((e + 14) << 10) | (bits - 1024);
}
static NSData *bytes(const std::vector<uint16_t> &v) {
    return [NSData dataWithBytes:v.data() length:v.size() * 2];
}
static NSDictionary *affine(NSUInteger base, NSDictionary *terms = @{}) {
    return @{@"base": @(base), @"terms": terms};
}
static NSDictionary *slice(NSString *name, id offset, NSUInteger count,
                            uint16_t padding = 0) {
    return @{@"tensor": name, @"offset": offset, @"count": @(count),
             @"paddingBits": @(padding)};
}
static NSDictionary *loop(NSString *var, NSUInteger count, NSArray *body) {
    return @{@"loop": var, @"count": @(count), @"body": body};
}
static NSDictionary *call(NSUInteger kernel, NSArray *inputs, NSDictionary *output) {
    return @{@"kernel": @(kernel), @"inputs": inputs, @"output": output};
}

template<class T> static NSDictionary *layout(const T &t) {
    NSMutableArray *dims = [NSMutableArray array];
    for (auto x : t.nchw) [dims addObject:@(x)];
    return @{@"index": @(t.index), @"nchw": dims,
             @"allocationBytes": @(t.allocationBytes)};
}

struct Builder {
    bool h14;
    NSURL *root;
    ANEDiagnosticEngine *diagnostics;
    NSMutableArray *kernels = [NSMutableArray array];
    NSMutableArray *payloads = [NSMutableArray array];
    NSMutableArray *plan = [NSMutableArray array];
    NSMutableDictionary *tensors = [NSMutableDictionary dictionary];
    NSMutableDictionary *constantData = [NSMutableDictionary dictionary];
    NSMutableDictionary *cache = [NSMutableDictionary dictionary];
    NSMutableDictionary *scalarTensors = [NSMutableDictionary dictionary];
    NSUInteger serial = 0;
    Builder(bool second, NSURL *model, ANEDiagnosticEngine *d)
        : h14(second), root(model), diagnostics(d) {}

    NSString *temp(NSUInteger n) {
        NSString *name = [NSString stringWithFormat:@"$spatial.%lu", (unsigned long)serial++];
        tensors[name] = @{@"shape": @[@(n)], @"elements": @(n), @"role": @"scratch"};
        return name;
    }
    void value(ANEGraphValue *v, NSString *role) {
        NSUInteger n = elements(v);
        need(![v.name hasPrefix:@"$spatial."], "spatial.reserved-name");
        tensors[v.name] = @{@"shape": v.type.shape, @"elements": @(n), @"role": role};
    }
    NSData *constant(ANEGraphValue *v) {
        if (constantData[v.name]) return constantData[v.name];
        NSUInteger n = elements(v);
        need([v.producer.operationName isEqualToString:@"const"],
             "spatial.requires-constant-weight");
        ANEGraphArgument *a = v.producer.attributes[@"val"];
        NSData *data = nil;
        ANEGraphArgument *payload = scalar(a);
        if (payload.kind == ANEGraphArgumentKindList ||
            payload.kind == ANEGraphArgumentKindFloatingPoint ||
            payload.kind == ANEGraphArgumentKindInteger) {
            NSArray *items = payload.kind == ANEGraphArgumentKindList ? payload.elements : @[payload];
            need(items.count == n, "spatial.constant-count");
            std::vector<uint16_t> result;
            for (ANEGraphArgument *item in items) result.push_back(half(number(item)));
            data = bytes(result);
        } else {
            data = [ANEBlobResolver loadConstantForOperation:v.producer expectedBytes:n * 2
                modelRoot:root diagnostics:diagnostics];
            need(data != nil, "spatial.invalid-weight-blob");
        }
        const uint16_t *p = static_cast<const uint16_t *>(data.bytes);
        for (NSUInteger i = 0; i < n; ++i)
            need((p[i] & 0x7c00) != 0x7c00, "spatial.nonfinite-constant");
        constantData[v.name] = data;
        return data;
    }
    NSString *constantTensor(NSData *data) {
        NSString *name = temp(data.length / 2);
        tensors[name] = @{@"shape": @[@(data.length / 2)], @"elements": @(data.length / 2),
                          @"role": @"constant"};
        constantData[name] = data;
        return name;
    }
    template<class P> NSUInteger save(P &p, NSDictionary *semantics,
                                      const std::vector<uint8_t> &encoded,
                                      NSUInteger constants, NSUInteger tasks) {
        NSData *payload = [NSData dataWithBytes:encoded.data() length:encoded.size()];
        // NSData's content hash and equality deduplicate repeated weight tiles,
        // independent of the names and locations used by dispatch bindings.
        NSNumber *previous = cache[payload];
        if (previous) return previous.unsignedIntegerValue;
        NSMutableArray *inputs = [NSMutableArray array];
        for (const auto &input : p.inputs) [inputs addObject:layout(input)];
        NSUInteger index = kernels.count;
        NSMutableDictionary *record = [semantics mutableCopy];
        record[@"file"] = [NSString stringWithFormat:@"program-%lu.anec", (unsigned long)index];
        record[@"bytes"] = @(payload.length);
        record[@"inputs"] = inputs;
        record[@"output"] = layout(p.output);
        record[@"constantOffset"] = @(constants);
        record[@"constantBytes"] = @(p.constants.size());
        record[@"taskDescriptors"] = @(tasks);
        // ANEC's tile table starts at byte 40; channel 3 is private scratch.
        uint64_t scratch = 0;
        for (NSUInteger byte = 0; byte < 4; ++byte)
            scratch |= static_cast<uint64_t>(encoded[52 + byte]) << (byte * 8);
        record[@"scratchBytes"] = @(scratch * ane::h13::tileBytes);
        [kernels addObject:record]; [payloads addObject:payload]; cache[payload] = @(index);
        return index;
    }
    NSUInteger binary(NSString *name) {
        auto op = [name isEqualToString:@"add"] ? ane::h13::BinaryOperation::Add :
            [name isEqualToString:@"minimum"] ? ane::h13::BinaryOperation::Minimum :
            [name isEqualToString:@"mul"] ? ane::h13::BinaryOperation::Multiply :
            ane::h13::BinaryOperation::Maximum;
        NSDictionary *s = @{@"kind": name};
        if (h14) {
            auto p = ane::h14::encodeElementwise(ane::h14::Generation::H14,
                static_cast<ane::h14::BinaryOperation>(op), {256,1,1}, {256,1,1});
            return save(p, s, ane::h14::encodeANEC(p), p.constantOffsetBytes, p.taskCount);
        }
        auto p = ane::h13::encodeElementwise(op, {256,1,1});
        return save(p, s, ane::h13::encodeANEC(p), p.constantOffsetBytes, p.taskCount);
    }
    NSUInteger matrix(NSData *data, NSUInteger size = 256) {
        need(data.length == size * size * 2, "spatial.matrix-size");
        auto w = static_cast<const uint8_t *>(data.bytes);
        NSDictionary *s = @{@"kind": @"matrix", @"size": @(size)};
        if (h14) {
            auto p = ane::h14::encodeMatvecParity({1, (uint32_t)size, (uint32_t)size}, w, data.length);
            return save(p, s, ane::h14::encodeANEC(p), p.constantOffsetBytes, p.taskCount);
        }
        auto p = ane::h13::encodeMatmulParity({1, (uint32_t)size, (uint32_t)size}, w, data.length);
        return save(p, s, ane::h13::encodeANEC(p), p.constantOffsetBytes, p.taskCount);
    }
    NSUInteger convKernel(NSData *weights, NSUInteger k, bool depthwise, NSUInteger side) {
        NSDictionary *s = @{@"kind": @"conv", @"kernel": @(k), @"side": @(side),
                             @"groups": @(depthwise ? 64 : 1)};
        auto w = static_cast<const uint8_t *>(weights.bytes);
        if (h14) {
            ane::h14::ConvShape shape{(uint32_t)k,(uint32_t)k,1,depthwise ? 64u : 1u,false,
                {64,(uint32_t)side,(uint32_t)side},{64,(uint32_t)side,(uint32_t)side}};
            auto p = ane::h14::encodeConvParity(shape,w,weights.length);
            return save(p,s,ane::h14::encodeANEC(p),p.constantOffsetBytes,p.taskCount);
        }
        ane::h13::ConvShape shape{(uint32_t)k,(uint32_t)k,1,depthwise ? 64u : 1u,false,
            {64,(uint32_t)side,(uint32_t)side},{64,(uint32_t)side,(uint32_t)side}};
        auto p = ane::h13::encodeConvParity(shape,w,weights.length);
        return save(p,s,ane::h13::encodeANEC(p),p.constantOffsetBytes,p.taskCount);
    }
    NSDictionary *copy(NSDictionary *src, NSDictionary *dst) {
        return call(binary(@"maximum"), @[src,src], dst);
    }
    void copies(NSMutableArray *body, NSString *src, NSUInteger si,
                 NSString *dst, NSUInteger di, NSUInteger n) {
        // Loops bound descriptor size even for full-frame channel planes.
        NSUInteger chunks = n / 256, tail = n % 256;
        if (chunks) [body addObject:loop(@"copy",chunks,@[copy(
            slice(src,affine(si,@{@"copy": @256}),256),
            slice(dst,affine(di,@{@"copy": @256}),256))])];
        if (tail) [body addObject:copy(slice(src,@(si + chunks * 256),tail),
                                      slice(dst,@(di + chunks * 256),tail))];
    }
    NSString *scalarTensor(uint16_t x) {
        if (!scalarTensors[@(x)])
            scalarTensors[@(x)] = constantTensor(bytes(std::vector<uint16_t>(256,x)));
        return scalarTensors[@(x)];
    }
    void fill(NSMutableArray *body, NSString *dst, NSUInteger n) {
        NSString *zero = scalarTensor(0);
        [body addObject:loop(@"fill", n / 256, @[copy(slice(zero,@0,256),
            slice(dst,affine(0,@{@"fill": @256}),256))])];
    }
    void unary(ANEGraphOperation *op) {
        ANEGraphValue *x = op.operands[@"x"].value, *y = op.results[0];
        NSUInteger n = elements(x);
        need([x.type.shape isEqualToArray:y.type.shape], "spatial.elementwise-shape");
        NSString *name = op.operationName;
        if ([name isEqualToString:@"relu"] || [name isEqualToString:@"clip"]) {
            double low = [name isEqualToString:@"relu"] ? 0 : number(op.arguments[@"alpha"]);
            double high = [name isEqualToString:@"clip"] ? number(op.arguments[@"beta"]) : 0;
            need(![name isEqualToString:@"clip"] || low <= high, "spatial.clip-bounds");
            NSString *lower = scalarTensor(half(low));
            NSString *middle = [name isEqualToString:@"clip"] ? temp(n) : y.name;
            auto emit = [&](NSUInteger offset, NSUInteger count) {
                id at = affine(offset,@{@"element": @256});
                NSMutableArray *body = [NSMutableArray arrayWithObject:call(binary(@"maximum"),
                    @[slice(x.name,at,count),slice(lower,@0,count)],slice(middle,at,count))];
                if ([name isEqualToString:@"clip"]) {
                    NSString *upper = scalarTensor(half(high));
                    [body addObject:call(binary(@"minimum"),@[slice(middle,at,count),slice(upper,@0,count)],
                        slice(y.name,at,count))];
                }
                return body;
            };
            if (n / 256) [plan addObject:loop(@"element",n / 256,emit(0,256))];
            if (n % 256) [plan addObject:loop(@"element",1,emit(n / 256 * 256,n % 256))];
        } else throw std::invalid_argument("spatial.unsupported-unary");
    }
    void concat(ANEGraphOperation *op) {
        auto sources = list(op.arguments[@"values"]);
        ANEGraphValue *y = op.results[0];
        NSUInteger rank = y.type.shape.count;
        NSInteger axis = integer(op.arguments[@"axis"]);
        if (axis < 0) axis += rank;
        need(axis >= 0 && axis < (NSInteger)rank && sources.count >= 2 &&
             !boolean(op.arguments[@"interleave"],NO), "spatial.invalid-concat");
        NSUInteger outer = 1, inner = 1, total = 0;
        for (NSInteger d = 0; d < axis; ++d) outer *= y.type.shape[d].unsignedIntegerValue;
        for (NSUInteger d = axis + 1; d < rank; ++d) inner *= y.type.shape[d].unsignedIntegerValue;
        for (ANEGraphArgument *a in sources) {
            ANEGraphValue *x = a.value;
            elements(x);
            need(x.type.shape.count == rank, "spatial.concat-rank");
            for (NSUInteger d = 0; d < rank; ++d)
                need(d == (NSUInteger)axis || [x.type.shape[d] isEqual:y.type.shape[d]],
                     "spatial.concat-shape");
            total += x.type.shape[axis].unsignedIntegerValue;
        }
        need(total == y.type.shape[axis].unsignedIntegerValue, "spatial.concat-extent");
        NSUInteger position = 0;
        for (ANEGraphArgument *a in sources) {
            NSUInteger span = a.value.type.shape[axis].unsignedIntegerValue * inner;
            NSMutableArray *body = [NSMutableArray array];
            if (span / 256)
                [body addObject:loop(@"block",span / 256,@[copy(
                    slice(a.value.name,affine(0,@{@"outer": @(span),@"block": @256}),256),
                    slice(y.name,affine(position,@{@"outer": @(total * inner),@"block": @256}),256))])];
            if (span % 256)
                [body addObject:copy(
                    slice(a.value.name,affine(span / 256 * 256,@{@"outer": @(span)}),span % 256),
                    slice(y.name,affine(position + span / 256 * 256,@{@"outer": @(total * inner)}),span % 256))];
            [plan addObject:loop(@"outer",outer,body)]; position += span;
        }
    }
    void convolution(ANEGraphOperation *op) {
        ANEGraphValue *x = op.operands[@"x"].value, *y = op.results[0];
        ANEGraphValue *w = op.operands[@"weight"].value, *bias = op.operands[@"bias"].value;
        elements(x); elements(w); elements(y);
        auto xs = x.type.shape, ys = y.type.shape, ws = w.type.shape;
        need(xs.count == 4 && ys.count == 4 && ws.count == 4 &&
             [xs[0] isEqual:ys[0]], "spatial.conv-rank");
        NSUInteger batch = xs[0].unsignedIntegerValue, ci = xs[1].unsignedIntegerValue,
            ih = xs[2].unsignedIntegerValue, iw = xs[3].unsignedIntegerValue,
            co = ys[1].unsignedIntegerValue, oh = ys[2].unsignedIntegerValue,
            ow = ys[3].unsignedIntegerValue, k = ws[2].unsignedIntegerValue;
        auto strides = list(op.arguments[@"strides"]), dilation = list(op.arguments[@"dilations"]),
            pads = list(op.arguments[@"pad"]);
        need(strides.count == 2 && dilation.count == 2 && pads.count == 4,
             "spatial.conv-parameters");
        long long stride = integer(strides[0]), groups = integer(op.arguments[@"groups"]);
        need((stride == 1 || stride == 2) && integer(strides[1]) == stride &&
             integer(dilation[0]) == 1 && integer(dilation[1]) == 1 &&
             (k == 1 || k == 3) && ws[3].unsignedIntegerValue == k &&
             ws[0].unsignedIntegerValue == co && ci <= 1024 && co <= 2048,
             "spatial.conv-outside-envelope");
        bool depthwise = groups == (long long)ci && co == ci && k == 3;
        need(groups == 1 || depthwise, "spatial.conv-groups-outside-envelope");
        need(ws[1].unsignedIntegerValue == ci / groups, "spatial.conv-weight-shape");
        for (ANEGraphArgument *a in pads) need(integer(a) == 0, "spatial.conv-explicit-padding");
        NSString *padding = string(op.arguments[@"pad_type"]);
        bool same = [padding isEqualToString:@"same"];
        need(same || [padding isEqualToString:@"valid"], "spatial.conv-padding");
        need(same || (ih >= k && iw >= k), "spatial.conv-valid-extent");
        need(oh == (same ? (ih + stride - 1) / stride : (ih - k) / stride + 1) &&
             ow == (same ? (iw + stride - 1) / stride : (iw - k) / stride + 1),
             "spatial.conv-result-shape");
        NSData *weightData = constant(w), *biasData = bias ? constant(bias) : nil;
        if (bias) need([bias.type.shape isEqualToArray:@[@(co)]], "spatial.conv-bias-shape");
        auto weights = static_cast<const uint16_t *>(weightData.bytes);
        auto biases = static_cast<const uint16_t *>(biasData.bytes);
        NSUInteger side = depthwise || std::max(ih,iw) <= 16 ? 16 : 64;
        NSUInteger crop = k / 2, core = (side - 2 * crop + stride - 1) / stride;
        NSInteger padTop = same ? std::max<NSInteger>(0,(NSInteger)((oh - 1) * stride + k) - (NSInteger)ih) / 2 : 0;
        NSInteger padLeft = same ? std::max<NSInteger>(0,(NSInteger)((ow - 1) * stride + k) - (NSInteger)iw) / 2 : 0;
        NSUInteger patchElements = 64 * side * side;
        NSString *patch = temp(patchElements), *partial = temp(patchElements),
            *accum = temp(patchElements), *sum = temp(patchElements), *rowResult = temp(256);
        NSUInteger add = binary(@"add");
        std::vector<uint16_t> sampling(256 * 256,0);
        for (NSUInteger col = 0; col < core; ++col)
            sampling[col * 256 + crop + col * stride] = 0x3c00;
        NSUInteger sampleKernel = stride == 2 ? matrix(bytes(sampling)) : 0;
        NSMutableArray *batchBody = [NSMutableArray array];
        for (NSUInteger out = 0; out < co; out += 64) {
            NSUInteger outputChannels = MIN((NSUInteger)64,co - out);
            NSMutableArray *weightKernels = [NSMutableArray array];
            NSUInteger reductionChannels = depthwise ? 1 : ci;
            for (NSUInteger in = 0; in < reductionChannels; in += 64) {
                std::vector<uint16_t> block(64 * (depthwise ? 1 : 64) * k * k,0);
                for (NSUInteger oc = 0; oc < outputChannels; ++oc)
                    for (NSUInteger ic = 0; ic < (depthwise ? 1 : MIN((NSUInteger)64,ci - in)); ++ic)
                        for (NSUInteger tap = 0; tap < k * k; ++tap)
                            block[(oc * (depthwise ? 1 : 64) + ic) * k * k + tap] =
                                weights[((out + oc) * (depthwise ? 1 : ci) + (depthwise ? 0 : in + ic)) * k * k + tap];
                [weightKernels addObject:@(convKernel(bytes(block),k,depthwise,side))];
            }
            NSString *expandedBias = nil;
            if (bias) {
                std::vector<uint16_t> expanded(outputChannels * 256);
                for (NSUInteger c = 0; c < outputChannels; ++c)
                    std::fill(expanded.begin() + c * 256,expanded.begin() + (c + 1) * 256,biases[out + c]);
                expandedBias = constantTensor(bytes(expanded));
            }
            for (NSUInteger top = 0; top < oh; top += core) {
                NSUInteger rows = MIN(core,oh - top);
                for (NSUInteger left = 0; left < ow; left += core) {
                    NSUInteger cols = MIN(core,ow - left);
                    NSInteger inputY = (NSInteger)(top * stride) - padTop;
                    NSInteger inputX = (NSInteger)(left * stride) - padLeft;
                    NSUInteger sy = std::max<NSInteger>(0,inputY), sx = std::max<NSInteger>(0,inputX);
                    NSUInteger py = sy - inputY, px = sx - inputX;
                    NSUInteger gatherRows = sy < ih ? MIN(side - py,ih - sy) : 0;
                    NSUInteger gatherCols = sx < iw ? MIN(side - px,iw - sx) : 0;
                    for (NSUInteger chunk = 0; chunk < weightKernels.count; ++chunk) {
                        NSUInteger firstChannel = depthwise ? out : chunk * 64;
                        NSUInteger channels = MIN((NSUInteger)64,ci - firstChannel);
                        fill(batchBody,patch,patchElements);
                        if (gatherRows && gatherCols) {
                            auto src = slice(x.name,affine(firstChannel * ih * iw + sy * iw + sx,
                                @{@"batch": @(ci * ih * iw), @"channel": @(ih * iw), @"row": @(iw)}),gatherCols);
                            auto dst = slice(patch,affine(py * side + px,
                                @{@"channel": @(side * side), @"row": @(side)}),gatherCols);
                            [batchBody addObject:loop(@"channel",channels,@[loop(@"row",gatherRows,@[copy(src,dst)])])];
                        }
                        NSString *result = chunk == 0 ? accum : partial;
                        [batchBody addObject:call([weightKernels[chunk] unsignedIntegerValue],
                            @[slice(patch,@0,patchElements)],slice(result,@0,patchElements))];
                        if (chunk) {
                            id at = affine(0,@{@"sum": @256});
                            [batchBody addObject:loop(@"sum",patchElements / 256,@[call(add,
                                @[slice(accum,at,256),slice(partial,at,256)],slice(sum,at,256))])];
                            std::swap(accum,sum);
                        }
                    }
                    id sourceOffset = affine(crop * side + (stride == 2 ? 0 : crop),
                        @{@"channel": @(side * side), @"row": @(stride * side)});
                    id destinationOffset = affine(out * oh * ow + top * ow + left,
                        @{@"batch": @(co * oh * ow), @"channel": @(oh * ow), @"row": @(ow)});
                    NSMutableArray *store = [NSMutableArray array];
                    NSDictionary *sample = slice(accum,sourceOffset,stride == 2 ? side : cols);
                    if (stride == 2) {
                        [store addObject:call(sampleKernel,@[sample],slice(rowResult,@0,cols))];
                        sample = slice(rowResult,@0,cols);
                    }
                    if (bias) [store addObject:call(add,@[sample,slice(expandedBias,
                        affine(0,@{@"channel": @256}),cols)],slice(y.name,destinationOffset,cols))];
                    else [store addObject:copy(sample,slice(y.name,destinationOffset,cols))];
                    [batchBody addObject:loop(@"channel",outputChannels,@[loop(@"row",rows,store)])];
                }
            }
        }
        [plan addObject:loop(@"batch",batch,batchBody)];
    }
    void mean(ANEGraphOperation *op) {
        ANEGraphValue *x = op.operands[@"x"].value, *y = op.results[0];
        auto shape = x.type.shape;
        need(shape.count == 4, "spatial.mean-rank");
        auto axes = list(op.arguments[@"axes"]);
        need(axes.count == 2 && integer(axes[0]) == 2 && integer(axes[1]) == 3,
             "spatial.mean-axes");
        NSUInteger rows = shape[0].unsignedIntegerValue * shape[1].unsignedIntegerValue;
        NSUInteger span = shape[2].unsignedIntegerValue * shape[3].unsignedIntegerValue;
        need(span <= 256 && elements(y) == rows, "spatial.mean-outside-envelope");
        BOOL keep = boolean(op.arguments[@"keep_dims"],NO);
        need([y.type.shape isEqualToArray:keep ? @[shape[0],shape[1],@1,@1] : @[shape[0],shape[1]]],
             "spatial.mean-result-shape");
        std::vector<uint16_t> weights(256 * 256,0);
        for (NSUInteger i = 0; i < span; ++i) weights[i] = half(1. / span);
        [plan addObject:loop(@"mean",rows,@[call(matrix(bytes(weights)),
            @[slice(x.name,affine(0,@{@"mean": @(span)}),span)],
            slice(y.name,affine(0,@{@"mean": @1}),1))])];
    }
    void linear(ANEGraphOperation *op) {
        ANEGraphValue *x = op.operands[@"x"].value, *y = op.results[0],
            *w = op.operands[@"weight"].value, *bias = op.operands[@"bias"].value;
        NSUInteger k = x.type.shape.lastObject.unsignedIntegerValue,
            n = y.type.shape.lastObject.unsignedIntegerValue, rows = elements(x) / k;
        NSMutableArray *expected = [x.type.shape mutableCopy]; expected[expected.count - 1] = @(n);
        need([y.type.shape isEqualToArray:expected] &&
             [w.type.shape isEqualToArray:@[@(n),@(k)]] && k <= 1024 && n <= 1024,
             "spatial.linear-outside-envelope");
        NSUInteger size = std::max(k,n) <= 256 ? 256 : (std::max(k,n) <= 512 ? 512 : 1024);
        NSData *data = constant(w);
        const uint16_t *src = static_cast<const uint16_t *>(data.bytes);
        std::vector<uint16_t> padded(size * size,0);
        for (NSUInteger c = 0; c < n; ++c)
            std::copy(src + c * k,src + (c + 1) * k,padded.begin() + c * size);
        NSString *middle = bias ? temp(rows * n) : y.name;
        NSMutableArray *body = [NSMutableArray arrayWithObject:call(matrix(bytes(padded),size),
            @[slice(x.name,affine(0,@{@"linear": @(k)}),k)],
            slice(middle,affine(0,@{@"linear": @(n)}),n))];
        if (bias) {
            need([bias.type.shape isEqualToArray:@[@(n)]], "spatial.linear-bias-shape");
            NSString *b = constantTensor(constant(bias));
            for (NSUInteger off = 0; off < n; off += 256) {
                NSUInteger count = MIN((NSUInteger)256,n - off);
                id at = affine(off,@{@"linear": @(n)});
                [body addObject:call(binary(@"add"),@[slice(middle,at,count),slice(b,@(off),count)],slice(y.name,at,count))];
            }
        }
        [plan addObject:loop(@"linear",rows,body)];
    }
    void softmax(ANEGraphOperation *op) {
        ANEGraphValue *x = op.operands[@"x"].value, *y = op.results[0];
        NSUInteger n = x.type.shape.lastObject.unsignedIntegerValue, rank = x.type.shape.count;
        long long axis = integer(op.arguments[@"axis"]); if (axis < 0) axis += rank;
        need(axis == (long long)rank - 1 && n <= 1024 &&
             [x.type.shape isEqualToArray:y.type.shape], "spatial.softmax-outside-envelope");
        NSUInteger index;
        NSDictionary *s = @{@"kind": @"softmax"};
        if (h14) {
            auto p = ane::h14::encodeNormParity(ane::h14::NormOperation::Softmax,
                {{1024,1,1},{1024,1,1},2,true});
            index = save(p,s,ane::h14::encodeANEC(p),p.constantOffsetBytes,p.taskCount);
        } else {
            auto p = ane::h13::encodeNormParity(ane::h13::NormOperation::Softmax,
                {{1024,1,1},{1024,1,1},2,true});
            index = save(p,s,ane::h13::encodeANEC(p),p.constantOffsetBytes,p.taskCount);
        }
        // Excluded lanes must contribute exactly zero to exp/sum. Zero
        // padding would change every probability in a 1000-way softmax.
        [plan addObject:loop(@"softmax",elements(x) / n,@[call(index,
            @[slice(x.name,affine(0,@{@"softmax": @(n)}),n,0xfc00)],
            slice(y.name,affine(0,@{@"softmax": @(n)}),n))])];
    }
    void resize(ANEGraphOperation *op) {
        ANEGraphValue *x = op.operands[@"x"].value, *y = op.results[0];
        NSUInteger rank = x.type.shape.count;
        need(rank >= 3 && rank == y.type.shape.count, "spatial.resize-rank");
        for (NSUInteger d = 0; d + 2 < rank; ++d)
            need([x.type.shape[d] isEqual:y.type.shape[d]], "spatial.resize-leading-shape");
        NSUInteger ih = x.type.shape[rank - 2].unsignedIntegerValue,
            iw = x.type.shape[rank - 1].unsignedIntegerValue,
            oh = y.type.shape[rank - 2].unsignedIntegerValue,
            ow = y.type.shape[rank - 1].unsignedIntegerValue;
        NSString *name = op.operationName, *mode = @"DEFAULT";
        bool nearest = [name containsString:@"nearest"], up = [name hasPrefix:@"upsample"];
        if (up) {
            double sh = op.arguments[@"scale_factor_height"] ? number(op.arguments[@"scale_factor_height"]) : 1;
            double sw = op.arguments[@"scale_factor_width"] ? number(op.arguments[@"scale_factor_width"]) : 1;
            need(sh > 0 && sw > 0 && std::floor(ih * sh) == oh && std::floor(iw * sw) == ow,
                 "spatial.resize-scale");
            need(!nearest || (sh == std::floor(sh) && sw == std::floor(sw)),
                 "spatial.nearest-fractional-scale-outside-envelope");
            BOOL align = boolean(op.arguments[@"align_corners"],YES);
            BOOL hp = boolean(op.arguments[@"half_pixel_centers"],!align);
            need(!(align && hp), "spatial.resize-coordinate-conflict");
            mode = align ? @"STRICT_ALIGN_CORNERS" : hp ? @"UNALIGN_CORNERS" : @"DEFAULT";
        } else {
            need(integer(op.arguments[@"target_size_height"]) == (long long)oh &&
                 integer(op.arguments[@"target_size_width"]) == (long long)ow,
                 "spatial.resize-result-shape");
            if (op.arguments[@"sampling_mode"]) mode = string(op.arguments[@"sampling_mode"]);
        }
        need(!nearest || (oh % ih == 0 && ow % iw == 0),
             "spatial.nearest-fractional-scale-outside-envelope");
        need([mode isEqualToString:@"DEFAULT"] || [mode isEqualToString:@"STRICT_ALIGN_CORNERS"] ||
             [mode isEqualToString:@"UNALIGN_CORNERS"], "spatial.resize-sampling-mode-outside-envelope");
        auto coord = [&](NSUInteger i, NSUInteger n, NSUInteger out) {
            if (nearest) return (double)(i * n / out);
            if ([mode isEqualToString:@"STRICT_ALIGN_CORNERS"])
                return out > 1 ? (double)i * (n - 1) / (out - 1) : 0.;
            double f = [mode isEqualToString:@"UNALIGN_CORNERS"]
                ? (i + .5) * n / out - .5 : (double)i * n / out;
            return std::min((double)n - 1,std::max(0.,f));
        };
        NSString *a = temp(256), *b = temp(256);
        NSMutableArray *body = [NSMutableArray array];
        for (NSUInteger row = 0; row < oh; ++row) {
            double fy = coord(row,ih,oh), dy = fy - std::floor(fy);
            NSUInteger y0 = (NSUInteger)fy, y1 = MIN(ih - 1,y0 + 1);
            for (NSUInteger left = 0; left < ow;) {
                NSUInteger first = (NSUInteger)coord(left,iw,ow), count = 0, last = first;
                while (left + count < ow && count < 256) {
                    double fx = coord(left + count,iw,ow);
                    NSUInteger end = nearest ? (NSUInteger)fx : MIN(iw - 1,(NSUInteger)fx + 1);
                    if (end - first >= 256) break;
                    last = end; ++count;
                }
                need(count > 0, "spatial.resize-window");
                std::vector<uint16_t> wa(256 * 256,0), wb(256 * 256,0);
                for (NSUInteger i = 0; i < count; ++i) {
                    double fx = coord(left + i,iw,ow), dx = fx - std::floor(fx);
                    NSUInteger x0 = (NSUInteger)fx, x1 = MIN(iw - 1,x0 + 1);
                    auto put = [&](std::vector<uint16_t> &w, double factor) {
                        // Merge clamped taps before quantization.
                        if (nearest || x0 == x1) w[i * 256 + x0 - first] = half(factor);
                        else {
                            w[i * 256 + x0 - first] = half(factor * (1 - dx));
                            w[i * 256 + x1 - first] = half(factor * dx);
                        }
                    };
                    put(wa,nearest || y0 == y1 ? 1 : 1 - dy);
                    if (!nearest && y0 != y1) put(wb,dy);
                }
                id out = affine(row * ow + left,@{@"plane": @(oh * ow)});
                bool two = !nearest && y0 != y1 && dy != 0;
                [body addObject:call(matrix(bytes(wa)),@[slice(x.name,
                    affine(y0 * iw + first,@{@"plane": @(ih * iw)}),last - first + 1)],
                    slice(two ? a : y.name,two ? (id)@0 : out,count))];
                if (two) {
                    [body addObject:call(matrix(bytes(wb)),@[slice(x.name,
                        affine(y1 * iw + first,@{@"plane": @(ih * iw)}),last - first + 1)],slice(b,@0,count))];
                    [body addObject:call(binary(@"add"),@[slice(a,@0,count),slice(b,@0,count)],slice(y.name,out,count))];
                }
                left += count;
            }
        }
        [plan addObject:loop(@"plane",elements(x) / (ih * iw),body)];
    }
    void operation(ANEGraphOperation *op) {
        NSString *name = op.operationName;
        if ([name isEqualToString:@"const"]) return;
        NSDictionary *arguments = @{
            @"conv": @[@"x",@"weight",@"bias",@"strides",@"dilations",@"pad",@"pad_type",@"groups"],
            @"concat": @[@"values",@"axis",@"interleave"],
            @"resize_bilinear": @[@"x",@"target_size_height",@"target_size_width",@"sampling_mode"],
            @"resize_nearest_neighbor": @[@"x",@"target_size_height",@"target_size_width"],
            @"upsample_bilinear": @[@"x",@"scale_factor_height",@"scale_factor_width",@"align_corners",@"half_pixel_centers"],
            @"upsample_nearest_neighbor": @[@"x",@"scale_factor_height",@"scale_factor_width"],
            @"reduce_mean": @[@"x",@"axes",@"keep_dims"],
            @"linear": @[@"x",@"weight",@"bias"], @"softmax": @[@"x",@"axis"],
            @"relu": @[@"x"], @"clip": @[@"x",@"alpha",@"beta"],
        };
        need(arguments[name] != nil, "spatial.unsupported-operation");
        for (NSString *key in op.arguments)
            need([arguments[name] containsObject:key], "spatial.unsupported-argument");
        need(op.results.count == 1, "spatial.multi-result-operation");
        ANEGraphValue *y = op.results[0]; value(y,@"intermediate");
        if ([name isEqualToString:@"conv"]) convolution(op);
        else if ([name isEqualToString:@"concat"]) concat(op);
        else if ([name hasPrefix:@"resize_"] || [name hasPrefix:@"upsample_"]) {
            need([@[@"resize_bilinear",@"resize_nearest_neighbor",@"upsample_bilinear",
                   @"upsample_nearest_neighbor"] containsObject:name], "spatial.unsupported-resize");
            resize(op);
        } else if ([name isEqualToString:@"reduce_mean"]) mean(op);
        else if ([name isEqualToString:@"linear"]) linear(op);
        else if ([name isEqualToString:@"softmax"]) softmax(op);
        else if ([name isEqualToString:@"relu"] || [name isEqualToString:@"clip"]) unary(op);
        else throw std::invalid_argument("spatial.unsupported-operation");
    }
};
} // namespace

@implementation ANESpatialCompiler
+ (BOOL)compileMILData:(NSData *)milData modelRoot:(NSURL *)modelRoot
                target:(NSString *)target format:(NSString *)format
       outputDirectory:(NSURL *)directory diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error {
    ANEGraphOperation *current = nil;
    @try {
        try {
            need(([target isEqualToString:@"H13"] || [target isEqualToString:@"H14"]) &&
                 [format isEqualToString:@"anec"], "spatial.requires-h13-or-h14-anec");
            MILLexer *lexer = [[MILLexer alloc] initWithData:milData diagnostics:diagnostics];
            MILParser *parser = [[MILParser alloc] initWithTokens:lexer.lexAllTokens diagnostics:diagnostics];
            if (diagnostics.errorCount) return NO;
            MILProgramSyntax *syntax = parser.parseProgram;
            ANEGraphModule *module = syntax ? [MILGraphImporter importProgram:syntax diagnostics:diagnostics] : nil;
            if (!module || diagnostics.errorCount || ![ANEGraphVerifier verifyModule:module diagnostics:diagnostics]) return NO;
            need(module.functions.count == 1, "spatial.requires-one-function");
            ANEGraphFunction *function = module.functions[0];
            Builder builder([target isEqualToString:@"H14"],modelRoot,diagnostics);
            for (ANEGraphValue *input in function.inputs) builder.value(input,@"input");
            for (ANEGraphOperation *op in function.operations) {
                current = op;
                builder.operation(op);
            }
            need(builder.plan.count > 0, "spatial.empty-program");
            NSMutableArray *outputs = [NSMutableArray array];
            for (ANEGraphValue *v in function.returnValues) {
                need(![outputs containsObject:v.name], "spatial.duplicate-results");
                need([builder.tensors[v.name][@"role"] isEqual:@"intermediate"], "spatial.unproduced-result");
                builder.value(v,@"output"); [outputs addObject:v.name];
            }
            need(outputs.count > 0, "spatial.empty-results");
            NSMutableDictionary *constantFiles = [NSMutableDictionary dictionary];
            NSMutableDictionary *files = [NSMutableDictionary dictionary];
            for (NSUInteger i = 0; i < builder.kernels.count; ++i)
                files[builder.kernels[i][@"file"]] = builder.payloads[i];
            NSUInteger index = 0;
            for (NSString *name in [builder.constantData.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
                if (![builder.tensors[name][@"role"] isEqual:@"constant"]) continue;
                NSString *file = [NSString stringWithFormat:@"constant-%lu.fp16",(unsigned long)index++];
                constantFiles[name] = file; files[file] = builder.constantData[name];
            }
            NSDictionary *manifest = @{@"schema": @"mil-hwxc.spatial-package.v1", @"target": target,
                @"artifactFormat": @"anec", @"schedule": @"spatial", @"kernels": builder.kernels,
                @"dispatchPlan": builder.plan, @"tensors": builder.tensors, @"constants": constantFiles,
                @"outputs": outputs, @"arithmetic": @"fp16-partials"};
            NSData *json = [NSJSONSerialization dataWithJSONObject:manifest
                options:NSJSONWritingSortedKeys error:error];
            if (!json) return NO;
            NSFileManager *manager = NSFileManager.defaultManager;
            if ([manager fileExistsAtPath:directory.path]) {
                NSArray *contents = [manager contentsOfDirectoryAtPath:directory.path error:error];
                if (!contents) return NO;
                need(contents.count == 0, "spatial.output-directory-not-empty");
            } else if (![manager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:error]) return NO;
            for (NSString *file in files)
                if (![files[file] writeToURL:[directory URLByAppendingPathComponent:file]
                    options:NSDataWritingAtomic error:error]) return NO;
            return [json writeToURL:[directory URLByAppendingPathComponent:@"manifest.json"]
                options:NSDataWritingAtomic error:error];
        } catch (const std::exception &e) {
            NSString *message = [NSString stringWithUTF8String:e.what()];
            ANESourceLocation start = ANESourceLocationMake(0,1,1);
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                code:[message hasPrefix:@"spatial."] ? message : @"spatial.encoder-outside-envelope"
                message:message range:current ? current.range : ANESourceRangeMake(start,start)];
            return NO;
        }
    } @catch (NSException *exception) {
        ANESourceLocation start = ANESourceLocationMake(0,1,1);
        [diagnostics emitSeverity:ANEDiagnosticSeverityError code:@"spatial.invalid-program"
            message:exception.reason range:current ? current.range : ANESourceRangeMake(start,start)];
        return NO;
    }
}
@end
