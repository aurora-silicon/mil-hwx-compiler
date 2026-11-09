#import "H16GLUTEncoder.h"

#include "H16GRulePacket.h"

#include <cmath>
#include <cstring>
#include <limits>

static NSString *const H16GLUTEncoderErrorDomain = @"ANE.H16G.LUTEncoder";

typedef NS_ENUM(NSUInteger, H16GUnaryPacketFamily) {
    H16GUnaryPacketFamilyTable,
    H16GUnaryPacketFamilyDirectReLU,
    H16GUnaryPacketFamilyReciprocalUnit,
};

// The descriptor carries span selectors, not process addresses. Assign those
// spans here so a packet never inherits an allocation from an observed stream.
typedef struct { uint32_t offset; uint32_t length; } H16GRuleAllocation;

class H16GRuleAllocator {
public:
    H16GRuleAllocation allocate(uint64_t length, uint64_t alignment) {
        uint64_t start = (_cursor + alignment - 1) / alignment * alignment;
        uint64_t end = start + length;
        if (end > std::numeric_limits<uint32_t>::max()) return {0,0};
        _cursor = end;
        return {(uint32_t)start,(uint32_t)length};
    }
private:
    uint64_t _cursor = 0;
};

typedef struct {
    double lower;
    double upper;
    uint16_t operationCode;
    uint16_t tailFlags;
} H16GLUTRule;

static NSString *canonicalOperation(NSString *operation) {
    if ([operation isEqualToString:@"inverse"] ||
        [operation isEqualToString:@"recip"])
        return @"reciprocal";
    return operation;
}

static BOOL isTableOperation(NSString *operation) {
    return [@[@"sigmoid",@"tanh",@"gelu",@"silu",@"exp",@"log",
              @"sqrt",@"rsqrt",@"reciprocal"] containsObject:operation];
}

static H16GUnaryPacketFamily familyForOperation(NSString *operation,
                                                 BOOL *supported) {
    if ([operation isEqualToString:@"relu"]) {
        *supported = YES;
        return H16GUnaryPacketFamilyDirectReLU;
    }
    if ([operation isEqualToString:@"rsqrt"] ||
        [operation isEqualToString:@"reciprocal"]) {
        *supported = YES;
        return H16GUnaryPacketFamilyReciprocalUnit;
    }
    if (isTableOperation(operation)) {
        *supported = YES;
        return H16GUnaryPacketFamilyTable;
    }
    *supported = NO;
    return H16GUnaryPacketFamilyTable;
}

static BOOL validSquareShape(NSArray<NSNumber *> *shape, NSUInteger *size) {
    if (shape.count != 4 || shape[0].unsignedIntegerValue != 1 ||
        shape[1].unsignedIntegerValue != 1 ||
        shape[2].unsignedIntegerValue != shape[3].unsignedIntegerValue)
        return NO;
    NSUInteger candidate = shape[2].unsignedIntegerValue;
    if (candidate < 128 || candidate > 2048 || candidate % 64 != 0)
        return NO;
    if (size) *size = candidate;
    return YES;
}

static NSUInteger alignUp(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static void setError(NSError **error, NSString *message) {
    if (error) *error = [NSError errorWithDomain:H16GLUTEncoderErrorDomain
        code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
}

static uint32_t unaryControl(H16GUnaryPacketFamily family) {
    uint32_t tensorCount = family == H16GUnaryPacketFamilyReciprocalUnit ? 1 : 5;
    return (tensorCount << 16) | 9;
}

static uint32_t unaryOpcode(NSString *operation,
                            H16GUnaryPacketFamily family) {
    if (family == H16GUnaryPacketFamilyDirectReLU) return 0x00013c0c;
    if (family == H16GUnaryPacketFamilyReciprocalUnit)
        return [operation isEqualToString:@"rsqrt"] ? 0x00013c08 : 0x00013c09;
    return 0x00023c0c;
}

static H16GEncodedTDProgram *encodeUnaryPacket(NSString *operation,
                                               NSUInteger size,
                                               H16GUnaryPacketFamily family,
                                               NSError **error) {
    uint64_t elements = (uint64_t)size * size;
    uint64_t surfaceBytes = elements * sizeof(uint16_t);
    uint32_t rowBytes = (uint32_t)(size * sizeof(uint16_t));
    uint32_t tiles = (uint32_t)((elements + 0x3fff) / 0x4000);

    H16GRuleAllocator allocator;
    H16GRuleAllocation input = allocator.allocate(surfaceBytes,0x4000);
    H16GRuleAllocation output = allocator.allocate(surfaceBytes,0x4000);
    H16GRuleAllocation table = family == H16GUnaryPacketFamilyDirectReLU
        ? H16GRuleAllocation{0,0} : allocator.allocate(0x80,0x40);
    if (input.length == 0 || output.length == 0 ||
        (family != H16GUnaryPacketFamilyDirectReLU && table.length == 0)) {
        setError(error,@"unary packet allocations exceed the H16G field width");
        return nil;
    }

    // Register addresses are ABI fields; the record headers and word order are
    // constructed from this map by H16GRulePacket.
    std::map<uint32_t,uint32_t> registers;
    registers[0x0004]=(uint32_t)size;
    registers[0x0008]=(uint32_t)size;
    registers[0x000c]=1;
    registers[0x0014]=(uint32_t)size;
    registers[0x0018]=(uint32_t)size;
    registers[0x001c]=1;
    registers[0x0034]=tiles;
    registers[0x003c]=(uint32_t)((table.offset + table.length + 0x3fff) / 0x4000);
    registers[0x0040]=(uint32_t)((output.offset + output.length) / 0x4000);
    registers[0x004c]=0x00100000;
    registers[0x4104]=0x00500172;
    registers[0x4108]=0x00500130;
    registers[0x4114]=rowBytes;
    registers[0x4118]=(uint32_t)surfaceBytes;
    registers[0x411c]=(uint32_t)surfaceBytes;
    registers[0x4120]=(uint32_t)surfaceBytes;
    registers[0x4148]=0x0050017a;
    registers[0x4150]=rowBytes;
    registers[0x4154]=(uint32_t)surfaceBytes;
    registers[0x4158]=(uint32_t)surfaceBytes;
    registers[0x415c]=(uint32_t)surfaceBytes;
    registers[0x4900]=family == H16GUnaryPacketFamilyDirectReLU
        ? 0x00000000 : 0x00010082;
    registers[0x4904]=unaryOpcode(operation,family);
    registers[0x4910]=[operation isEqualToString:@"exp"] ? 0x3dc5 : 0x3c00;
    registers[0x4d00]=0x21;
    registers[0x4d08]=0xce;
    registers[0x4d10]=0;
    registers[0x4d14]=0;
    registers[0x4d18]=rowBytes;
    registers[0x4d1c]=(uint32_t)surfaceBytes;
    registers[0x4d68]=0x01002031;
    registers[0x5100]=0x31;
    registers[0x5108]=0;
    registers[0x510c]=0;
    registers[0x5110]=rowBytes;
    registers[0x5114]=(uint32_t)surfaceBytes;
    registers[0x5138]=0x01302031;
    registers[0x5900]=0x12;
    registers[0x5904]=0x00a000a0;
    registers[0x5918]=0;
    registers[0x5928]=0x007f0000;
    registers[0x592c]=0x20010701;
    BOOL hasTable=family != H16GUnaryPacketFamilyDirectReLU;
    if(hasTable){registers[0x5500]=0x000102c0;registers[0x5510]=0;
        registers[0x5514]=0;}
    try {
        ane::h16::rule::Packet packet=ane::h16::rule::build(
            registers,unaryControl(family),hasTable);
        NSMutableArray<NSNumber *> *relocations=[NSMutableArray array];
        for(std::size_t offset:packet.kernelRelocations)
            [relocations addObject:@(offset)];
        NSData *data=[NSData dataWithBytes:packet.bytes.data()
                                   length:packet.bytes.size()];
        return [[H16GEncodedTDProgram alloc] initWithData:data
            kernelRelocationOffsets:relocations
            programRecordCount:packet.recordCount
            programFormatCode:tiles-1 scratchByteLength:0];
    } catch(const std::exception &exception) {
        setError(error,[NSString stringWithUTF8String:exception.what()]);
        return nil;
    }
}

static uint16_t floatToHalf(float value) {
    uint32_t bits = 0;
    memcpy(&bits,&value,sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int exponent = (int)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffff;
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        if (mantissa & 0x1000) mantissa += 0x2000;
        return (uint16_t)(sign | (mantissa >> 13));
    }
    if (exponent >= 31)
        return (uint16_t)(sign | 0x7c00 | (mantissa ? 0x0200 : 0));
    if (mantissa & 0x1000) {
        mantissa += 0x2000;
        if (mantissa & 0x800000) {
            mantissa = 0;
            if (++exponent >= 31) return (uint16_t)(sign | 0x7c00);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | (mantissa >> 13));
}

static uint32_t packHalves(double low, double high) {
    return (uint32_t)floatToHalf((float)low) |
        ((uint32_t)floatToHalf((float)high) << 16);
}

static BOOL ruleForOperation(NSString *operation, H16GLUTRule *rule) {
    if ([operation isEqualToString:@"sigmoid"]) *rule = {-8.0,8.0,1,0};
    else if ([operation isEqualToString:@"tanh"]) *rule = {-4.0,4.0,2,0};
    else if ([operation isEqualToString:@"gelu"]) *rule = {-4.0,4.0,3,3};
    else if ([operation isEqualToString:@"silu"]) *rule = {-8.0,8.0,4,3};
    else if ([operation isEqualToString:@"exp"]) *rule = {0.0,1.0,5,2};
    else if ([operation isEqualToString:@"log"]) *rule = {0.5,1.0,6,1};
    else if ([operation isEqualToString:@"sqrt"]) *rule = {0.0,4.0,7,2};
    else if ([operation isEqualToString:@"rsqrt"]) *rule = {1.0,2.0,8,1};
    else if ([operation isEqualToString:@"reciprocal"])
        *rule = {1.0,2.0,9,1};
    else return NO;
    return YES;
}

static double evaluateTable(NSString *operation, double x) {
    if ([operation isEqualToString:@"sigmoid"]) return 1.0/(1.0+std::exp(-x));
    if ([operation isEqualToString:@"tanh"]) return std::tanh(x);
    if ([operation isEqualToString:@"gelu"])
        return 0.5*x*(1.0+std::erf(x/std::sqrt(2.0)));
    if ([operation isEqualToString:@"silu"]) return x/(1.0+std::exp(-x));
    if ([operation isEqualToString:@"exp"]) return std::exp2(x);
    if ([operation isEqualToString:@"log"]) return std::log2(x);
    if ([operation isEqualToString:@"sqrt"]) return std::sqrt(x);
    if ([operation isEqualToString:@"rsqrt"]) return 1.0/std::sqrt(x);
    return 1.0/x;
}

static NSData *buildKernelTable(NSString *operation, NSError **error) {
    H16GLUTRule rule;
    if (!ruleForOperation(operation,&rule)) {
        setError(error,@"operation has no rule-built H16G unary table");
        return nil;
    }
    NSMutableData *data = [NSMutableData dataWithLength:0x80];
    uint32_t bounds = packHalves(rule.lower,rule.upper);
    [data replaceBytesInRange:NSMakeRange(0,4) withBytes:&bounds];
    uint32_t tails = packHalves(evaluateTable(operation,rule.lower),
                               evaluateTable(operation,rule.upper));
    [data replaceBytesInRange:NSMakeRange(4,4) withBytes:&tails];
    for (NSUInteger index = 0; index < 33; ++index) {
        double t = (double)index / 32.0;
        double x = rule.lower + (rule.upper-rule.lower)*t;
        uint16_t sample = floatToHalf((float)evaluateTable(operation,x));
        [data replaceBytesInRange:NSMakeRange(8+index*2,2) withBytes:&sample];
    }
    uint32_t metadata = ((uint32_t)rule.operationCode << 16) | rule.tailFlags;
    uint32_t sampleCount = 33;
    [data replaceBytesInRange:NSMakeRange(0x50,4) withBytes:&metadata];
    [data replaceBytesInRange:NSMakeRange(0x54,4) withBytes:&sampleCount];
    return data;
}

@implementation H16GLUTEncoder
+ (BOOL)supportsOperationName:(NSString *)operationName
                   inputShape:(NSArray<NSNumber *> *)inputShape {
    BOOL supported = NO;
    familyForOperation(canonicalOperation(operationName),&supported);
    return supported && validSquareShape(inputShape,NULL);
}

+ (NSUInteger)squareViewSizeForOperationName:(NSString *)operationName
                                elementCount:(NSUInteger)elementCount {
    BOOL supported = NO;
    familyForOperation(canonicalOperation(operationName),&supported);
    if (!supported || elementCount == 0) return NSNotFound;
    NSUInteger root = (NSUInteger)std::ceil(std::sqrt((double)elementCount));
    NSUInteger size = MAX((NSUInteger)128,alignUp(root,64));
    return size <= 2048 ? size : NSNotFound;
}

+ (H16GEncodedTDProgram *)encodeOperationName:(NSString *)operationName
                                    inputShape:(NSArray<NSNumber *> *)inputShape
                                         error:(NSError **)error {
    NSString *operation = canonicalOperation(operationName);
    BOOL supported = NO;
    H16GUnaryPacketFamily family = familyForOperation(operation,&supported);
    NSUInteger size = 0;
    if (!supported || !validSquareShape(inputShape,&size)) {
        setError(error,@"unary pointwise operation requires a supported operation and square [1,1,N,N], N a multiple of 64 in [128,2048]");
        return nil;
    }
    return encodeUnaryPacket(operation,size,family,error);
}

+ (NSUInteger)inputScaleByteOffsetForOperationName:(NSString *)operationName
                                        inputShape:(NSArray<NSNumber *> *)inputShape {
    NSString *operation = canonicalOperation(operationName);
    NSUInteger size=0;
    if(!validSquareShape(inputShape,&size)||!isTableOperation(operation))
        return NSNotFound;
    BOOL supported=NO;
    H16GUnaryPacketFamily family=familyForOperation(operation,&supported);
    H16GEncodedTDProgram *program=encodeUnaryPacket(operation,size,family,nil);
    if(!program)return NSNotFound;
    std::size_t offset=ane::h16::rule::registerValueOffset(
        (const uint8_t *)program.data.bytes,program.data.length,0x4910);
    return offset==SIZE_MAX?NSNotFound:(NSUInteger)offset;
}

+ (double)inputScaleValueForOperationName:(NSString *)operationName {
    NSString *operation = canonicalOperation(operationName);
    if (!isTableOperation(operation)) return NAN;
    return [operation isEqualToString:@"exp"] ? M_LOG2E : 1.0;
}

+ (NSData *)constantRegionForOperationName:(NSString *)operationName
                                      error:(NSError **)error {
    NSString *operation = canonicalOperation(operationName);
    if ([operation isEqualToString:@"relu"]) return [NSData data];
    return buildKernelTable(operation,error);
}
@end
