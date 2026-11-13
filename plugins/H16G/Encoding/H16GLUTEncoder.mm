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

static uint32_t nextPowerOfTwo(uint32_t value) {
    uint32_t result=1;
    while(result<value) result<<=1;
    return result;
}

static void setError(NSError **error, NSString *message) {
    if (error) *error = [NSError errorWithDomain:H16GLUTEncoderErrorDomain
        code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
}

static uint16_t floatToHalf(float value);

static uint32_t unaryEngineCount(H16GUnaryPacketFamily family) {
    // Table/direct packets enable five NE lanes; reciprocal-family packets
    // use the single reciprocal execution lane.
    return family == H16GUnaryPacketFamilyReciprocalUnit ? 1 : 5;
}

static uint32_t unaryOpcode(NSString *operation,
                            H16GUnaryPacketFamily family) {
    using namespace ane::h16::rule::fields;
    if (family == H16GUnaryPacketFamilyDirectReLU)
        return unaryCommand(1,UnaryOpcode::Direct);
    if (family == H16GUnaryPacketFamilyReciprocalUnit)
        return unaryCommand(1,UnaryOpcode::Reciprocal,
            ![operation isEqualToString:@"rsqrt"]);
    return unaryCommand(2,UnaryOpcode::Direct);
}

static H16GEncodedTDProgram *encodeUnaryPacket(NSString *operation,
                                               NSUInteger size,
                                               H16GUnaryPacketFamily family,
                                               NSError **error) {
    uint64_t elements = (uint64_t)size * size;
    uint64_t surfaceBytes = elements * sizeof(uint16_t);
    uint32_t rowBytes = (uint32_t)(size * sizeof(uint16_t));
    // L2 walks one hardware tile at a time.  The row is power-of-two padded,
    // but a tile never exposes more than 256 fp16 lanes; whole-surface byte
    // counts here make every 32nd row bypass the unary unit.
    uint32_t l2RowBytes=std::min(nextPowerOfTwo(rowBytes),uint32_t(512));
    constexpr uint32_t allocationPageBytes=uint32_t(1)<<14;
    constexpr uint32_t tableBytes=128;
    constexpr uint32_t tableAlignment=64;
    uint32_t tiles = (uint32_t)((elements + allocationPageBytes-1) /
                                allocationPageBytes);

    H16GRuleAllocator allocator;
    H16GRuleAllocation input = allocator.allocate(surfaceBytes,allocationPageBytes);
    H16GRuleAllocation output = allocator.allocate(surfaceBytes,allocationPageBytes);
    H16GRuleAllocation table = family == H16GUnaryPacketFamilyDirectReLU
        ? H16GRuleAllocation{0,0} : allocator.allocate(tableBytes,tableAlignment);
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
    registers[0x003c]=(uint32_t)((table.offset + table.length +
        allocationPageBytes-1) / allocationPageBytes);
    registers[0x0040]=(uint32_t)((output.offset + output.length) /
        allocationPageBytes);
    using namespace ane::h16::rule::fields;
    registers[0x004c]=surfaceLimit(1u << 20);
    registers[0x4104]=l2Transfer(5,7,2);
    registers[0x4108]=l2Transfer(5,3,0);
    registers[0x4114]=l2RowBytes;
    registers[0x4118]=l2RowBytes;
    registers[0x411c]=l2RowBytes;
    registers[0x4120]=l2RowBytes;
    registers[0x4148]=l2Transfer(5,7,10);
    registers[0x414c]=unaryOutputTileBase();
    registers[0x4900]=family == H16GUnaryPacketFamilyDirectReLU
        ? 0u : neuralInput();
    registers[0x4904]=unaryOpcode(operation,family);
    registers[0x4910]=floatToHalf([operation isEqualToString:@"exp"]
        ? (float)M_LOG2E : 1.0f);
    registers[0x4d00]=endpoint(2);
    registers[0x4d08]=sourceDMAFlags();
    registers[0x4d10]=0;
    registers[0x4d14]=0;
    registers[0x4d18]=rowBytes;
    registers[0x4d1c]=(uint32_t)surfaceBytes;
    registers[0x4d68]=surfaceDescriptor(0,2);
    registers[0x5100]=resultEndpoint(3);
    registers[0x5108]=0;
    registers[0x510c]=0;
    registers[0x5110]=rowBytes;
    registers[0x5114]=(uint32_t)surfaceBytes;
    registers[0x5138]=surfaceDescriptor(3,2);
    registers[0x5900]=storeFlags();
    registers[0x5904]=storeGeometry();
    registers[0x5918]=0;
    registers[0x5928]=storeCompletionMask();
    registers[0x592c]=outputSurface();
    BOOL hasTable=family != H16GUnaryPacketFamilyDirectReLU;
    if(hasTable){
        registers[0x5500]=lookupKernelCommand();
        registers[0x5610]=lookupKernelStream(2);
        registers[0x5614]=tableBytes;
        registers[0x5618]=0;
        registers[0x561c]=0;
    }
    try {
        ane::h16::rule::Packet packet=ane::h16::rule::build(
            registers,unaryEngineCount(family),hasTable);
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
    constexpr uint32_t floatMantissaMask=(uint32_t(1)<<23)-1;
    constexpr uint32_t floatHiddenBit=uint32_t(1)<<23;
    constexpr uint32_t halfSignMask=uint32_t(1)<<15;
    constexpr uint32_t halfRoundBit=uint32_t(1)<<12;
    constexpr uint32_t halfRoundIncrement=uint32_t(1)<<13;
    constexpr uint32_t halfInfinity=uint32_t(31)<<10;
    constexpr uint32_t halfQuietNaN=uint32_t(1)<<9;
    uint32_t bits = 0;
    memcpy(&bits,&value,sizeof(bits));
    uint32_t sign = (bits >> 16) & halfSignMask;
    int exponent = (int)((bits >> 23) & ((uint32_t(1)<<8)-1)) - 127 + 15;
    uint32_t mantissa = bits & floatMantissaMask;
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;
        mantissa = (mantissa | floatHiddenBit) >> (1 - exponent);
        if (mantissa & halfRoundBit) mantissa += halfRoundIncrement;
        return (uint16_t)(sign | (mantissa >> 13));
    }
    if (exponent >= 31)
        return (uint16_t)(sign | halfInfinity |
                          (mantissa ? halfQuietNaN : 0));
    if (mantissa & halfRoundBit) {
        mantissa += halfRoundIncrement;
        if (mantissa & floatHiddenBit) {
            mantissa = 0;
            if (++exponent >= 31) return (uint16_t)(sign | halfInfinity);
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

static NSData *buildStandaloneKernelTable(NSString *operation,
                                           NSError **error) {
    H16GLUTRule rule;
    if (!ruleForOperation(operation,&rule)) {
        setError(error,@"operation has no rule-built H16G unary table");
        return nil;
    }
    NSMutableData *data = [NSMutableData dataWithLength:128];
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
    [data replaceBytesInRange:NSMakeRange(80,4) withBytes:&metadata];
    [data replaceBytesInRange:NSMakeRange(84,4) withBytes:&sampleCount];
    return data;
}

static NSData *buildConsumerCompatibleKernelTable(NSString *operation,
                                                   NSError **error) {
    NSMutableData *data=[buildStandaloneKernelTable(operation,error) mutableCopy];
    if (!data) return nil;
    if ([operation isEqualToString:@"exp"]) {
        // The consumer-facing exponential table uses range reduction before
        // interpolation. These are, in order: input clamp low, input clamp
        // high, output below range, and output above range. The samples still
        // cover 2^x on [0,1].
        uint32_t inputClamps=packHalves(-25.0,16.0);
        uint32_t outputTails=packHalves(0.0,
            std::numeric_limits<double>::infinity());
        uint32_t operationControl=(uint32_t)5 << 16;
        uint32_t rangeReductionControl=10;
        [data replaceBytesInRange:NSMakeRange(0,4) withBytes:&inputClamps];
        [data replaceBytesInRange:NSMakeRange(4,4) withBytes:&outputTails];
        [data replaceBytesInRange:NSMakeRange(80,4) withBytes:&operationControl];
        [data replaceBytesInRange:NSMakeRange(84,4)
                        withBytes:&rangeReductionControl];
    } else if ([operation isEqualToString:@"reciprocal"]) {
        // Reciprocal consumers likewise range-reduce their input. These are
        // input clamp low/high followed by output below/above range. The
        // reduced table stores 2/x for x in [1,2].
        uint32_t inputClamps=packHalves(0.0,
            std::numeric_limits<double>::infinity());
        uint32_t outputTails=packHalves(
            std::numeric_limits<double>::infinity(),0.0);
        [data replaceBytesInRange:NSMakeRange(0,4) withBytes:&inputClamps];
        [data replaceBytesInRange:NSMakeRange(4,4) withBytes:&outputTails];
        for (NSUInteger index=0;index<33;++index) {
            double x=1.0+(double)index/32.0;
            uint16_t sample=floatToHalf((float)(2.0/x));
            [data replaceBytesInRange:NSMakeRange(8+index*2,2)
                            withBytes:&sample];
        }
        uint32_t operationControl=(uint32_t)6 << 16;
        uint32_t rangeReductionControl=103;
        [data replaceBytesInRange:NSMakeRange(80,4) withBytes:&operationControl];
        [data replaceBytesInRange:NSMakeRange(84,4)
                        withBytes:&rangeReductionControl];
    }
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
    return buildConsumerCompatibleKernelTable(operation,error);
}

+ (NSData *)standaloneConstantRegionForOperationName:(NSString *)operationName
                                               error:(NSError **)error {
    NSString *operation=canonicalOperation(operationName);
    if ([operation isEqualToString:@"relu"])return [NSData data];
    return buildStandaloneKernelTable(operation,error);
}
@end
