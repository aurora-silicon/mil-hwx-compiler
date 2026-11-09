#import "H16GReduceEncoder.h"

#include "H16GRulePacket.h"

#include <cstring>
#include <limits>

static NSString *const H16GReduceEncoderErrorDomain = @"ANE.H16G.ReduceEncoder";

typedef NS_ENUM(uint32_t, H16GReductionKind) {
    H16GReductionKindSum = 0,
    H16GReductionKindMean = 1,
    H16GReductionKindMax = 2,
};

typedef struct { uint32_t offset; uint32_t length; } H16GRuleAllocation;

class H16GReductionAllocator {
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

@interface H16GReduceEncoding ()
@property(nonatomic, readwrite, strong) H16GEncodedTDProgram *tdProgram;
@property(nonatomic, readwrite, copy) NSArray<NSNumber *> *outputShape;
@property(nonatomic, readwrite) NSUInteger taskCount;
@property(nonatomic, readwrite) NSUInteger inputRowStrideBytes;
@property(nonatomic, readwrite) NSUInteger inputPlaneStrideBytes;
@property(nonatomic, readwrite) NSUInteger inputBatchStrideBytes;
@property(nonatomic, readwrite) NSUInteger inputStorageByteLength;
@property(nonatomic, readwrite) NSUInteger outputRowStrideBytes;
@property(nonatomic, readwrite) NSUInteger outputPlaneStrideBytes;
@property(nonatomic, readwrite) NSUInteger outputBatchStrideBytes;
@property(nonatomic, readwrite) NSUInteger outputStorageByteLength;
@end

@implementation H16GReduceEncoding
@end

static NSUInteger alignUp(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static void setError(NSError **error, NSString *message) {
    if (error) *error = [NSError errorWithDomain:H16GReduceEncoderErrorDomain
        code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
}

static BOOL reductionKind(NSString *operation, H16GReductionKind *kind) {
    if ([operation isEqualToString:@"reduce_sum"])
        *kind = H16GReductionKindSum;
    else if ([operation isEqualToString:@"reduce_mean"])
        *kind = H16GReductionKindMean;
    else if ([operation isEqualToString:@"reduce_max"])
        *kind = H16GReductionKindMax;
    else return NO;
    return YES;
}

static BOOL validGeometry(NSArray<NSNumber *> *shape, NSUInteger axis) {
    if (shape.count != 4 || shape[0].unsignedIntegerValue != 1 ||
        axis < 1 || axis > 3) return NO;
    uint64_t elements = 1;
    for (NSNumber *dimension in shape) {
        NSUInteger value = dimension.unsignedIntegerValue;
        if (value == 0 || value > UINT16_MAX) return NO;
        elements *= value;
        if (elements > UINT32_MAX / sizeof(uint16_t)) return NO;
    }
    return shape[axis].unsignedIntegerValue > 1;
}

static NSArray<NSNumber *> *reducedShape(NSArray<NSNumber *> *inputShape,
                                          NSUInteger axis) {
    NSMutableArray<NSNumber *> *shape = [inputShape mutableCopy];
    shape[axis] = @1;
    return shape;
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
    if (exponent >= 31) return (uint16_t)(sign | 0x7c00);
    if (mantissa & 0x1000) mantissa += 0x2000;
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | (mantissa >> 13));
}

static H16GEncodedTDProgram *encodeReductionPacket(
    H16GReductionKind kind, NSArray<NSNumber *> *inputShape,
    NSArray<NSNumber *> *outputShape, NSUInteger axis,
    NSUInteger inputRowBytes, NSUInteger inputPlaneBytes,
    NSUInteger inputBatchBytes, NSUInteger outputRowBytes,
    NSUInteger outputPlaneBytes, NSUInteger outputBatchBytes,
    NSError **error) {
    NSUInteger channels = inputShape[1].unsignedIntegerValue;
    NSUInteger height = inputShape[2].unsignedIntegerValue;
    NSUInteger width = inputShape[3].unsignedIntegerValue;
    NSUInteger outputChannels = outputShape[1].unsignedIntegerValue;
    NSUInteger outputHeight = outputShape[2].unsignedIntegerValue;
    NSUInteger outputWidth = outputShape[3].unsignedIntegerValue;
    NSUInteger reductionLength = inputShape[axis].unsignedIntegerValue;
    uint64_t inputElements = (uint64_t)channels * height * width;
    uint32_t tiles = (uint32_t)((inputElements + 0x3fff) / 0x4000);

    H16GReductionAllocator allocator;
    H16GRuleAllocation input = allocator.allocate(inputBatchBytes,0x4000);
    H16GRuleAllocation partial = allocator.allocate(outputBatchBytes,0x4000);
    H16GRuleAllocation output = allocator.allocate(outputBatchBytes,0x4000);
    if (input.length == 0 || partial.length == 0 || output.length == 0) {
        setError(error,@"reduction packet allocations exceed the H16G field width");
        return nil;
    }

    std::map<uint32_t,uint32_t> registers;
    registers[0x0004]=(uint32_t)width;
    registers[0x0008]=(uint32_t)height;
    registers[0x000c]=(uint32_t)channels;
    registers[0x0014]=(uint32_t)outputWidth;
    registers[0x0018]=(uint32_t)outputHeight;
    registers[0x001c]=(uint32_t)outputChannels;
    registers[0x0024]=(uint32_t)axis;
    registers[0x0034]=(uint32_t)(partial.offset/0x40);
    registers[0x003c]=(uint32_t)(output.offset/0x40);
    registers[0x0040]=tiles;
    registers[0x004c]=0x00100000;
    registers[0x4104]=0x00500172;
    registers[0x4108]=0x00500130;
    registers[0x4114]=(uint32_t)inputRowBytes;
    registers[0x4118]=(uint32_t)inputPlaneBytes;
    registers[0x411c]=(uint32_t)inputBatchBytes;
    registers[0x4120]=(uint32_t)inputBatchBytes;
    registers[0x4148]=0x0050017a;
    registers[0x4150]=(uint32_t)outputRowBytes;
    registers[0x4154]=(uint32_t)outputPlaneBytes;
    registers[0x4158]=(uint32_t)outputBatchBytes;
    registers[0x415c]=(uint32_t)outputBatchBytes;
    registers[0x4900]=(uint32_t)axis;
    registers[0x4904]=0x000103c0 | ((uint32_t)kind << 2);
    registers[0x4910]=kind == H16GReductionKindMean
        ? floatToHalf(1.0f/(float)reductionLength) : floatToHalf(1.0f);
    registers[0x4d00]=0x21;
    registers[0x4d08]=0xce;
    registers[0x4d10]=0;
    registers[0x4d14]=0;
    registers[0x4d18]=(uint32_t)inputRowBytes;
    registers[0x4d1c]=(uint32_t)inputPlaneBytes;
    registers[0x4d24]=(uint32_t)inputBatchBytes;
    registers[0x4d68]=0x01002031;
    registers[0x5100]=0x31;
    registers[0x5108]=0;
    registers[0x510c]=0;
    registers[0x5110]=(uint32_t)outputRowBytes;
    registers[0x5114]=(uint32_t)outputPlaneBytes;
    registers[0x511c]=(uint32_t)outputBatchBytes;
    registers[0x5138]=0x01302031;
    registers[0x5900]=0x12;
    registers[0x5904]=0x00a000a0;
    registers[0x5918]=0;
    registers[0x5928]=0x007f0000;
    registers[0x592c]=0x20010701;
    try {
        uint32_t control=((kind==H16GReductionKindMax?1u:5u)<<16)|9u;
        ane::h16::rule::Packet packet=ane::h16::rule::build(
            registers,control,false);
        NSData *data=[NSData dataWithBytes:packet.bytes.data()
                                   length:packet.bytes.size()];
        return [[H16GEncodedTDProgram alloc] initWithData:data
            kernelRelocationOffsets:@[] programRecordCount:packet.recordCount
            programFormatCode:tiles-1 scratchByteLength:0];
    } catch(const std::exception &exception) {
        setError(error,[NSString stringWithUTF8String:exception.what()]);
        return nil;
    }
}

@implementation H16GReduceEncoder
+ (BOOL)supportsOperationName:(NSString *)operationName
                   inputShape:(NSArray<NSNumber *> *)inputShape
                         axis:(NSUInteger)axis {
    H16GReductionKind kind;
    return reductionKind(operationName,&kind) && validGeometry(inputShape,axis);
}

+ (H16GReduceEncoding *)encodeOperationName:(NSString *)operationName
                                    inputShape:(NSArray<NSNumber *> *)inputShape
                                          axis:(NSUInteger)axis
                                         error:(NSError **)error {
    H16GReductionKind kind;
    if (!reductionKind(operationName,&kind) || !validGeometry(inputShape,axis)) {
        setError(error,@"reduction requires reduce_sum/reduce_mean/reduce_max, a non-empty [1,C,H,W] fp16 shape, and one non-unit axis in {1,2,3}");
        return nil;
    }
    NSArray<NSNumber *> *outputShape = reducedShape(inputShape,axis);
    NSUInteger inputChannels=inputShape[1].unsignedIntegerValue;
    NSUInteger inputHeight=inputShape[2].unsignedIntegerValue;
    NSUInteger inputWidth=inputShape[3].unsignedIntegerValue;
    NSUInteger inputRowBytes=alignUp(inputWidth * sizeof(uint16_t),64);
    NSUInteger inputPlaneBytes=inputHeight * inputRowBytes;
    NSUInteger inputBatchBytes=inputChannels * inputPlaneBytes;
    NSUInteger channels=outputShape[1].unsignedIntegerValue;
    NSUInteger height=outputShape[2].unsignedIntegerValue;
    NSUInteger width=outputShape[3].unsignedIntegerValue;
    NSUInteger rowBytes=alignUp(width * sizeof(uint16_t),64);
    NSUInteger planeBytes=height * rowBytes;
    NSUInteger batchBytes=channels * planeBytes;
    H16GEncodedTDProgram *program=encodeReductionPacket(kind,inputShape,
        outputShape,axis,inputRowBytes,inputPlaneBytes,inputBatchBytes,
        rowBytes,planeBytes,batchBytes,error);
    if (!program) return nil;
    H16GReduceEncoding *encoding = [H16GReduceEncoding new];
    encoding.tdProgram=program; encoding.outputShape=outputShape;
    encoding.taskCount=1;
    encoding.inputRowStrideBytes=inputRowBytes;
    encoding.inputPlaneStrideBytes=inputPlaneBytes;
    encoding.inputBatchStrideBytes=inputBatchBytes;
    encoding.inputStorageByteLength=inputBatchBytes;
    encoding.outputRowStrideBytes=rowBytes;
    encoding.outputPlaneStrideBytes=planeBytes;
    encoding.outputBatchStrideBytes=batchBytes;
    encoding.outputStorageByteLength=batchBytes;
    return encoding;
}
@end
