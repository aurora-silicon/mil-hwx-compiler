#import <Foundation/Foundation.h>
#ifdef __APPLE__
#import <CommonCrypto/CommonDigest.h>
#else
#include <openssl/sha.h>
#define CC_SHA256_DIGEST_LENGTH SHA256_DIGEST_LENGTH
using CC_LONG = size_t;
#define CC_SHA256(data, length, digest) \
    SHA256((const unsigned char *)(data), (length), (digest))
#endif

#import "H16GConvEncoder.h"
#import "H16GConvChainEncoder.h"
#import "H16GALUEncoder.h"
#import "H16GBroadcastALUEncoder.h"
#import "H16GDepthwiseEncoder.h"
#import "H16GLayoutEncoder.h"
#import "H16GLayoutConvChainEncoder.h"
#import "H16GLUTEncoder.h"
#import "H16GMatmulEncoder.h"
#import "H16GMatrixRowDivisionEncoder.h"
#import "H16GRegularConvEncoder.h"
#import "H16GReduceEncoder.h"
#include "H16GRulePacket.h"
#import "H16GTDWriter.h"
#import "H16GTarget.h"
#import "H16GTaskEncoder.h"

#include <stdio.h>
#include <cstring>

static int failures = 0;
static NSString *sha256(NSData *data);
static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    failures++;
}

static void testWriterBoundsAndZeroInitialization(void) {
    H16GTDWriter *writer = [[H16GTDWriter alloc] initWithByteLength:16];
    NSError *error = nil;
    expect([writer writeUInt32:0x11223344 atOffset:4 field:@"test" error:&error],
           @"named field write succeeds");
    NSData *data = writer.data;
    const uint32_t *words = (const uint32_t *)data.bytes;
    expect(words[0] == 0 && words[1] == 0x11223344 && words[2] == 0 && words[3] == 0,
           @"descriptor starts zeroed and only named field changes");
    expect(![writer writeUInt32:1 atOffset:14 field:@"overflow" error:&error],
           @"unaligned/out-of-range field write is rejected");
}

static void testConvReluDescriptorIsFullyFieldEncoded(void) {
    NSError *error = nil;
    NSData *generated = [H16GConvEncoder encodeConv1x1WithInputChannels:64
        outputChannels:64 spatial:64 bytesPerWeight:2
        numericMode:ANELegalNumericModeFP16 reluEpilogue:YES error:&error];
    expect(generated != nil && generated.length == 0x1d8,
           @"Conv encoder emits one complete 472-byte TD");
    expect([[sha256(generated) lowercaseString] isEqualToString:
        @"62a90244a4edc71a70d6f0630a2d72371b7dcb39d5194d22369781cb54094bff"],
           @"zero-buffer field encoding matches the measured TD hash");
}

static void testGeometryChangesAreComputed(void) {
    NSError *error = nil;
    NSData *td = [H16GConvEncoder encodeConv1x1WithInputChannels:64
        outputChannels:64 spatial:32 bytesPerWeight:2
        numericMode:ANELegalNumericModeFP16 reluEpilogue:NO error:&error];
    const uint32_t *words = (const uint32_t *)td.bytes;
    expect(words[0x104 / 4] == 32 && words[0x138 / 4] == 64,
           @"spatial and row-byte fields derive from requested geometry");
    expect(words[0x13c / 4] == 32 * 32 * 2,
           @"plane bytes derive from geometry");
    expect(words[0x84 / 4] == 64 * 64 / 8,
           @"weight tap step derives from the decoded H16G tap geometry");
    expect(words[0x174 / 4] == 0x00103c00,
           @"plain Conv does not acquire the ReLU epilogue bit");
}

typedef struct {
    NSUInteger inputChannels;
    NSUInteger outputChannels;
    NSUInteger spatial;
    uint32_t descriptorCount;
    uint32_t bankRows;
    uint32_t tileMode;
    uint32_t tileGroups;
    uint32_t window;
    uint32_t strideA;
    uint32_t strideB;
    uint32_t inputBytes;
} ConvGeometryExpectation;

static void testMeasuredConvGeometryTable(void) {
    static const ConvGeometryExpectation rows[] = {
        {128,256,32,0x0c,0x20,0x220000,0x4,0x2050,0x2000,0x2000,0x40a00},
        {160,160,64,0x27,0x40,0x210000,0x4,0x5000,0x5000,0x5000,0x140000},
        {192,192,64,0x2f,0x40,0x210000,0x4,0x6000,0x6000,0x6000,0x180000},
        {256,256,128,0xf7,0x01,0x200000,0x4,0x10000,0x10000,0x10000,0x10000},
        {256,256,32,0x11,0x20,0x220000,0x4,0x4050,0x4000,0x4000,0x80a00},
        {256,256,64,0x3f,0x02,0x210000,0x4,0x8000,0x8000,0x8000,0x10000},
        {320,320,64,0x4f,0x02,0x210000,0x4,0xa000,0xa000,0xa000,0x14000},
        {32,32,64,0x07,0x04,0x1a0000,0xa,0x1050,0x1000,0x1000,0x4140},
        {384,384,64,0x60,0x02,0x210000,0x4,0xc000,0xc000,0xc000,0x18000},
        {448,448,64,0x71,0x22,0x210000,0x4,0xe000,0xe000,0xe000,0x1dc000},
        {512,512,128,0x1f2,0x01,0x200000,0x4,0x20000,0x20000,0x20000,0x20000},
        {512,512,32,0x26,0x20,0x222200,0x4,0x8050,0x8000,0x8000,0x100a00},
        {512,512,64,0x82,0x02,0x210000,0x4,0x10000,0x10000,0x10000,0x20000},
        {64,64,128,0x3d,0x80,0x211100,0x2,0x2000,0x2000,0x2000,0x100000},
        {64,64,32,0x03,0x20,0x220000,0x2,0x1050,0x1000,0x1000,0x20a00},
        {64,64,64,0x0f,0x40,0x210000,0x2,0x2000,0x2000,0x2000,0x80000},
    };
    for (const ConvGeometryExpectation &row : rows) {
        NSError *error = nil;
        NSData *td = [H16GConvEncoder
            encodeConv1x1WithInputChannels:row.inputChannels
            outputChannels:row.outputChannels spatial:row.spatial
            bytesPerWeight:2 numericMode:ANELegalNumericModeFP16
            reluEpilogue:NO error:&error];
        NSString *label = [NSString stringWithFormat:@"C%lux%lu S%lu",
            (unsigned long)row.inputChannels,
            (unsigned long)row.outputChannels,
            (unsigned long)row.spatial];
        expect(td != nil, [label stringByAppendingString:@" is admitted"]);
        if (!td) continue;
        const uint32_t *words = (const uint32_t *)td.bytes;
        BOOL matches = words[0x014/4] == row.descriptorCount &&
            words[0x120/4] == row.bankRows &&
            words[0x124/4] == row.tileMode &&
            words[0x128/4] == row.tileGroups &&
            words[0x15c/4] == row.window &&
            words[0x160/4] == row.strideA &&
            words[0x164/4] == row.strideB &&
            words[0x16c/4] == row.inputBytes &&
            words[0x084/4] == row.inputChannels * row.outputChannels / 8;
        expect(matches, [label stringByAppendingString:
            @" selects its measured SRAM and tap fields"]);
    }
    NSError *error = nil;
    expect(![H16GConvEncoder supportsConv1x1WithInputChannels:96
        outputChannels:96 spatial:64] &&
        [H16GConvEncoder encodeConv1x1WithInputChannels:96
            outputChannels:96 spatial:64 bytesPerWeight:2
            numericMode:ANELegalNumericModeFP16 reluEpilogue:NO
            error:&error] == nil && error != nil,
        @"an unmeasured Conv geometry fails closed");
}

static void testDepthwisePacketsMatchMeasuredFamilies(void) {
    NSArray<NSNumber *> *channelCases = @[@64,@128,@256,@512];
    NSArray<NSString *> *hashes = @[
        @"b3d0e231a447aa05d25f8323351e93b3ecc52bb823c53dd56a11b9861c918f12",
        @"3523d9b8bdfa9ebaec1846b374f786c5e9e8e8e236af8a26c141741d6e85a6b2",
        @"5e9e1ffee51bbd55733b290c11220019c4cd3b9c051a5f9e37b1e54c0a8ba061",
        @"ba8485a730bfdd5ccc6139ce845e68b3d71f07e63c4cedd6b2308b93b851eb7d",
    ];
    for (NSUInteger caseIndex = 0; caseIndex < channelCases.count; ++caseIndex) {
        NSError *error = nil;
        NSUInteger channels = channelCases[caseIndex].unsignedIntegerValue;
        NSData *td = [H16GDepthwiseEncoder encode3x3WithChannels:channels
            spatial:64 error:&error];
        expect(td.length == 0x1e0 && error == nil &&
            [[sha256(td) lowercaseString] isEqualToString:hashes[caseIndex]],
            @"depthwise 3x3 fields reproduce an independent complete TD oracle");
    }
    NSError *error = nil;
    expect([H16GDepthwiseEncoder encode3x3WithChannels:96 spatial:64
        error:&error] == nil && error != nil,
        @"unmeasured depthwise geometry fails closed");
}

static void testRegularConvPacketsMatchMeasuredFamilies(void) {
    NSArray<NSArray<NSNumber *> *> *cases = @[
        @[@64,@32,@3], @[@64,@64,@3], @[@64,@64,@5],
        @[@128,@32,@3], @[@128,@64,@3], @[@128,@64,@5],
    ];
    NSArray<NSString *> *hashes = @[
        @"b3e901114155bb7f0ee600db47051c4c9175dfbece5cf741760dc272e4a6bb9a",
        @"36e2027bb14cd552b2166f1f11762f7cfae4511673548d808a11e78d1619dc59",
        @"d1531aee412df2780fb527869138c6f5068895b3824253b936a718ea10a08987",
        @"721c64c7602564bb8f25be6fad60636b50919d40b5a044e5e6fa1cecf7264638",
        @"5a6ee548fc5a37a809a02f1a6faf93188b5b2f3de8d0e6f177e7c3ab1adcdf96",
        @"e228b964f0c1b91df3229a4f20bc3d8dc46ecee1a8b46760acf37d59d3662eb4",
    ];
    for (NSUInteger caseIndex = 0; caseIndex < cases.count; ++caseIndex) {
        NSUInteger channels = cases[caseIndex][0].unsignedIntegerValue;
        NSUInteger spatial = cases[caseIndex][1].unsignedIntegerValue;
        NSUInteger kernel = cases[caseIndex][2].unsignedIntegerValue;
        NSError *error = nil;
        NSData *td = [H16GRegularConvEncoder encodeWithChannels:channels
            spatial:spatial kernel:kernel error:&error];
        expect(td != nil && error == nil &&
            [[sha256(td) lowercaseString] isEqualToString:hashes[caseIndex]],
            @"regular 3x3/5x5 fields reproduce an independent complete TD oracle");
    }
    NSError *error = nil;
    expect([H16GRegularConvEncoder encodeWithChannels:96 spatial:64
        kernel:3 error:&error] == nil && error != nil,
        @"unmeasured regular Conv geometry fails closed");
}

static void testSquareMatmulPacketsFollowDecodedFamilies(void) {
    NSArray<NSNumber *> *singleSizes = @[@128,@256,@512];
    NSArray<NSString *> *singleHashes = @[
        @"bf43e48293566a79477756859f746877f91777189ca9e1cb242e4dda24066c60",
        @"c0c92aa4f1bbda7fc917d01914c854fd45869c84663cb7d278a23ee301cfbe73",
        @"1090b6fb996e4d129079342a4b006ab249bd0cccb6b09ffdadac9c24ff7f990e",
    ];
    for (NSUInteger index = 0; index < singleSizes.count; ++index) {
        NSError *error = nil;
        H16GEncodedTDProgram *program = [H16GMatmulEncoder
            encodeSquareSize:singleSizes[index].unsignedIntegerValue error:&error];
        expect(program != nil && error == nil &&
               [[sha256(program.data) lowercaseString]
                    isEqualToString:singleHashes[index]] &&
               program.programRecordCount == 31 &&
               program.kernelRelocationOffsets.count == 0,
               @"single-tile matmul reproduces its complete decoded TD family");
    }

    NSError *error = nil;
    H16GEncodedTDProgram *tiled = [H16GMatmulEncoder
        encodeSquareSize:768 error:&error];
    expect(tiled != nil && error == nil && tiled.data.length == 0x6d4 &&
           tiled.programRecordCount == 46 &&
           tiled.scratchByteLength == 0x1e0000 &&
           [H16GMatmulEncoder tileCountForSquareSize:768] == 3 &&
           [H16GMatmulEncoder rowsPerTileForSquareSize:768] == 256,
           @"tiled matmul derives packet count, row slabs and working set from N");
    expect([H16GMatmulEncoder tileCountForSquareSize:2176] == 17 &&
           [H16GMatmulEncoder rowsPerTileForSquareSize:2176] == 128,
           @"large matmul selects the decoded 128-row packet family");

    for (NSNumber *unsupported in @[@384,@640,@5000]) {
        error = nil;
        expect([H16GMatmulEncoder encodeSquareSize:
            unsupported.unsignedIntegerValue error:&error] == nil && error != nil,
            @"matmul shapes outside decoded packet families fail closed");
    }
}

static void testBinaryALUPacketsFollowMeasuredGeometry(void) {
    NSArray<NSNumber *> *sizes=@[@128,@256,@512,@1024,@2048];
    NSArray<NSString *> *hashes=@[
        @"f771fd68b79650d7f250b08b454d4295f8f210d5edc411b99a1ea9eeea0e9c7e",
        @"fc41a0c84737f7ffb0bd2a47c8c00dec2d62404371f3a42d8896d1ab1a6d1b15",
        @"99f8af8df237cdd5adee04acdf13d1e23d148b8dab89e5e77a1a2b029db87f92",
        @"7f06e3525c7b9b874d9ebbb1e4adec88107ef152362b6306f754666c08b14bb9",
        @"d29df2570fb8d1f26ab4bfe760e95a21a9990e51a016cb4528f68635aafb69bd",
    ];
    for(NSUInteger index=0;index<sizes.count;++index){
        NSError *error=nil;
        H16GEncodedTDProgram *program=[H16GALUEncoder encodeOperationName:@"add"
            squareSize:sizes[index].unsignedIntegerValue error:&error];
        expect(program!=nil&&error==nil&&
               [[sha256(program.data) lowercaseString] isEqualToString:hashes[index]]&&
               program.programRecordCount==16&&program.scratchByteLength==0,
               @"add packet reproduces its complete measured TD geometry");
    }
    NSError *error=nil;
    H16GEncodedTDProgram *multiply=[H16GALUEncoder encodeOperationName:@"mul"
        squareSize:512 error:&error];
    expect([[sha256(multiply.data) lowercaseString] isEqualToString:
        @"94f47ed6bbac46d76eda16037dc6b0734abb41e178b602ebc56766b6b87d41c0"]&&
        ((const uint8_t *)multiply.data.bytes)[0xcc]==0x04,
        @"ALU selector changes add to multiply without changing the packet family");
    for(NSString *operation in @[@"max",@"min"]){
        H16GEncodedTDProgram *program=[H16GALUEncoder encodeOperationName:operation
            squareSize:512 error:&error];
        expect(program!=nil,@"hardware-verified max/min selectors are admitted");
    }
    expect([H16GALUEncoder encodeOperationName:@"sub" squareSize:512
        error:&error]==nil&&[H16GALUEncoder encodeOperationName:@"add"
        squareSize:384 error:&error]==nil,
        @"unknown ALU selectors and unmeasured geometry fail closed");
}

static uint32_t word(NSData *data, NSUInteger offset) {
    uint32_t value = 0;
    [data getBytes:&value range:NSMakeRange(offset, sizeof(value))];
    return value;
}

static uint32_t packTestHalves(float low,float high) {
    _Float16 halves[2]={(_Float16)low,(_Float16)high};
    uint32_t packed=0;
    static_assert(sizeof(halves)==sizeof(packed));
    memcpy(&packed,halves,sizeof(packed));
    return packed;
}

static NSUInteger linearTaskCount(NSData *data, BOOL *headersValid) {
    NSUInteger count=0,position=16;
    BOOL valid=data.length>=16&&((const uint8_t *)data.bytes)[0]==1;
    while(valid&&position<data.length) {
        position=(position+15)&~(NSUInteger)15;
        if(position==data.length)break;
        if(position+36>data.length){valid=NO;break;}
        uint32_t header=word(data,position);
        NSUInteger words=header>>16;
        if(words<9||position+words*4>data.length||
           (header&65535)!=count){valid=NO;break;}
        BOOL last=position+words*4==data.length;
        uint32_t output=word(data,position+8);
        uint32_t dependencies=word(data,position+28);
        uint32_t control=word(data,position+32);
        valid&=!!(output&(1u<<26))==last;
        valid&=!!(control&(1u<<3))==last;
        valid&=dependencies==(count?((1u<<count)-1):0);
        position+=words*4;
        ++count;
    }
    if(headersValid)*headersValid=valid&&position==data.length;
    return count;
}

static void testUnaryPointwisePacketsAndTablesAreRuleBuilt(void) {
    // 192 and 320 were not among the captured programs used by the old
    // position-selected encoder. They exercise the common constructor.
    NSArray<NSNumber *> *sizes = @[@128,@192,@320,@512,@1024,@2048];
    NSMutableArray<NSData *> *packets=[NSMutableArray array];
    for (NSNumber *sizeNumber in sizes) {
        NSError *error = nil;
        NSUInteger size = sizeNumber.unsignedIntegerValue;
        H16GEncodedTDProgram *program = [H16GLUTEncoder
            encodeOperationName:@"sigmoid" inputShape:@[@1,@1,@(size),@(size)]
            error:&error];
        NSUInteger tiles = (size*size + 0x3fff) / 0x4000;
        expect(program != nil && error == nil && program.data.length >= 0x80 &&
               ((const uint8_t *)program.data.bytes)[0] == 1 &&
               program.programFormatCode == tiles-1 &&
               program.programRecordCount > 0 &&
               program.kernelRelocationOffsets.count == 1,
               @"LUT register records are derived from square geometry");
        if(program)[packets addObject:program.data];
    }
    expect([[NSSet setWithArray:packets] count] == sizes.count,
           @"each LUT geometry produces a distinct rule-built packet");
    for (NSString *operation in @[@"sigmoid",@"tanh",@"gelu",@"silu",
            @"exp",@"log",@"sqrt",@"rsqrt",@"reciprocal"]) {
        NSError *error = nil;
        NSData *first = [H16GLUTEncoder
            standaloneConstantRegionForOperationName:operation error:&error];
        NSData *second = [H16GLUTEncoder
            standaloneConstantRegionForOperationName:operation error:&error];
        expect(first.length == 0x80 && error == nil &&
               [first isEqualToData:second] && word(first,0x54) == 33,
               @"unary constants are deterministic 33-sample rule-built tables");
    }
    NSError *error=nil;
    NSData *consumerExp=[H16GLUTEncoder
        constantRegionForOperationName:@"exp" error:&error];
    expect(word(consumerExp,0)==packTestHalves(-25.0f,16.0f) &&
           word(consumerExp,4)==packTestHalves(0.0f,INFINITY) &&
           word(consumerExp,80)==((uint32_t)5<<16) &&
           word(consumerExp,84)==10,
           @"composed exp consumers keep the proven clamp and tail contract");
    H16GEncodedTDProgram *expProgram=[H16GLUTEncoder
        encodeOperationName:@"exp" inputShape:@[@1,@1,@128,@128]
        error:&error];
    std::size_t resultEndpointOffset=ane::h16::rule::registerValueOffset(
        (const uint8_t *)expProgram.data.bytes,expProgram.data.length,0x5100);
    expect(resultEndpointOffset!=SIZE_MAX &&
           word(expProgram.data,resultEndpointOffset)==
               ane::h16::rule::fields::resultEndpoint(3),
           @"rule-built ResultDMA is enabled so Store can observe completion");
    std::size_t kernelStreamOffset=ane::h16::rule::registerValueOffset(
        (const uint8_t *)expProgram.data.bytes,expProgram.data.length,0x5610);
    std::size_t kernelBytesOffset=ane::h16::rule::registerValueOffset(
        (const uint8_t *)expProgram.data.bytes,expProgram.data.length,0x5614);
    std::size_t kernelBaseOffset=ane::h16::rule::registerValueOffset(
        (const uint8_t *)expProgram.data.bytes,expProgram.data.length,0x5618);
    expect(kernelStreamOffset!=SIZE_MAX && kernelBytesOffset!=SIZE_MAX &&
           kernelBaseOffset!=SIZE_MAX &&
           word(expProgram.data,kernelStreamOffset)==
               ane::h16::rule::fields::lookupKernelStream(2) &&
           word(expProgram.data,kernelBytesOffset)==128 &&
           expProgram.kernelRelocationOffsets.count==1 &&
           expProgram.kernelRelocationOffsets[0].unsignedIntegerValue==
               kernelBaseOffset,
           @"rule-built KernelDMA relocates the named lookup-table base");
    error = nil;
    H16GEncodedTDProgram *log = [H16GLUTEncoder encodeOperationName:@"log"
        inputShape:@[@1,@1,@192,@192] error:&error];
    H16GEncodedTDProgram *relu = [H16GLUTEncoder encodeOperationName:@"relu"
        inputShape:@[@1,@1,@320,@320] error:&error];
    expect(log.data.length >= 0x80 && log.programRecordCount > 0 &&
           log.kernelRelocationOffsets.count == 1,
           @"log uses the common rule-built table packet");
    expect(relu.data.length >= 0x80 && relu.kernelRelocationOffsets.count == 0 &&
           [H16GLUTEncoder constantRegionForOperationName:@"relu"
               error:&error].length == 0,
           @"standalone ReLU is generated without a constant table");
    expect([H16GLUTEncoder encodeOperationName:@"sigmoid"
        inputShape:@[@1,@1,@160,@160] error:&error] == nil && error != nil,
        @"pointwise geometry outside the 64-lane rule fails closed");
}

static void testReductionPacketsAreRuleBuilt(void) {
    NSArray<NSArray *> *cases = @[
        @[@"reduce_sum",@96,@12,@20,@1,@2],
        @[@"reduce_mean",@7,@33,@19,@2,@3],
        @[@"reduce_max",@5,@11,@37,@3,@1],
        @[@"reduce_sum",@64,@8,@8,@1,@2],
    ];
    for (NSArray *row in cases) {
        NSError *error = nil;
        NSUInteger axis=[row[4] unsignedIntegerValue];
        NSArray<NSNumber *> *shape = @[@1,row[1],row[2],row[3]];
        H16GReduceEncoding *encoding = [H16GReduceEncoder
            encodeOperationName:row[0] inputShape:shape axis:axis error:&error];
        NSMutableArray<NSNumber *> *output=[shape mutableCopy]; output[axis]=@1;
        BOOL headersValid=NO;
        NSUInteger streamTasks=encoding ?
            linearTaskCount(encoding.tdProgram.data,&headersValid) : 0;
        expect(encoding != nil && error == nil &&
               encoding.taskCount == [row[5] unsignedIntegerValue] &&
               streamTasks == encoding.taskCount && headersValid &&
               encoding.tdProgram.data.length >= 0x80 &&
               encoding.tdProgram.programRecordCount > 0 &&
               encoding.tdProgram.kernelRelocationOffsets.count == 0 &&
               [encoding.outputShape isEqualToArray:output],
               @"reduction records derive from operation, shape, and axis");
    }
    NSError *error = nil;
    H16GReduceEncoding *channel = [H16GReduceEncoder
        encodeOperationName:@"reduce_sum" inputShape:@[@1,@96,@12,@20]
        axis:1 error:&error];
    H16GReduceEncoding *height = [H16GReduceEncoder
        encodeOperationName:@"reduce_mean" inputShape:@[@1,@7,@33,@19]
        axis:2 error:&error];
    expect(channel.inputRowStrideBytes == 64 &&
           channel.inputPlaneStrideBytes == 768 &&
           channel.outputStorageByteLength == 768,
           @"channel reduction derives padded physical strides");
    expect(height.outputRowStrideBytes == 64 &&
           height.outputBatchStrideBytes == 448 &&
           height.outputStorageByteLength == 448,
           @"height reduction derives output storage without a shape table");
    expect([H16GReduceEncoder encodeOperationName:@"reduce_sum"
        inputShape:@[@1,@1,@8,@8] axis:1 error:&error] == nil && error != nil,
        @"unit reduction axes fail closed");
}

static NSString *sha256(NSData *data) {
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *text = [NSMutableString stringWithCapacity:64];
    for (NSUInteger i = 0; i < sizeof(digest); ++i)
        [text appendFormat:@"%02x", digest[i]];
    return text;
}

static void testLayoutPacketsAreGeneratedFromGeometry(void) {
    NSError *error = nil;
    H16GEncodedTDProgram *s2d = [H16GLayoutEncoder
        encodeOperationName:@"space_to_depth"
        inputShape:@[@1,@32,@64,@64] outputShape:@[@1,@512,@16,@16]
        blockSize:4 strategy:ANETileStrategyLayoutDMA3 error:&error];
    expect(s2d != nil && s2d.data.length == 0x2ec,
           @"S2D emits the complete measured three-TD packet family");
    expect([[sha256(s2d.data) lowercaseString] isEqualToString:
        @"55e3fb0aedb8838f1b3378b7187d75a4f5d322caad7bbfc5ed6fdf4a32f18ee5"],
        @"shape-derived S2D fields reproduce the independent full-TD oracle");

    error = nil;
    H16GEncodedTDProgram *wideS2D = [H16GLayoutEncoder
        encodeOperationName:@"space_to_depth"
        inputShape:@[@1,@8,@128,@128] outputShape:@[@1,@128,@32,@32]
        blockSize:4 strategy:ANETileStrategyLayoutDMA3 error:&error];
    expect(wideS2D != nil && wideS2D.data.length == 0x2f4,
           @"S2D selects the wider measured packet variant from geometry");
    NSString *wideHash = [sha256(wideS2D.data) lowercaseString];
    BOOL wideMatches = [wideHash isEqualToString:
        @"2d629cb293349bb4652b99a072ac7680855ff8a19a8e590d72e496f52588aae1"];
    if (!wideMatches)
        fprintf(stderr, "wide S2D hash got %s\n", wideHash.UTF8String);
    expect(wideMatches,
        @"wide S2D fields reproduce an independently minted Apple TD oracle");

    error = nil;
    H16GEncodedTDProgram *unmeasuredS2D = [H16GLayoutEncoder
        encodeOperationName:@"space_to_depth"
        inputShape:@[@1,@8,@64,@64] outputShape:@[@1,@128,@16,@16]
        blockSize:4 strategy:ANETileStrategyLayoutDMA3 error:&error];
    expect(unmeasuredS2D == nil && error != nil,
           @"nearby S2D geometry fails closed instead of reusing the wrong packet variant");

    error = nil;
    H16GEncodedTDProgram *d2s = [H16GLayoutEncoder
        encodeOperationName:@"depth_to_space"
        inputShape:@[@1,@256,@32,@32] outputShape:@[@1,@16,@128,@128]
        blockSize:4 strategy:ANETileStrategyLayoutDMA3 error:&error];
    expect(d2s != nil && d2s.data.length == 0x290,
           @"D2S emits its complete measured packet family");
    expect([[sha256(d2s.data) lowercaseString] isEqualToString:
        @"26a63ec93b2e897911fc3b5a7276a36e6c9aa6b0d5690a75b4d4282185342c06"],
        @"shape-derived D2S fields reproduce the independent full-TD oracle");
}

static void testFusedLayoutConvChainIsGeneratedFromGeometry(void) {
    NSError *error = nil;
    H16GEncodedTDProgram *c8 = [H16GLayoutConvChainEncoder
        encodeNaturalChannels:8 spatial:128 blockSize:4 error:&error];
    expect(c8 != nil && c8.data.length == 0x8ac,
           @"fused S2D-Conv-D2S emits the measured seven-task packet family");
    expect([[sha256(c8.data) lowercaseString] isEqualToString:
        @"6ed394127f6d3b8161cc3be9986f084fad01f83595c39f590ad478155df9b973"],
        @"C8 fused layout-compute fields reproduce the independent TD oracle");
    expect(c8.programRecordCount == 62 && c8.programFormatCode == 0x8a &&
           c8.scratchByteLength == 0x800000 &&
           [c8.kernelRelocationOffsets isEqualToArray:
            @[@0x41c,@0x594,@0x704,@0x890]],
           @"fused packet metadata is derived from its seven-task schedule");

    error = nil;
    H16GEncodedTDProgram *c16 = [H16GLayoutConvChainEncoder
        encodeNaturalChannels:16 spatial:128 blockSize:4 error:&error];
    expect(c16 != nil && [[sha256(c16.data) lowercaseString] isEqualToString:
        @"afc31ea732628def5e34d0fdd63a3953120514b22cbac08792b0e9510c2f77f4"],
        @"the same field grammar scales to the independently minted C16 oracle");

    NSArray<NSNumber *> *largeChannels = @[@24,@32];
    NSArray<NSString *> *largeHashes = @[
        @"9e75d41457f05cbb5c61979c4ecaf99887cd6b4c1e13f3465ea9db66df17500e",
        @"85d30e10f1e0a16247dabf11a6bdd1aa5377e7f9d173ca006557b042e9a0e709",
    ];
    for (NSUInteger i = 0; i < largeChannels.count; ++i) {
        error = nil;
        H16GEncodedTDProgram *large = [H16GLayoutConvChainEncoder
            encodeNaturalChannels:largeChannels[i].unsignedIntegerValue
            spatial:128 blockSize:4 error:&error];
        NSString *largeHash = [sha256(large.data) lowercaseString];
        BOOL largeMatches = large != nil && large.data.length == 0x8a8 &&
            [largeHash isEqualToString:largeHashes[i]] &&
            [large.kernelRelocationOffsets isEqualToArray:
                @[@0x418,@0x590,@0x700,@0x88c]];
        if (!largeMatches)
            fprintf(stderr,"large C%lu TD length=0x%lx hash=%s error=%s\n",
                (unsigned long)largeChannels[i].unsignedIntegerValue,
                (unsigned long)large.data.length,largeHash.UTF8String,
                error.description.UTF8String);
        expect(largeMatches,
            @"larger fused channels select and reproduce the measured 0x8a8 family");
    }

    error = nil;
    expect([H16GLayoutConvChainEncoder encodeNaturalChannels:40
        spatial:128 blockSize:4 error:&error] == nil && error != nil,
        @"unmeasured fused channel families fail closed");
}

static void testW8A8ChainIsGeneratedFromDepthAndLayerState(void) {
    NSError *error = nil;
    H16GEncodedTDProgram *depth3 = [H16GConvChainEncoder
        encodeW8A8C64S64WithDepth:3 error:&error];
    H16GEncodedTDProgram *depth6 = [H16GConvChainEncoder
        encodeW8A8C64S64WithDepth:6 error:&error];
    H16GEncodedTDProgram *depth7 = [H16GConvChainEncoder
        encodeW8A8C64S64WithDepth:7 error:&error];
    expect(depth3.data.length == 0x4c0 && depth6.data.length == 0x910,
           @"chain size follows first, middle and final block lengths");
    expect([depth3.kernelRelocationOffsets isEqualToArray:
        @[@0x18c, @0x30c, @0x4a4]],
           @"depth-three relocations follow scheduled Conv tasks");
    expect([depth6.kernelRelocationOffsets isEqualToArray:
        @[@0x18c, @0x30c, @0x480, @0x5f0, @0x760, @0x8f4]],
           @"depth-six relocations scale without a depth-specific row");
    expect(word(depth6.data, 0x114) == 0x93418005 &&
           word(depth6.data, 0x2b4) == 0xb3418005 &&
           word(depth6.data, 0x8f4) == 0x00005000,
           @"boundary, packed and per-layer kernel state are encoded");
    expect(word(depth6.data, 0x1d8) == 0x00020240 &&
           word(depth6.data, 0x348) == 0x00030240 &&
           word(depth6.data, 0x4b8) == 0x00040240 &&
           word(depth6.data, 0x628) == 0x00050240,
           @"each distinct interior weight receives its own task control slot");
    expect(word(depth6.data, 0x30c) == 0x00001000 &&
           word(depth6.data, 0x480) == 0x00002000 &&
           word(depth6.data, 0x5f0) == 0x00003000 &&
           word(depth6.data, 0x760) == 0x00004000,
           @"kernel addends follow independently packed layer constants");
    expect(depth7 == nil,
           @"unmeasured chain depths fail closed instead of overflowing state fields");

    H16GEncodedTDProgram *depth4 = [H16GConvChainEncoder
        encodeW8A8C64S64WithDepth:4 error:&error];
    expect(depth4.programRecordCount == 0x3b &&
           depth4.programFormatCode == 7,
           @"chain encoder reports its measured program-record fields");
    expect([[sha256(depth4.data) lowercaseString] isEqualToString:
        @"185de64f2a80436a319e7b7ef561a9ee05b9fba45142ddfc11e409e055ae69f8"],
           @"generated depth-four fields reproduce the measured TD hash");
}

static H16GTaskEncodingRequest *matmulGELURequest(NSUInteger size,
                                                  BOOL gelu) {
    NSArray<NSNumber *> *matrix = @[@1, @(size), @(size)];
    NSArray<NSNumber *> *image = @[@1, @1, @(size), @(size)];
    return [[H16GTaskEncodingRequest alloc]
        initWithStageOperations:gelu ? @[@"gelu"] : @[@"matmul", @"reshape"]
        inputIdentifiers:gelu ? @[@"matrix"] : @[@"a", @"b"]
        inputShapes:gelu ? @[image] : @[matrix, matrix]
        outputIdentifier:gelu ? @"y" : @"matrix"
        outputShape:image numericMode:ANELegalNumericModeFP16
        outputStorage:ANEScheduledBridgeStorageExternal];
}

static void testTaskCapabilitiesAreGeometryAndStorageRows(void) {
    H16GTarget *target = [H16GTarget currentTarget];
    for (NSNumber *sizeNumber in @[@128, @256]) {
        NSUInteger size = sizeNumber.unsignedIntegerValue;
        H16GTaskEncodingRequest *matmul = matmulGELURequest(size, NO);
        H16GTaskEncodingRequest *gelu = matmulGELURequest(size, YES);
        H16GTaskCapability *matmulCapability = [target
            taskCapabilityForStageOperations:matmul.stageOperations
            inputShapes:matmul.inputShapes outputShape:matmul.outputShape
            numericMode:matmul.numericMode
            outputStorage:matmul.outputStorage];
        H16GTaskCapability *geluCapability = [target
            taskCapabilityForStageOperations:gelu.stageOperations
            inputShapes:gelu.inputShapes outputShape:gelu.outputShape
            numericMode:gelu.numericMode outputStorage:gelu.outputStorage];
        expect(matmulCapability.packetFamily ==
                   H16GTaskPacketFamilySquareMatmul &&
               matmulCapability.geometry == size,
               @"matmul capability row carries its decoded square geometry");
        expect(geluCapability.packetFamily == H16GTaskPacketFamilyUnaryLUT &&
               geluCapability.geometry == size,
               @"GELU capability row carries its decoded unary geometry");
    }

    H16GTaskEncodingRequest *unsupported = matmulGELURequest(192, NO);
    expect([target taskCapabilityForStageOperations:unsupported.stageOperations
        inputShapes:unsupported.inputShapes outputShape:unsupported.outputShape
        numericMode:unsupported.numericMode
        outputStorage:unsupported.outputStorage] == nil,
        @"nearby unmeasured matmul geometry has no capability row");
    H16GTaskEncodingRequest *internal = [[H16GTaskEncodingRequest alloc]
        initWithStageOperations:@[@"matmul", @"reshape"]
        inputIdentifiers:@[@"a", @"b"]
        inputShapes:@[@[@1,@128,@128], @[@1,@128,@128]]
        outputIdentifier:@"matrix" outputShape:@[@1,@1,@128,@128]
        numericMode:ANELegalNumericModeFP16
        outputStorage:ANEScheduledBridgeStorageSRAM];
    H16GTaskCapability *internalCapability = [target
        taskCapabilityForStageOperations:internal.stageOperations
        inputShapes:internal.inputShapes outputShape:internal.outputShape
        numericMode:internal.numericMode
        outputStorage:internal.outputStorage];
    expect(internalCapability != nil &&
           internalCapability.outputStorage == ANEScheduledBridgeStorageSRAM,
           @"composable matmul has a separate SRAM capability row");
}

static void testProgramCompositionCapabilitiesAreExactRows(void) {
    H16GTarget *target = [H16GTarget currentTarget];
    H16GProgramCompositionCapability *tiledPostOperation = [target
        programCompositionCapabilityFrom:H16GTaskPacketFamilySquareMatmul
        to:H16GTaskPacketFamilyUnaryLUT producerOperationName:@"matmul"
        consumerOperationName:@"gelu" producerGeometry:128
        consumerGeometry:128 bridgeStorage:ANEScheduledBridgeStorageSRAM];
    expect(tiledPostOperation != nil &&
           tiledPostOperation.action ==
               H16GProgramCompositionActionTiledPostOperation,
           @"square matmul and unary LUT have an exact tiled post-operation row");

    expect([target
        programCompositionCapabilityFrom:H16GTaskPacketFamilySquareMatmul
        to:H16GTaskPacketFamilyUnaryLUT producerOperationName:@"matmul"
        consumerOperationName:@"gelu" producerGeometry:128
        consumerGeometry:256
        bridgeStorage:ANEScheduledBridgeStorageSRAM] == nil,
        @"composition rows do not cross decoded geometries");
    expect([target
        programCompositionCapabilityFrom:H16GTaskPacketFamilySquareALU
        to:H16GTaskPacketFamilyUnaryLUT producerOperationName:@"add"
        consumerOperationName:@"gelu" producerGeometry:128
        consumerGeometry:128
        bridgeStorage:ANEScheduledBridgeStorageSRAM] == nil,
        @"an unmeasured packet transition has no composition row");
    expect([target
        programCompositionCapabilityFrom:H16GTaskPacketFamilySquareMatmul
        to:H16GTaskPacketFamilyUnaryLUT producerOperationName:@"matmul"
        consumerOperationName:@"exp" producerGeometry:128
        consumerGeometry:128
        bridgeStorage:ANEScheduledBridgeStorageSRAM] == nil,
        @"a measured GELU row is not reused for another unary operation");
    expect(target.maximumProgramInputCount.available &&
           target.maximumProgramInputCount.value == 3 &&
           target.maximumProgramInputCount.provenance.length != 0,
           @"program input limit respects the five-resource object bound");
    expect(target.maximumProgramTaskCount.available &&
           target.maximumProgramTaskCount.value == 8 &&
           target.maximumProgramTaskCount.provenance.length != 0 &&
           target.maximumTaskDescriptorByteLength == 0x3fc0,
           @"composition uses conservative task and descriptor bounds");
}

static void testTaskEncoderUsesPrimitiveLeaves(void) {
    for (NSNumber *sizeNumber in @[@128, @256]) {
        NSUInteger size = sizeNumber.unsignedIntegerValue;
        NSError *error = nil;
        H16GEncodedTask *matmul = [H16GTaskEncoder
            encodeRequest:matmulGELURequest(size, NO)
            target:[H16GTarget currentTarget] error:&error];
        H16GEncodedTDProgram *matmulLeaf = [H16GMatmulEncoder
            encodeSquareSize:size error:&error];
        expect(matmul != nil && error == nil &&
               [matmul.tdProgram.data isEqualToData:matmulLeaf.data],
               @"task encoder delegates square matmul bits to the matmul leaf");
        expect(matmul.taskCount ==
                   [H16GMatmulEncoder tileCountForSquareSize:size] + 1 &&
               matmul.scratchBacked,
               @"matmul task fragment retains program and scratch metadata");
        expect(matmul.packetFamily == H16GTaskPacketFamilySquareMatmul &&
               matmul.geometry == size &&
               matmul.numericMode == ANELegalNumericModeFP16 &&
               matmul.outputStorage == ANEScheduledBridgeStorageExternal,
               @"matmul task exposes its target capability metadata");

        error = nil;
        H16GEncodedTask *gelu = [H16GTaskEncoder
            encodeRequest:matmulGELURequest(size, YES)
            target:[H16GTarget currentTarget] error:&error];
        H16GEncodedTDProgram *geluLeaf = [H16GLUTEncoder
            encodeOperationName:@"gelu"
            inputShape:@[@1,@1,@(size),@(size)] error:&error];
        NSData *geluConstants = [H16GLUTEncoder
            constantRegionForOperationName:@"gelu" error:&error];
        expect(gelu != nil && error == nil &&
               [gelu.tdProgram.data isEqualToData:geluLeaf.data] &&
               [gelu.constantRegion isEqualToData:geluConstants],
               @"task encoder delegates GELU bits and constants to the LUT leaf");
        expect(gelu.taskCount == 1 && !gelu.scratchBacked,
               @"GELU fragment retains its linear one-task metadata");
        expect(gelu.packetFamily == H16GTaskPacketFamilyUnaryLUT &&
               gelu.geometry == size &&
               gelu.numericMode == ANELegalNumericModeFP16 &&
               gelu.outputStorage == ANEScheduledBridgeStorageExternal,
               @"GELU task exposes its target capability metadata");
    }
}

static void testMatmulPostOperationUsesDecodedFieldRows(void) {
    NSDictionary<NSNumber *, NSString *> *expectedHashes = @{
        @128: @"720ff02eb5395f30ce2f0d9ff572e24fc9529cf2724bd07d0338bc6989ccd649",
        @256: @"a980c4a72b9d59313361cf3a54e1827bcdfa474b4df66b1c633d4fe49cfffe91",
    };
    NSError *error = nil;
    NSData *geluKernel = [H16GLUTEncoder
        constantRegionForOperationName:@"gelu" error:&error];
    for (NSNumber *sizeNumber in @[@128, @256]) {
        H16GMatmulPostOperationEncoding *encoding = [H16GMatmulEncoder
            encodeSquareSize:sizeNumber.unsignedIntegerValue
            postOperationName:@"gelu" kernelRegion:geluKernel error:&error];
        expect(encoding != nil && error == nil &&
               encoding.kernelRegion.length == 86,
               @"matmul post-operation keeps the decoded GELU kernel length");
        expect([[sha256(encoding.tdProgram.data) lowercaseString]
                   isEqualToString:expectedHashes[sizeNumber]],
               @"matmul post-operation fields match the controlled M4 pair");
    }
    error = nil;
    expect([H16GMatmulEncoder encodeSquareSize:128
        postOperationName:@"exp" kernelRegion:geluKernel error:&error] == nil &&
        error != nil,
        @"an unmeasured matmul post-operation fails closed");
}

static void testBroadcastALUUsesDecodedFieldFamilies(void) {
    NSError *error = nil;
    H16GEncodedTDProgram *scale = [H16GBroadcastALUEncoder
        encodeScalarScaleForMatrixRows:128 columns:128
        scalar:0.08838834764831845 error:&error];
    expect(scale != nil && error == nil && scale.programRecordCount == 16 &&
           scale.programFormatCode == 0 &&
           [[sha256(scale.data) lowercaseString] isEqualToString:
               @"9669e2040c7eb684ae93cf880fccc9cdb10444a157a28dc3c2e6e860c904b35c"],
           @"scalar scale fields reproduce the controlled N128 oracle");
    // Apple-compiled scalar multiply oracles for 0.25 and 0.7 differ from the
    // 0.0884 oracle only in the operand word at 0xac: single precision
    // rounded to ten mantissa bits.
    NSDictionary<NSNumber *, NSNumber *> *operandWords = @{
        @0.08838834764831845: @0x3db50000u,
        @0.25: @0x3e800000u,
        @0.7: @0x3f334000u,
    };
    for (NSNumber *operand in operandWords) {
        H16GEncodedTDProgram *program = [H16GBroadcastALUEncoder
            encodeScalarScaleForMatrixRows:128 columns:128
            scalar:operand.doubleValue error:&error];
        uint32_t word = 0;
        if (program) [program.data getBytes:&word range:NSMakeRange(0xac, 4)];
        expect(program != nil && program.data.length == scale.data.length &&
               word == operandWords[operand].unsignedIntValue,
               @"scalar scale writes the measured operand word");
    }
    expect(H16GALUScalarWord(0.7) == 0x3f334000u &&
           H16GALUScalarWord(0.25) == 0x3e800000u &&
           H16GALUScalarWord(0.08838834764831845) == 0x3db50000u &&
           H16GALUScalarWord(1.0) == 0x3f800000u &&
           H16GALUScalarWord(-2.0) == 0xc0000000u &&
           H16GALUScalarWord(1.0 + 1.0 / 2048.0) == 0x3f800000u &&
           H16GALUScalarWord(1.0 + 3.0 / 2048.0) == 0x3f804000u,
           @"operand rounding keeps ten mantissa bits with ties to even");
    expect([H16GBroadcastALUEncoder encodeScalarScaleForMatrixRows:128
               columns:128 scalar:INFINITY error:&error] == nil,
           @"a non-finite scalar operand fails closed");

    NSDictionary<NSString *, NSString *> *matrixRowHashes = @{
        @"sub": @"d5c7cb371bd20b5ba6a8338a4bca9ad6c8962f44a9d106d15ff095a30883a17e",
        @"mul": @"bc6533a25cf7d731696fc1060cd38d89512df2d6b1f45861119a22c77dfd660c",
    };
    for (NSString *operation in matrixRowHashes) {
        H16GEncodedTDProgram *program = [H16GBroadcastALUEncoder
            encodeMatrixRowOperation:operation rows:128 columns:128
            error:&error];
        expect(program.programFormatCode == 1 &&
               [[sha256(program.data) lowercaseString]
                   isEqualToString:matrixRowHashes[operation]],
               @"matrix-row operation selects its decoded opcode field");
    }
    NSDictionary<NSString *, NSString *> *rowHashes = @{
        @"add": @"1248c4fe690bdada5fb8eed2750d52ceeea6906c644289ad5ce67041e022209f",
        @"mul": @"a5f5fc4a6b54a57ac0fdd46ce3572a8c35150affe5c7dac6ad50cc07c94e31e8",
        @"max": @"eaf12663a09d6e919b0495af25b7228b14ee64d3d2e322a2e723eae8bd99e735",
    };
    for (NSString *operation in rowHashes) {
        H16GEncodedTDProgram *program = [H16GBroadcastALUEncoder
            encodeRowOperation:operation rows:128 error:&error];
        expect(program.programFormatCode == 0 &&
               [[sha256(program.data) lowercaseString]
                   isEqualToString:rowHashes[operation]],
               @"row-state operation selects its decoded opcode field");
    }
    expect([H16GBroadcastALUEncoder encodeMatrixRowOperation:@"add"
        rows:128 columns:128 error:&error] == nil &&
        [H16GBroadcastALUEncoder encodeRowOperation:@"mul" rows:96
        error:&error] == nil,
        @"unmeasured broadcast operations and geometries fail closed");
    error = nil;
    NSArray<NSNumber *> *square = @[@1,@1,@128,@128];
    expect([H16GTaskEncoder encodeRequest:[[H16GTaskEncodingRequest alloc]
                initWithStageOperations:@[@"mul"] inputIdentifiers:@[@"x"]
                inputShapes:@[square] outputIdentifier:@"y"
                outputShape:square numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal]
            target:[H16GTarget currentTarget] error:&error] == nil &&
           error != nil,
           @"a scalar scale request without its operand fails closed");

    NSArray<NSNumber *> *matrix = @[@1,@1,@128,@128];
    NSArray<NSNumber *> *row = @[@1,@1,@128,@1];
    NSArray<H16GTaskEncodingRequest *> *requests = @[
        [[H16GTaskEncodingRequest alloc] initWithStageOperations:@[@"mul"]
            inputIdentifiers:@[@"scores"] inputShapes:@[matrix]
            outputIdentifier:@"scaled" outputShape:matrix
            numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageExternal
            scalarOperand:@0.08838834764831845],
        [[H16GTaskEncodingRequest alloc] initWithStageOperations:@[@"sub"]
            inputIdentifiers:@[@"scores", @"maximum"]
            inputShapes:@[matrix,row] outputIdentifier:@"centered"
            outputShape:matrix numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageExternal],
        [[H16GTaskEncodingRequest alloc] initWithStageOperations:@[@"max"]
            inputIdentifiers:@[@"previous", @"tile"]
            inputShapes:@[row,row] outputIdentifier:@"next"
            outputShape:row numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageExternal],
    ];
    for (H16GTaskEncodingRequest *request in requests) {
        error = nil;
        H16GEncodedTask *task = [H16GTaskEncoder encodeRequest:request
            target:[H16GTarget currentTarget] error:&error];
        expect(task != nil && error == nil && task.taskCount == 1 &&
               !task.scratchBacked,
               @"broadcast capability row lowers through the shared task interface");
        if (request.scalarOperand)
            expect([task.scalarOperand isEqualToNumber:request.scalarOperand],
                   @"the scalar scale task keeps its operand for composition");
        else
            expect(task.scalarOperand == nil,
                   @"tasks without a scalar operand carry none");
        if (request.inputIdentifiers.count == 2)
            expect([task.inputIdentifiers isEqualToArray:
                       [[request.inputIdentifiers reverseObjectEnumerator]
                           allObjects]],
                   @"broadcast task records the measured request input order");
    }
}

static void testMatrixRowDivisionUsesMeasuredTwoTaskFamily(void) {
    NSError *error = nil;
    H16GMatrixRowDivisionEncoding *division =
        [H16GMatrixRowDivisionEncoder encodeRows:128 columns:128
                                           error:&error];
    expect(division != nil && error == nil && division.taskCount == 2 &&
           division.tdProgram.programRecordCount == 31 &&
           division.tdProgram.programFormatCode == 0 &&
           [division.tdProgram.kernelRelocationOffsets
               isEqualToArray:@[@0xd0]],
           @"matrix-row division records the measured two-task metadata");
    expect([[sha256(division.tdProgram.data) lowercaseString]
               isEqualToString:
               @"accfb4fae347c972bb7d0465a1788e4f32ce77a0ef35cfd859cfcae35d5656bc"],
           @"matrix-row division fields reproduce the N128 descriptor");
    NSData *reciprocal = [H16GLUTEncoder
        constantRegionForOperationName:@"reciprocal" error:&error];
    expect([division.constantRegion isEqualToData:reciprocal],
           @"matrix-row division uses the rule-built reciprocal table");
    error = nil;
    expect([H16GMatrixRowDivisionEncoder encodeRows:96 columns:128
        error:&error] == nil && error != nil,
        @"unmeasured matrix-row division geometry fails closed");
}

static H16GTaskEncodingRequest *primitiveRequest(
    NSArray<NSString *> *operations, NSArray<NSString *> *identifiers,
    NSArray<NSArray<NSNumber *> *> *inputShapes, NSString *output,
    NSArray<NSNumber *> *outputShape) {
    return [[H16GTaskEncodingRequest alloc]
        initWithStageOperations:operations inputIdentifiers:identifiers
        inputShapes:inputShapes outputIdentifier:output outputShape:outputShape
        numericMode:ANELegalNumericModeFP16
        outputStorage:ANEScheduledBridgeStorageExternal];
}

static void testOnlineReductionPrimitiveRows(void) {
    NSArray<NSNumber *> *matrix = @[@1,@1,@128,@128];
    NSArray<NSNumber *> *row = @[@1,@1,@128,@1];
    NSArray<H16GTaskEncodingRequest *> *requests = @[
        primitiveRequest(@[@"mul"], @[@"state", @"factor"],
                         @[matrix,matrix], @"product", matrix),
        primitiveRequest(@[@"add"], @[@"product", @"update"],
                         @[matrix,matrix], @"state.next", matrix),
        primitiveRequest(@[@"matmul"], @[@"q", @"k"], @[matrix,matrix],
                         @"scores", matrix),
        primitiveRequest(@[@"exp"], @[@"centered"], @[matrix],
                         @"exponential", matrix),
        primitiveRequest(@[@"reduce_max"], @[@"scaled"], @[matrix],
                         @"maximum", row),
        primitiveRequest(@[@"reduce_sum"], @[@"exponential"], @[matrix],
                         @"sum", row),
        primitiveRequest(@[@"real_div"], @[@"exponential", @"sum"],
                         @[matrix,row], @"probabilities", matrix),
    ];
    for (H16GTaskEncodingRequest *request in requests) {
        NSError *error = nil;
        H16GEncodedTask *task = [H16GTaskEncoder encodeRequest:request
            target:[H16GTarget currentTarget] error:&error];
        expect(task != nil && error == nil,
               [NSString stringWithFormat:@"%@ has an exact primitive row",
                    request.stageOperations.firstObject]);
        if ([request.stageOperations.firstObject isEqualToString:@"mul"] ||
            [request.stageOperations.firstObject isEqualToString:@"add"])
            expect(task.outputResourceIndex == 1,
                   @"square ALU places output between its operands");
    }
    NSError *error = nil;
    H16GEncodedTask *division = [H16GTaskEncoder
        encodeRequest:requests.lastObject target:[H16GTarget currentTarget]
        error:&error];
    expect(division.taskCount == 2 &&
           [division.inputIdentifiers
               isEqualToArray:@[@"sum", @"exponential"]],
           @"shared task encoding preserves division task metadata and order");
    H16GTaskEncodingRequest *unsupported = primitiveRequest(
        @[@"reduce_sum"], @[@"x"], @[@[@1,@1,@96,@96]], @"y",
        @[@1,@1,@96,@1]);
    expect([H16GTaskEncoder encodeRequest:unsupported
        target:[H16GTarget currentTarget] error:&error] == nil,
        @"nearby unmeasured reduction geometry has no primitive row");
}

int main(void) {
    @autoreleasepool {
        testWriterBoundsAndZeroInitialization();
        testConvReluDescriptorIsFullyFieldEncoded();
        testGeometryChangesAreComputed();
        testMeasuredConvGeometryTable();
        testDepthwisePacketsMatchMeasuredFamilies();
        testRegularConvPacketsMatchMeasuredFamilies();
        testSquareMatmulPacketsFollowDecodedFamilies();
        testBinaryALUPacketsFollowMeasuredGeometry();
        testUnaryPointwisePacketsAndTablesAreRuleBuilt();
        testReductionPacketsAreRuleBuilt();
        testW8A8ChainIsGeneratedFromDepthAndLayerState();
        testLayoutPacketsAreGeneratedFromGeometry();
        testFusedLayoutConvChainIsGeneratedFromGeometry();
        testTaskCapabilitiesAreGeometryAndStorageRows();
        testProgramCompositionCapabilitiesAreExactRows();
        testTaskEncoderUsesPrimitiveLeaves();
        testMatmulPostOperationUsesDecodedFieldRows();
        testBroadcastALUUsesDecodedFieldFamilies();
        testMatrixRowDivisionUsesMeasuredTwoTaskFamily();
        testOnlineReductionPrimitiveRows();
        printf("structured TD encoding: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
