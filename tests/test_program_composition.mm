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

#import "H16GTaskComposer.h"
#import "H16GTaskEncoder.h"
#import "H16GSRAMChainEncoder.h"
#import "H16GLUTEncoder.h"

static int failures = 0;

static void expect(BOOL condition, NSString *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message.UTF8String);
        failures++;
    }
}

static NSString *sha256(NSData *data) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *result = [NSMutableString string];
    for (NSUInteger index = 0; index < sizeof(digest); ++index)
        [result appendFormat:@"%02x", digest[index]];
    return result;
}

static NSData *trimmedKernel(NSData *data) {
    const uint8_t *bytes = (const uint8_t *)data.bytes;
    NSUInteger length = data.length;
    while (length > 0 && bytes[length - 1] == 0) --length;
    length = MIN(data.length,(length + 1) & ~(NSUInteger)1);
    return [data subdataWithRange:NSMakeRange(0,length)];
}

static H16GTaskEncodingRequest *request(NSUInteger size, BOOL gelu) {
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

static void testTiledPostOperationComposition(void) {
    NSDictionary<NSNumber *, NSString *> *expectedHashes = @{
        @128: @"720ff02eb5395f30ce2f0d9ff572e24fc9529cf2724bd07d0338bc6989ccd649",
        @256: @"a980c4a72b9d59313361cf3a54e1827bcdfa474b4df66b1c633d4fe49cfffe91",
    };
    H16GTarget *target = [H16GTarget currentTarget];
    for (NSNumber *sizeNumber in @[@128, @256]) {
        NSError *error = nil;
        H16GEncodedTask *producer = [H16GTaskEncoder
            encodeRequest:request(sizeNumber.unsignedIntegerValue, NO)
            target:target error:&error];
        H16GEncodedTask *consumer = [H16GTaskEncoder
            encodeRequest:request(sizeNumber.unsignedIntegerValue, YES)
            target:target error:&error];
        H16GProgramCompositionCapability *capability = [target
            programCompositionCapabilityFrom:producer.packetFamily
            to:consumer.packetFamily producerOperationName:@"matmul"
            consumerOperationName:@"gelu"
            producerGeometry:producer.geometry
            consumerGeometry:consumer.geometry
            bridgeStorage:ANEScheduledBridgeStorageSRAM];
        H16GEncodedTask *composed = [H16GTaskComposer
            composeProducer:producer consumer:consumer
            capability:capability error:&error];
        expect(composed != nil && error == nil,
               @"measured matmul and GELU tasks compose");
        expect([[sha256(composed.tdProgram.data) lowercaseString]
                   isEqualToString:expectedHashes[sizeNumber]],
               @"composed task stream matches the controlled M4 pair");
        NSData *unpaddedGELU = trimmedKernel(consumer.constantRegion);
        expect([composed.constantRegion isEqualToData:unpaddedGELU],
               @"composed task uses the constructively trimmed GELU kernel");
        expect([composed.inputIdentifiers isEqualToArray:@[@"a", @"b"]] &&
               [composed.outputIdentifier isEqualToString:@"y"] &&
               [composed.stageOperations
                   isEqualToArray:@[@"matmul", @"reshape", @"gelu"]],
               @"composed task retains partition boundary identifiers");
        expect(composed.taskCount == producer.taskCount &&
               composed.outputResourceIndex == producer.outputResourceIndex &&
               composed.packetFamily == consumer.packetFamily,
               @"composed metadata follows the fused producer program and result");
    }
}

static void testUnmeasuredCompositionDeclines(void) {
    H16GTarget *target = [H16GTarget currentTarget];
    NSError *error = nil;
    H16GEncodedTask *producer = [H16GTaskEncoder
        encodeRequest:request(128, NO) target:target error:&error];
    H16GEncodedTask *consumer = [H16GTaskEncoder
        encodeRequest:request(128, YES) target:target error:&error];
    expect([H16GTaskComposer composeProducer:producer consumer:consumer
        capability:nil error:&error] == nil && error != nil,
        @"missing composition capability declines without emitting bytes");
}

static H16GEncodedTask *encode(H16GTaskEncodingRequest *request) {
    NSError *error = nil;
    return [H16GTaskEncoder encodeRequest:request
                                   target:[H16GTarget currentTarget]
                                    error:&error];
}

static H16GTaskEncodingRequest *matmul128(NSString *output) {
    NSArray<NSNumber *> *matrix = @[@1, @1, @128, @128];
    return [[H16GTaskEncodingRequest alloc]
        initWithStageOperations:@[@"matmul"]
        inputIdentifiers:@[@"a", @"b"] inputShapes:@[matrix, matrix]
        outputIdentifier:output outputShape:matrix
        numericMode:ANELegalNumericModeFP16
        outputStorage:ANEScheduledBridgeStorageExternal];
}

static H16GTaskEncodingRequest *scale128(NSString *input, NSString *output,
                                         double scalar) {
    NSArray<NSNumber *> *matrix = @[@1, @1, @128, @128];
    return [[H16GTaskEncodingRequest alloc]
        initWithStageOperations:@[@"mul"]
        inputIdentifiers:@[input] inputShapes:@[matrix]
        outputIdentifier:output outputShape:matrix
        numericMode:ANELegalNumericModeFP16
        outputStorage:ANEScheduledBridgeStorageExternal
        scalarOperand:@(scalar)];
}

static H16GTaskEncodingRequest *unary128(NSString *operation,
                                         NSString *input, NSString *output) {
    NSArray<NSNumber *> *matrix = @[@1, @1, @128, @128];
    return [[H16GTaskEncodingRequest alloc]
        initWithStageOperations:@[operation]
        inputIdentifiers:@[input] inputShapes:@[matrix]
        outputIdentifier:output outputShape:matrix
        numericMode:ANELegalNumericModeFP16
        outputStorage:ANEScheduledBridgeStorageExternal];
}

static H16GTaskEncodingRequest *reduceSum128(NSString *input,
                                             NSString *output) {
    NSArray<NSNumber *> *matrix = @[@1, @1, @128, @128];
    NSArray<NSNumber *> *row = @[@1, @1, @128, @1];
    return [[H16GTaskEncodingRequest alloc]
        initWithStageOperations:@[@"reduce_sum"]
        inputIdentifiers:@[input] inputShapes:@[matrix]
        outputIdentifier:output outputShape:row
        numericMode:ANELegalNumericModeFP16
        outputStorage:ANEScheduledBridgeStorageExternal];
}

static H16GProgramCompositionCapability *capabilityFor(
    H16GEncodedTask *producer, H16GEncodedTask *consumer) {
    return [[H16GTarget currentTarget]
        programCompositionCapabilityFrom:producer.packetFamily
        to:consumer.packetFamily
        producerOperationName:producer.compositionOperationName
        consumerOperationName:consumer.compositionOperationName
        producerGeometry:producer.geometry consumerGeometry:consumer.geometry
        bridgeStorage:ANEScheduledBridgeStorageSRAM];
}

static uint32_t wordAt(NSData *data, NSUInteger byteOffset) {
    uint32_t value = 0;
    [data getBytes:&value range:NSMakeRange(byteOffset, sizeof(value))];
    return value;
}

static BOOL dataEqualExceptWord(NSData *left, NSData *right,
                                NSUInteger byteOffset) {
    if (left.length != right.length) return NO;
    for (NSUInteger offset = 0; offset < left.length; offset += 4)
        if (offset != byteOffset && wordAt(left, offset) != wordAt(right, offset))
            return NO;
    return YES;
}

static void testMatmulOutputScaleFold(void) {
    H16GEncodedTask *matmul = encode(matmul128(@"t"));
    H16GEncodedTask *plain = encode(matmul128(@"t"));
    // Apple's matmul-then-mul(0.0884) oracle differs from the plain matmul
    // oracle in exactly word 129: 0x3c00 -> 0x2da8. mul(0.25) gives 0x3400.
    NSDictionary<NSNumber *, NSNumber *> *foldedWords = @{
        @0.08838834764831845: @0x00002da8u,
        @0.25: @0x00003400u,
    };
    for (NSNumber *scalar in foldedWords) {
        H16GEncodedTask *scale = encode(scale128(@"t", @"y",
                                                 scalar.doubleValue));
        expect(scale != nil && [scale.scalarOperand isEqualToNumber:scalar],
               @"the standalone scale task carries its scalar operand");
        H16GProgramCompositionCapability *capability =
            capabilityFor(matmul, scale);
        expect(capability != nil &&
               capability.action ==
                   H16GProgramCompositionActionMatmulOutputScale &&
               capability.consumerTaskCountContribution == 0,
               @"matmul then scalar mul has an output-scale composition row");
        NSError *error = nil;
        H16GEncodedTask *composed = [H16GTaskComposer
            composeProducer:matmul consumer:scale capability:capability
            error:&error];
        expect(composed != nil && error == nil,
               @"matmul then scalar mul composes into one program");
        if (!composed) continue;
        expect(wordAt(composed.tdProgram.data, 129 * 4) ==
                   foldedWords[scalar].unsignedIntValue &&
               wordAt(plain.tdProgram.data, 129 * 4) == 0x00003c00u &&
               dataEqualExceptWord(composed.tdProgram.data,
                                   plain.tdProgram.data, 129 * 4),
               @"the fold changes only the decoded output-scale word");
        expect([composed.inputIdentifiers isEqualToArray:@[@"a", @"b"]] &&
               [composed.outputIdentifier isEqualToString:@"y"] &&
               [composed.stageOperations
                   isEqualToArray:@[@"matmul", @"mul"]] &&
               composed.taskCount == matmul.taskCount &&
               composed.packetFamily == H16GTaskPacketFamilySquareMatmul &&
               composed.scalarOperand == nil &&
               [composed.constantRegion isEqualToData:matmul.constantRegion],
               @"the composed program keeps the matmul's structure and the scale's output");
    }

    H16GEncodedTask *scale = encode(scale128(@"t", @"y", 0.25));
    NSError *error = nil;
    expect(capabilityFor(scale, matmul) == nil,
           @"scalar mul then matmul has no composition row");
    H16GEncodedTask *unrelated = encode(scale128(@"other", @"y", 0.25));
    expect([H16GTaskComposer composeProducer:matmul consumer:unrelated
               capability:capabilityFor(matmul, unrelated)
               error:&error] == nil && error != nil,
           @"a scale that does not read the matmul output declines");
    H16GEncodedTask *prefolded = [H16GTaskComposer composeProducer:matmul
        consumer:scale capability:capabilityFor(matmul, scale) error:&error];
    error = nil;
    expect([H16GTaskComposer composeProducer:prefolded consumer:scale
               capability:capabilityFor(matmul, scale) error:&error] == nil &&
           error != nil,
           @"a matmul whose output scale is already folded declines a second fold");
}

static void testLUTInputScaleFold(void) {
    H16GEncodedTask *exponential = encode(unary128(@"exp", @"t", @"y"));
    H16GEncodedTask *plain = encode(unary128(@"exp", @"t", @"y"));
    NSUInteger scaleOffset=[H16GLUTEncoder
        inputScaleByteOffsetForOperationName:@"exp"
        inputShape:@[@1,@1,@128,@128]];
    NSDictionary<NSNumber *, NSNumber *> *foldedWords = @{
        @0.08838834764831845: @0x00003015u,
        @0.25: @0x000035c5u,
    };
    for (NSNumber *scalar in foldedWords) {
        H16GEncodedTask *scale = encode(scale128(@"x", @"t",
                                                 scalar.doubleValue));
        H16GProgramCompositionCapability *capability =
            capabilityFor(scale, exponential);
        expect(capability != nil &&
               capability.action == H16GProgramCompositionActionLUTInputScale,
               @"scalar mul then exp has an input-scale composition row");
        NSError *error = nil;
        H16GEncodedTask *composed = [H16GTaskComposer
            composeProducer:scale consumer:exponential capability:capability
            error:&error];
        expect(composed != nil && error == nil,
               @"scalar mul then exp composes into one program");
        if (!composed) continue;
        expect(wordAt(composed.tdProgram.data, scaleOffset) ==
                   foldedWords[scalar].unsignedIntValue &&
               wordAt(plain.tdProgram.data, scaleOffset) == 15813u &&
               dataEqualExceptWord(composed.tdProgram.data,
                                   plain.tdProgram.data, scaleOffset),
               @"the fold changes only the constructed input-scale field");
        expect([composed.inputIdentifiers isEqualToArray:@[@"x"]] &&
               [composed.outputIdentifier isEqualToString:@"y"] &&
               [composed.stageOperations isEqualToArray:@[@"mul", @"exp"]] &&
               composed.taskCount == 1 &&
               composed.packetFamily == H16GTaskPacketFamilyUnaryLUT &&
               [composed.constantRegion
                   isEqualToData:exponential.constantRegion],
               @"the composed program keeps the exp table and the scale's input");
    }

    H16GEncodedTask *scale = encode(scale128(@"x", @"t", 0.25));
    expect(capabilityFor(exponential, scale) == nil,
           @"exp then scalar mul has no composition row");
    H16GEncodedTask *gelu = encode(unary128(@"gelu", @"t", @"y"));
    expect(gelu != nil && capabilityFor(scale, gelu) == nil,
           @"scalar mul then an unmeasured table unary has no composition row");
    NSError *error = nil;
    H16GEncodedTask *unrelated = encode(unary128(@"exp", @"other", @"y"));
    expect([H16GTaskComposer composeProducer:scale consumer:unrelated
               capability:capabilityFor(scale, unrelated) error:&error] == nil &&
           error != nil,
           @"an exp that does not read the scale output declines");
}

static void testUnaryReductionSRAMChain(void) {
    H16GEncodedTask *exponential = encode(unary128(@"exp", @"x", @"t"));
    H16GEncodedTask *sum = encode(reduceSum128(@"t", @"y"));
    H16GProgramCompositionCapability *capability =
        capabilityFor(exponential, sum);
    expect(exponential != nil && sum != nil,
           @"standalone unary and reduction tasks encode");
    expect(capability != nil &&
           capability.consumerTaskCountContribution == 1,
           @"unary to reduction has a measured SRAM task-chain row");

    NSError *error = nil;
    H16GEncodedTask *composed = [H16GTaskComposer
        composeProducer:exponential consumer:sum capability:capability
        error:&error];
    expect(composed != nil && error == nil,
           @"unary output remains in SRAM for the reduction");
    if (!composed) return;

    expect([[sha256(composed.tdProgram.data) lowercaseString]
               isEqualToString:
                   @"8353e335a95323211dcf8a90d136e8373c52e8378250a0d81051a92e57749929"] &&
           [composed.tdProgram.kernelRelocationOffsets
               isEqualToArray:@[@0xd0]] &&
           composed.tdProgram.programRecordCount == 31,
           @"the encoded task stream matches the measured two-task program");
    expect([composed.constantRegion isEqualToData:exponential.constantRegion],
           @"the composed program keeps the rule-built unary table");
    expect([composed.stageOperations
               isEqualToArray:@[@"exp", @"reduce_sum"]] &&
           [composed.inputIdentifiers isEqualToArray:@[@"x"]] &&
           [composed.outputIdentifier isEqualToString:@"y"] &&
           [composed.outputShape isEqualToArray:@[@1, @1, @128, @1]] &&
           composed.taskCount == 2 &&
           composed.packetFamily == H16GTaskPacketFamilyReduction,
           @"the chain exposes only its graph input and reduced output");

    H16GEncodedTask *unrelated = encode(reduceSum128(@"other", @"y"));
    error = nil;
    expect([H16GTaskComposer composeProducer:exponential consumer:unrelated
               capability:capabilityFor(exponential, unrelated)
               error:&error] == nil && error != nil,
           @"a reduction that does not consume the unary result declines");
}

static void testRowNormalizationSRAMProgram(void) {
    NSArray<NSString *> *operations = @[
        @"reduce_max", @"sub", @"exp", @"reduce_sum", @"reciprocal", @"mul"];
    NSError *error = nil;
    H16GEncodedTDProgram *program = [H16GSRAMChainEncoder
        encodeStageOperations:operations rows:128 columns:128 error:&error];
    NSData *constants = [H16GSRAMChainEncoder
        constantRegionForStageOperations:operations error:&error];
    expect(program != nil && constants != nil && error == nil,
           @"the measured row-normalization operation chain encodes");
    expect(program != nil &&
           [[sha256(program.data) lowercaseString]
               isEqualToString:
                   @"b4f2dab3553c1bf9c8381621f0cba736b7092be04d5c382e622f1265b00628ee"] &&
           [program.kernelRelocationOffsets
               isEqualToArray:@[@0x2c4, @0x434]] &&
           program.programRecordCount == 88 && program.programFormatCode == 1,
           @"the row-normalization task stream matches the measured program");
    NSMutableData *expectedConstants = [NSMutableData data];
    [expectedConstants appendData:[H16GLUTEncoder
        constantRegionForOperationName:@"exp" error:&error]];
    [expectedConstants appendData:[H16GLUTEncoder
        constantRegionForOperationName:@"reciprocal" error:&error]];
    expect(constants != nil && [constants isEqualToData:expectedConstants] &&
           [H16GSRAMChainEncoder taskCountForStageOperations:operations] == 7,
           @"row normalization uses constructed constants and task count");

    error = nil;
    expect([H16GSRAMChainEncoder encodeStageOperations:operations
        rows:256 columns:256 error:&error] == nil && error != nil,
        @"an unmeasured row-normalization geometry declines");
    error = nil;
    expect([H16GSRAMChainEncoder encodeStageOperations:
        @[@"reduce_max", @"sub", @"exp", @"reduce_sum", @"mul"]
        rows:128 columns:128 error:&error] == nil && error != nil,
        @"a different operation sequence declines");
}

int main(void) {
    @autoreleasepool {
        testTiledPostOperationComposition();
        testUnmeasuredCompositionDeclines();
        testMatmulOutputScaleFold();
        testLUTInputScaleFold();
        testUnaryReductionSRAMChain();
        testRowNormalizationSRAMProgram();
        printf("program composition: %s\n",
               failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
