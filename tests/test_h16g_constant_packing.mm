#import <Foundation/Foundation.h>
#include "sha256_compat.h"

#import "H16GConstantPacker.h"

#include <stdio.h>

static int failures = 0;

static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    ++failures;
}

static NSString *sha256(NSData *data) {
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *text = [NSMutableString stringWithCapacity:64];
    for (NSUInteger i = 0; i < sizeof(digest); ++i)
        [text appendFormat:@"%02x", digest[i]];
    return text;
}

static void testConvPackingTraversesInputAndOutputBlocks(void) {
    const NSUInteger inputChannels = 8;
    const NSUInteger outputChannels = 16;
    NSMutableData *weights = [NSMutableData dataWithLength:
        inputChannels * outputChannels * sizeof(uint16_t)];
    uint16_t *source = (uint16_t *)weights.mutableBytes;
    for (NSUInteger i = 0; i < inputChannels * outputChannels; ++i)
        source[i] = (uint16_t)i;

    NSError *error = nil;
    NSData *packed = [H16GConstantPacker packConv1x1Weights:weights
        inputChannels:inputChannels outputChannels:outputChannels
        bytesPerWeight:sizeof(uint16_t) error:&error];
    const uint16_t *actual = (const uint16_t *)packed.bytes;
    expect(packed.length == weights.length && error == nil,
           @"packing preserves the complete weight payload");
    expect(actual[0] == 0 && actual[3] == 3 && actual[4] == 8,
           @"packing walks four input lanes inside each output lane");
    expect(actual[31] == 59 && actual[32] == 64,
           @"packing advances to the next eight-output block");
    expect(actual[63] == 123 && actual[64] == 4,
           @"packing returns to the next four-input block");
    expect(actual[127] == 127,
           @"packing reaches the final output and input block");
}

static void testConvPackingRejectsUndecodedShapes(void) {
    NSError *error = nil;
    NSData *weights = [NSMutableData dataWithLength:
        15 * 8 * sizeof(uint16_t)];
    NSData *packed = [H16GConstantPacker packConv1x1Weights:weights
        inputChannels:8 outputChannels:15 bytesPerWeight:sizeof(uint16_t)
        error:&error];
    expect(packed == nil && error != nil,
           @"non-eight-channel output groups fail closed");

    error = nil;
    weights = [NSMutableData dataWithLength:
        16 * 8 * sizeof(uint16_t) - 1];
    packed = [H16GConstantPacker packConv1x1Weights:weights
        inputChannels:8 outputChannels:16 bytesPerWeight:sizeof(uint16_t)
        error:&error];
    expect(packed == nil && error != nil,
           @"truncated constant payloads fail closed");
}

static void testW8A8PackingUsesFourByFourLaneOrder(void) {
    NSMutableData *weights = [NSMutableData dataWithLength:8 * 8];
    uint8_t *source = (uint8_t *)weights.mutableBytes;
    for (NSUInteger i = 0; i < weights.length; ++i)
        source[i] = (uint8_t)i;

    NSError *error = nil;
    NSData *packed = [H16GConstantPacker packConv1x1Weights:weights
        inputChannels:8 outputChannels:8 bytesPerWeight:1
        packingFormat:H16GConvWeightPackingFormatW8A8 error:&error];
    const uint8_t *actual = (const uint8_t *)packed.bytes;
    expect(packed.length == 64 && error == nil,
           @"W8A8 packing preserves one complete int8 matrix");
    expect(actual[0] == 0 && actual[1] == 8 && actual[3] == 24 &&
           actual[4] == 1 && actual[15] == 27,
           @"W8A8 packing interleaves four output lanes for each input lane");
    expect(actual[16] == 4 && actual[31] == 31 &&
           actual[32] == 32 && actual[63] == 63,
           @"W8A8 packing advances through input and output blocks");
}

static void testUnknownPackingFormatFailsClosed(void) {
    NSError *error = nil;
    NSData *weights = [NSMutableData dataWithLength:8 * 8];
    NSData *packed = [H16GConstantPacker packConv1x1Weights:weights
        inputChannels:8 outputChannels:8 bytesPerWeight:1
        packingFormat:(H16GConvWeightPackingFormat)999 error:&error];
    expect(packed == nil && error != nil,
           @"unknown packing formats fail closed");
}

static void testFusedLayoutConvPackingMatchesHardwareTiles(void) {
    NSArray<NSNumber *> *channelCases = @[@128,@256,@384,@512];
    NSArray<NSString *> *hashes = @[
        @"7f32283eb4ded99eb90fb20f2af535129a350449b1f885503fe165413f46bd8b",
        @"72788b9a1baf1e1afb33b2199d4ff1f5aaffc412ac78f1ce76417795d4764111",
        @"b0b27b290f779910db29baf632b1be6259bb47144bd55e6bdbc07f091f311f07",
        @"bf1e32cfcace7277e232612c9b02e23188276f3f1a9590579cc21b6e55528e14",
    ];
    for (NSUInteger caseIndex = 0; caseIndex < channelCases.count; ++caseIndex) {
        NSUInteger channels = channelCases[caseIndex].unsignedIntegerValue;
        NSMutableData *weights = [NSMutableData dataWithLength:
            channels * channels * sizeof(_Float16)];
        _Float16 *values = (_Float16 *)weights.mutableBytes;
        for (NSUInteger i = 0; i < channels; ++i)
            values[i * channels + i] = (_Float16)1.0f;
        NSError *error = nil;
        NSData *packed = [H16GConstantPacker packConv1x1Weights:weights
            inputChannels:channels outputChannels:channels bytesPerWeight:2
            packingFormat:H16GConvWeightPackingFormatLayoutConv error:&error];
        expect(packed != nil && error == nil && [[sha256(packed) lowercaseString]
            isEqualToString:hashes[caseIndex]],
            @"fused layout Conv packing reproduces its independent kernel oracle");
    }
}

static void testDepthwisePackingMatchesIndependentHardwareSections(void) {
    NSArray<NSNumber *> *channelCases = @[@64,@128,@256,@512];
    NSArray<NSString *> *hashes = @[
        @"3f310516445b6427da23fd31433b9f2cab21ace8d471ccbf718c6c34804b121b",
        @"effd4da043eb9b735bc06b4fd4db7b356bfde34b73244259717ca9efb9f862c9",
        @"2c56466159476b5922d1d330fbcb9cc14621b1547f41891606e8094a412aeb10",
        @"e6f5d1980bfa2db0c64033f7d8525ece5fa96ac41a34721fd15db70f1b3d1296",
    ];
    for (NSUInteger caseIndex = 0; caseIndex < channelCases.count; ++caseIndex) {
        NSUInteger channels = channelCases[caseIndex].unsignedIntegerValue;
        NSMutableData *weights = [NSMutableData dataWithLength:
            channels * 9 * sizeof(_Float16)];
        _Float16 *values = (_Float16 *)weights.mutableBytes;
        for (NSUInteger index = 0; index < channels * 9; ++index)
            values[index] = (_Float16)(((NSInteger)(index % 7) - 3) * 0.125f);
        NSError *error = nil;
        NSData *packed = [H16GConstantPacker
            packDepthwise3x3Weights:weights channels:channels error:&error];
        NSString *actualHash = [sha256(packed) lowercaseString];
        BOOL matches = packed.length == 16 * (channels + 64) && error == nil &&
            [actualHash isEqualToString:hashes[caseIndex]];
        if (!matches)
            fprintf(stderr,"depthwise C%lu hash got=%s want=%s\n",
                (unsigned long)channels,actualHash.UTF8String,
                hashes[caseIndex].UTF8String);
        expect(matches,
            @"depthwise 3x3 packing reproduces its independent H16G kernel section");
    }
    NSError *error = nil;
    NSData *invalid = [H16GConstantPacker
        packDepthwise3x3Weights:[NSMutableData dataWithLength:96*9*2]
        channels:96 error:&error];
    expect(invalid == nil && error != nil,
        @"unmeasured depthwise channel geometry fails closed");
}

static void testRegularConvPackingMatchesIndependentHardwareSections(void) {
    NSArray<NSArray<NSNumber *> *> *cases = @[
        @[@64,@3], @[@64,@5], @[@128,@3], @[@128,@5],
    ];
    NSArray<NSString *> *hashes = @[
        @"3e8c357cc7ea0f21e0c4eab4c98d3a3365eeca9c280e2555566aedf364c890fb",
        @"5fd716152521ab710b836cae1b4995a6542c1b2d60716bd44bf36183d2003d1c",
        @"320e0d79b583bff58ce7b63d01d73a9f4be19c7c61fd8440b4a14e90ea21652d",
        @"21bc283b7617e473b5c9d17f1fdf827e03bba6fc5b48c0390b9ba54b4862d045",
    ];
    for (NSUInteger caseIndex = 0; caseIndex < cases.count; ++caseIndex) {
        NSUInteger channels = cases[caseIndex][0].unsignedIntegerValue;
        NSUInteger kernel = cases[caseIndex][1].unsignedIntegerValue;
        NSUInteger count = channels * channels * kernel * kernel;
        NSMutableData *weights = [NSMutableData dataWithLength:count*2];
        _Float16 *values = (_Float16 *)weights.mutableBytes;
        for (NSUInteger index = 0; index < count; ++index)
            values[index] = (_Float16)(((NSInteger)(index % 7)-3)*0.125f);
        NSError *error = nil;
        NSData *packed = [H16GConstantPacker packRegularConvWeights:weights
            inputChannels:channels outputChannels:channels
            kernelHeight:kernel kernelWidth:kernel error:&error];
        expect(packed != nil && error == nil &&
            [[sha256(packed) lowercaseString] isEqualToString:hashes[caseIndex]],
            @"regular 3x3/5x5 packing reproduces its independent H16G kernel section");
    }
}

int main(void) {
    @autoreleasepool {
        testConvPackingTraversesInputAndOutputBlocks();
        testConvPackingRejectsUndecodedShapes();
        testW8A8PackingUsesFourByFourLaneOrder();
        testFusedLayoutConvPackingMatchesHardwareTiles();
        testDepthwisePackingMatchesIndependentHardwareSections();
        testRegularConvPackingMatchesIndependentHardwareSections();
        testUnknownPackingFormatFailsClosed();
        printf("H16G constant packing: %s\n",
               failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
