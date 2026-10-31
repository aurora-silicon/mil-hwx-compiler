// SPDX-License-Identifier: MIT
#import "ANEGraphIR.h"

// Find a single-use ReLU edge inside a larger DAG. Encoding admission remains
// the target's responsibility: a graph edge alone never proves DMA routing.
static inline ANEGraphOperation *ANEReluRegionConsumer(
    NSArray<ANEGraphOperation *> *operations, ANEGraphOperation *producer,
    NSSet<NSString *> *returnedStorageNames) {
    if (producer.results.count != 1 ||
        ![@[@"add", @"mul", @"matmul"] containsObject:producer.operationName])
        return nil;
    ANEGraphValue *value = producer.results[0];
    if ([returnedStorageNames containsObject:value.name]) return nil;
    ANEGraphOperation *consumer = nil;
    NSUInteger uses = 0;
    BOOL afterProducer = NO;
    for (ANEGraphOperation *candidate in operations) {
        if (candidate == producer) {
            afterProducer = YES;
            continue;
        }
        for (ANEGraphArgument *argument in candidate.operands.allValues) {
            if (![argument.value.name isEqualToString:value.name]) continue;
            if (!afterProducer) return nil;
            ++uses;
            consumer = candidate;
        }
    }
    if (uses != 1 || ![consumer.operationName isEqualToString:@"relu"] ||
        consumer.arguments.count != 1 || consumer.results.count != 1 ||
        ![consumer.operands[@"x"].value.name isEqualToString:value.name] ||
        ![value.type isEqualToValueType:consumer.results[0].type])
        return nil;
    return consumer;
}
