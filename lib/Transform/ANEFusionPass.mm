#import "ANEFusionPass.h"

static NSUInteger byteWidth(ANEElementType type) {
    switch (type) {
        case ANEElementTypeFP16: return 2;
        case ANEElementTypeFP32: return 4;
        case ANEElementTypeInt8: return 1;
        case ANEElementTypeInt32: return 4;
        case ANEElementTypeUInt64: return 8;
        case ANEElementTypeBool: return 1;
        case ANEElementTypeString:
        case ANEElementTypeInvalid: return 0;
    }
}

static NSUInteger valueBytes(ANEOperationNode *node) {
    NSUInteger elements = 1;
    for (NSNumber *dimension in node.outputType.shape)
        elements *= dimension.unsignedIntegerValue;
    return elements * byteWidth(node.outputType.elementType);
}

static void collectReachableOutputs(ANEOperationNode *node,
                                    NSSet<NSString *> *graphOutputs,
                                    NSMutableSet<NSString *> *found,
                                    NSMutableSet<ANEOperationNode *> *visited) {
    if ([visited containsObject:node]) return;
    [visited addObject:node];
    if ([graphOutputs containsObject:node.identifier]) [found addObject:node.identifier];
    for (ANEOperationNode *user in node.users)
        collectReachableOutputs(user, graphOutputs, found, visited);
}

static NSUInteger reachableOutputCount(ANEOperationNode *node,
                                       NSSet<NSString *> *outputs) {
    NSMutableSet<NSString *> *found = [NSMutableSet set];
    collectReachableOutputs(node, outputs, found, [NSMutableSet set]);
    return found.count;
}

static NSUInteger findRoot(NSMutableArray<NSNumber *> *parents, NSUInteger value) {
    NSUInteger parent = parents[value].unsignedIntegerValue;
    if (parent == value) return value;
    NSUInteger root = findRoot(parents, parent);
    parents[value] = @(root);
    return root;
}

static void unite(NSMutableArray<NSNumber *> *parents, NSUInteger a, NSUInteger b) {
    NSUInteger rootA = findRoot(parents, a);
    NSUInteger rootB = findRoot(parents, b);
    if (rootA == rootB) return;
    if (rootA < rootB) parents[rootB] = @(rootA);
    else parents[rootA] = @(rootB);
}

static NSUInteger peakLiveBytes(NSArray<ANEOperationNode *> *nodes,
                                NSSet<NSString *> *graphOutputs) {
    NSSet<ANEOperationNode *> *members = [NSSet setWithArray:nodes];
    NSMutableDictionary<NSNumber *, NSNumber *> *remaining =
        [NSMutableDictionary dictionary];
    for (ANEOperationNode *node in nodes) {
        NSUInteger uses = 0;
        for (ANEOperationNode *user in node.users)
            if ([members containsObject:user]) uses++;
        if ([graphOutputs containsObject:node.identifier]) uses++;
        remaining[@(node.ordinal)] = @(uses);
    }
    NSUInteger live = 0;
    NSUInteger peak = 0;
    for (ANEOperationNode *node in nodes) {
        live += valueBytes(node);
        peak = MAX(peak, live);
        for (ANEOperationNode *input in node.inputs) {
            if (![members containsObject:input]) continue;
            NSNumber *inputKey = @(input.ordinal);
            NSUInteger uses = remaining[inputKey].unsignedIntegerValue;
            if (uses == 0) continue;
            uses--;
            remaining[inputKey] = @(uses);
            if (uses == 0) live -= valueBytes(input);
        }
    }
    return peak;
}

static BOOL hasRank3FusedGELU(NSArray<ANEOperationNode *> *nodes) {
    NSSet<ANEOperationNode *> *members=[NSSet setWithArray:nodes];
    for (ANEOperationNode *node in nodes) {
        if (![node.operationName isEqualToString:@"gelu"] ||
            node.inputs.count!=1) continue;
        ANEOperationNode *producer=node.inputs.firstObject;
        if ([members containsObject:producer] &&
            producer.outputType.shape.count==3 &&
            (producer.kind==ANEOperationKindConv ||
             [producer.operationName isEqualToString:@"linear"])) return YES;
    }
    return NO;
}

@implementation ANEFusionPass
+ (BOOL)runOnGraph:(ANEOperationGraph *)graph
             target:(H16GTarget *)target
        diagnostics:(ANEDiagnosticEngine *)diagnostics {
    (void)diagnostics;
    NSMutableArray<NSNumber *> *parents = [NSMutableArray array];
    for (NSUInteger i = 0; i < graph.nodes.count; ++i) [parents addObject:@(i)];
    NSSet<NSString *> *outputs = [NSSet setWithArray:graph.outputValueNames];

    for (ANEOperationNode *node in graph.nodes) {
        if (![target supportsOperationKind:node.kind]) continue;
        for (ANEOperationNode *input in node.inputs) {
            if (![target supportsOperationKind:input.kind]) continue;
            if (input.users.count > 1 && reachableOutputCount(input, outputs) > 1)
                continue;
            unite(parents, input.ordinal, node.ordinal);
        }
    }

    NSMutableDictionary<NSNumber *, NSMutableArray<ANEOperationNode *> *> *groups =
        [NSMutableDictionary dictionary];
    for (ANEOperationNode *node in graph.nodes) {
        NSNumber *root = @(findRoot(parents, node.ordinal));
        if (!groups[root]) groups[root] = [NSMutableArray array];
        [groups[root] addObject:node];
    }

    NSMutableArray<ANEGraphRegion *> *regions = [NSMutableArray array];
    NSArray<NSNumber *> *roots = [[groups allKeys] sortedArrayUsingSelector:@selector(compare:)];
    NSUInteger regionIndex = 0;
    for (NSNumber *root in roots) {
        NSArray<ANEOperationNode *> *members = groups[root];
        NSArray<NSArray<ANEOperationNode *> *> *pieces;
        if (peakLiveBytes(members, outputs) <= target.workingSetBytes ||
            hasRank3FusedGELU(members)) {
            pieces = @[members];
        } else {
            NSMutableArray<NSArray<ANEOperationNode *> *> *singletons =
                [NSMutableArray array];
            for (ANEOperationNode *node in members) [singletons addObject:@[node]];
            pieces = singletons;
        }
        for (NSArray<ANEOperationNode *> *piece in pieces) {
            NSSet *pieceSet = [NSSet setWithArray:piece];
            NSMutableArray<NSString *> *materialized = [NSMutableArray array];
            for (ANEOperationNode *node in piece) {
                BOOL escapes = [outputs containsObject:node.identifier];
                for (ANEOperationNode *user in node.users)
                    if (![pieceSet containsObject:user]) escapes = YES;
                if (escapes) [materialized addObject:node.identifier];
            }
            [regions addObject:[[ANEGraphRegion alloc]
                initWithIdentifier:[NSString stringWithFormat:@"region%lu",
                    (unsigned long)regionIndex++]
                nodes:piece materializedValues:materialized]];
        }
    }
    [graph setPlannedRegions:regions];
    return YES;
}
@end
