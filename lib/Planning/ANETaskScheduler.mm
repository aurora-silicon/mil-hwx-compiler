#import "ANETaskScheduler.h"
#import "ANEComposedRegionPlanner.h"
#import "ANEMemoryPlanner.h"
#import "ANETilePlanner.h"

static BOOL isConstantNode(ANEOperationNode *node) {
    return node.kind == ANEOperationKindConstant ||
        [node.operationName isEqualToString:@"constexpr_affine_dequantize"];
}
static BOOL argumentContainsBlob(ANEGraphArgument *argument) {
    if (!argument) return NO;
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"BLOBFILE"]) return YES;
    for (ANEGraphNamedArgument *named in argument.callArguments)
        if (argumentContainsBlob(named.value)) return YES;
    for (ANEGraphArgument *element in argument.elements)
        if (argumentContainsBlob(element)) return YES;
    return NO;
}
static BOOL isRuntimeConstantNode(ANEOperationNode *node) {
    if (!isConstantNode(node) || !node.sourceOperation) return NO;
    for (ANEGraphArgument *argument in node.sourceOperation.attributes.allValues)
        if (argumentContainsBlob(argument)) return YES;
    return NO;
}
static ANEOperationNode *effectiveProducer(ANEOperationNode *node) {
    if (!node.foldedIntoNumericBoundary || isConstantNode(node)) return node;
    for (ANEOperationNode *input in node.inputs)
        if (!isConstantNode(input)) return effectiveProducer(input);
    return node;
}
static BOOL quantizedOutput(ANEOperationNode *node) {
    for (ANEOperationNode *user in node.users)
        if ([user.operationName isEqualToString:@"quantize"]) return YES;
    return NO;
}
static ANEScheduledCommand *command(ANEScheduledCommandKind kind,
                                    NSArray *inputs, NSArray *outputs) {
    return [[ANEScheduledCommand alloc] initWithKind:kind inputs:inputs outputs:outputs];
}

static NSArray<NSString *> *stageInputs(ANEOperationNode *node) {
    NSMutableArray<NSString *> *inputs = [NSMutableArray array];
    NSMutableSet<NSString *> *seen = [NSMutableSet set];
    if ([node.sourceOperation.operationName isEqualToString:
            node.operationName]) {
        for (NSString *operandName in @[@"x", @"y"]) {
            ANEGraphValue *value =
                node.sourceOperation.operands[operandName].value;
            if (!value || value.type.kind != ANEValueTypeKindTensor ||
                [seen containsObject:value.name]) continue;
            [inputs addObject:value.name];
            [seen addObject:value.name];
        }
    }
    for (ANEOperationNode *input in node.inputs)
        if (![seen containsObject:input.identifier]) {
            [inputs addObject:input.identifier];
            [seen addObject:input.identifier];
        }
    for (NSString *external in node.externalValueNames)
        if (![seen containsObject:external]) {
            [inputs addObject:external];
            [seen addObject:external];
        }
    return [inputs copy];
}

static ANEScheduledBridgeStorage bridgeStorageForSurface(
    ANEScheduledSurface *surface) {
    switch (surface.role) {
        case ANEScheduledSurfaceRoleExternalInput:
        case ANEScheduledSurfaceRoleOutput:
            return ANEScheduledBridgeStorageExternal;
        case ANEScheduledSurfaceRoleConstant:
            return ANEScheduledBridgeStorageConstant;
        case ANEScheduledSurfaceRoleIntermediate:
            return ANEScheduledBridgeStorageSRAM;
        case ANEScheduledSurfaceRoleCarry:
            return ANEScheduledBridgeStorageCarry;
    }
}

static ANEScheduledStage *stageForNode(
    ANEOperationNode *node,
    NSDictionary<NSString *, ANEScheduledSurface *> *surfaceByName) {
    ANEScheduledSurface *surface = surfaceByName[node.identifier];
    ANEScheduledBridgeStorage storage = surface
        ? bridgeStorageForSurface(surface) : ANEScheduledBridgeStorageSRAM;
    return [[ANEScheduledStage alloc]
        initWithSourceNodeIdentifier:node.identifier
        operationName:node.operationName operationKind:node.kind
        inputIdentifiers:stageInputs(node) outputIdentifier:node.identifier
        bridgeStorage:storage];
}

static NSUInteger groupIndexContaining(
    NSArray<NSMutableArray<ANEOperationNode *> *> *groups,
    ANEOperationNode *node) {
    for (NSUInteger i = 0; i < groups.count; ++i)
        if ([groups[i] containsObject:node]) return i;
    return NSNotFound;
}

static void applyOnlineMemoryPlans(
    ANEOperationGraph *graph,
    NSArray<ANEScheduledRegionPlan *> *plans,
    NSMutableArray<ANEScheduledSurface *> *surfaces) {
    for (ANEScheduledRegionPlan *plan in plans) {
        if (plan.topology != ANEScheduledTopologyOnlineReduction ||
            plan.taskIndexes.count == 0 ||
            plan.carriedSurfaceIdentifiers.count != 3) continue;
        NSSet<NSString *> *elided = [NSSet setWithArray:
            plan.elidedSurfaceIdentifiers];
        NSIndexSet *remove = [surfaces indexesOfObjectsPassingTest:
            ^BOOL(ANEScheduledSurface *surface, NSUInteger index, BOOL *stop) {
                (void)index;
                (void)stop;
                return surface.role == ANEScheduledSurfaceRoleIntermediate &&
                    [elided containsObject:surface.identifier];
            }];
        [surfaces removeObjectsAtIndexes:remove];
        ANEOperationNode *output = [graph nodeForValueName:
            plan.stageIdentifiers.lastObject];
        if (!output || output.outputType.shape.count == 0) continue;
        NSMutableArray<NSNumber *> *rowShape =
            [output.outputType.shape mutableCopy];
        rowShape[rowShape.count - 1] = @1;
        NSUInteger first = plan.taskIndexes.firstObject.unsignedIntegerValue;
        NSUInteger last = plan.taskIndexes.lastObject.unsignedIntegerValue;
        NSArray<NSArray<NSNumber *> *> *shapes = @[
            rowShape, rowShape, output.outputType.shape,
        ];
        for (NSUInteger index = 0;
             index < plan.carriedSurfaceIdentifiers.count; ++index) {
            ANEScheduledSurface *carry = [[ANEScheduledSurface alloc]
                initWithIdentifier:plan.carriedSurfaceIdentifiers[index]
                role:ANEScheduledSurfaceRoleCarry
                elementType:output.outputType.elementType
                shape:shapes[index] firstTask:first];
            [carry extendLifetimeThroughTask:last];
            [surfaces addObject:carry];
        }
    }
}

static void mergeTaskGroups(
    NSMutableArray<NSMutableArray<ANEOperationNode *> *> *groups,
    ANEOperationNode *first, ANEOperationNode *second) {
    NSUInteger a = groupIndexContaining(groups, first);
    NSUInteger b = groupIndexContaining(groups, second);
    if (a == NSNotFound || b == NSNotFound || a == b) return;
    NSUInteger destination = MIN(a, b);
    NSUInteger source = MAX(a, b);
    [groups[destination] addObjectsFromArray:groups[source]];
    [groups[destination] sortUsingComparator:
        ^NSComparisonResult(ANEOperationNode *left, ANEOperationNode *right) {
            if (left.ordinal < right.ordinal) return NSOrderedAscending;
            if (left.ordinal > right.ordinal) return NSOrderedDescending;
            return NSOrderedSame;
        }];
    [groups removeObjectAtIndex:source];
}

static BOOL sharesExternalInput(ANEOperationNode *a, ANEOperationNode *b) {
    for (NSString *left in a.externalValueNames)
        if ([b.externalValueNames containsObject:left]) return YES;
    return NO;
}

static NSArray<ANEOperationNode *> *nonConstantInputs(ANEOperationNode *node) {
    NSMutableArray<ANEOperationNode *> *inputs = [NSMutableArray array];
    for (ANEOperationNode *input in node.inputs)
        if (!isConstantNode(input)) [inputs addObject:input];
    return inputs;
}

static BOOL nodesShareRegion(ANEOperationNode *left, ANEOperationNode *right,
                             NSDictionary<NSNumber *, NSString *> *regions) {
    NSString *leftRegion = regions[@(left.ordinal)];
    NSString *rightRegion = regions[@(right.ordinal)];
    return leftRegion && [leftRegion isEqualToString:rightRegion];
}

static NSArray<NSArray<ANEOperationNode *> *> *formTaskGroups(
    NSArray<ANEOperationNode *> *active,
    NSDictionary<NSNumber *, NSString *> *regions) {
    NSMutableArray<NSMutableArray<ANEOperationNode *> *> *groups =
        [NSMutableArray array];
    for (ANEOperationNode *node in active)
        [groups addObject:[NSMutableArray arrayWithObject:node]];

    for (NSUInteger i = 0; i < active.count; ++i) {
        ANEOperationNode *left = active[i];
        if (left.kind != ANEOperationKindLayout ||
            ![left.operationName isEqualToString:@"slice_by_size"]) continue;
        for (NSUInteger j = i + 1; j < active.count; ++j) {
            ANEOperationNode *right = active[j];
            if (right.kind == ANEOperationKindLayout &&
                [right.operationName isEqualToString:@"slice_by_size"] &&
                nodesShareRegion(left,right,regions) &&
                sharesExternalInput(left, right))
                mergeTaskGroups(groups, left, right);
        }
    }
    for (NSUInteger i = 0; i < active.count; ++i) {
        ANEOperationNode *left = active[i];
        if (left.kind != ANEOperationKindLayout || left.users.count != 1)
            continue;
        for (NSUInteger j = i + 1; j < active.count; ++j) {
            ANEOperationNode *right = active[j];
            if (right.kind == ANEOperationKindLayout && right.users.count == 1 &&
                nodesShareRegion(left,right,regions) &&
                left.users[0] == right.users[0] &&
                left.users[0].kind == ANEOperationKindMatmul)
                mergeTaskGroups(groups, left, right);
        }
    }
    for (ANEOperationNode *node in active) {
        if (node.kind != ANEOperationKindLUT || node.inputs.count != 1) continue;
        ANEOperationNode *producer = node.inputs[0];
        if (producer.kind == ANEOperationKindALU && producer.users.count == 1 &&
            nodesShareRegion(producer,node,regions))
            mergeTaskGroups(groups, producer, node);
    }
    for (ANEOperationNode *node in active) {
        if (node.kind != ANEOperationKindMatmul) continue;
        NSArray<ANEOperationNode *> *inputs = nonConstantInputs(node);
        if (inputs.count != 2) continue;
        ANEOperationNode *alu = nil;
        ANEOperationNode *layout = nil;
        for (ANEOperationNode *input in inputs) {
            if (input.kind == ANEOperationKindALU) alu = input;
            if (input.kind == ANEOperationKindLayout) layout = input;
        }
        if (alu && layout && nodesShareRegion(alu,layout,regions))
            mergeTaskGroups(groups, alu, layout);
    }
    for (ANEOperationNode *node in active) {
        if (node.kind != ANEOperationKindMatmul) continue;
        ANEOperationNode *tail = node;
        while (tail.users.count == 1 &&
               tail.users[0].kind == ANEOperationKindLayout) {
            ANEOperationNode *next = tail.users[0];
            if (!nodesShareRegion(node,next,regions)) break;
            mergeTaskGroups(groups, node, next);
            tail = next;
        }
    }
    return groups;
}

static ANEValueType *typeForIdentifier(ANEOperationGraph *graph,
                                       NSString *identifier) {
    for (ANEGraphValue *input in graph.sourceFunction.inputs)
        if ([input.name isEqualToString:identifier]) return input.type;
    return [graph nodeForValueName:identifier].outputType;
}

static BOOL targetSupportsGroup(NSArray<ANEOperationNode *> *group,
                                ANEOperationGraph *graph,
                                H16GTarget *target) {
    if (group.count == 0) return NO;
    NSMutableSet<NSString *> *internal = [NSMutableSet set];
    NSMutableArray<NSString *> *operations = [NSMutableArray array];
    for (ANEOperationNode *node in group) {
        [internal addObject:node.identifier];
        [operations addObject:node.operationName];
    }
    NSMutableArray<NSArray<NSNumber *> *> *inputShapes =
        [NSMutableArray array];
    NSMutableSet<NSString *> *seen = [NSMutableSet set];
    for (ANEOperationNode *node in group)
        for (NSString *identifier in stageInputs(node)) {
            if ([internal containsObject:identifier] ||
                [seen containsObject:identifier]) continue;
            ANEValueType *type = typeForIdentifier(graph, identifier);
            if (!type || type.kind != ANEValueTypeKindTensor) continue;
            [seen addObject:identifier];
            [inputShapes addObject:type.shape];
        }
    ANEOperationNode *output = group.lastObject;
    for (NSNumber *storage in @[
            @(ANEScheduledBridgeStorageSRAM),
            @(ANEScheduledBridgeStorageExternal)])
        if ([target taskCapabilityForStageOperations:operations
            inputShapes:inputShapes outputShape:output.outputType.shape
            numericMode:output.numericMode
            outputStorage:(ANEScheduledBridgeStorage)
                storage.unsignedIntegerValue]) return YES;
    return NO;
}

static NSArray<NSArray<ANEOperationNode *> *> *splitUnsupportedTaskGroups(
    NSArray<NSArray<ANEOperationNode *> *> *groups,
    ANEOperationGraph *graph, H16GTarget *target) {
    NSMutableArray<NSArray<ANEOperationNode *> *> *result =
        [NSMutableArray array];
    for (NSArray<ANEOperationNode *> *group in groups) {
        BOOL aluLUTPair = group.count == 2 &&
            group.firstObject.kind == ANEOperationKindALU &&
            group.lastObject.kind == ANEOperationKindLUT;
        if (!aluLUTPair || targetSupportsGroup(group, graph, target)) {
            [result addObject:group];
            continue;
        }
        BOOL everyStageSupported = YES;
        for (ANEOperationNode *node in group)
            everyStageSupported = everyStageSupported &&
                targetSupportsGroup(@[node], graph, target);
        if (!everyStageSupported) {
            [result addObject:group];
            continue;
        }
        for (ANEOperationNode *node in group) [result addObject:@[node]];
    }
    return [result copy];
}

@implementation ANETaskScheduler
+ (ANEScheduledGraph *)scheduleGraph:(ANEOperationGraph *)graph
                               target:(H16GTarget *)target
                          diagnostics:(ANEDiagnosticEngine *)diagnostics {
    if (graph.regions.count == 0) {
        [diagnostics emitSeverity:ANEDiagnosticSeverityError
                             code:@"h16g.schedule.no-regions"
                          message:@"fusion regions must be formed before scheduling"
                            range:ANESourceRangeMake(ANESourceLocationMake(0,1,1),
                                                     ANESourceLocationMake(0,1,1))];
        return nil;
    }
    NSMutableDictionary<NSNumber *, NSString *> *regionForNode = [NSMutableDictionary dictionary];
    for (ANEGraphRegion *region in graph.regions)
        for (ANEOperationNode *node in region.nodes)
            regionForNode[@(node.ordinal)] = region.identifier;

    NSMutableDictionary<NSNumber *, ANEOperationNode *> *epilogueForProducer =
        [NSMutableDictionary dictionary];
    NSMutableSet<NSNumber *> *foldedEpilogues = [NSMutableSet set];
    for (ANEOperationNode *node in graph.nodes) {
        if ((node.kind != ANEOperationKindALU &&
             node.kind != ANEOperationKindLUT) || node.inputs.count != 1 ||
            node.externalValueNames.count != 0) continue;
        ANEOperationNode *producer = node.inputs[0];
        BOOL sameRegion = [regionForNode[@(producer.ordinal)]
            isEqualToString:regionForNode[@(node.ordinal)]];
        if (sameRegion && producer.users.count == 1 &&
            [target supportsFusedEpilogue:node.operationName
                               producerKind:producer.kind]) {
            epilogueForProducer[@(producer.ordinal)] = node;
            [foldedEpilogues addObject:@(node.ordinal)];
        }
    }

    NSMutableArray<ANEOperationNode *> *active = [NSMutableArray array];
    for (ANEOperationNode *node in graph.nodes)
        if (!isConstantNode(node) && !node.foldedIntoNumericBoundary &&
            ![foldedEpilogues containsObject:@(node.ordinal)]) [active addObject:node];
    NSArray<NSArray<ANEOperationNode *> *> *taskGroups =
        splitUnsupportedTaskGroups(formTaskGroups(active,regionForNode),
                                   graph, target);
    NSMutableDictionary<NSNumber *, NSNumber *> *taskForNode = [NSMutableDictionary dictionary];
    for (NSUInteger i = 0; i < taskGroups.count; ++i) {
        for (ANEOperationNode *node in taskGroups[i]) {
            taskForNode[@(node.ordinal)] = @(i);
            ANEOperationNode *epilogue = epilogueForProducer[@(node.ordinal)];
            if (epilogue) taskForNode[@(epilogue.ordinal)] = @(i);
        }
    }

    NSMutableDictionary<NSString *, ANEScheduledSurface *> *surfaceByName =
        [NSMutableDictionary dictionary];
    NSMutableArray<ANEScheduledSurface *> *surfaces = [NSMutableArray array];
    NSSet<NSString *> *outputs = [NSSet setWithArray:graph.outputValueNames];
    for (ANEOperationNode *node in active) {
        NSUInteger task = taskForNode[@(node.ordinal)].unsignedIntegerValue;
        ANEOperationNode *resultNode = epilogueForProducer[@(node.ordinal)] ?: node;
        ANEElementType type = quantizedOutput(resultNode) ? ANEElementTypeInt8
                                              : resultNode.outputType.elementType;
        ANEScheduledSurfaceRole role = [outputs containsObject:resultNode.identifier]
            ? ANEScheduledSurfaceRoleOutput : ANEScheduledSurfaceRoleIntermediate;
        ANEScheduledSurface *surface = [[ANEScheduledSurface alloc]
            initWithIdentifier:resultNode.identifier role:role elementType:type
            shape:resultNode.outputType.shape firstTask:task];
        surfaceByName[resultNode.identifier] = surface; [surfaces addObject:surface];
    }

    NSMutableArray<ANEScheduledTask *> *tasks = [NSMutableArray array];
    for (NSUInteger taskIndex = 0; taskIndex < taskGroups.count; ++taskIndex) {
        NSArray<ANEOperationNode *> *group = taskGroups[taskIndex];
        ANEOperationNode *anchor = group.firstObject;
        NSString *region = regionForNode[@(anchor.ordinal)] ?: @"unfused";
        ANEOperationNode *lastNode = group.lastObject;
        ANEOperationNode *taskResult =
            epilogueForProducer[@(lastNode.ordinal)] ?: lastNode;
        NSMutableArray<ANEScheduledCommand *> *commands = [NSMutableArray array];
        NSMutableSet<NSNumber *> *dependencySet = [NSMutableSet set];
        NSMutableSet<NSString *> *seenInputs = [NSMutableSet set];
        NSMutableArray<NSString *> *computeOutputs = [NSMutableArray array];
        NSMutableArray<NSString *> *sourceNodes = [NSMutableArray array];
        NSMutableArray<ANEScheduledStage *> *stages = [NSMutableArray array];
        for (ANEOperationNode *node in group) {
            ANEOperationNode *resultNode =
                epilogueForProducer[@(node.ordinal)] ?: node;
            [computeOutputs addObject:resultNode.identifier];
            [sourceNodes addObject:node.identifier];
            [stages addObject:stageForNode(node, surfaceByName)];
            if (resultNode != node) {
                [sourceNodes addObject:resultNode.identifier];
                [stages addObject:stageForNode(resultNode, surfaceByName)];
            }
            for (ANEOperationNode *rawInput in node.inputs) {
                if (isConstantNode(rawInput) && !isRuntimeConstantNode(rawInput))
                    continue;
                ANEOperationNode *input = effectiveProducer(rawInput);
                NSNumber *producerTask = taskForNode[@(input.ordinal)];
                if (producerTask &&
                    producerTask.unsignedIntegerValue == taskIndex) continue;
                NSString *name = input.identifier;
                if ([seenInputs containsObject:name]) continue;
                [seenInputs addObject:name];
                ANEScheduledSurface *surface = surfaceByName[name];
                if (!surface) {
                    surface = [[ANEScheduledSurface alloc] initWithIdentifier:name
                        role:ANEScheduledSurfaceRoleConstant
                        elementType:input.outputType.elementType
                        shape:input.outputType.shape firstTask:taskIndex];
                    surfaceByName[name] = surface;
                    [surfaces addObject:surface];
                }
                BOOL internal = producerTask &&
                    [regionForNode[@(input.ordinal)] isEqualToString:region];
                [commands addObject:command(
                    internal ? ANEScheduledCommandKindDMAInter
                             : ANEScheduledCommandKindDMALoad,
                    @[name], @[resultNode.identifier])];
                if (producerTask) {
                    [dependencySet addObject:producerTask];
                    [surface extendLifetimeThroughTask:taskIndex];
                }
            }
            for (NSString *external in node.externalValueNames) {
                if ([seenInputs containsObject:external]) continue;
                [seenInputs addObject:external];
                ANEScheduledSurface *surface = surfaceByName[external];
                if (!surface) {
                    ANEValueType *externalType = nil;
                    for (ANEGraphValue *value in graph.sourceFunction.inputs)
                        if ([value.name isEqualToString:external]) {
                            externalType = value.type;
                            break;
                        }
                    ANEValueType *type = externalType ?: node.outputType;
                    surface = [[ANEScheduledSurface alloc]
                        initWithIdentifier:external
                        role:ANEScheduledSurfaceRoleExternalInput
                        elementType:type.elementType
                        shape:type.shape firstTask:taskIndex];
                    surfaceByName[external] = surface;
                    [surfaces addObject:surface];
                }
                [commands addObject:command(ANEScheduledCommandKindDMALoad,
                    @[external], @[resultNode.identifier])];
            }
        }
        [commands addObject:command(ANEScheduledCommandKindCompute, @[],
                                    computeOutputs)];
        BOOL store = [outputs containsObject:taskResult.identifier];
        if (!store)
            for (ANEGraphRegion *candidate in graph.regions)
                if ([candidate.identifier isEqualToString:region] &&
                    [candidate.materializedValues
                        containsObject:taskResult.identifier]) store = YES;
        if (store) [commands addObject:command(ANEScheduledCommandKindDMAStore,
                @[taskResult.identifier], @[taskResult.identifier])];
        NSArray<NSNumber *> *dependencies = [[dependencySet allObjects]
            sortedArrayUsingSelector:@selector(compare:)];
        NSUInteger waveIndex = 0;
        for (NSNumber *dependency in dependencies) {
            NSUInteger dependencyIndex = dependency.unsignedIntegerValue;
            if (dependencyIndex < tasks.count)
                waveIndex = MAX(waveIndex,
                    tasks[dependencyIndex].waveIndex + 1);
        }
        [tasks addObject:[[ANEScheduledTask alloc] initWithIndex:taskIndex
            regionIdentifier:region nodeIdentifier:taskResult.identifier
            sourceNodeIdentifiers:sourceNodes
            operationKind:anchor.kind numericMode:anchor.numericMode
            tilePlan:[ANETilePlanner tilePlanForNode:anchor target:target]
            commands:commands dependencies:dependencies stages:stages
            waveIndex:waveIndex topology:ANEScheduledTopologyDirect]];
    }
    NSArray<ANEScheduledRegionPlan *> *composedPlans =
        [ANEComposedRegionPlanner plansForGraph:graph tasks:tasks];
    applyOnlineMemoryPlans(graph, composedPlans, surfaces);
    NSUInteger peak = [ANEMemoryPlanner allocateSurfaces:surfaces target:target];
    if (peak > target.workingSetBytes) {
        [diagnostics emitSeverity:ANEDiagnosticSeverityError
                             code:@"h16g.schedule.sram-overflow"
                          message:[NSString stringWithFormat:@"planned SRAM %lu exceeds %lu",
                              (unsigned long)peak, (unsigned long)target.workingSetBytes]
                            range:ANESourceRangeMake(ANESourceLocationMake(0,1,1),
                                                     ANESourceLocationMake(0,1,1))];
        return nil;
    }
    return [[ANEScheduledGraph alloc] initWithSurfaces:surfaces tasks:tasks
        composedPlans:composedPlans peakSRAMBytes:peak];
}
@end
