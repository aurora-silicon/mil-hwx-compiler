#import "ANEOperationGraph.h"

static ANEOperationKind classifyOperation(NSString *name) {
    static NSDictionary<NSString *, NSNumber *> *kinds = [] {
        return @{
            @"const": @(ANEOperationKindConstant),
            @"conv": @(ANEOperationKindConv),
            @"matmul": @(ANEOperationKindMatmul),
            @"linear": @(ANEOperationKindMatmul),
            @"relu": @(ANEOperationKindALU),
            @"add": @(ANEOperationKindALU),
            @"mul": @(ANEOperationKindALU),
            @"multiply": @(ANEOperationKindALU),
            @"sub": @(ANEOperationKindALU),
            @"max": @(ANEOperationKindALU),
            @"min": @(ANEOperationKindALU),
            @"square": @(ANEOperationKindALU),
            @"reciprocal": @(ANEOperationKindALU),
            @"rsqrt": @(ANEOperationKindALU),
            @"sqrt": @(ANEOperationKindLUT),
            @"sigmoid": @(ANEOperationKindLUT),
            @"tanh": @(ANEOperationKindLUT),
            @"gelu": @(ANEOperationKindLUT),
            @"silu": @(ANEOperationKindLUT),
            @"exp": @(ANEOperationKindLUT),
            @"log": @(ANEOperationKindLUT),
            @"reduce_sum": @(ANEOperationKindReduce),
            @"reduce_mean": @(ANEOperationKindReduce),
            @"reduce_max": @(ANEOperationKindReduce),
            @"transpose": @(ANEOperationKindLayout),
            @"reshape": @(ANEOperationKindLayout),
            @"slice_by_size": @(ANEOperationKindLayout),
            @"slice_by_index": @(ANEOperationKindLayout),
            @"space_to_depth": @(ANEOperationKindLayout),
            @"depth_to_space": @(ANEOperationKindLayout),
            @"pad": @(ANEOperationKindLayout),
            @"tile": @(ANEOperationKindLayout),
            @"quantize": @(ANEOperationKindQuantize),
            @"dequantize": @(ANEOperationKindDequantize),
            @"constexpr_affine_dequantize": @(ANEOperationKindDequantize),
            @"softmax": @(ANEOperationKindHighLevel),
            @"layer_norm": @(ANEOperationKindHighLevel),
            @"less": @(ANEOperationKindMask),
            @"floor": @(ANEOperationKindMask),
            @"select": @(ANEOperationKindMask),
            @"floor_div": @(ANEOperationKindMask),
            @"logical_and": @(ANEOperationKindMask),
            @"logical_not": @(ANEOperationKindMask),
        };
    }();
    NSNumber *kind = kinds[name];
    return kind ? (ANEOperationKind)kind.unsignedIntegerValue
                : ANEOperationKindUnsupported;
}

static NSString *kindName(ANEOperationKind kind) {
    switch (kind) {
        case ANEOperationKindConstant: return @"constant";
        case ANEOperationKindConv: return @"conv";
        case ANEOperationKindMatmul: return @"matmul";
        case ANEOperationKindALU: return @"alu";
        case ANEOperationKindLUT: return @"lut";
        case ANEOperationKindReduce: return @"reduce";
        case ANEOperationKindLayout: return @"layout";
        case ANEOperationKindQuantize: return @"quantize";
        case ANEOperationKindDequantize: return @"dequantize";
        case ANEOperationKindHighLevel: return @"high-level";
        case ANEOperationKindMask: return @"mask";
        case ANEOperationKindUnsupported: return @"unsupported";
    }
}

@interface ANEOperationNode ()
@property(nonatomic) NSUInteger ordinal;
@property(nonatomic, copy) NSString *identifier;
@property(nonatomic, copy) NSString *operationName;
@property(nonatomic) ANEOperationKind kind;
@property(nonatomic) ANEValueType *outputType;
@property(nonatomic, nullable) ANEGraphOperation *sourceOperation;
@property(nonatomic, copy) NSArray<ANEOperationNode *> *inputs;
@property(nonatomic, copy) NSArray<NSString *> *externalValueNames;
@property(nonatomic) NSPointerArray *weakUsers;
@property(nonatomic) ANEPhysicalLayout physicalLayout;
@property(nonatomic) ANELegalNumericMode numericMode;
@property(nonatomic) BOOL foldedIntoNumericBoundary;
- (void)addUser:(ANEOperationNode *)user;
- (void)clearUsers;
@end

@implementation ANEOperationNode
- (instancetype)initWithIdentifier:(NSString *)identifier
                      operationName:(NSString *)operationName
                               kind:(ANEOperationKind)kind
                         outputType:(ANEValueType *)outputType
                             inputs:(NSArray<ANEOperationNode *> *)inputs
                 externalValueNames:(NSArray<NSString *> *)externalValueNames
                    sourceOperation:(ANEGraphOperation *)sourceOperation {
    self = [super init];
    if (self) {
        _identifier = [identifier copy];
        _operationName = [operationName copy];
        _kind = kind;
        _outputType = outputType;
        _inputs = [inputs copy];
        _externalValueNames = [externalValueNames copy];
        _sourceOperation = sourceOperation;
        _physicalLayout = ANEPhysicalLayoutUnknown;
        _numericMode = ANELegalNumericModeFP16;
        _weakUsers = [NSPointerArray weakObjectsPointerArray];
    }
    return self;
}
- (void)addUser:(ANEOperationNode *)user { [_weakUsers addPointer:(__bridge void *)user]; }
- (void)clearUsers { _weakUsers = [NSPointerArray weakObjectsPointerArray]; }
- (void)applyPhysicalLayout:(ANEPhysicalLayout)layout
                numericMode:(ANELegalNumericMode)numericMode {
    _physicalLayout = layout;
    _numericMode = numericMode;
}
- (void)markFoldedIntoNumericBoundary { _foldedIntoNumericBoundary = YES; }
- (NSArray<ANEOperationNode *> *)users {
    [_weakUsers compact];
    NSMutableArray<ANEOperationNode *> *result = [NSMutableArray array];
    for (NSUInteger i = 0; i < _weakUsers.count; ++i) {
        ANEOperationNode *node = (__bridge ANEOperationNode *)[_weakUsers pointerAtIndex:i];
        if (node) [result addObject:node];
    }
    return [result copy];
}
@end

@implementation ANEGraphRegion
- (instancetype)initWithIdentifier:(NSString *)identifier
                              nodes:(NSArray<ANEOperationNode *> *)nodes
                 materializedValues:(NSArray<NSString *> *)materializedValues {
    self = [super init];
    if (self) {
        _identifier = [identifier copy];
        _nodes = [nodes copy];
        _materializedValues = [materializedValues copy];
    }
    return self;
}
@end

@implementation ANEOperationGraph {
    NSMutableDictionary<NSString *, ANEOperationNode *> *_nodesByValue;
}

- (instancetype)initWithFunction:(ANEGraphFunction *)function
                      diagnostics:(ANEDiagnosticEngine *)diagnostics {
    self = [super init];
    if (!self) return nil;
    _sourceFunction = function;
    for (ANEGraphOperation *operation in function.operations) {
        if (operation.results.count != 1) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                 code:@"ane.graph.unsupported-multi-result-operation"
                              message:[NSString stringWithFormat:
                                  @"operation '%@' has %lu results; the operation graph requires exactly one",
                                  operation.operationName,
                                  (unsigned long)operation.results.count]
                                range:operation.range];
            return nil;
        }
    }
    NSMutableArray<ANEOperationNode *> *nodes = [NSMutableArray array];
    NSMutableDictionary<NSString *, ANEOperationNode *> *byValue =
        [NSMutableDictionary dictionary];
    NSUInteger ordinal = 0;
    for (ANEGraphOperation *operation in function.operations) {
        ANEGraphValue *result = operation.results[0];
        ANEOperationNode *node = [[ANEOperationNode alloc]
            initWithIdentifier:result.name
                 operationName:operation.operationName
                          kind:classifyOperation(operation.operationName)
                    outputType:result.type
                        inputs:@[]
            externalValueNames:@[]
               sourceOperation:operation];
        node.ordinal = ordinal++;
        if (byValue[node.identifier]) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                 code:@"ane.graph.duplicate-result"
                              message:[NSString stringWithFormat:
                                  @"duplicate graph result '%@'", node.identifier]
                                range:operation.range];
            return nil;
        }
        byValue[node.identifier] = node;
        [nodes addObject:node];
    }
    for (ANEOperationNode *node in nodes) {
        NSMutableArray<ANEOperationNode *> *inputs = [NSMutableArray array];
        NSMutableArray<NSString *> *external = [NSMutableArray array];
        NSArray<NSString *> *keys = [[node.sourceOperation.operands allKeys]
            sortedArrayUsingSelector:@selector(compare:)];
        for (NSString *key in keys) {
            ANEGraphValue *value = node.sourceOperation.operands[key].value;
            if (!value.producer) {
                [external addObject:value.name];
                continue;
            }
            ANEOperationNode *producer = byValue[value.name];
            if (!producer) {
                [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                     code:@"ane.graph.missing-producer"
                                  message:[NSString stringWithFormat:
                                      @"producer for '%@' is outside function '%@'",
                                      value.name, function.name]
                                    range:node.sourceOperation.range];
                return nil;
            }
            [inputs addObject:producer];
            [producer addUser:node];
        }
        node.inputs = inputs;
        node.externalValueNames = external;
    }
    _nodes = [nodes copy];
    _nodesByValue = [byValue mutableCopy];
    _regions = @[];
    NSMutableArray<NSString *> *outputs = [NSMutableArray array];
    for (ANEGraphValue *value in function.returnValues) [outputs addObject:value.name];
    _outputValueNames = [outputs copy];
    [self rebuildUseLists];
    return self;
}

- (void)rebuildUseLists {
    [_nodesByValue removeAllObjects];
    NSUInteger ordinal = 0;
    for (ANEOperationNode *node in _nodes) {
        node.ordinal = ordinal++;
        [node clearUsers];
        _nodesByValue[node.identifier] = node;
    }
    for (ANEOperationNode *node in _nodes)
        for (ANEOperationNode *input in node.inputs) [input addUser:node];
}

- (ANEOperationNode *)nodeForValueName:(NSString *)valueName {
    return _nodesByValue[valueName];
}

- (BOOL)replaceNode:(ANEOperationNode *)node
          withNodes:(NSArray<ANEOperationNode *> *)newNodes
    replacementNode:(ANEOperationNode *)replacement
         diagnostics:(ANEDiagnosticEngine *)diagnostics {
    NSUInteger index = [_nodes indexOfObjectIdenticalTo:node];
    if (index == NSNotFound || ![newNodes containsObject:replacement]) return NO;
    NSMutableSet<NSString *> *identifiers = [NSMutableSet set];
    for (ANEOperationNode *candidate in newNodes) {
        if ([identifiers containsObject:candidate.identifier] ||
            (_nodesByValue[candidate.identifier] &&
             ![candidate.identifier isEqualToString:node.identifier])) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                 code:@"ane.graph.rewrite-duplicate"
                              message:[NSString stringWithFormat:
                                  @"rewrite produced duplicate value '%@'",
                                  candidate.identifier]
                                range:node.sourceOperation ? node.sourceOperation.range
                                                           : ANESourceRangeMake(
                                                               ANESourceLocationMake(0, 1, 1),
                                                               ANESourceLocationMake(0, 1, 1))];
            return NO;
        }
        [identifiers addObject:candidate.identifier];
    }
    NSMutableArray<ANEOperationNode *> *updated = [_nodes mutableCopy];
    [updated replaceObjectsInRange:NSMakeRange(index, 1) withObjectsFromArray:newNodes];
    for (ANEOperationNode *candidate in updated) {
        if (candidate == node || [newNodes containsObject:candidate]) continue;
        NSMutableArray<ANEOperationNode *> *inputs = [candidate.inputs mutableCopy];
        for (NSUInteger i = 0; i < inputs.count; ++i)
            if (inputs[i] == node) inputs[i] = replacement;
        candidate.inputs = inputs;
    }
    _nodes = [updated copy];
    [self rebuildUseLists];
    return YES;
}

- (void)removeNodes:(NSArray<ANEOperationNode *> *)removed {
    NSSet<ANEOperationNode *> *set = [NSSet setWithArray:removed];
    NSMutableArray<ANEOperationNode *> *updated = [NSMutableArray array];
    for (ANEOperationNode *node in _nodes)
        if (![set containsObject:node]) [updated addObject:node];
    _nodes = [updated copy];
    [self rebuildUseLists];
}

- (void)setPlannedRegions:(NSArray<ANEGraphRegion *> *)regions {
    _regions = [regions copy];
}

- (NSString *)textualDescription {
    NSMutableString *text = [NSMutableString stringWithFormat:@"graph %@\n",
        _sourceFunction.name];
    for (ANEOperationNode *node in _nodes) {
        NSMutableArray<NSString *> *inputs = [NSMutableArray array];
        for (ANEOperationNode *input in node.inputs) [inputs addObject:input.identifier];
        NSMutableArray<NSString *> *allInputs = [inputs mutableCopy];
        [allInputs addObjectsFromArray:node.externalValueNames];
        [text appendFormat:@"  n%lu %%%@ = %@[%@](%@)\n",
            (unsigned long)node.ordinal, node.identifier, kindName(node.kind),
            node.operationName, [allInputs componentsJoinedByString:@", "]];
    }
    return [text copy];
}
@end
