// The formula-built convolution path, kept separate from decoded parity
// lowering so unrelated H14/H17/H18 encodings remain byte-stable.
static BOOL emitConvGraph(ANEGraphFunction *function,
                          NSArray<ANEGraphOperation *> *operations,
                          NSURL *modelRoot, NSURL *directory,
                          NSString *format, BOOL linked,
                          ANEDiagnosticEngine *diagnostics, NSError **error) {
    if (![format isEqualToString:@"anec"])
        return reject(diagnostics, @"H14 formula-built convolution currently emits ANEC; HWX descriptor cost fields remain unresolved",
                      nil, @"h14.conv-outside-envelope");
    if (function.inputs.count != 1 || function.returnValues.count != 1)
        return reject(diagnostics, @"H14 convolution needs one input and one result",
                      nil, @"h14.linked-outside-envelope");
    std::vector<ane::h14::Program> layers;
    NSMutableArray<ANEGraphValue *> *results = [NSMutableArray array];
    NSMutableArray<ANEGraphValue *> *inputs = [NSMutableArray array];
    ANEGraphValue *current = function.inputs[0];
    try {
        for (ANEGraphOperation *operation in operations) {
            if (operation.operands[@"x"].value != current)
                return reject(diagnostics, @"H14 convolution scheduling requires a straight chain",
                              operation, @"h14.linked-outside-envelope");
            if ([operation.operationName isEqualToString:@"relu"]) {
                if (layers.empty() || operation.arguments.count != 1 ||
                    ![current.type.shape isEqualToArray:operation.results[0].type.shape])
                    return reject(diagnostics, @"H14 relu must immediately follow its convolution",
                                  operation, @"h14.conv-outside-envelope");
                ane::h14::fuseConvRelu(layers.back());
                current = operation.results[0];
                results[results.count - 1] = current;
                continue;
            }
            H14ConvPlan plan{};
            if (!convParityPlan(operation, &plan, YES))
                return reject(diagnostics, @"H14 formula-built convolution is outside its dense/depthwise 1x1/3x3 envelope",
                              operation, @"h14.conv-outside-envelope");
            const auto reduction = plan.shape.input.channels / plan.shape.groups *
                plan.shape.kernel * plan.shape.kernelWidth;
            NSData *weights = [ANEBlobResolver loadConstantForOperation:plan.weight.producer
                expectedBytes:static_cast<NSUInteger>(plan.shape.output.channels) * reduction * 2
                modelRoot:modelRoot diagnostics:diagnostics];
            if (!weights) return NO;
            NSData *bias = nil;
            if (plan.bias) {
                bias = [ANEBlobResolver loadConstantForOperation:plan.bias.producer
                    expectedBytes:plan.shape.output.channels * 2
                    modelRoot:modelRoot diagnostics:diagnostics];
                if (!bias) return NO;
            }
            layers.push_back(ane::h14::encodeConvGeneric(plan.shape,
                static_cast<const std::uint8_t *>(weights.bytes), weights.length,
                static_cast<const std::uint8_t *>(bias.bytes), bias.length));
            [inputs addObject:current];
            current = operation.results[0];
            [results addObject:current];
        }
        if (layers.empty() || function.returnValues[0] != current)
            return reject(diagnostics, @"H14 convolution must return the last result",
                          nil, @"h14.linked-outside-envelope");
        std::vector<ane::h14::LinkedLayer> layout;
        std::vector<ane::h14::Program> programs;
        if (linked) programs.push_back(ane::h14::linkConvChain(layers, &layout));
        else programs = layers;
        NSMutableArray *records = [NSMutableArray array];
        NSMutableArray *payloads = [NSMutableArray array];
        NSMutableArray *dispatch = [NSMutableArray array];
        NSMutableArray *tasks = [NSMutableArray array];
        NSMutableArray *intermediates = [NSMutableArray array];
        NSMutableDictionary *regions = [NSMutableDictionary dictionary];
        for (NSUInteger i = 0; i < programs.size(); ++i) {
            const auto &p = programs[i];
            NSData *payload = nil;
            const auto anec = ane::h14::encodeANEC(p);
            payload = [NSData dataWithBytes:anec.data() length:anec.size()];
            if (!payload) return NO;
            NSDictionary *record = @{
                @"file": [NSString stringWithFormat:@"program-%lu.%@", (unsigned long)i, format],
                @"bytes": @(payload.length), @"taskDescriptors": @(p.taskCount),
                @"encoder": linked ? @"h14-linked-conv-chain" : @"h14-generic-conv",
                @"operation": linked ? @"conv_chain" : @"conv",
                @"inputs": @[binding(linked ? function.inputs[0] : inputs[i], p.inputs[0])],
                @"outputs": @[binding(linked ? current : results[i], p.output)],
                @"constantInputs": @{}, @"constantOffset": @(p.constantOffsetBytes),
                @"constantBytes": @(p.constants.size()), @"firstTaskBytes": @(p.firstTaskBytes)};
            [records addObject:record]; [payloads addObject:payload]; [dispatch addObject:@(i)];
        }
        if (linked) for (NSUInteger i = 0; i < layout.size(); ++i) {
            const auto &l = layout[i];
            for (unsigned j = 0; j < l.taskCount; ++j)
                [tasks addObject:@{@"index": @(l.firstTask + j), @"layer": @(i),
                    @"operation": @"conv", @"result": results[i].name}];
            if (i + 1 < layout.size()) regions[results[i].name] = @{
                @"offset": @(l.outputOffset), @"bytes": @(layers[i].output.allocationBytes)};
        }
        if (!linked) for (NSUInteger i = 0; i + 1 < results.count; ++i)
            [intermediates addObject:results[i].name];
        NSMutableDictionary *manifest = [@{
            @"schema": @"mil-hwxc.h14-anec-package.v1", @"target": @"H14",
            @"artifactFormat": format, @"programs": records, @"dispatchPlan": dispatch,
            @"intermediates": intermediates, @"schedule": linked ? @"linked" : @"per-op",
            @"scratch": @{@"bytes": @(linked ? programs[0].scratchAllocationBytes : 0),
                          @"regions": regions}, @"tasks": tasks} mutableCopy];
        if (records.count == 1) [manifest addEntriesFromDictionary:records[0]];
        NSData *metadata = [NSJSONSerialization dataWithJSONObject:manifest
            options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:error];
        if (!metadata) return NO;
        NSFileManager *fm = NSFileManager.defaultManager;
        if ([fm fileExistsAtPath:directory.path] &&
            [fm contentsOfDirectoryAtPath:directory.path error:error].count) {
            if (error) *error = [NSError errorWithDomain:@"dev.aurora.H14" code:1
                userInfo:@{NSLocalizedDescriptionKey: @"H14 output directory must be empty"}];
            return NO;
        }
        if (![fm createDirectoryAtURL:directory withIntermediateDirectories:YES
                           attributes:nil error:error]) return NO;
        for (NSUInteger i = 0; i < payloads.count; ++i)
            if (![payloads[i] writeToURL:[directory URLByAppendingPathComponent:records[i][@"file"]]
                                options:NSDataWritingAtomic error:error]) return NO;
        return [metadata writeToURL:[directory URLByAppendingPathComponent:@"manifest.json"]
                           options:NSDataWritingAtomic error:error];
    } catch (const std::exception &exception) {
        return reject(diagnostics, [NSString stringWithUTF8String:exception.what()],
                      nil, @"h14.conv-outside-envelope");
    }
}
