#import "ANEStagedCompiler.h"

#import "ANEDecomposePass.h"
#import "ANEExecutableBundle.h"
#import "ANEFusionPass.h"
#import "ANEGraphVerifier.h"
#import "ANEH16GLegalizePass.h"
#import "ANEHWXArtifact.h"
#import "ANENormalizePass.h"
#import "ANEOperationGraph.h"
#import "ANETaskScheduler.h"
#import "H16GProgramEncoder.h"
#import "H16GProgramAssembler.h"
#import "H16GTarget.h"
#import "MILLexer.h"
#import "MILGraphImporter.h"
#import "MILParser.h"

static ANESourceRange syntheticRange(void) {
    ANESourceLocation location = ANESourceLocationMake(0, 1, 1);
    return ANESourceRangeMake(location, location);
}

@implementation ANEStagedCompiler
+ (ANEExecutableBundle *)compileMILData:(NSData *)milData
                                modelRoot:(NSURL *)modelRoot
                                   target:(NSString *)targetName
                              diagnostics:(ANEDiagnosticEngine *)diagnostics {
    if (![targetName isEqualToString:@"H16G"]) {
        [diagnostics emitSeverity:ANEDiagnosticSeverityError
            code:@"ane.driver.unsupported-target"
            message:[NSString stringWithFormat:@"unsupported target '%@'", targetName]
            range:syntheticRange()];
        return nil;
    }
    MILLexer *lexer = [[MILLexer alloc] initWithData:milData
                                         diagnostics:diagnostics];
    MILParser *parser = [[MILParser alloc] initWithTokens:lexer.lexAllTokens
                                              diagnostics:diagnostics];
    MILProgramSyntax *syntax = parser.parseProgram;
    ANEGraphModule *module = syntax
        ? [MILGraphImporter importProgram:syntax diagnostics:diagnostics] : nil;
    if (!module || module.functions.count != 1 ||
        ![ANEGraphVerifier verifyModule:module diagnostics:diagnostics]) return nil;

    ANEOperationGraph *graph = [[ANEOperationGraph alloc]
        initWithFunction:module.functions[0] diagnostics:diagnostics];
    H16GTarget *target = [H16GTarget currentTarget];
    if (!graph ||
        ![ANENormalizePass runOnGraph:graph diagnostics:diagnostics] ||
        ![ANEDecomposePass runOnGraph:graph diagnostics:diagnostics] ||
        ![ANEFusionPass runOnGraph:graph target:target diagnostics:diagnostics] ||
        ![ANEH16GLegalizePass runOnGraph:graph target:target
                              diagnostics:diagnostics] ||
        ![ANEFusionPass runOnGraph:graph target:target diagnostics:diagnostics])
        return nil;
    // The native last-axis layer_norm formula owns its complete multi-task
    // schedule and bounded DRAM scratch.  The generic decomposed scheduler
    // counts all intermediate tensors simultaneously and rejects Whisper's
    // long sequence before the program encoder can apply that measured plan.
    BOOL standaloneLayerNorm=graph.sourceFunction.inputs.count==1 &&
        graph.outputValueNames.count==1;
    NSUInteger layerNormNodes=0,activeNodes=0;
    for(ANEOperationNode *node in graph.nodes) {
        if(node.kind==ANEOperationKindConstant) continue;
        ++activeNodes;
        layerNormNodes += [node.sourceOperation.operationName
            isEqualToString:@"layer_norm"];
    }
    standaloneLayerNorm = standaloneLayerNorm && activeNodes==7 &&
        layerNormNodes==7;
    if(standaloneLayerNorm) {
        ANEHWXArtifact *artifact=[H16GProgramEncoder encodeGraph:graph
            scheduled:nil modelRoot:modelRoot diagnostics:diagnostics];
        if(!artifact) return nil;
        return [[ANEExecutableBundle alloc] initWithTarget:targetName
            artifacts:@[artifact] dispatchPlan:@[@0] passTrace:@[
                @"mil.import-operation-graph", @"ane.normalize", @"ane.decompose",
                @"ane.fuse", @"h16g.legalize", @"h16g.native-layer-norm",
                @"h16g.write-object",
            ] compositionTrace:@[
                @"native layer_norm formula owns its measured multi-task schedule",
            ]];
    }
    ANEScheduledGraph *scheduled = [ANETaskScheduler scheduleGraph:graph
        target:target diagnostics:diagnostics];
    if (!scheduled) return nil;
    H16GAssembledProgram *assembled = [H16GProgramAssembler
        assembleGraph:graph scheduled:scheduled target:target
        diagnostics:diagnostics];
    NSArray<ANEHWXArtifact *> *artifacts = assembled.artifacts;
    NSArray<NSNumber *> *dispatchPlan = assembled.dispatchPlan;
    NSArray<NSString *> *compositionTrace = assembled.compositionTrace ?: @[];
    if (!assembled) {
        ANEHWXArtifact *artifact = [H16GProgramEncoder encodeGraph:graph
            scheduled:scheduled modelRoot:modelRoot diagnostics:diagnostics];
        if (!artifact) return nil;
        artifacts = @[artifact];
        dispatchPlan = @[@0];
        compositionTrace = @[@"single-program encoder path; no partition planning"];
    }
    return [[ANEExecutableBundle alloc] initWithTarget:targetName
        artifacts:artifacts dispatchPlan:dispatchPlan passTrace:@[
            @"mil.import-operation-graph", @"ane.normalize", @"ane.decompose",
            @"ane.fuse", @"h16g.legalize", @"h16g.plan", @"h16g.encode",
            @"h16g.write-object",
        ] compositionTrace:compositionTrace];
}
@end
