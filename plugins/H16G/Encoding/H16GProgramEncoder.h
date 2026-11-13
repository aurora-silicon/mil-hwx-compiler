#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEOperationGraph.h"
#import "ANEScheduledGraph.h"

@class ANEHWXArtifact;
@class H16GAssembledProgram;

NS_ASSUME_NONNULL_BEGIN

@interface H16GProgramEncoder : NSObject
+ (nullable ANEHWXArtifact *)encodeExactResNetGraph:(ANEOperationGraph *)graph
                                          modelRoot:(NSURL *)modelRoot
                                        diagnostics:(ANEDiagnosticEngine *)diagnostics;
+ (nullable H16GAssembledProgram *)encodeProvenAttentionChainGraph:
                                      (ANEOperationGraph *)graph
                                      diagnostics:(ANEDiagnosticEngine *)diagnostics;
+ (nullable ANEHWXArtifact *)encodeGraph:(ANEOperationGraph *)graph
                                scheduled:(nullable ANEScheduledGraph *)scheduled
                                modelRoot:(NSURL *)modelRoot
                              diagnostics:(ANEDiagnosticEngine *)diagnostics;
@end

NS_ASSUME_NONNULL_END
