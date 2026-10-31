#import <Foundation/Foundation.h>
#import "ANEDiagnostic.h"

/// Explicit multi-dispatch schedule. Existing per-op/chain packages retain
/// their original contracts; spatial packages require a spatial-aware runner.
@interface ANESpatialCompiler : NSObject
+ (BOOL)compileMILData:(NSData *)milData
             modelRoot:(NSURL *)modelRoot
                target:(NSString *)target
                format:(NSString *)format
       outputDirectory:(NSURL *)directory
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error;
@end
