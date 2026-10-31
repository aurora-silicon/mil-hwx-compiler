#import <Foundation/Foundation.h>
#import "ANEDiagnostic.h"

NS_ASSUME_NONNULL_BEGIN

@interface ANEH13Compiler : NSObject
+ (BOOL)compileMILData:(NSData *)milData
             modelRoot:(NSURL *)modelRoot
                format:(NSString *)format
       outputDirectory:(NSURL *)directory
              schedule:(NSString *)schedule
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error;
// Bounds materialized programs before payload encoding, including the
// temporary per-op programs used by a chain. Does not expand any ISA envelope.
+ (BOOL)compileMILData:(NSData *)milData
             modelRoot:(NSURL *)modelRoot
                format:(NSString *)format
       outputDirectory:(NSURL *)directory
              schedule:(NSString *)schedule
           maxPrograms:(NSUInteger)maxPrograms
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
