#pragma once

#import <Foundation/Foundation.h>
#import "ANEDiagnostic.h"

NS_ASSUME_NONNULL_BEGIN

/// The decoded-oracle parity compiler for the H14 task format. `target` is
/// H14, or H17 or H18, which keep that format and cover only the decoded
/// elementwise, unary, and scalar-constant families as HWX.
@interface ANEH14Compiler : NSObject
+ (BOOL)compileMILData:(NSData *)milData
             modelRoot:(NSURL *)modelRoot
                target:(NSString *)target
                format:(NSString *)format
       outputDirectory:(NSURL *)directory
              schedule:(NSString *)schedule
researchCompactTaskMinimumBytes:(nullable NSString *)researchCompactTaskMinimumBytes
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
