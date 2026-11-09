#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GLUTEncoder : NSObject
+ (BOOL)supportsOperationName:(NSString *)operationName
                   inputShape:(NSArray<NSNumber *> *)inputShape;
/// Smallest rule-built square N with N*N >= elementCount, or NSNotFound.
/// Squares are rounded to the packet's 64-lane granularity.
+ (NSUInteger)squareViewSizeForOperationName:(NSString *)operationName
                                elementCount:(NSUInteger)elementCount;
+ (nullable H16GEncodedTDProgram *)encodeOperationName:(NSString *)operationName
                                             inputShape:(NSArray<NSNumber *> *)inputShape
                                                  error:(NSError **)error;
+ (nullable NSData *)constantRegionForOperationName:(NSString *)operationName
                                               error:(NSError **)error;
/// Byte offset of the fp16 input-scale word in a table unary task stream, or
/// NSNotFound when the operation is not a table unary or the geometry is not
/// encodable. The plain program carries 1.0 there (log2 e for `exp`).
+ (NSUInteger)inputScaleByteOffsetForOperationName:(NSString *)operationName
                                        inputShape:(NSArray<NSNumber *> *)inputShape;
/// The exact value the plain program's input-scale word represents:
/// log2 e for `exp` (its table is a base-two exponential) and 1.0 for every
/// other table unary. NAN for operations without a table.
+ (double)inputScaleValueForOperationName:(NSString *)operationName;
@end

NS_ASSUME_NONNULL_END
