#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GALUEncoder : NSObject
+ (BOOL)supportsOperationName:(NSString *)operationName squareSize:(NSUInteger)size;
/// Smallest measured square N with N*N >= elementCount, or NSNotFound.
/// A standalone elementwise operation on contiguous external tensors runs
/// as the square [1,1,N,N] program over the same bytes; lanes past
/// elementCount are padding and their results are discarded.
+ (NSUInteger)squareViewSizeForElementCount:(NSUInteger)elementCount;
+ (nullable H16GEncodedTDProgram *)encodeOperationName:(NSString *)operationName
                                            squareSize:(NSUInteger)size
                                                 error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
