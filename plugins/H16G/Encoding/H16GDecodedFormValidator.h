#import <Foundation/Foundation.h>

#import "ANEOperationGraph.h"
#import "ANEScheduledGraph.h"

NS_ASSUME_NONNULL_BEGIN

/// Number of elements of a static shape; 0 for an empty, zero-sized or
/// overflowing shape.
FOUNDATION_EXPORT NSUInteger H16GElementCount(NSArray<NSNumber *> *shape);

@interface H16GDecodedFormValidator : NSObject
+ (BOOL)validateFP16ConvGraph:(ANEOperationGraph *)graph
                     scheduled:(ANEScheduledGraph *)scheduled
                         error:(NSError **)error;
+ (BOOL)validateFP16DepthwiseGraph:(ANEOperationGraph *)graph
                           scheduled:(ANEScheduledGraph *)scheduled
                               error:(NSError **)error;
+ (BOOL)validateFP16RegularConvGraph:(ANEOperationGraph *)graph
                            scheduled:(ANEScheduledGraph *)scheduled
                                error:(NSError **)error;
+ (BOOL)validateFP16SquareMatmulGraph:(ANEOperationGraph *)graph
                              scheduled:(ANEScheduledGraph *)scheduled
                                  error:(NSError **)error;
+ (BOOL)validateFP16BinaryALUGraph:(ANEOperationGraph *)graph
                           scheduled:(ANEScheduledGraph *)scheduled
                               error:(NSError **)error;
+ (BOOL)validateFP16UnaryPointwiseGraph:(ANEOperationGraph *)graph
                                scheduled:(ANEScheduledGraph *)scheduled
                                    error:(NSError **)error;
+ (BOOL)validateFP16ReductionGraph:(ANEOperationGraph *)graph
                           scheduled:(ANEScheduledGraph *)scheduled
                               error:(NSError **)error;
+ (BOOL)validateW8A8ConvGraph:(ANEOperationGraph *)graph
                      scheduled:(ANEScheduledGraph *)scheduled
                          error:(NSError **)error;
+ (BOOL)validateMixedGraph:(ANEOperationGraph *)graph
                   scheduled:(ANEScheduledGraph *)scheduled
                       error:(NSError **)error;
+ (BOOL)validateLayoutConvChainGraph:(ANEOperationGraph *)graph
                             scheduled:(ANEScheduledGraph *)scheduled
                                 error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
