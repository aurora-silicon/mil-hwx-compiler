#import <Foundation/Foundation.h>

#import "ANEGraphIR.h"

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, HWXObjectBindingRole) {
    HWXObjectBindingRoleInput,
    HWXObjectBindingRoleOutput,
};

@interface HWXObjectBinding : NSObject
@property(nonatomic, readonly, copy) NSString *symbol;
@property(nonatomic, readonly, copy) NSString *shortName;
@property(nonatomic, readonly) HWXObjectBindingRole role;
@property(nonatomic, readonly) ANEElementType elementType;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *shape;
@property(nonatomic, readonly) NSUInteger rowStrideBytes;
@property(nonatomic, readonly) NSUInteger planeStrideBytes;
@property(nonatomic, readonly) NSUInteger batchStrideBytes;
@property(nonatomic, readonly) NSUInteger storageByteLength;
/// Bytes the surface occupies in the object's VM layout. Apple's batched
/// surfaces record `storageByteLength` as one batch element's stride in the
/// tensor descriptor while spacing surfaces by the whole allocation, so this
/// is settable and defaults to `storageByteLength`.
@property(nonatomic) NSUInteger allocationByteLength;
- (instancetype)initWithSymbol:(NSString *)symbol
                      shortName:(NSString *)shortName
                           role:(HWXObjectBindingRole)role
                    elementType:(ANEElementType)elementType
                          shape:(NSArray<NSNumber *> *)shape;
- (instancetype)initWithSymbol:(NSString *)symbol
                      shortName:(NSString *)shortName
                           role:(HWXObjectBindingRole)role
                    elementType:(ANEElementType)elementType
                          shape:(NSArray<NSNumber *> *)shape
                 rowStrideBytes:(NSUInteger)rowStrideBytes
               planeStrideBytes:(NSUInteger)planeStrideBytes
               batchStrideBytes:(NSUInteger)batchStrideBytes
              storageByteLength:(NSUInteger)storageByteLength;
@end

typedef NS_ENUM(NSUInteger, HWXProgramDescriptorLayout) {
    HWXProgramDescriptorLayoutLinear,
    HWXProgramDescriptorLayoutScratchBackedMixed,
};

@interface HWXObjectProgramInfo : NSObject
@property(nonatomic, readonly) NSUInteger taskCount;
@property(nonatomic, readonly) NSUInteger firstTaskByteLength;
@property(nonatomic, readonly) NSUInteger recordCount;
@property(nonatomic, readonly) uint32_t formatCode;
@property(nonatomic, readonly) NSUInteger scratchByteLength;
/// Writable so an H13 or H14 linear program can declare Apple's
/// __DATA/__bss scratch allocation without a new initializer.
@property(nonatomic) NSUInteger scratchAllocationByteLength;
@property(nonatomic, readonly) HWXProgramDescriptorLayout descriptorLayout;
/// H14 program descriptors carry one word at command offset 0x880 that the
/// oracle campaign resolves no formula for; parity copies the decoded value.
@property(nonatomic) uint32_t unresolvedDescriptorWord;
/// H14 program descriptors carry a second unmodelled word at command offset
/// 0x858, which Apple sets to the input surface byte count for the three-task
/// layer_norm form; parity copies the decoded value without allocating for it.
@property(nonatomic) uint32_t h14ScratchDescriptorWord;
/// H17 and H18 text-constant objects only: the kernel table (LUT and scalar
/// operands) Apple writes to its own __KERN_0 segment right after __TEXT,
/// with that address as the resource slot after the surfaces. Nil for none.
@property(nonatomic, copy, nullable) NSData *kernelTable;
/// Formula-built H16G/H17 convolution: constantRegion is the __KERN_0
/// weight section; __TEXT/__const is a zero allocation, never the weights.
@property(nonatomic) BOOL kernelRegionConstants;
- (instancetype)initWithTaskCount:(NSUInteger)taskCount
                       recordCount:(NSUInteger)recordCount
                        formatCode:(uint32_t)formatCode
                 scratchByteLength:(NSUInteger)scratchByteLength
                  descriptorLayout:(HWXProgramDescriptorLayout)descriptorLayout;
- (instancetype)initWithTaskCount:(NSUInteger)taskCount
               firstTaskByteLength:(NSUInteger)firstTaskByteLength
                       recordCount:(NSUInteger)recordCount
                        formatCode:(uint32_t)formatCode
                 scratchByteLength:(NSUInteger)scratchByteLength
                  descriptorLayout:(HWXProgramDescriptorLayout)descriptorLayout;
- (instancetype)initWithTaskCount:(NSUInteger)taskCount
                       recordCount:(NSUInteger)recordCount
                        formatCode:(uint32_t)formatCode
                 scratchByteLength:(NSUInteger)scratchByteLength
       scratchAllocationByteLength:(NSUInteger)scratchAllocationByteLength
                  descriptorLayout:(HWXProgramDescriptorLayout)descriptorLayout;
@end

/// The Mach-O CPU subtype of each generation's HWX. H17 and H18 keep
/// constants in __TEXT/__const like H13 and H14 and use the 0x8a0-byte
/// program descriptor of the H16G layout.
typedef NS_ENUM(NSUInteger, HWXObjectArchitecture) {
    HWXObjectArchitectureH13 = 4,
    HWXObjectArchitectureH14 = 5,
    HWXObjectArchitectureH16G = 7,
    HWXObjectArchitectureH17 = 9,
    HWXObjectArchitectureH18 = 10,
};

@interface HWXObjectWriter : NSObject
+ (nullable NSData *)buildObjectWithTaskDescriptor:(NSData *)taskDescriptor
                                     constantRegion:(NSData *)constantRegion
                                            bindings:(NSArray<HWXObjectBinding *> *)bindings
                                               error:(NSError **)error;
+ (nullable NSData *)buildObjectWithTaskDescriptor:(NSData *)taskDescriptor
                                     constantRegion:(NSData *)constantRegion
                                            bindings:(NSArray<HWXObjectBinding *> *)bindings
                             kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
                                  programRecordCount:(NSUInteger)programRecordCount
                                   programFormatCode:(uint32_t)programFormatCode
                                               error:(NSError **)error;
+ (nullable NSData *)buildObjectWithTaskDescriptor:(NSData *)taskDescriptor
                                     constantRegion:(NSData *)constantRegion
                                            bindings:(NSArray<HWXObjectBinding *> *)bindings
                             kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
                                         programInfo:(HWXObjectProgramInfo *)programInfo
                                               error:(NSError **)error;
+ (nullable NSData *)buildObjectForArchitecture:(HWXObjectArchitecture)architecture
                                     taskDescriptor:(NSData *)taskDescriptor
                                      constantRegion:(NSData *)constantRegion
                                             bindings:(NSArray<HWXObjectBinding *> *)bindings
                              kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
                                         programInfo:(HWXObjectProgramInfo *)programInfo
                                               error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
