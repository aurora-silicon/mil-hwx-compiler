#import "H16GLUTEncoder.h"

static NSString *const H16GLUTEncoderErrorDomain = @"ANE.H16G.LUTEncoder";

#include "H16GLUTEncoderData.inc"

#include <cmath>

typedef NS_ENUM(NSUInteger, H16GUnaryPacketFamily) {
    H16GUnaryPacketFamilyTable,
    H16GUnaryPacketFamilyDirectReLU,
    H16GUnaryPacketFamilyReciprocalUnit,
};

typedef struct {
    NSUInteger size;
    const uint32_t *tableWords;
    NSUInteger tableWordCount;
    const uint32_t *reluWords;
    NSUInteger reluWordCount;
    const uint32_t *reciprocalWords;
    NSUInteger reciprocalWordCount;
    uint32_t tableFormatCode;
    uint32_t reciprocalFormatCode;
    NSUInteger tableRelocationOffset;
    NSUInteger reciprocalRelocationOffset;
    NSUInteger exponentialScaleOffset;
} H16GUnaryGeometry;

#define WORD_COUNT(array) (sizeof(array) / sizeof((array)[0]))
static const H16GUnaryGeometry kUnaryGeometries[] = {
    {128,kSigmoid128TDWords,WORD_COUNT(kSigmoid128TDWords),
         kRelu128TDWords,WORD_COUNT(kRelu128TDWords),
         kRsqrt128TDWords,WORD_COUNT(kRsqrt128TDWords),
         0x00,0x01,0xf0,0xf4,0xc4},
    {256,kSigmoid256TDWords,WORD_COUNT(kSigmoid256TDWords),
         kRelu256TDWords,WORD_COUNT(kRelu256TDWords),
         kRsqrt256TDWords,WORD_COUNT(kRsqrt256TDWords),
         0x03,0x04,0xd8,0xe4,0xac},
    {512,kSigmoid512TDWords,WORD_COUNT(kSigmoid512TDWords),
         kRelu512TDWords,WORD_COUNT(kRelu512TDWords),
         kRsqrt512TDWords,WORD_COUNT(kRsqrt512TDWords),
         0x0f,0x13,0xec,0xf4,0xc0},
    {1024,kSigmoid1024TDWords,WORD_COUNT(kSigmoid1024TDWords),
          kRelu1024TDWords,WORD_COUNT(kRelu1024TDWords),
          kRsqrt1024TDWords,WORD_COUNT(kRsqrt1024TDWords),
          0x3d,0x4c,0xec,0xf4,0xc0},
    {2048,kSigmoid2048TDWords,WORD_COUNT(kSigmoid2048TDWords),
          kRelu2048TDWords,WORD_COUNT(kRelu2048TDWords),
          kRsqrt2048TDWords,WORD_COUNT(kRsqrt2048TDWords),
          0xf5,0x130,0xec,0xf4,0xc0},
};
#undef WORD_COUNT

static NSString *canonicalOperation(NSString *operation) {
    if ([operation isEqualToString:@"inverse"] ||
        [operation isEqualToString:@"recip"])
        return @"reciprocal";
    return operation;
}

static BOOL isTableOperation(NSString *operation) {
    return [@[@"sigmoid",@"tanh",@"gelu",@"silu",@"exp",@"sqrt"]
        containsObject:operation];
}

static H16GUnaryPacketFamily familyForOperation(NSString *operation,
                                                 BOOL *supported) {
    if (isTableOperation(operation)) {
        *supported = YES;
        return H16GUnaryPacketFamilyTable;
    }
    if ([operation isEqualToString:@"relu"]) {
        *supported = YES;
        return H16GUnaryPacketFamilyDirectReLU;
    }
    if ([operation isEqualToString:@"rsqrt"] ||
        [operation isEqualToString:@"reciprocal"]) {
        *supported = YES;
        return H16GUnaryPacketFamilyReciprocalUnit;
    }
    *supported = NO;
    return H16GUnaryPacketFamilyTable;
}

static const H16GUnaryGeometry *geometryForShape(NSArray<NSNumber *> *shape) {
    if (shape.count != 4 || shape[0].unsignedIntegerValue != 1 ||
        shape[1].unsignedIntegerValue != 1 ||
        shape[2].unsignedIntegerValue != shape[3].unsignedIntegerValue)
        return NULL;
    NSUInteger size = shape[2].unsignedIntegerValue;
    for (const H16GUnaryGeometry &row : kUnaryGeometries)
        if (row.size == size) return &row;
    return NULL;
}

static NSData *dataFromWords(const uint32_t *words, NSUInteger count) {
    return [NSData dataWithBytes:words length:count * sizeof(uint32_t)];
}

static NSData *kernelTable(NSString *operation) {
    struct TableRow { NSString *__unsafe_unretained operation;
                      const uint32_t *words; NSUInteger count; };
#define TABLE_ROW(name, array) {name,array,sizeof(array)/sizeof(array[0])}
    static const TableRow rows[] = {
        TABLE_ROW(@"sigmoid",kSigmoidKERNWords),
        TABLE_ROW(@"tanh",kTanhKERNWords),
        TABLE_ROW(@"gelu",kGeluKERNWords),
        TABLE_ROW(@"silu",kSiluKERNWords),
        TABLE_ROW(@"exp",kExpKERNWords),
        TABLE_ROW(@"log",kLogKERNWords),
        TABLE_ROW(@"sqrt",kSqrtKERNWords),
        TABLE_ROW(@"rsqrt",kRsqrtKERNWords),
        TABLE_ROW(@"reciprocal",kRecipKERNWords),
    };
#undef TABLE_ROW
    for (const TableRow &row : rows)
        if ([operation isEqualToString:row.operation])
            return dataFromWords(row.words,row.count);
    return nil;
}

static void setError(NSError **error, NSString *message) {
    if (error) *error = [NSError errorWithDomain:H16GLUTEncoderErrorDomain
        code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
}

@implementation H16GLUTEncoder
+ (BOOL)supportsOperationName:(NSString *)operationName
                   inputShape:(NSArray<NSNumber *> *)inputShape {
    NSString *operation = canonicalOperation(operationName);
    if ([operation isEqualToString:@"log"])
        return [inputShape isEqualToArray:@[@1,@1,@256,@256]];
    BOOL supported = NO;
    familyForOperation(operation,&supported);
    return supported && geometryForShape(inputShape) != NULL;
}

+ (NSUInteger)squareViewSizeForOperationName:(NSString *)operationName
                                elementCount:(NSUInteger)elementCount {
    if (elementCount == 0) return NSNotFound;
    for (const H16GUnaryGeometry &row : kUnaryGeometries) {
        NSArray<NSNumber *> *view = @[@1,@1,@(row.size),@(row.size)];
        if (row.size*row.size >= elementCount &&
            [self supportsOperationName:operationName inputShape:view])
            return row.size;
    }
    return NSNotFound;
}

+ (H16GEncodedTDProgram *)encodeOperationName:(NSString *)operationName
                                    inputShape:(NSArray<NSNumber *> *)inputShape
                                         error:(NSError **)error {
    NSString *operation = canonicalOperation(operationName);
    if ([operation isEqualToString:@"log"]) {
        if (![inputShape isEqualToArray:@[@1,@1,@256,@256]]) {
            setError(error,@"log currently requires the decoded [1,1,256,256] two-task family");
            return nil;
        }
        return [[H16GEncodedTDProgram alloc]
            initWithData:dataFromWords(kLog256TDWords,
                sizeof(kLog256TDWords)/sizeof(kLog256TDWords[0]))
            kernelRelocationOffsets:@[@0xc8] programRecordCount:31
            programFormatCode:4 scratchByteLength:0];
    }
    BOOL supported = NO;
    H16GUnaryPacketFamily family = familyForOperation(operation,&supported);
    const H16GUnaryGeometry *row = geometryForShape(inputShape);
    if (!supported || !row) {
        setError(error,@"unary pointwise operation requires a decoded operation and square [1,1,N,N], N in {128,256,512,1024,2048}");
        return nil;
    }
    const uint32_t *words = NULL;
    NSUInteger count = 0, relocation = NSNotFound;
    uint32_t format = 0;
    switch (family) {
        case H16GUnaryPacketFamilyTable:
            words=row->tableWords; count=row->tableWordCount;
            relocation=row->tableRelocationOffset; format=row->tableFormatCode;
            break;
        case H16GUnaryPacketFamilyDirectReLU:
            words=row->reluWords; count=row->reluWordCount;
            format=row->tableFormatCode;
            break;
        case H16GUnaryPacketFamilyReciprocalUnit:
            words=row->reciprocalWords; count=row->reciprocalWordCount;
            relocation=row->reciprocalRelocationOffset;
            format=row->reciprocalFormatCode;
            break;
    }
    NSMutableData *data=[dataFromWords(words,count) mutableCopy];
    if ([operation isEqualToString:@"exp"]) {
        uint32_t log2eFP16 = 0x00003dc5;
        [data replaceBytesInRange:NSMakeRange(row->exponentialScaleOffset,4)
                        withBytes:&log2eFP16];
    }
    NSArray<NSNumber *> *relocations = relocation == NSNotFound
        ? @[] : @[@(relocation)];
    return [[H16GEncodedTDProgram alloc] initWithData:data
        kernelRelocationOffsets:relocations programRecordCount:16
        programFormatCode:format scratchByteLength:0];
}

+ (NSUInteger)inputScaleByteOffsetForOperationName:(NSString *)operationName
                                        inputShape:(NSArray<NSNumber *> *)inputShape {
    NSString *operation = canonicalOperation(operationName);
    const H16GUnaryGeometry *row = geometryForShape(inputShape);
    if (!row || !isTableOperation(operation)) return NSNotFound;
    return row->exponentialScaleOffset;
}

+ (double)inputScaleValueForOperationName:(NSString *)operationName {
    NSString *operation = canonicalOperation(operationName);
    if (!isTableOperation(operation)) return NAN;
    return [operation isEqualToString:@"exp"] ? M_LOG2E : 1.0;
}

+ (NSData *)constantRegionForOperationName:(NSString *)operationName
                                      error:(NSError **)error {
    NSString *operation = canonicalOperation(operationName);
    if ([operation isEqualToString:@"relu"]) return [NSData data];
    NSData *table = kernelTable(operation);
    if (!table) {
        setError(error,@"operation has no decoded H16G unary constant table");
        return nil;
    }
    return table;
}
@end
