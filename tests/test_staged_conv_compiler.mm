#import <Foundation/Foundation.h>
#include "sha256_compat.h"
#import "HWXMachOFormat.h"

#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"
#import "ANEHWXArtifact.h"
#import "ANEStagedCompiler.h"
#import "H16GReduceEncoder.h"
#import "HWXImage.h"

#include <stdio.h>

static int failures = 0;

static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    ++failures;
}

static void expectRejectedMutation(NSString *fixturePath,
                                   NSString *modelPath,
                                   NSString *needle,
                                   NSString *replacement,
                                   NSString *message) {
    NSError *readError = nil;
    NSMutableString *source = [NSMutableString stringWithContentsOfFile:fixturePath
        encoding:NSUTF8StringEncoding error:&readError];
    NSRange match = [source rangeOfString:needle];
    expect(source != nil && readError == nil && match.location != NSNotFound,
           [message stringByAppendingString:@" fixture mutation is applicable"]);
    if (!source || match.location == NSNotFound) return;
    [source replaceCharactersInRange:match withString:replacement];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = [ANEStagedCompiler
        compileMILData:[source dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:[NSURL fileURLWithPath:modelPath isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics];
    expect(bundle == nil && diagnostics.errorCount > 0, message);
}

static NSString *firstFVMLIBSectionName(NSData *imageData) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *segment =
                (const struct segment_command_64 *)command;
            if (strncmp(segment->segname, "__FVMLIB", 16) == 0 &&
                segment->nsects == 1) {
                const struct section_64 *section =
                    (const struct section_64 *)(segment + 1);
                return [[NSString alloc] initWithBytes:section->sectname
                    length:strnlen(section->sectname, 16)
                    encoding:NSUTF8StringEncoding];
            }
        }
        offset += command->cmdsize;
    }
    return nil;
}

static NSString *sha256(NSData *data) {
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *text = [NSMutableString stringWithCapacity:64];
    for (NSUInteger i = 0; i < sizeof(digest); ++i)
        [text appendFormat:@"%02x", digest[i]];
    return text;
}

static uint32_t tdWord(NSData *data, NSUInteger offset) {
    uint32_t value = 0;
    [data getBytes:&value range:NSMakeRange(offset, sizeof(value))];
    return value;
}

static uint32_t programDescriptorWord(NSData *imageData,
                                      NSUInteger relativeOffset) {
    const uint8_t *bytes = (const uint8_t *)imageData.bytes;
    const struct mach_header_64 *header =
        (const struct mach_header_64 *)bytes;
    NSUInteger offset = sizeof(*header);
    for (uint32_t index = 0; index < header->ncmds; ++index) {
        const struct load_command *command =
            (const struct load_command *)(bytes + offset);
        if (command->cmd == 0x04 && command->cmdsize >= relativeOffset+4 &&
            *(const uint32_t *)(bytes+offset+8) == 4)
            return *(const uint32_t *)(bytes+offset+relativeOffset);
        offset += command->cmdsize;
    }
    return UINT32_MAX;
}

static void testConvReluTraversesTheStagedCompiler(void) {
    NSData *mil = [NSData dataWithContentsOfFile:@"tests/fixtures/conv_relu.mil"];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = [ANEStagedCompiler compileMILData:mil
        modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                             isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics];
    expect(bundle != nil && diagnostics.errorCount == 0,
           @"Conv and ReLU compile through the staged path");
    expect([bundle.passTrace isEqualToArray:@[
        @"mil.import-operation-graph", @"ane.normalize", @"ane.decompose",
        @"ane.fuse", @"h16g.legalize", @"h16g.plan", @"h16g.encode",
        @"h16g.write-object",
    ]], @"bundle records every semantic compiler stage");
    expect(bundle.artifacts.count == 1 &&
           [bundle.dispatchPlan isEqualToArray:@[@0]],
           @"one fused region becomes one dispatchable object");

    NSError *error = nil;
    HWXImage *actual = [HWXImage imageWithData:bundle.artifacts[0].image
                                         error:&error];
    expect(actual != nil && error == nil,
           @"staged object reparses");
    expect([[sha256([[actual firstSectionNamed:@"__text"
        inSegment:@"__TEXT"] data]) lowercaseString] isEqualToString:
        @"62a90244a4edc71a70d6f0630a2d72371b7dcb39d5194d22369781cb54094bff"],
           @"staged Conv encoder reproduces the measured TD hash");
    expect([[sha256([[actual firstSectionNamed:@"__kern_0"
        inSegment:@"__KERN_0"] data]) lowercaseString] isEqualToString:
        @"44f8459ee020ef8a01d767df4e238eef58953984af4be906101e9f7bc1a4a1bb"],
           @"staged constant packing reproduces the measured kernel hash");
}

static NSURL *prepareDenseConvModel(NSUInteger inputChannels,
                                    NSUInteger outputChannels,
                                    NSString *name) {
    NSString *root = [@"build/" stringByAppendingString:name];
    NSString *weightsDirectory =
        [root stringByAppendingPathComponent:@"weights"];
    NSError *error = nil;
    [NSFileManager.defaultManager createDirectoryAtPath:weightsDirectory
        withIntermediateDirectories:YES attributes:nil error:&error];
    NSUInteger weightBytes = inputChannels * outputChannels * sizeof(_Float16);
    NSMutableData *blob = [NSMutableData dataWithLength:128 + weightBytes];
    uint8_t *bytes = (uint8_t *)blob.mutableBytes;
    uint32_t magic = 0xDEADBEEF;
    uint64_t payloadLength = weightBytes;
    uint64_t payloadOffset = 128;
    memcpy(bytes + 64, &magic, sizeof(magic));
    memcpy(bytes + 72, &payloadLength, sizeof(payloadLength));
    memcpy(bytes + 80, &payloadOffset, sizeof(payloadOffset));
    _Float16 *weights = (_Float16 *)(bytes + payloadOffset);
    for (NSUInteger output = 0; output < outputChannels; ++output)
        for (NSUInteger input = 0; input < inputChannels; ++input)
            weights[output * inputChannels + input] =
                (_Float16)(output == input ? 1.0f : 0.0f);
    [blob writeToFile:[weightsDirectory stringByAppendingPathComponent:@"weight.bin"]
             options:NSDataWritingAtomic error:&error];
    expect(error == nil, [name stringByAppendingString:@" model blob is prepared"]);
    return [NSURL fileURLWithPath:root isDirectory:YES];
}

static void testMeasuredConvShapesTraverseTheStagedCompiler(void) {
    typedef struct { NSUInteger inputChannels, outputChannels, spatial; } Shape;
    static const Shape shapes[] = {{32,32,64},{128,256,32}};
    for (const Shape &shape : shapes) {
        NSString *name = [NSString stringWithFormat:@"test-conv-%lu-%lu-%lu",
            (unsigned long)shape.inputChannels,
            (unsigned long)shape.outputChannels,
            (unsigned long)shape.spatial];
        NSString *source = [NSString stringWithFormat:
            @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
             "  func main<ios18>(tensor<fp16, [1, %lu, %lu, %lu]> x) {\n"
             "    string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
             "    tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
             "    tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
             "    tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
             "    int32 gp = const()[name = string(\"gp\"), val = int32(1)];\n"
             "    tensor<fp16, [%lu, %lu, 1, 1]> w = const()[name = string(\"w\"), val = tensor<fp16, [%lu, %lu, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
             "    tensor<fp16, [1, %lu, %lu, %lu]> y = conv(dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string(\"y\")];\n"
             "  } -> (y);\n}\n",
            (unsigned long)shape.inputChannels,(unsigned long)shape.spatial,
            (unsigned long)shape.spatial,(unsigned long)shape.outputChannels,
            (unsigned long)shape.inputChannels,(unsigned long)shape.outputChannels,
            (unsigned long)shape.inputChannels,(unsigned long)shape.outputChannels,
            (unsigned long)shape.spatial,(unsigned long)shape.spatial];
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEExecutableBundle *bundle = [ANEStagedCompiler
            compileMILData:[source dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:prepareDenseConvModel(shape.inputChannels,
                shape.outputChannels,name)
            target:@"H16G" diagnostics:diagnostics];
        expect(bundle != nil && diagnostics.errorCount == 0,
            [name stringByAppendingString:@" compiles from MIL without a fused epilogue"]);
        if (!bundle) continue;
        NSError *error = nil;
        HWXImage *image = [HWXImage imageWithData:bundle.artifacts[0].image
                                             error:&error];
        NSData *td = [[image firstSectionNamed:@"__text" inSegment:@"__TEXT"] data];
        expect(image != nil && td.length == 0x1d8,
            [name stringByAppendingString:@" emits one complete Conv TD"]);
        expect(tdWord(td,0x10c) == shape.inputChannels &&
               tdWord(td,0x11c) == shape.outputChannels &&
               tdWord(td,0x104) == shape.spatial,
            [name stringByAppendingString:@" carries graph-derived shape fields"]);
    }
}

static NSURL *prepareDepthwiseModel(NSUInteger channels, NSString *name) {
    NSString *root = [@"build/" stringByAppendingString:name];
    NSString *weightsDirectory =
        [root stringByAppendingPathComponent:@"weights"];
    NSError *error = nil;
    [NSFileManager.defaultManager createDirectoryAtPath:weightsDirectory
        withIntermediateDirectories:YES attributes:nil error:&error];
    NSUInteger weightBytes = channels * 9 * sizeof(_Float16);
    NSMutableData *blob = [NSMutableData dataWithLength:128 + weightBytes];
    uint8_t *bytes = (uint8_t *)blob.mutableBytes;
    uint32_t magic = 0xDEADBEEF;
    uint64_t payloadLength = weightBytes;
    uint64_t payloadOffset = 128;
    memcpy(bytes + 64, &magic, sizeof(magic));
    memcpy(bytes + 72, &payloadLength, sizeof(payloadLength));
    memcpy(bytes + 80, &payloadOffset, sizeof(payloadOffset));
    _Float16 *weights = (_Float16 *)(bytes + payloadOffset);
    for (NSUInteger index = 0; index < channels * 9; ++index)
        weights[index] = (_Float16)(((NSInteger)(index % 7) - 3) * 0.125f);
    [blob writeToFile:[weightsDirectory stringByAppendingPathComponent:@"weight.bin"]
             options:NSDataWritingAtomic error:&error];
    expect(error == nil, [name stringByAppendingString:@" model blob is prepared"]);
    return [NSURL fileURLWithPath:root isDirectory:YES];
}

static void testDepthwiseFamiliesTraverseTheStagedCompiler(void) {
    NSArray<NSNumber *> *channelCases = @[@64,@128,@256,@512];
    NSArray<NSString *> *tdHashes = @[
        @"b3d0e231a447aa05d25f8323351e93b3ecc52bb823c53dd56a11b9861c918f12",
        @"3523d9b8bdfa9ebaec1846b374f786c5e9e8e8e236af8a26c141741d6e85a6b2",
        @"5e9e1ffee51bbd55733b290c11220019c4cd3b9c051a5f9e37b1e54c0a8ba061",
        @"ba8485a730bfdd5ccc6139ce845e68b3d71f07e63c4cedd6b2308b93b851eb7d",
    ];
    NSArray<NSString *> *kernelHashes = @[
        @"3f310516445b6427da23fd31433b9f2cab21ace8d471ccbf718c6c34804b121b",
        @"effd4da043eb9b735bc06b4fd4db7b356bfde34b73244259717ca9efb9f862c9",
        @"2c56466159476b5922d1d330fbcb9cc14621b1547f41891606e8094a412aeb10",
        @"e6f5d1980bfa2db0c64033f7d8525ece5fa96ac41a34721fd15db70f1b3d1296",
    ];
    for (NSUInteger caseIndex = 0; caseIndex < channelCases.count; ++caseIndex) {
        NSUInteger channels = channelCases[caseIndex].unsignedIntegerValue;
        NSString *name = [NSString stringWithFormat:@"test-depthwise-%lu",
            (unsigned long)channels];
        NSString *source = [NSString stringWithFormat:
            @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
             "  func main<ios18>(tensor<fp16, [1, %lu, 64, 64]> x) {\n"
             "    string pt = const()[name = string(\"pt\"), val = string(\"same\")];\n"
             "    tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
             "    tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
             "    tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
             "    int32 gp = const()[name = string(\"gp\"), val = int32(%lu)];\n"
             "    tensor<fp16, [%lu, 1, 3, 3]> w = const()[name = string(\"w\"), val = tensor<fp16, [%lu, 1, 3, 3]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
             "    tensor<fp16, [1, %lu, 64, 64]> y = conv(dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string(\"y\")];\n"
             "  } -> (y);\n}\n",
             (unsigned long)channels,(unsigned long)channels,
             (unsigned long)channels,(unsigned long)channels,
             (unsigned long)channels];
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEExecutableBundle *bundle = [ANEStagedCompiler
            compileMILData:[source dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:prepareDepthwiseModel(channels,name)
            target:@"H16G" diagnostics:diagnostics];
        expect(bundle != nil && diagnostics.errorCount == 0,
            [name stringByAppendingString:@" compiles through the shared staged path"]);
        if (!bundle) {
            for (ANEDiagnostic *diagnostic in diagnostics.diagnostics)
                fprintf(stderr,"  %s: %s\n",diagnostic.code.UTF8String,
                    diagnostic.message.UTF8String);
            continue;
        }
        NSError *error = nil;
        HWXImage *image = [HWXImage imageWithData:bundle.artifacts[0].image
                                             error:&error];
        NSData *td = [[image firstSectionNamed:@"__text" inSegment:@"__TEXT"] data];
        NSData *kernel = [[image firstSectionNamed:@"__kern_0"
            inSegment:@"__KERN_0"] data];
        expect([[sha256(td) lowercaseString] isEqualToString:tdHashes[caseIndex]] &&
               [[sha256(kernel) lowercaseString] isEqualToString:kernelHashes[caseIndex]],
            [name stringByAppendingString:
                @" reproduces independent TD and packed-kernel sections"]);
    }
}

static NSURL *prepareRegularConvModel(NSUInteger channels, NSUInteger kernel,
                                      NSString *name) {
    NSString *root = [@"build/" stringByAppendingString:name];
    NSString *weightsDirectory =
        [root stringByAppendingPathComponent:@"weights"];
    NSError *error = nil;
    [NSFileManager.defaultManager createDirectoryAtPath:weightsDirectory
        withIntermediateDirectories:YES attributes:nil error:&error];
    NSUInteger count = channels*channels*kernel*kernel;
    NSUInteger weightBytes = count*sizeof(_Float16);
    NSMutableData *blob = [NSMutableData dataWithLength:128+weightBytes];
    uint8_t *bytes = (uint8_t *)blob.mutableBytes;
    uint32_t magic = 0xDEADBEEF;
    uint64_t payloadLength = weightBytes, payloadOffset = 128;
    memcpy(bytes+64,&magic,4); memcpy(bytes+72,&payloadLength,8);
    memcpy(bytes+80,&payloadOffset,8);
    _Float16 *weights = (_Float16 *)(bytes+payloadOffset);
    for (NSUInteger index=0;index<count;++index)
        weights[index]=(_Float16)(((NSInteger)(index%7)-3)*0.125f);
    [blob writeToFile:[weightsDirectory stringByAppendingPathComponent:@"weight.bin"]
             options:NSDataWritingAtomic error:&error];
    expect(error == nil,[name stringByAppendingString:@" model blob is prepared"]);
    return [NSURL fileURLWithPath:root isDirectory:YES];
}

static void testRegularConvFamiliesTraverseTheStagedCompiler(void) {
    NSArray<NSArray<NSNumber *> *> *cases = @[
        @[@64,@32,@3], @[@64,@64,@3], @[@64,@64,@5],
        @[@128,@32,@3], @[@128,@64,@3], @[@128,@64,@5],
    ];
    NSArray<NSString *> *tdHashes = @[
        @"b3e901114155bb7f0ee600db47051c4c9175dfbece5cf741760dc272e4a6bb9a",
        @"36e2027bb14cd552b2166f1f11762f7cfae4511673548d808a11e78d1619dc59",
        @"d1531aee412df2780fb527869138c6f5068895b3824253b936a718ea10a08987",
        @"721c64c7602564bb8f25be6fad60636b50919d40b5a044e5e6fa1cecf7264638",
        @"5a6ee548fc5a37a809a02f1a6faf93188b5b2f3de8d0e6f177e7c3ab1adcdf96",
        @"e228b964f0c1b91df3229a4f20bc3d8dc46ecee1a8b46760acf37d59d3662eb4",
    ];
    NSArray<NSString *> *kernelHashes = @[
        @"3e8c357cc7ea0f21e0c4eab4c98d3a3365eeca9c280e2555566aedf364c890fb",
        @"3e8c357cc7ea0f21e0c4eab4c98d3a3365eeca9c280e2555566aedf364c890fb",
        @"5fd716152521ab710b836cae1b4995a6542c1b2d60716bd44bf36183d2003d1c",
        @"320e0d79b583bff58ce7b63d01d73a9f4be19c7c61fd8440b4a14e90ea21652d",
        @"320e0d79b583bff58ce7b63d01d73a9f4be19c7c61fd8440b4a14e90ea21652d",
        @"21bc283b7617e473b5c9d17f1fdf827e03bba6fc5b48c0390b9ba54b4862d045",
    ];
    for (NSUInteger caseIndex=0;caseIndex<cases.count;++caseIndex) {
        NSUInteger channels=cases[caseIndex][0].unsignedIntegerValue;
        NSUInteger spatial=cases[caseIndex][1].unsignedIntegerValue;
        NSUInteger kernel=cases[caseIndex][2].unsignedIntegerValue;
        NSString *name=[NSString stringWithFormat:@"test-conv-c%lu-s%lu-k%lu",
            (unsigned long)channels,(unsigned long)spatial,(unsigned long)kernel];
        NSString *source=[NSString stringWithFormat:
            @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
             "  func main<ios18>(tensor<fp16, [1, %lu, %lu, %lu]> x) {\n"
             "    string pt = const()[name = string(\"pt\"), val = string(\"same\")];\n"
             "    tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
             "    tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
             "    tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
             "    int32 gp = const()[name = string(\"gp\"), val = int32(1)];\n"
             "    tensor<fp16, [%lu, %lu, %lu, %lu]> w = const()[name = string(\"w\"), val = tensor<fp16, [%lu, %lu, %lu, %lu]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
             "    tensor<fp16, [1, %lu, %lu, %lu]> y = conv(dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string(\"y\")];\n"
             "  } -> (y);\n}\n",
            (unsigned long)channels,(unsigned long)spatial,(unsigned long)spatial,
            (unsigned long)channels,(unsigned long)channels,
            (unsigned long)kernel,(unsigned long)kernel,
            (unsigned long)channels,(unsigned long)channels,
            (unsigned long)kernel,(unsigned long)kernel,
            (unsigned long)channels,(unsigned long)spatial,(unsigned long)spatial];
        ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc] init];
        ANEExecutableBundle *bundle=[ANEStagedCompiler
            compileMILData:[source dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:prepareRegularConvModel(channels,kernel,name)
            target:@"H16G" diagnostics:diagnostics];
        expect(bundle != nil && diagnostics.errorCount == 0,
            [name stringByAppendingString:@" compiles through the staged path"]);
        if (!bundle) {
            for (ANEDiagnostic *diagnostic in diagnostics.diagnostics)
                fprintf(stderr,"  %s: %s\n",diagnostic.code.UTF8String,
                    diagnostic.message.UTF8String);
            continue;
        }
        NSError *error=nil;
        HWXImage *image=[HWXImage imageWithData:bundle.artifacts[0].image error:&error];
        NSData *td=[[image firstSectionNamed:@"__text" inSegment:@"__TEXT"] data];
        NSData *kernelData=[[image firstSectionNamed:@"__kern_0"
            inSegment:@"__KERN_0"] data];
        expect([[sha256(td) lowercaseString] isEqualToString:tdHashes[caseIndex]] &&
               [[sha256(kernelData) lowercaseString] isEqualToString:kernelHashes[caseIndex]],
            [name stringByAppendingString:@" reproduces independent TD and KERN sections"]);
    }
}

static void testW8A8ConvChainTraversesTheSameStagedCompiler(void) {
    NSData *mil = [NSData dataWithContentsOfFile:
        @"tests/fixtures/w8a8_conv_chain.mil"];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = [ANEStagedCompiler compileMILData:mil
        modelRoot:[NSURL fileURLWithPath:@"tests/models/w8a8"
                             isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics];
    expect(bundle != nil && diagnostics.errorCount == 0,
           @"a quantized Conv chain compiles through the shared staged path");
    if (!bundle) return;
    expect([bundle.passTrace isEqualToArray:@[
        @"mil.import-operation-graph", @"ane.normalize", @"ane.decompose",
        @"ane.fuse", @"h16g.legalize", @"h16g.plan", @"h16g.encode",
        @"h16g.write-object",
    ]], @"quantized Conv uses the same semantic pass sequence");
    expect(bundle.artifacts.count == 1 &&
           [bundle.dispatchPlan isEqualToArray:@[@0]],
           @"the scheduled Conv chain becomes one dispatchable object");
    expect([firstFVMLIBSectionName(bundle.artifacts[0].image)
            isEqualToString:@"__data"],
           @"the chain encoder carries its decoded output-first binding slots");

    NSError *error = nil;
    HWXImage *actual = [HWXImage imageWithData:bundle.artifacts[0].image
                                         error:&error];
    if (!actual)
        fprintf(stderr, "W8A8 parse error: %s\n",
                error.description.UTF8String ?: "unknown");
    expect(actual != nil && error == nil,
           @"staged W8A8 object reparses");
    expect([[sha256([[actual firstSectionNamed:@"__text"
        inSegment:@"__TEXT"] data]) lowercaseString] isEqualToString:
        @"185de64f2a80436a319e7b7ef561a9ee05b9fba45142ddfc11e409e055ae69f8"],
           @"scheduled numeric modes reproduce the measured W8A8 TD hash");
    expect([[sha256([[actual firstSectionNamed:@"__kern_0"
        inSegment:@"__KERN_0"] data]) lowercaseString] isEqualToString:
        @"54e18123ce030cef0fa8ac906f358e6bcba6ce19ee083173875ee21717407155"],
           @"scheduled constants reproduce the measured W8A8 kernel hash");
}

static void testMixedTaskGraphTraversesTheSameStagedCompiler(void) {
    NSData *mil = [NSData dataWithContentsOfFile:@"tests/fixtures/attention.mil"];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = [ANEStagedCompiler compileMILData:mil
        modelRoot:[NSURL fileURLWithPath:@"tests/models/attention"
                             isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics];
    expect(bundle != nil && diagnostics.errorCount == 0,
           @"a mixed layout/matmul/ALU/reduce/LUT graph compiles through the staged path");
    if (!bundle) {
        for (ANEDiagnostic *diagnostic in diagnostics.diagnostics)
            fprintf(stderr, "  %s: %s\n", diagnostic.code.UTF8String,
                    diagnostic.message.UTF8String);
        return;
    }
    expect([bundle.passTrace isEqualToArray:@[
        @"mil.import-operation-graph", @"ane.normalize", @"ane.decompose",
        @"ane.fuse", @"h16g.legalize", @"h16g.plan", @"h16g.encode",
        @"h16g.write-object",
    ]], @"mixed primitives use the same semantic pass sequence");
    expect(bundle.artifacts.count == 1 &&
           [bundle.dispatchPlan isEqualToArray:@[@0]],
           @"the ten scheduled tasks become one dispatchable object");

    NSError *error = nil;
    HWXImage *actual = [HWXImage imageWithData:bundle.artifacts[0].image
                                         error:&error];
    expect(actual != nil && error == nil,
           @"staged mixed-task object reparses");
    NSData *actualTD=[[actual firstSectionNamed:@"__text"
        inSegment:@"__TEXT"] data];
    expect([[sha256(actualTD) lowercaseString] isEqualToString:
        @"a2d9c65e6440b7fcd1b01c40934c2fdedb5a76575d9c0a5bb86f2becd2e9a95c"],
           @"scheduled primitive groups reproduce the measured mixed-task TD hash");
    expect([[sha256([[actual firstSectionNamed:@"__kern_0"
        inSegment:@"__KERN_0"] data]) lowercaseString] isEqualToString:
        @"b7b6085a1edc7def0f0bb2fc1fe345f1ba9d9a1a47e55b4516273149323b54d2"],
           @"mixed-task constants reproduce the measured kernel hash");
}

static void testD2STraversesTheProductionCompiler(void) {
    NSString *source =
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 256, 32, 32]> x) {\n"
         "    int32 b = const()[name = string(\"b\"), val = int32(4)];\n"
         "    tensor<fp16, [1, 16, 128, 128]> y = "
         "depth_to_space(x = x, block_size = b)[name = string(\"unpack\")];\n"
         "  } -> (y);\n}\n";
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = [ANEStagedCompiler
        compileMILData:[source dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                             isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics];
    expect(bundle != nil && diagnostics.errorCount == 0,
           @"D2S compiles through the shared production pass sequence");
    if (!bundle) return;
    expect(bundle.artifacts.count == 1 &&
           [bundle.dispatchPlan isEqualToArray:@[@0]],
           @"one planned layout task becomes one clean HWX object");
    NSError *error = nil;
    HWXImage *image = [HWXImage imageWithData:bundle.artifacts[0].image
                                         error:&error];
    NSData *td = [[image firstSectionNamed:@"__text" inSegment:@"__TEXT"] data];
    expect(image != nil && error == nil && td.length == 0x290,
           @"the emitted D2S object reparses with its complete TD stream");
    expect([[sha256(td) lowercaseString] isEqualToString:
        @"26a63ec93b2e897911fc3b5a7276a36e6c9aa6b0d5690a75b4d4282185342c06"],
        @"production D2S emission retains the independent full-TD identity");
    NSArray<ANEHWXBinding *> *bindings = bundle.artifacts[0].bindings;
    expect(bindings.count == 2 &&
           bindings[0].logicalByteLength == 256*32*32*2 &&
           bindings[1].logicalByteLength == 16*128*128*2,
           @"runtime bindings are derived from MIL input and output shapes");
}

static NSString *standaloneLayoutMIL(BOOL depthToSpace, NSUInteger channels,
                                     NSUInteger spatial, NSUInteger block) {
    NSUInteger outputChannels=depthToSpace
        ? channels/(block*block) : channels*block*block;
    NSUInteger outputSpatial=depthToSpace ? spatial*block : spatial/block;
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, %lu, %lu]> x) {\n"
         "    int32 b = const()[name = string(\"b\"), val = int32(%lu)];\n"
         "    tensor<fp16, [1, %lu, %lu, %lu]> y = %@"
         "(x = x, block_size = b)[name = string(\"layout\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)channels,(unsigned long)spatial,(unsigned long)spatial,
        (unsigned long)block,(unsigned long)outputChannels,
        (unsigned long)outputSpatial,(unsigned long)outputSpatial,
        depthToSpace?@"depth_to_space":@"space_to_depth"];
}

static void testStandaloneLayoutFamiliesTraverseTheProductionCompiler(void) {
    NSArray<NSArray<NSNumber *> *> *cases=@[
        @[@0,@8,@128,@4], @[@0,@32,@64,@4], @[@0,@8,@128,@8],
        @[@1,@256,@32,@4], @[@1,@512,@16,@8],
    ];
    for (NSArray<NSNumber *> *row in cases) {
        BOOL d2s=row[0].boolValue;
        NSUInteger channels=row[1].unsignedIntegerValue;
        NSUInteger spatial=row[2].unsignedIntegerValue;
        NSUInteger block=row[3].unsignedIntegerValue;
        ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc]init];
        ANEExecutableBundle *bundle=[ANEStagedCompiler compileMILData:
            [standaloneLayoutMIL(d2s,channels,spatial,block)
                dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                                 isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics];
        expect(bundle!=nil&&diagnostics.errorCount==0&&
               bundle.artifacts.count==1,
            @"each measured standalone S2D/D2S packet family traverses the staged compiler");
        if (!bundle) continue;
        NSArray<ANEHWXBinding *> *bindings=bundle.artifacts[0].bindings;
        expect(bindings.count==2&&bindings[0].rowStrideBytes>=64&&
               bindings[1].rowStrideBytes>=64,
            @"standalone layout bindings retain logical shapes with physical row padding");
    }
}

static NSURL *prepareLayoutConvModel(void) {
    NSString *root = @"build/test-layout-conv-model";
    NSString *weightsDirectory =
        [root stringByAppendingPathComponent:@"weights"];
    NSError *error = nil;
    [NSFileManager.defaultManager createDirectoryAtPath:weightsDirectory
        withIntermediateDirectories:YES attributes:nil error:&error];
    const NSUInteger channels = 128;
    const NSUInteger weightBytes = channels * channels * sizeof(_Float16);
    NSMutableData *blob = [NSMutableData dataWithLength:128 + weightBytes];
    uint8_t *bytes = (uint8_t *)blob.mutableBytes;
    uint32_t magic = 0xDEADBEEF;
    uint64_t payloadLength = weightBytes;
    uint64_t payloadOffset = 128;
    memcpy(bytes + 64, &magic, sizeof(magic));
    memcpy(bytes + 72, &payloadLength, sizeof(payloadLength));
    memcpy(bytes + 80, &payloadOffset, sizeof(payloadOffset));
    _Float16 *weights = (_Float16 *)(bytes + payloadOffset);
    for (NSUInteger output = 0; output < channels; ++output)
        for (NSUInteger input = 0; input < channels; ++input)
            weights[output * channels + input] =
                (_Float16)(output == input ? 1.0f : 0.0f);
    [blob writeToFile:[weightsDirectory
        stringByAppendingPathComponent:@"weight.bin"]
             options:NSDataWritingAtomic error:&error];
    expect(error == nil, @"layout-chain model blob is prepared");
    return [NSURL fileURLWithPath:root isDirectory:YES];
}

static void testS2DConvD2SCompilesAsOneHardwarePipeline(void) {
    NSString *source =
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 8, 128, 128]> x) {\n"
         "    int32 b = const()[name = string(\"b\"), val = int32(4)];\n"
         "    string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
         "    tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "    tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    int32 gp = const()[name = string(\"gp\"), val = int32(1)];\n"
         "    tensor<fp16, [1, 128, 32, 32]> packed = space_to_depth(x = x, block_size = b)[name = string(\"packed\")];\n"
         "    tensor<fp16, [128, 128, 1, 1]> w = const()[name = string(\"w\"), val = tensor<fp16, [128, 128, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
         "    tensor<fp16, [1, 128, 32, 32]> c = conv(dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = packed)[name = string(\"c\")];\n"
         "    tensor<fp16, [1, 8, 128, 128]> y = depth_to_space(x = c, block_size = b)[name = string(\"y\")];\n"
         "  } -> (y);\n}\n";
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = [ANEStagedCompiler
        compileMILData:[source dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:prepareLayoutConvModel() target:@"H16G"
        diagnostics:diagnostics];
    expect(bundle != nil && diagnostics.errorCount == 0,
           @"S2D, Conv and D2S compile as one scheduled hardware pipeline");
    if (!bundle) return;
    expect(bundle.artifacts.count == 1 &&
           [bundle.dispatchPlan isEqualToArray:@[@0]],
           @"layout intermediates stay inside one dispatchable HWX object");
    NSError *error = nil;
    HWXImage *image = [HWXImage imageWithData:bundle.artifacts[0].image
                                         error:&error];
    NSData *td = [[image firstSectionNamed:@"__text" inSegment:@"__TEXT"] data];
    expect(td.length == 0x8ac &&
           [[sha256(td) lowercaseString] isEqualToString:
            @"6ed394127f6d3b8161cc3be9986f084fad01f83595c39f590ad478155df9b973"],
           @"fused layout pipeline emits the measured DMA_INTER packet grammar");
}

static NSString *squareMatmulMIL(NSUInteger size, BOOL transpose) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, %lu]> a, tensor<fp16, [1, %lu, %lu]> b) {\n"
         "    bool f = const()[name = string(\"f\"), val = bool(%s)];\n"
         "    tensor<fp16, [1, %lu, %lu]> y = matmul(transpose_x = f, transpose_y = f, x = a, y = b)[name = string(\"mm\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)size,(unsigned long)size,
        (unsigned long)size,(unsigned long)size,
        transpose ? "true" : "false",
        (unsigned long)size,(unsigned long)size];
}

static void testSquareMatmulShapesTraverseTheProductionCompiler(void) {
    NSArray<NSNumber *> *sizes = @[@128,@256,@512,@768,@1024,@2048,@2176,@4096];
    NSArray<NSNumber *> *tiles = @[@1,@1,@1,@3,@4,@8,@17,@32];
    NSArray<NSNumber *> *rows = @[@128,@256,@512,@256,@256,@256,@128,@128];
    NSArray<NSString *> *singleHashes = @[
        @"bf43e48293566a79477756859f746877f91777189ca9e1cb242e4dda24066c60",
        @"c0c92aa4f1bbda7fc917d01914c854fd45869c84663cb7d278a23ee301cfbe73",
        @"1090b6fb996e4d129079342a4b006ab249bd0cccb6b09ffdadac9c24ff7f990e",
    ];
    for (NSUInteger index = 0; index < sizes.count; ++index) {
        NSUInteger size = sizes[index].unsignedIntegerValue;
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEExecutableBundle *bundle = [ANEStagedCompiler compileMILData:
            [squareMatmulMIL(size,NO) dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                                 isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics];
        NSString *label = [NSString stringWithFormat:@"N%lu matmul",
            (unsigned long)size];
        expect(bundle != nil && diagnostics.errorCount == 0,
            [label stringByAppendingString:@" compiles through the staged path"]);
        if (!bundle) continue;
        ANEHWXArtifact *artifact = bundle.artifacts.firstObject;
        NSError *error = nil;
        HWXImage *image = [HWXImage imageWithData:artifact.image error:&error];
        NSData *td = [image firstSectionNamed:@"__text" inSegment:@"__TEXT"].data;
        NSUInteger tileCount = tiles[index].unsignedIntegerValue;
        NSUInteger expectedWords = tileCount == 1 ? (size == 256 ? 179 : 185)
            : 4+53+93+(tileCount-2)*124+131+32;
        expect(image != nil && error == nil && td.length == expectedWords*4 &&
               [image firstSectionNamed:@"__kern_0" inSegment:@"__KERN_0"] == nil,
            [label stringByAppendingString:@" emits a constant-free TD object"]);
        expect(artifact.bindings.count == 3 &&
               artifact.bindings[0].role == ANESurfaceRoleInput &&
               artifact.bindings[1].role == ANESurfaceRoleInput &&
               artifact.bindings[2].role == ANESurfaceRoleOutput &&
               artifact.bindings[0].ioSurfaceIndex == 0 &&
               artifact.bindings[1].ioSurfaceIndex == 1 &&
               artifact.bindings[2].ioSurfaceIndex == 2,
            [label stringByAppendingString:@" exposes two inputs and one output"]);
        NSUInteger rowCount = rows[index].unsignedIntegerValue;
        NSUInteger tensorBytes = size*size*2;
        NSUInteger workingSet = tileCount == 1 ? tensorBytes
            : tensorBytes+(tileCount-1)*rowCount*size*2;
        if (workingSet > 0x2000000) workingSet = tensorBytes;
        expect(programDescriptorWord(artifact.image,0x830) == tileCount+1 &&
               programDescriptorWord(artifact.image,0x860) ==
                    (tileCount == 1 ? 31 : 14*tileCount+4) &&
               programDescriptorWord(artifact.image,0x858) == workingSet,
            [label stringByAppendingString:
                @" carries its derived task, record and row-slab schedule"]);
        if (index < singleHashes.count)
            expect([[sha256(td) lowercaseString]
                    isEqualToString:singleHashes[index]],
                [label stringByAppendingString:@" reproduces its full Apple TD"]);
    }

    for (NSNumber *unsupported in @[@384,@640]) {
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEExecutableBundle *bundle = [ANEStagedCompiler compileMILData:
            [squareMatmulMIL(unsupported.unsignedIntegerValue,NO)
                dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                                 isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics];
        expect(bundle == nil && diagnostics.errorCount > 0,
            @"unmeasured single-tile matmul sizes fail closed");
    }
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    expect([ANEStagedCompiler compileMILData:
        [squareMatmulMIL(256,YES) dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                             isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics] == nil &&
        diagnostics.errorCount > 0,
        @"transpose flags outside the decoded matmul form fail closed");
}

static NSString *binaryALUMIL(NSString *operation,NSUInteger size) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 1, %lu, %lu]> x, tensor<fp16, [1, 1, %lu, %lu]> z) {\n"
         "    tensor<fp16, [1, 1, %lu, %lu]> y = %@(x = x, y = z)[name = string(\"op\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)size,(unsigned long)size,
        (unsigned long)size,(unsigned long)size,
        (unsigned long)size,(unsigned long)size,operation];
}

static void testBinaryALUShapesTraverseTheProductionCompiler(void) {
    NSArray<NSNumber *> *sizes=@[@128,@256,@512,@1024,@2048];
    NSArray<NSString *> *hashes=@[
        @"f771fd68b79650d7f250b08b454d4295f8f210d5edc411b99a1ea9eeea0e9c7e",
        @"fc41a0c84737f7ffb0bd2a47c8c00dec2d62404371f3a42d8896d1ab1a6d1b15",
        @"99f8af8df237cdd5adee04acdf13d1e23d148b8dab89e5e77a1a2b029db87f92",
        @"7f06e3525c7b9b874d9ebbb1e4adec88107ef152362b6306f754666c08b14bb9",
        @"d29df2570fb8d1f26ab4bfe760e95a21a9990e51a016cb4528f68635aafb69bd",
    ];
    for(NSUInteger index=0;index<sizes.count;++index){
        ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc]init];
        ANEExecutableBundle *bundle=[ANEStagedCompiler compileMILData:
            [binaryALUMIL(@"add",sizes[index].unsignedIntegerValue)
                dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                                 isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics];
        expect(bundle!=nil&&diagnostics.errorCount==0,
            @"measured add geometry compiles through the staged path");
        if(!bundle)continue;
        HWXImage *image=[HWXImage imageWithData:bundle.artifacts[0].image error:nil];
        NSData *td=[image firstSectionNamed:@"__text" inSegment:@"__TEXT"].data;
        expect([[sha256(td) lowercaseString]isEqualToString:hashes[index]]&&
               bundle.artifacts[0].bindings.count==3,
               @"staged add preserves its complete TD and three-surface contract");
    }
    for(NSString *operation in @[@"mul",@"max",@"min"]){
        ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc]init];
        expect([ANEStagedCompiler compileMILData:
            [binaryALUMIL(operation,512)dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                                 isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics]!=nil&&
            diagnostics.errorCount==0,
            @"hardware-verified binary ALU selector compiles through shared passes");
    }
    for(NSString *source in @[binaryALUMIL(@"sub",512),binaryALUMIL(@"add",384)]){
        ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc]init];
        expect([ANEStagedCompiler compileMILData:
            [source dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                                 isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics]==nil&&diagnostics.errorCount>0,
            @"unmapped ALU semantics fail before object emission");
    }
}

static NSString *unaryMIL(NSString *operation, NSUInteger size) {
    NSString *arguments = @"x = x";
    if ([operation isEqualToString:@"gelu"])
        arguments = @"x = x, mode = string(\"EXACT\")";
    if ([operation isEqualToString:@"rsqrt"] ||
        [operation isEqualToString:@"reciprocal"] ||
        [operation isEqualToString:@"log"])
        arguments = @"x = x, epsilon = fp32(0.000001)";
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 1, %lu, %lu]> x) {\n"
         "    tensor<fp16, [1, 1, %lu, %lu]> y = %@(%@)"
         "[name = string(\"pointwise\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)size,(unsigned long)size,
        (unsigned long)size,(unsigned long)size,operation,arguments];
}

static void testUnaryPointwiseShapesTraverseTheProductionCompiler(void) {
    NSArray<NSNumber *> *sizes = @[@128,@256,@512,@1024,@2048];
    NSArray<NSString *> *hashes = @[
        @"0816aeafe502da3e027587d730cba455a3424c5589cf333010824243a56b4ed3",
        @"6e86fb37ce458b7f5a5952c94e4f38ca28d458e6afb22e75884a738825a7d4f0",
        @"b3d7ce3f51c6d693d56430c8775b7b0892d554a12564a6836bd5a5ba4bf8235a",
        @"c0624039ec55ef0b8c8f675dbaa5e458e04b6e6c00c13ffc1dcde4085b21cea3",
        @"6171aa5214af2a262ec8dac74d55001f8cd3c4f5d0862d311d944b56908c8151",
    ];
    for (NSUInteger index=0; index<sizes.count; ++index) {
        ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc]init];
        ANEExecutableBundle *bundle=[ANEStagedCompiler compileMILData:
            [unaryMIL(@"sigmoid",sizes[index].unsignedIntegerValue)
                dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                                 isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics];
        expect(bundle!=nil&&diagnostics.errorCount==0,
            @"measured sigmoid geometry compiles through the staged path");
        if(!bundle)continue;
        HWXImage *image=[HWXImage imageWithData:bundle.artifacts[0].image error:nil];
        NSData *td=[image firstSectionNamed:@"__text" inSegment:@"__TEXT"].data;
        NSData *table=[image firstSectionNamed:@"__kern_0"
            inSegment:@"__KERN_0"].data;
        expect([[sha256(td)lowercaseString]isEqualToString:hashes[index]]&&
               table.length==0x80&&bundle.artifacts[0].bindings.count==2,
            @"staged sigmoid preserves its TD, table and two-surface contract");
    }
    for(NSString *operation in @[@"relu",@"tanh",@"gelu",@"silu",@"exp",
                                  @"sqrt",@"rsqrt",@"reciprocal"]){
        ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc]init];
        ANEExecutableBundle *bundle=[ANEStagedCompiler compileMILData:
            [unaryMIL(operation,256)dataUsingEncoding:NSUTF8StringEncoding]
            modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                                 isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics];
        expect(bundle!=nil&&diagnostics.errorCount==0,
            @"decoded unary operation compiles through the shared staged path");
    }
    ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc]init];
    expect([ANEStagedCompiler compileMILData:
        [unaryMIL(@"sigmoid",384)dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                             isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics]==nil&&diagnostics.errorCount>0,
        @"unmeasured unary geometry fails before object emission");
}

static NSString *reductionMIL(NSString *operation, NSUInteger channels,
                              NSUInteger height, NSUInteger width,
                              NSUInteger axis) {
    NSUInteger outChannels=axis == 1 ? 1 : channels;
    NSUInteger outHeight=axis == 2 ? 1 : height;
    NSUInteger outWidth=axis == 3 ? 1 : width;
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, %lu, %lu, %lu]> x) {\n"
         "    tensor<int32, [1]> ax = const()[name = string(\"ax\"), "
         "val = tensor<int32, [1]>([%lu])];\n"
         "    tensor<fp16, [1, %lu, %lu, %lu]> y = %@"
         "(x = x, axes = ax, keep_dims = bool(true))"
         "[name = string(\"reduce\")];\n"
         "  } -> (y);\n}\n",
        (unsigned long)channels,(unsigned long)height,(unsigned long)width,
        (unsigned long)axis,(unsigned long)outChannels,
        (unsigned long)outHeight,(unsigned long)outWidth,operation];
}

static void testReductionFamiliesTraverseTheProductionCompiler(void) {
    NSArray<NSArray<NSNumber *> *> *geometries=@[
        @[@32,@8,@8,@1], @[@64,@8,@8,@1], @[@128,@8,@8,@1],
        @[@64,@16,@16,@1], @[@64,@32,@32,@1],
        @[@1,@64,@64,@3], @[@1,@128,@128,@3], @[@32,@64,@16,@2],
    ];
    for (NSString *operation in @[@"reduce_sum",@"reduce_mean",@"reduce_max"]) {
        for (NSArray<NSNumber *> *geometry in geometries) {
            NSUInteger channels=geometry[0].unsignedIntegerValue;
            NSUInteger height=geometry[1].unsignedIntegerValue;
            NSUInteger width=geometry[2].unsignedIntegerValue;
            NSUInteger axis=geometry[3].unsignedIntegerValue;
            ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc]init];
            ANEExecutableBundle *bundle=[ANEStagedCompiler compileMILData:
                [reductionMIL(operation,channels,height,width,axis)
                    dataUsingEncoding:NSUTF8StringEncoding]
                modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                                     isDirectory:YES]
                target:@"H16G" diagnostics:diagnostics];
            expect(bundle!=nil&&diagnostics.errorCount==0,
                @"measured reduction compiles through the shared staged path");
            if (!bundle) continue;
            NSError *error=nil;
            H16GReduceEncoding *expected=[H16GReduceEncoder
                encodeOperationName:operation
                inputShape:@[@1,@(channels),@(height),@(width)]
                axis:axis error:&error];
            HWXImage *image=[HWXImage imageWithData:bundle.artifacts[0].image
                                             error:&error];
            NSData *td=[image firstSectionNamed:@"__text" inSegment:@"__TEXT"].data;
            NSArray<ANEHWXBinding *> *bindings=bundle.artifacts[0].bindings;
            expect(image!=nil&&error==nil&&[td isEqualToData:expected.tdProgram.data]&&
                   [image firstSectionNamed:@"__kern_0" inSegment:@"__KERN_0"]==nil,
                @"staged reduction emits the exact clean TD and no kernel segment");
            expect(bindings.count==2&&
                   bindings[0].rowStrideBytes==expected.inputRowStrideBytes&&
                   bindings[0].planeStrideBytes==expected.inputPlaneStrideBytes&&
                   bindings[0].batchStrideBytes==expected.inputBatchStrideBytes&&
                   bindings[1].rowStrideBytes==64&&
                   bindings[1].planeStrideBytes==expected.outputPlaneStrideBytes&&
                   bindings[1].batchStrideBytes==expected.outputBatchStrideBytes&&
                   bindings[1].allocationByteLength>=expected.outputStorageByteLength,
                @"staged reduction preserves semantic output with measured physical padding");
        }
    }
    ANEDiagnosticEngine *diagnostics=[[ANEDiagnosticEngine alloc]init];
    expect([ANEStagedCompiler compileMILData:
        [reductionMIL(@"reduce_sum",96,8,8,1)
            dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                             isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics]==nil&&diagnostics.errorCount>0,
        @"unmeasured reduction geometry fails before object emission");
}

static void testDecodedFormsRejectNearbyButUnencodedSemantics(void) {
    expectRejectedMutation(@"tests/fixtures/conv_relu.mil",
        @"tests/models/conv_relu", @"([1, 1])", @"([2, 1])",
        @"Conv stride mutation is rejected before descriptor emission");
    expectRejectedMutation(@"tests/fixtures/w8a8_conv_chain.mil",
        @"tests/models/w8a8", @"scale = fp16(0x1p-3)",
        @"scale = fp16(0x1p-2)",
        @"W8A8 scale mutation is rejected before descriptor emission");
    expectRejectedMutation(@"tests/fixtures/attention.mil",
        @"tests/models/attention", @"fp16(0.125)", @"fp16(0.25)",
        @"attention scale mutation is rejected before descriptor emission");
    expectRejectedMutation(@"tests/fixtures/attention.mil",
        @"tests/models/attention", @"int32(-1)", @"int32(-2)",
        @"softmax-axis mutation is rejected before descriptor emission");
    expectRejectedMutation(@"tests/fixtures/attention.mil",
        @"tests/models/attention", @"([0, 2, 1, 3])",
        @"([0, 1, 2, 3])",
        @"K-transpose mutation is rejected before descriptor emission");
    expectRejectedMutation(@"tests/fixtures/attention.mil",
        @"tests/models/attention", @"bool(false)", @"bool(true)",
        @"matmul transpose mutation is rejected before descriptor emission");
    expectRejectedMutation(@"tests/fixtures/attention.mil",
        @"tests/models/attention", @"([0, 0, 0, 64])",
        @"([0, 0, 0, 32])",
        @"slice-bound mutation is rejected before descriptor emission");
}

int main(void) {
    @autoreleasepool {
        testConvReluTraversesTheStagedCompiler();
        testMeasuredConvShapesTraverseTheStagedCompiler();
        testDepthwiseFamiliesTraverseTheStagedCompiler();
        testRegularConvFamiliesTraverseTheStagedCompiler();
        testW8A8ConvChainTraversesTheSameStagedCompiler();
        testMixedTaskGraphTraversesTheSameStagedCompiler();
        testD2STraversesTheProductionCompiler();
        testStandaloneLayoutFamiliesTraverseTheProductionCompiler();
        testS2DConvD2SCompilesAsOneHardwarePipeline();
        testSquareMatmulShapesTraverseTheProductionCompiler();
        testBinaryALUShapesTraverseTheProductionCompiler();
        testUnaryPointwiseShapesTraverseTheProductionCompiler();
        testReductionFamiliesTraverseTheProductionCompiler();
        testDecodedFormsRejectNearbyButUnencodedSemantics();
        printf("staged compiler: %s\n",
               failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
