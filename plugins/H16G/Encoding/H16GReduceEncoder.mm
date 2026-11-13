#import "H16GReduceEncoder.h"

#include "H16GRulePacket.h"

#include <cstring>
#include <limits>

static NSString *const H16GReduceEncoderErrorDomain = @"ANE.H16G.ReduceEncoder";

typedef NS_ENUM(uint32_t, H16GReductionKind) {
    H16GReductionKindSum = 0,
    H16GReductionKindMean = 1,
    H16GReductionKindMax = 2,
};

typedef struct { uint32_t offset; uint32_t length; } H16GRuleAllocation;

class H16GReductionAllocator {
public:
    H16GRuleAllocation allocate(uint64_t length, uint64_t alignment) {
        uint64_t start = (_cursor + alignment - 1) / alignment * alignment;
        uint64_t end = start + length;
        if (end > std::numeric_limits<uint32_t>::max()) return {0,0};
        _cursor = end;
        return {(uint32_t)start,(uint32_t)length};
    }
private:
    uint64_t _cursor = 0;
};

@interface H16GReduceEncoding ()
@property(nonatomic, readwrite, strong) H16GEncodedTDProgram *tdProgram;
@property(nonatomic, readwrite, copy) NSArray<NSNumber *> *outputShape;
@property(nonatomic, readwrite) NSUInteger taskCount;
@property(nonatomic, readwrite) NSUInteger inputRowStrideBytes;
@property(nonatomic, readwrite) NSUInteger inputPlaneStrideBytes;
@property(nonatomic, readwrite) NSUInteger inputBatchStrideBytes;
@property(nonatomic, readwrite) NSUInteger inputStorageByteLength;
@property(nonatomic, readwrite) NSUInteger outputRowStrideBytes;
@property(nonatomic, readwrite) NSUInteger outputPlaneStrideBytes;
@property(nonatomic, readwrite) NSUInteger outputBatchStrideBytes;
@property(nonatomic, readwrite) NSUInteger outputStorageByteLength;
@end

@implementation H16GReduceEncoding
@end

static NSUInteger alignUp(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static void setError(NSError **error, NSString *message) {
    if (error) *error = [NSError errorWithDomain:H16GReduceEncoderErrorDomain
        code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
}

static BOOL reductionKind(NSString *operation, H16GReductionKind *kind) {
    if ([operation isEqualToString:@"reduce_sum"])
        *kind = H16GReductionKindSum;
    else if ([operation isEqualToString:@"reduce_mean"])
        *kind = H16GReductionKindMean;
    else if ([operation isEqualToString:@"reduce_max"])
        *kind = H16GReductionKindMax;
    else return NO;
    return YES;
}

static BOOL validGeometry(NSArray<NSNumber *> *shape, NSUInteger axis) {
    if (shape.count != 4 || shape[0].unsignedIntegerValue != 1 ||
        axis < 1 || axis > 3) return NO;
    uint64_t elements = 1;
    for (NSNumber *dimension in shape) {
        NSUInteger value = dimension.unsignedIntegerValue;
        if (value == 0 || value > UINT16_MAX) return NO;
        elements *= value;
        if (elements > UINT32_MAX / sizeof(uint16_t)) return NO;
    }
    return shape[axis].unsignedIntegerValue > 1;
}

static NSArray<NSNumber *> *reducedShape(NSArray<NSNumber *> *inputShape,
                                          NSUInteger axis) {
    NSMutableArray<NSNumber *> *shape = [inputShape mutableCopy];
    shape[axis] = @1;
    return shape;
}

static uint16_t floatToHalf(float value) {
    constexpr uint32_t floatMantissaMask=(uint32_t(1)<<23)-1;
    constexpr uint32_t floatHiddenBit=uint32_t(1)<<23;
    constexpr uint32_t halfSignMask=uint32_t(1)<<15;
    constexpr uint32_t halfRoundBit=uint32_t(1)<<12;
    constexpr uint32_t halfRoundIncrement=uint32_t(1)<<13;
    constexpr uint32_t halfInfinity=uint32_t(31)<<10;
    uint32_t bits = 0;
    memcpy(&bits,&value,sizeof(bits));
    uint32_t sign = (bits >> 16) & halfSignMask;
    int exponent = (int)((bits >> 23) & ((uint32_t(1)<<8)-1)) - 127 + 15;
    uint32_t mantissa = bits & floatMantissaMask;
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;
        mantissa = (mantissa | floatHiddenBit) >> (1 - exponent);
        if (mantissa & halfRoundBit) mantissa += halfRoundIncrement;
        return (uint16_t)(sign | (mantissa >> 13));
    }
    if (exponent >= 31) return (uint16_t)(sign | halfInfinity);
    if (mantissa & halfRoundBit) mantissa += halfRoundIncrement;
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | (mantissa >> 13));
}

typedef NS_ENUM(NSUInteger, H16GReductionPhase) {
    H16GReductionPhasePartial,
    H16GReductionPhaseReorder,
    H16GReductionPhaseFinalize,
    H16GReductionPhaseStore,
    H16GReductionPhaseDirect,
};

typedef struct {
    NSUInteger channelStart;
    NSUInteger channels;
    H16GReductionPhase phase;
} H16GReductionTaskRule;

static uint32_t reductionEngineCount(H16GReductionPhase phase,
                                     H16GReductionKind kind,
                                     NSUInteger axis,
                                     bool writesExternal) {
    // The task-control engine set describes the pipeline phase, not the NE
    // opcode.  Direct/final tasks use no additional engine set, a reorder and
    // an axis-1 partial use the vector set, and the axis-2 store has its own
    // set.  A one-task channel maximum is the only external max partial.
    switch(phase) {
        case H16GReductionPhaseDirect:
        case H16GReductionPhaseFinalize: return 0;
        case H16GReductionPhaseReorder: return 5;
        case H16GReductionPhaseStore: return 4;
        case H16GReductionPhasePartial:
            if(kind==H16GReductionKindMax && writesExternal) return 1;
            return axis==1 ? 5 : 0;
    }
    return 0;
}

static uint32_t ceilLog2(NSUInteger value) {
    uint32_t result=0;
    NSUInteger limit=1;
    while(limit<value) { limit<<=1; ++result; }
    return result;
}

static uint32_t packedPipelineWord(uint32_t family,
                                   uint32_t dimensionClass,
                                   uint32_t laneClass) {
    return (family<<24) | (dimensionClass<<16) | (laneClass<<8);
}

static constexpr uint32_t partialPipelineFamily=3u<<4;
static constexpr uint32_t storePipelineFamily=1u<<4;
static constexpr uint32_t partialDimensionClassBase=39;
static constexpr uint32_t reorderDimensionClassBase=41;
static constexpr uint32_t packedLaneClassStep=(1u<<4)|1u;

static uint32_t reductionAccumulatorBase(NSUInteger outputRowBytes) {
    // The accumulator follows one output row plus its four-word control area.
    return (uint32_t)(outputRowBytes+4*sizeof(uint32_t));
}

static uint32_t partialPipelineWord(NSUInteger channels,
                                    NSUInteger height) {
    const uint32_t dimensionClass=
        partialDimensionClassBase-ceilLog2(height);
    const uint32_t channelBits=ceilLog2(channels);
    const uint32_t channelGroups=channelBits>=8 ? 1u : 8u-channelBits;
    return packedPipelineWord(partialPipelineFamily,dimensionClass,
                              channelGroups*packedLaneClassStep);
}

static uint32_t reorderPipelineWord(NSUInteger height, NSUInteger width) {
    const uint32_t dimensionClass=
        reorderDimensionClassBase-ceilLog2(height);
    const uint32_t widthBits=ceilLog2(width);
    const uint32_t widthGroups=widthBits>=6 ? 1u : 6u-widthBits;
    return packedPipelineWord(partialPipelineFamily,dimensionClass,
                              widthGroups*packedLaneClassStep);
}

static uint32_t maximumPartialPipelineWord(NSUInteger height,
                                           NSUInteger width) {
    const NSUInteger largeSpatialTile=16*16;
    const uint32_t scheduleTier=height*width<=largeSpatialTile ? 4u : 3u;
    return scheduleTier<<16;
}

static uint32_t storePipelineWord() {
    // Store is pipeline family 0x10 with seven four-lane completion groups.
    return packedPipelineWord(storePipelineFamily,7u<<2,0);
}

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t outputWidth;
    uint32_t outputHeight;
    uint32_t outputChannels;
    uint32_t reductionExtent;
    uint32_t residentLayout;
    uint32_t tileOrPartialBase;
    uint32_t kernelOrOutputBase;
    uint32_t tileCount;
    uint32_t scheduleClass;
    uint32_t inputElementQuads;
    uint32_t surfaceLimit;
    bool emitsReductionExtent;
    bool emitsResidentLayout;
    bool emitsChannels;
    bool emitsOutputChannels;
    bool emitsTileCount;
    bool emitsScheduleClass;
    bool emitsInputElementQuads;
} H16GReductionCommonGeometry;

static H16GReductionCommonGeometry reductionCommonGeometry(
    H16GReductionPhase phase, H16GReductionKind kind, NSUInteger axis,
    NSUInteger channels, NSUInteger localChannels, NSUInteger height,
    NSUInteger width, NSUInteger outputRowBytes, bool writesExternal) {
    H16GReductionCommonGeometry result={};
    result.width=(uint32_t)width;
    result.height=(uint32_t)height;
    result.channels=(uint32_t)localChannels;
    result.outputWidth=(uint32_t)width;
    result.outputHeight=(uint32_t)height;
    result.outputChannels=(uint32_t)localChannels;
    result.tileOrPartialBase=1;
    result.kernelOrOutputBase=reductionAccumulatorBase(outputRowBytes);
    result.emitsChannels=true;
    result.emitsOutputChannels=true;
    switch(phase) {
        case H16GReductionPhaseDirect:
            // A width reduction presents H independent contiguous rows.  The
            // local Common height is one; writing the source H here was the
            // remaining 8e-3 direct-max error in round 6.
            result.height=1;
            result.channels=(uint32_t)height;
            result.outputWidth=(uint32_t)width;
            result.outputHeight=1;
            result.outputChannels=(uint32_t)height;
            result.scheduleClass=6;
            result.surfaceLimit=0;
            result.emitsScheduleClass=true;
            break;
        case H16GReductionPhasePartial:
            if(axis==1) {
                result.tileOrPartialBase=(uint32_t)height;
                result.outputWidth=(uint32_t)width;
                result.outputHeight=(uint32_t)height;
                result.outputChannels=(uint32_t)(writesExternal ? 1 :
                                                   localChannels);
                result.kernelOrOutputBase=
                    kind==H16GReductionKindMax && writesExternal
                    ? maximumPartialPipelineWord(height,width)
                    : partialPipelineWord(localChannels,height);
                result.residentLayout=writesExternal ? 0x00404000 : 0x00044000;
                result.emitsResidentLayout=true;
                result.tileCount=writesExternal ? 0 :
                    1+(uint32_t)(((uint64_t)localChannels*height*width+
                                  ((uint64_t(1)<<14)-1))/(uint64_t(1)<<14));
                result.emitsTileCount=true;
                result.surfaceLimit=1u<<20;
            } else {
                // The reordered partial reduces one H-wide row for each C;
                // its local extent selector is the original W, not axis 2.
                result.width=(uint32_t)height;
                result.height=1;
                result.outputWidth=(uint32_t)height;
                result.outputHeight=1;
                result.outputChannels=(uint32_t)localChannels;
                result.reductionExtent=(uint32_t)width;
                result.emitsReductionExtent=true;
                result.scheduleClass=6;
                result.emitsScheduleClass=true;
                result.surfaceLimit=0;
            }
            break;
        case H16GReductionPhaseReorder:
            // Reorder swaps C and H while preserving each W-wide row.  C is
            // the local height; using source H here made every axis-2 compute
            // phase wait on geometry that the reorder never produced.
            result.width=(uint32_t)width;
            result.height=(uint32_t)localChannels;
            result.channels=(uint32_t)height;
            result.outputWidth=(uint32_t)width;
            result.outputHeight=(uint32_t)localChannels;
            result.outputChannels=(uint32_t)height;
            result.residentLayout=0x00044000;
            result.emitsResidentLayout=true;
            result.tileOrPartialBase=(uint32_t)localChannels;
            result.kernelOrOutputBase=reorderPipelineWord(height,width);
            result.tileCount=(uint32_t)(((uint64_t)localChannels*height*width+
                                         ((uint64_t(1)<<14)-1)) /
                                        (uint64_t(1)<<14));
            result.emitsTileCount=true;
            result.surfaceLimit=1u<<20;
            break;
        case H16GReductionPhaseFinalize: {
            // Channel reductions pack C across a 64-wide single row, or
            // 32-wide rows once C exceeds one hardware row. W is the lane set.
            const NSUInteger packedWidth=channels<=64 ? channels : 32;
            result.width=(uint32_t)channels;
            result.height=1;
            result.channels=(uint32_t)width;
            result.outputWidth=(uint32_t)packedWidth;
            result.outputHeight=(uint32_t)((channels+packedWidth-1)/packedWidth);
            result.outputChannels=(uint32_t)width;
            result.reductionExtent=(uint32_t)height;
            result.emitsReductionExtent=true;
            result.scheduleClass=6;
            result.emitsScheduleClass=true;
            result.inputElementQuads=(uint32_t)(((uint64_t)channels*height*width)/4);
            result.emitsInputElementQuads=true;
            result.surfaceLimit=0;
            break;
        }
        case H16GReductionPhaseStore:
            result.width=1;
            result.height=(uint32_t)localChannels;
            result.emitsChannels=false;
            result.outputWidth=1;
            result.outputHeight=(uint32_t)localChannels;
            result.emitsOutputChannels=false;
            result.residentLayout=0x00014000;
            result.emitsResidentLayout=true;
            result.tileOrPartialBase=(uint32_t)width;
            result.kernelOrOutputBase=storePipelineWord();
            result.tileCount=0;
            result.emitsTileCount=true;
            result.surfaceLimit=1u<<20;
            break;
    }
    return result;
}

// One task may expose at most the Common.surface_limit resident bytes to L2.
// Axis 1 is accumulated in channel chunks and finalized; axis 2 uses the
// reorder/reduce/store pipeline because H is not contiguous in NCHW; axis 3
// is contiguous and therefore reduces directly.  This is a shape rule, not a
// captured-program-position rule.
static std::vector<H16GReductionTaskRule> reductionTaskRules(
    H16GReductionKind kind, NSUInteger channels, NSUInteger axis,
    NSUInteger inputPlaneBytes) {
    const NSUInteger residentBytes = NSUInteger(1) << 20;
    const NSUInteger channelsPerTask = MAX((NSUInteger)1,
        residentBytes / MAX((NSUInteger)1,inputPlaneBytes));
    std::vector<H16GReductionTaskRule> tasks;
    if (axis == 1) {
        for (NSUInteger first = 0; first < channels; first += channelsPerTask)
            tasks.push_back({first,MIN(channelsPerTask,channels-first),
                             H16GReductionPhasePartial});
        if (kind != H16GReductionKindMax || tasks.size() > 1)
            tasks.push_back({0,1,H16GReductionPhaseFinalize});
        return tasks;
    }
    for (NSUInteger first = 0; first < channels; first += channelsPerTask) {
        NSUInteger count=MIN(channelsPerTask,channels-first);
        if (axis == 2) {
            tasks.push_back({first,count,H16GReductionPhaseReorder});
            tasks.push_back({first,count,H16GReductionPhasePartial});
            tasks.push_back({first,count,H16GReductionPhaseStore});
        } else {
            tasks.push_back({first,count,H16GReductionPhaseDirect});
        }
    }
    return tasks;
}

static uint32_t reductionVectorCommand(H16GReductionKind kind) {
    // Vector partial/reorder/store phases share command 0x0c.  A direct
    // channel maximum selects the max sub-mode (0x0b).  Direct width and
    // post-reorder reductions instead use the Other selectors below.
    return (uint32_t(1)<<20) | (uint32_t(15)<<10) |
           (kind==H16GReductionKindMax ? 0x0b : 0x0c);
}

static uint32_t reductionCopyCommand() {
    return (uint32_t(1)<<20) | (uint32_t(15)<<10) | 0x0c;
}

static uint32_t meanReductionSelector() {
    // Mean is selected in Other as a packed fp16 unit-scale control.  The
    // reduction engine applies the dimension reciprocal; pre-scaling 0x4910
    // caused round 6's 17x/66x mean errors.
    return uint32_t(0x3c80)<<16;
}

static H16GEncodedTDProgram *encodeReductionPacket(
    H16GReductionKind kind, NSArray<NSNumber *> *inputShape,
    NSArray<NSNumber *> *outputShape, NSUInteger axis,
    NSUInteger inputRowBytes, NSUInteger inputPlaneBytes,
    NSUInteger inputBatchBytes, NSUInteger outputRowBytes,
    NSUInteger outputPlaneBytes, NSUInteger outputBatchBytes,
    NSUInteger *encodedTaskCount, NSError **error) {
    NSUInteger channels = inputShape[1].unsignedIntegerValue;
    NSUInteger height = inputShape[2].unsignedIntegerValue;
    NSUInteger width = inputShape[3].unsignedIntegerValue;
    NSUInteger outputChannels = outputShape[1].unsignedIntegerValue;
    NSUInteger outputHeight = outputShape[2].unsignedIntegerValue;
    NSUInteger outputWidth = outputShape[3].unsignedIntegerValue;
    std::vector<H16GReductionTaskRule> rules=reductionTaskRules(
        kind,channels,axis,inputPlaneBytes);
    constexpr uint32_t allocationPageBytes=uint32_t(1)<<14;
    H16GReductionAllocator allocator;
    H16GRuleAllocation input = allocator.allocate(inputBatchBytes,allocationPageBytes);
    const NSUInteger partialBytes=MAX(outputBatchBytes,rules.size()*128);
    H16GRuleAllocation partial = allocator.allocate(partialBytes,allocationPageBytes);
    H16GRuleAllocation output = allocator.allocate(outputBatchBytes,allocationPageBytes);
    if (input.length == 0 || partial.length == 0 || output.length == 0) {
        setError(error,@"reduction packet allocations exceed the H16G field width");
        return nil;
    }

    std::vector<ane::h16::rule::TaskSpec> tasks;
    uint64_t totalTiles=0;
    using namespace ane::h16::rule::fields;
    for (NSUInteger index=0;index<rules.size();++index) {
        const H16GReductionTaskRule &taskRule=rules[index];
        const bool finalize=taskRule.phase==H16GReductionPhaseFinalize;
        const bool readsExternal=
            taskRule.phase==H16GReductionPhaseDirect ||
            taskRule.phase==H16GReductionPhaseReorder ||
            (taskRule.phase==H16GReductionPhasePartial && axis==1);
        const bool writesExternal=
            taskRule.phase==H16GReductionPhaseDirect ||
            taskRule.phase==H16GReductionPhaseFinalize ||
            taskRule.phase==H16GReductionPhaseStore ||
            (taskRule.phase==H16GReductionPhasePartial &&
             kind==H16GReductionKindMax && rules.size()==1);
        const NSUInteger localChannels=finalize ? 1 : taskRule.channels;
        const NSUInteger localInputBytes=readsExternal
            ? localChannels*inputPlaneBytes : outputBatchBytes;
        const uint64_t localElements=finalize
            ? (uint64_t)outputChannels*outputHeight*outputWidth
            : (uint64_t)localChannels*height*width;
        const uint32_t tiles=(uint32_t)((localElements+
            ((uint64_t(1)<<14)-1))/(uint64_t(1)<<14));
        totalTiles+=tiles;
        const H16GReductionCommonGeometry common=reductionCommonGeometry(
            taskRule.phase,kind,axis,channels,localChannels,height,width,
            outputRowBytes,writesExternal);
        std::map<uint32_t,uint32_t> registers;
        registers[0x0004]=common.width;
        registers[0x0008]=common.height;
        if(common.emitsChannels) registers[0x000c]=common.channels;
        registers[0x0014]=common.outputWidth;
        registers[0x0018]=common.outputHeight;
        if(common.emitsOutputChannels)
            registers[0x001c]=common.outputChannels;
        if(common.emitsReductionExtent)
            registers[0x0024]=common.reductionExtent;
        if(common.emitsResidentLayout)
            registers[0x0030]=common.residentLayout;
        registers[0x0034]=common.tileOrPartialBase;
        registers[0x003c]=common.kernelOrOutputBase;
        if(common.emitsTileCount) registers[0x0040]=common.tileCount;
        if(common.emitsScheduleClass)
            registers[0x0044]=common.scheduleClass;
        if(common.emitsInputElementQuads)
            registers[0x0048]=common.inputElementQuads;
        registers[0x004c]=surfaceLimit(common.surfaceLimit);
        registers[0x4104]=l2Transfer(5,7,2);
        registers[0x4108]=l2Transfer(5,3,0);
        if (!readsExternal)
            registers[0x4110]=(uint32_t)(partial.offset+
                (index ? index-1 : 0)*128);
        registers[0x4114]=(uint32_t)inputRowBytes;
        registers[0x4118]=(uint32_t)inputPlaneBytes;
        registers[0x411c]=(uint32_t)localInputBytes;
        registers[0x4120]=(uint32_t)localInputBytes;
        registers[0x4148]=l2Transfer(5,7,10);
        if (!writesExternal)
            registers[0x414c]=(uint32_t)(partial.offset+index*128);
        registers[0x4150]=(uint32_t)outputRowBytes;
        registers[0x4154]=(uint32_t)outputPlaneBytes;
        registers[0x4158]=(uint32_t)outputBatchBytes;
        registers[0x415c]=(uint32_t)outputBatchBytes;
        const bool vectorPartial=taskRule.phase==H16GReductionPhasePartial &&
                                 axis==1;
        const bool copyPhase=taskRule.phase==H16GReductionPhaseReorder ||
                             taskRule.phase==H16GReductionPhaseStore;
        const bool selectorPhase=taskRule.phase==H16GReductionPhaseDirect ||
            taskRule.phase==H16GReductionPhaseFinalize ||
            (taskRule.phase==H16GReductionPhasePartial && axis==2);
        if(vectorPartial || copyPhase) {
            registers[0x4904]=vectorPartial
                ? reductionVectorCommand(kind) : reductionCopyCommand();
            registers[0x4910]=floatToHalf(1.0f);
        }
        if(selectorPhase && kind==H16GReductionKindMax)
            registers[0x4500]=(uint32_t)kind;
        else if(selectorPhase && kind==H16GReductionKindMean)
            registers[0x4514]=meanReductionSelector();
        if (readsExternal) {
            registers[0x4d00]=endpoint(2);
            registers[0x4d08]=sourceDMAFlags();
            registers[0x4d10]=(uint32_t)(taskRule.channelStart*inputPlaneBytes);
            registers[0x4d14]=0;
            if(axis==1) {
                registers[0x4d18]=(uint32_t)inputRowBytes;
                registers[0x4d1c]=(uint32_t)inputPlaneBytes;
            } else if(axis==2) {
                // Height reduction transposes row and plane traversal in its
                // external reorder phase.
                registers[0x4d18]=(uint32_t)inputPlaneBytes;
                registers[0x4d1c]=(uint32_t)inputRowBytes;
            } else {
                // A contiguous width reduction advances one padded input row
                // per output value and has no independent row stride.
                registers[0x4d1c]=(uint32_t)inputRowBytes;
            }
            registers[0x4d68]=surfaceDescriptor(0,2);
        }
        if (writesExternal) {
            registers[0x5100]=resultEndpoint(3);
            registers[0x5108]=(uint32_t)(taskRule.channelStart*outputPlaneBytes);
            registers[0x510c]=0;
            if(taskRule.phase==H16GReductionPhaseFinalize)
                registers[0x511c]=(uint32_t)outputRowBytes;
            else if(axis==3)
                registers[0x5114]=(uint32_t)outputRowBytes;
            else
                registers[0x5110]=(uint32_t)outputRowBytes;
            registers[0x5138]=surfaceDescriptor(3,2);
            registers[0x5900]=storeFlags();
            registers[0x5904]=storeGeometry();
            registers[0x5918]=0;
            registers[0x5928]=storeCompletionMask();
            registers[0x592c]=outputSurface();
        }
        ane::h16::rule::TaskSpec task={std::move(registers),
            reductionEngineCount(taskRule.phase,kind,axis,writesExternal),
            false};
        // Scheduler fields describe the local phase, not merely whether this
        // is the last task.  These values are geometry-independent and keep
        // dependency word 7 under the producer-mask rule proven in round 6.
        if(taskRule.phase==H16GReductionPhaseReorder) {
            task.executionCycles=1; task.outputRecordFormat=0x00000068;
            task.dispatchClassFormat=0x00000068;
        } else if(taskRule.phase==H16GReductionPhasePartial && axis==2) {
            task.executionCycles=2; task.outputRecordFormat=0;
            task.dispatchClassFormat=0x00fff820;
        } else if(taskRule.phase==H16GReductionPhaseStore) {
            task.executionCycles=0; task.outputRecordFormat=0x04000000;
            task.dispatchClassFormat=0x00000060;
        } else if(taskRule.phase==H16GReductionPhasePartial &&
                  axis==1 && !writesExternal) {
            task.executionCycles=0; task.outputRecordFormat=0x00000068;
            task.dispatchClassFormat=0x00000068;
        } else if(taskRule.phase==H16GReductionPhaseFinalize) {
            task.executionCycles=0; task.outputRecordFormat=0x04000000;
            task.dispatchClassFormat=0x00fff860;
        }
        tasks.push_back(std::move(task));
    }
    try {
        ane::h16::rule::Packet packet=ane::h16::rule::build(tasks);
        NSData *data=[NSData dataWithBytes:packet.bytes.data()
                                   length:packet.bytes.size()];
        if (encodedTaskCount) *encodedTaskCount=tasks.size();
        return [[H16GEncodedTDProgram alloc] initWithData:data
            kernelRelocationOffsets:@[] programRecordCount:packet.recordCount
            programFormatCode:(uint32_t)(totalTiles-1) scratchByteLength:0];
    } catch(const std::exception &exception) {
        setError(error,[NSString stringWithUTF8String:exception.what()]);
        return nil;
    }
}

@implementation H16GReduceEncoder
+ (BOOL)supportsOperationName:(NSString *)operationName
                   inputShape:(NSArray<NSNumber *> *)inputShape
                         axis:(NSUInteger)axis {
    H16GReductionKind kind;
    return reductionKind(operationName,&kind) && validGeometry(inputShape,axis);
}

+ (H16GReduceEncoding *)encodeOperationName:(NSString *)operationName
                                    inputShape:(NSArray<NSNumber *> *)inputShape
                                          axis:(NSUInteger)axis
                                         error:(NSError **)error {
    H16GReductionKind kind;
    if (!reductionKind(operationName,&kind) || !validGeometry(inputShape,axis)) {
        setError(error,@"reduction requires reduce_sum/reduce_mean/reduce_max, a non-empty [1,C,H,W] fp16 shape, and one non-unit axis in {1,2,3}");
        return nil;
    }
    NSArray<NSNumber *> *outputShape = reducedShape(inputShape,axis);
    NSUInteger inputChannels=inputShape[1].unsignedIntegerValue;
    NSUInteger inputHeight=inputShape[2].unsignedIntegerValue;
    NSUInteger inputWidth=inputShape[3].unsignedIntegerValue;
    NSUInteger inputRowBytes=alignUp(inputWidth * sizeof(uint16_t),64);
    NSUInteger inputPlaneBytes=inputHeight * inputRowBytes;
    NSUInteger inputBatchBytes=inputChannels * inputPlaneBytes;
    NSUInteger channels=outputShape[1].unsignedIntegerValue;
    NSUInteger height=outputShape[2].unsignedIntegerValue;
    NSUInteger width=outputShape[3].unsignedIntegerValue;
    NSUInteger rowBytes=alignUp(width * sizeof(uint16_t),64);
    NSUInteger planeBytes=height * rowBytes;
    NSUInteger batchBytes=channels * planeBytes;
    NSUInteger taskCount=0;
    H16GEncodedTDProgram *program=encodeReductionPacket(kind,inputShape,
        outputShape,axis,inputRowBytes,inputPlaneBytes,inputBatchBytes,
        rowBytes,planeBytes,batchBytes,&taskCount,error);
    if (!program) return nil;
    H16GReduceEncoding *encoding = [H16GReduceEncoding new];
    encoding.tdProgram=program; encoding.outputShape=outputShape;
    encoding.taskCount=taskCount;
    encoding.inputRowStrideBytes=inputRowBytes;
    encoding.inputPlaneStrideBytes=inputPlaneBytes;
    encoding.inputBatchStrideBytes=inputBatchBytes;
    encoding.inputStorageByteLength=inputBatchBytes;
    encoding.outputRowStrideBytes=rowBytes;
    encoding.outputPlaneStrideBytes=planeBytes;
    encoding.outputBatchStrideBytes=batchBytes;
    encoding.outputStorageByteLength=batchBytes;
    return encoding;
}
@end
