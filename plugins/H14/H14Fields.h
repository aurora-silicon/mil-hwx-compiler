#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ane::h14::fields {

// A Field is the public vocabulary for a documented H14 bit range.  Encoding
// through it makes overlap and overflow visible at the call site instead of
// hiding either inside an opaque 32-bit word.
struct Field {
    unsigned lsb;
    unsigned width;

    constexpr std::uint32_t value(std::uint32_t unshifted) const {
        const auto limit = width == 32 ? ~std::uint32_t(0)
                                       : (std::uint32_t(1) << width) - 1;
        return (unshifted & limit) << lsb;
    }
    constexpr std::uint32_t mask() const {
        return value(width == 32 ? ~std::uint32_t(0)
                                 : (std::uint32_t(1) << width) - 1);
    }
    constexpr std::uint32_t get(std::uint32_t encoded) const {
        return (encoded & mask()) >> lsb;
    }
};

// Packed-record header fields: address is a word address at bits 0:14,
// dense count-minus-one is bits 15:20, BAR slot is bits 23:28, BAR-present is
// bit 29, and scatter-present is bit 31.  A scatter mask occupies bits 15:30.
inline constexpr Field recordAddress{0, 15};
inline constexpr Field denseCountMinusOne{15, 6};
inline constexpr Field barSlot{23, 6};
inline constexpr Field barPresent{29, 1};
inline constexpr Field scatterMask{15, 16};
inline constexpr Field scatterPresent{31, 1};

// Task word 0: task id bits 0:15 and encoded word count bits 16:26.
inline constexpr Field taskIdentifier{0, 16};
inline constexpr Field taskWordCount{16, 11};
// Task word 1: low-half task mode/cost encoding.
inline constexpr Field taskMode{0, 16};
// A three-tap finite-clip task sets mode bit 1 so the coefficient engine
// consumes the prepended clip descriptor before kernel data.  One-tap clip
// tasks use mode zero; non-clip ring tasks retain their scheduler cost.
inline constexpr Field taskModeKernelClipDescriptor{1, 1};
// Task word 2: program-open bit 3, program-close bit 1 and final-command bit 5.
inline constexpr Field taskProgramOpen{3, 1};
inline constexpr Field taskProgramClose{1, 1};
inline constexpr Field taskFinalCommand{5, 1};
// Task word 4: the thirteen participating engines are bits 11:23; an edge
// task additionally selects endpoint controls at bits 3, 5 and 6.
inline constexpr Field taskEngineParticipation{11, 13};
inline constexpr Field taskEndpointControl0{3, 1};
inline constexpr Field taskEndpointControl1{5, 1};
inline constexpr Field taskEndpointControl2{6, 1};
// Task word 7: TSR bit 0, scheduler class bit 16, and completion bit 18.
inline constexpr Field taskTsr{0, 1};
inline constexpr Field taskSchedulerClass{16, 1};
inline constexpr Field taskCompletion{18, 1};

// Common convolution register fields.
inline constexpr Field surfaceWidth{0, 16};
inline constexpr Field surfaceHeight{16, 16};
inline constexpr Field convolutionKernelWidth{0, 5};
inline constexpr Field convolutionKernelHeight{6, 5};
inline constexpr Field convolutionStrideX{13, 2};
inline constexpr Field convolutionStrideY{15, 2};
inline constexpr Field convolutionPaddingX{17, 5};
inline constexpr Field convolutionPaddingY{22, 5};
inline constexpr Field convolutionFp16Accumulate{28, 1};
inline constexpr Field convolutionTraversal{30, 1};
inline constexpr Field outputChannelGroup{0, 3};
inline constexpr Field depthwiseInputGroup{0, 1};
inline constexpr Field depthwiseKernelMode{14, 1};
inline constexpr Field depthwiseOutputGroup{16, 1};

// Common schedule register fields: enable bit 0, three four-bit issue/repeat
// classes at bits 8, 12 and 16, and schedule-family bit 21.
inline constexpr Field scheduleEnable{0, 1};
inline constexpr Field scheduleLowWidthTail{2, 1};
inline constexpr Field scheduleInnerRepeat{8, 4};
inline constexpr Field scheduleOuterRepeat{12, 4};
inline constexpr Field scheduleIssueClass{16, 5};
inline constexpr Field scheduleFamily{21, 1};

// L2 SourceCfg/ResultCfg role byte.  Bits 0 and 7 are reserved and stay zero.
// Bit 1 admits a TileDMA transfer, bit 2 selects the compact/channel-
// interleaved source layout, bit 3 selects the channel-interleaved result
// layout, bit 4 marks a program boundary, bit 5 selects the wide source
// path, and bit 6 selects the ordinary linear-surface path.  Roles below
// compose these single-meaning controls instead of treating bits 0:7 as an
// opaque byte.
inline constexpr Field l2RoleReservedLow{0, 1};
inline constexpr Field l2RoleDmaTransfer{1, 1};
inline constexpr Field l2RoleCompactSource{2, 1};
inline constexpr Field l2RoleInterleavedResult{3, 1};
inline constexpr Field l2RoleProgramEdge{4, 1};
inline constexpr Field l2RoleWideSource{5, 1};
inline constexpr Field l2RoleLinearSurface{6, 1};
inline constexpr Field l2RoleReservedHigh{7, 1};
// Outer L2 controls: resident-edge bit 8, scratch-selection bit 11,
// external-DMA bit 20, and external-binding bit 22.
inline constexpr Field l2ResidentEdge{8, 1};
inline constexpr Field l2Scratch{11, 1};
inline constexpr Field l2ExternalDma{20, 1};
inline constexpr Field l2ExternalBinding{22, 1};
inline constexpr Field l2DmaLoadedResult{4, 2};

// NE KernelCfg: convolution format bit 1, kernel-engine bit 7, sparse-phase
// bit 8 and LUT-enable bit 16.
inline constexpr Field neConvolutionFormat{1, 1};
inline constexpr Field neKernelEnable{7, 1};
inline constexpr Field neSparsePhase{8, 1};
inline constexpr Field neLutEnable{16, 1};
// NE MacCfg: selector bit 4, three accumulator-bank enables at bits 10:12,
// MAC enable bit 20, ReLU bit 16 and finite-clip/LUT selector bit 17.  H14
// convolution feeds every accumulator bank, so each bank is enabled by name.
inline constexpr Field macSelector{4, 1};
inline constexpr Field macAccumulatorBank0{10, 1};
inline constexpr Field macAccumulatorBank1{11, 1};
inline constexpr Field macAccumulatorBank2{12, 1};
inline constexpr Field macEnable{20, 1};
inline constexpr Field macRelu{16, 1};
inline constexpr Field macFiniteClip{17, 1};

// DMA config fields shared by source and destination engines.  In the role
// nibble bit 4 selects a result write, bit 5 selects the linear surface path,
// bit 6 selects scratch, and bit 7 selects channel-interleaved result planes.
// Thus program source, scratch source, direct result and scratch result are
// composed from controls rather than encoded as role numbers.
inline constexpr Field dmaEnable{0, 1};
inline constexpr Field dmaRole{4, 4};
inline constexpr Field dmaRoleResultWrite{4, 1};
inline constexpr Field dmaRoleLinearSurface{5, 1};
inline constexpr Field dmaRoleScratchSurface{6, 1};
inline constexpr Field dmaRoleInterleavedResult{7, 1};
// Extension bit 8 is reserved.  Bits 9:11 tell TileDMA to consume the
// coordinate, stride and task extension portions of the descriptor.
inline constexpr Field dmaExtensionReserved{8, 1};
inline constexpr Field dmaExtensionCoordinates{9, 1};
inline constexpr Field dmaExtensionStrides{10, 1};
inline constexpr Field dmaExtensionTask{11, 1};
// The extension word repeats those three descriptor enables at bits 1:3.
// Bit 6 marks a source extension; bit 7 additionally marks program input.
inline constexpr Field dmaExtensionLowCoordinates{1, 1};
inline constexpr Field dmaExtensionLowStrides{2, 1};
inline constexpr Field dmaExtensionLowTask{3, 1};
inline constexpr Field dmaExtensionSource{6, 1};
inline constexpr Field dmaExtensionProgramInput{7, 1};
inline constexpr Field dmaTaskTag{16, 8};
inline constexpr Field dmaScratchDependency{28, 1};
inline constexpr Field dmaUpperByte{24, 8};
// Tile words select fp16 elements and NCHW traversal independently; both are
// set for convolution surfaces.  Remaining controls choose paired planes,
// direct/program I/O, or scratch addressing.
inline constexpr Field dmaTileFp16{4, 1};
inline constexpr Field dmaTileNchw{5, 1};
inline constexpr Field dmaTilePairedPlane{13, 1};
inline constexpr Field dmaTileDirect{20, 2};
inline constexpr Field dmaTileProgramInput{24, 1};
inline constexpr Field dmaTileScratch{27, 1};
inline constexpr Field dmaExternalSurface{24, 1};
// Kernel DMA header uses flags at bits 5, 6, 7 and 9 plus next-task tag
// bits 16:23; each lane config uses enable bit 0 and cache class bit 5.
inline constexpr Field kernelSparse{5, 1};
inline constexpr Field kernelBaseMode{6, 1};
inline constexpr Field kernelTerminal{7, 1};
inline constexpr Field kernelDescriptorMode{9, 1};
inline constexpr Field kernelNextTaskTag{16, 8};
inline constexpr Field kernelLaneEnable{0, 1};
inline constexpr Field kernelLaneCacheClass{5, 2};

// Masks used when decoding or replacing one named field.
inline constexpr std::uint32_t lowHalfMask = 0x0000ffffu; // bits 0:15
inline constexpr std::uint32_t highHalfMask = 0xffff0000u; // bits 16:31
inline constexpr std::uint32_t fp16MagnitudeMask = 0x0000007fu; // fp16 high-byte magnitude bits
inline constexpr std::uint32_t liveBankMask = 0x1fffffffu; // all 29 decoded live L2 banks

// Native-ring geometry limits: H14 exposes a 2 MiB L2 arena in 64 KiB banks.
// The compact terminal path covers a 128x56x56 fp16 surface plus one maximum
// compact-row allowance; larger wide-source terminals use staged strides.
inline constexpr std::uint64_t mobileL2Bytes = 2ull * 1024 * 1024;
inline constexpr std::uint64_t mobileL2BankBytes = 64ull * 1024;
// The convolution datapath consumes 16 fp16 values per execution lane.  L2
// row strides use this as their smallest positive skew quantum.
inline constexpr std::uint32_t mobileExecutionLaneValues = 16;
inline constexpr std::uint32_t mobileExecutionLaneBytes =
    mobileExecutionLaneValues * sizeof(std::uint16_t);
inline constexpr std::uint64_t mobileDirectResultMaxBytes =
    128ull * 56 * 56 * sizeof(std::uint16_t) + 128ull * 64;
// Scheduler cost is rounded to one MiMAC; terminal fp16 output drains in one
// L2-bank quantum.  Both are geometry units, never fixed cost codes.
inline constexpr std::uint64_t mobileCostMacQuantum = std::uint64_t(1) << 20;
inline constexpr std::uint64_t mobileCostDrainQuantumBytes = mobileL2BankBytes;

// H14 packed tasks begin with eight header words before their records.
inline constexpr std::size_t compactTaskHeaderWords = 8;
// Smallest padded task size accepted by our H14 device tests.  The
// probe generated by tools/h14-mobilenet/envelope_kit.py tests smaller
// candidates; keep this empirical limit until device runs fill its table.
inline constexpr std::size_t compactTaskMinimumBytes = 312;

// Named H14 register addresses.  They are byte addresses in packed records;
// the comments group them by the hardware block that owns the fields.
namespace registers {
inline constexpr std::uint32_t inputSurface = 0x0000u; // Common: H:W
inline constexpr std::uint32_t inputChannels = 0x000cu; // Common: input C
inline constexpr std::uint32_t outputChannels = 0x0010u; // Common: output C
inline constexpr std::uint32_t outputSurface = 0x0014u; // Common: H:W
inline constexpr std::uint32_t outputChannelGroup = 0x001cu; // Common: OCG
inline constexpr std::uint32_t convolution = 0x0020u; // Common: convolution geometry
inline constexpr std::uint32_t depthwise = 0x0028u; // Common: depthwise grouping
inline constexpr std::uint32_t tileHeight = 0x002cu; // Common: output tile rows
inline constexpr std::uint32_t schedule = 0x0038u; // Common: schedule classes
inline constexpr std::uint32_t macWork = 0x003cu; // Common: MAC work quantum
inline constexpr std::uint32_t sourceConfig = 0x0504u; // L2: SourceCfg
inline constexpr std::uint32_t sourceBase = 0x0508u; // L2: source base
inline constexpr std::uint32_t sourceChannelStride = 0x050cu; // L2: source channel stride
inline constexpr std::uint32_t sourceRowStride = 0x0510u; // L2: source row stride
inline constexpr std::uint32_t sourceDepthStride = 0x0514u; // L2: source depth stride
inline constexpr std::uint32_t sourcePlaneStride = 0x0518u; // L2: source plane stride
inline constexpr std::uint32_t resultConfig = 0x0530u; // L2: ResultCfg
inline constexpr std::uint32_t resultBase = 0x0534u; // L2: result base
inline constexpr std::uint32_t resultChannelStride = 0x0538u; // L2: result channel stride
inline constexpr std::uint32_t resultRowStride = 0x053cu; // L2: result row stride
inline constexpr std::uint32_t resultDepthStride = 0x0540u; // L2: result depth stride
inline constexpr std::uint32_t kernelConfig = 0x0d00u; // NE: KernelCfg
inline constexpr std::uint32_t macConfig = 0x0d04u; // NE: MacCfg/LUT selector
inline constexpr std::uint32_t unityScale = 0x0d10u; // NE: fp16 unity scale
inline constexpr std::uint32_t sourceDmaConfig = 0x1100u; // source TileDMA: config
inline constexpr std::uint32_t sourceDmaExtension = 0x1108u; // source TileDMA: extension
inline constexpr std::uint32_t sourceDmaBase = 0x1110u; // source TileDMA: BAR base
inline constexpr std::uint32_t sourceDmaRowStride = 0x1118u; // source TileDMA: row
inline constexpr std::uint32_t sourceDmaPlaneStride = 0x111cu; // source TileDMA: plane
inline constexpr std::uint32_t sourceDmaTile = 0x1150u; // source TileDMA: tile format
inline constexpr std::uint32_t resultDmaConfig = 0x1500u; // result TileDMA: config
inline constexpr std::uint32_t resultDmaBase = 0x1508u; // result TileDMA: BAR base
inline constexpr std::uint32_t resultDmaRowStride = 0x1510u; // result TileDMA: row
inline constexpr std::uint32_t resultDmaPlaneStride = 0x1514u; // result TileDMA: plane
inline constexpr std::uint32_t resultDmaTile = 0x1520u; // result TileDMA: tile format
inline constexpr std::uint32_t resultDmaDeadExtension = 0x1524u; // result TileDMA: ignored padding word
inline constexpr std::uint32_t kernelDmaConfig = 0x1900u; // coefficient DMA: config
inline constexpr std::uint32_t kernelDmaBase = 0x1908u; // coefficient DMA: BAR base
inline constexpr std::uint32_t kernelLaneConfig = 0x1918u; // coefficient DMA: lane 0 config
inline constexpr std::uint32_t kernelLaneConfigTail = 0x1944u; // coefficient DMA: lane 11 config
inline constexpr std::uint32_t kernelLaneOffset = 0x1958u; // coefficient DMA: lane 0 offset
inline constexpr std::uint32_t kernelLaneOffsetTail = 0x195cu; // coefficient DMA: lane 1 offset
inline constexpr std::uint32_t kernelLaneSize = 0x1998u; // coefficient DMA: lane 0 size
inline constexpr std::uint32_t kernelLaneSizeLast = 0x19d4u; // coefficient DMA: lane 15 size
} // namespace registers

// A coefficient engine has 16 lanes.  In the record rooted at KernelDmaCfg,
// scatter bit 0 addresses the following word and bit 15 the sixteenth.  Lane
// configs start six words after KernelDmaCfg, so lanes 0..10 fit in that
// scatter record; lanes 11..15 form the contiguous dense tail.
inline constexpr std::uint32_t kernelLaneCount = 16;
inline constexpr std::uint32_t kernelLaneConfigFirstWord =
    (registers::kernelLaneConfig - registers::kernelDmaConfig) /
    sizeof(std::uint32_t);
inline constexpr std::uint32_t kernelLaneScatterCount =
    scatterMask.width - kernelLaneConfigFirstWord + 1;
inline constexpr std::uint32_t kernelLaneDenseCount =
    kernelLaneCount - kernelLaneScatterCount;
static_assert(kernelLaneScatterCount == 11);
static_assert(kernelLaneDenseCount == 5);
static_assert(registers::kernelLaneConfigTail ==
    registers::kernelLaneConfig +
        kernelLaneScatterCount * sizeof(std::uint32_t));

constexpr std::uint32_t surface(std::uint32_t height, std::uint32_t width) {
    return surfaceHeight.value(height) | surfaceWidth.value(width);
}

constexpr std::uint32_t recordScatter(std::uint32_t address,
                                      std::uint32_t mask) {
    return scatterPresent.value(1) | scatterMask.value(mask) |
           recordAddress.value(address / 4);
}

constexpr std::uint32_t recordDense(std::uint32_t address,
                                    std::uint32_t count,
                                    int slot = -1) {
    return recordAddress.value(address / 4) |
           denseCountMinusOne.value(count - 1) |
           (slot < 0 ? 0u : barPresent.value(1) |
                              barSlot.value(static_cast<std::uint32_t>(slot)));
}

constexpr std::uint32_t taskHeader0(std::uint32_t words,
                                    std::uint32_t identifier) {
    return taskWordCount.value(words) | taskIdentifier.value(identifier);
}

constexpr std::uint32_t taskHeader2(bool first, bool last) {
    return (first ? taskProgramOpen.value(1) : 0u) |
           (last ? taskProgramClose.value(1) | taskFinalCommand.value(1) : 0u);
}

constexpr std::uint32_t taskHeader4(bool edge) {
    return taskEngineParticipation.value((std::uint32_t(1) << 13) - 1) |
           (edge ? taskEndpointControl0.value(1) |
                   taskEndpointControl1.value(1) |
                   taskEndpointControl2.value(1) : 0u);
}

constexpr std::uint32_t taskHeader7() {
    return taskTsr.value(1) | taskSchedulerClass.value(1) |
           taskCompletion.value(1);
}

constexpr std::uint32_t convolutionConfig(std::uint32_t kernelWidth,
                                          std::uint32_t kernelHeight,
                                          std::uint32_t stride,
                                          std::uint32_t padding) {
    return convolutionKernelWidth.value(kernelWidth) |
           convolutionKernelHeight.value(kernelHeight) |
           convolutionStrideX.value(stride) |
           convolutionStrideY.value(stride) |
           convolutionPaddingX.value(padding) |
           convolutionPaddingY.value(padding) |
           convolutionFp16Accumulate.value(1) |
           convolutionTraversal.value(1);
}

constexpr std::uint32_t depthwiseConfig() {
    return depthwiseInputGroup.value(1) | depthwiseKernelMode.value(1) |
           depthwiseOutputGroup.value(1);
}

constexpr std::uint32_t scheduleConfig(std::uint32_t issue,
                                       std::uint32_t outer,
                                       std::uint32_t inner,
                                       bool lowWidthTail) {
    return scheduleEnable.value(1) | scheduleFamily.value(1) |
           scheduleIssueClass.value(issue) | scheduleOuterRepeat.value(outer) |
           scheduleInnerRepeat.value(inner) |
           (lowWidthTail ? scheduleLowWidthTail.value(1) : 0u);
}

enum class L2Role {
    ProgramInput,
    ProgramInputResult,
    ResidentWideSource,
    ResidentCompactSource,
    ResidentResult,
    ScratchSource,
    ScratchResult,
    DirectResult,
    StagedDirectResult,
};

constexpr std::uint32_t l2Config(L2Role role) {
    constexpr auto dma = l2RoleDmaTransfer.value(1);
    constexpr auto compact = l2RoleCompactSource.value(1);
    constexpr auto interleaved = l2RoleInterleavedResult.value(1);
    constexpr auto program = l2RoleProgramEdge.value(1);
    constexpr auto wide = l2RoleWideSource.value(1);
    constexpr auto linear = l2RoleLinearSurface.value(1);
    switch (role) {
    case L2Role::ProgramInput:
        return l2ExternalBinding.value(1) | l2ExternalDma.value(1) |
               l2ResidentEdge.value(1) | dma | program | wide | linear;
    case L2Role::ProgramInputResult:
        return l2ExternalBinding.value(1) | l2ExternalDma.value(1) |
               l2ResidentEdge.value(1) | program | wide | linear;
    case L2Role::ResidentWideSource:
        return l2ResidentEdge.value(1) | wide;
    case L2Role::ResidentCompactSource:
        return l2ResidentEdge.value(1) | compact;
    case L2Role::ResidentResult:
        return l2ResidentEdge.value(1) | linear;
    case L2Role::ScratchSource:
        return l2Scratch.value(1) | dma | linear;
    case L2Role::ScratchResult:
        return l2Scratch.value(1) | dma | interleaved | linear;
    case L2Role::DirectResult:
        return l2ResidentEdge.value(1) | dma | interleaved | linear;
    case L2Role::StagedDirectResult:
        return l2ResidentEdge.value(1) | dma | linear;
    }
    return 0;
}

constexpr std::uint32_t neConfig(bool sparse, bool lut) {
    return neConvolutionFormat.value(1) | neKernelEnable.value(1) |
           (sparse ? neSparsePhase.value(1) : 0u) |
           (lut ? neLutEnable.value(1) : 0u);
}

enum class Activation { None, Relu, FiniteClip };

// Apply only the post-operation fields to an already valid MAC config.  The
// linked emitter and native ring both use this function, so selector and bank
// fields stay properties of the convolution form rather than the activation.
constexpr std::uint32_t applyMacActivation(std::uint32_t config,
                                           Activation activation) {
    return (config & ~(macRelu.mask() | macFiniteClip.mask())) |
           (activation == Activation::Relu ? macRelu.value(1) : 0u) |
           (activation == Activation::FiniteClip ? macFiniteClip.value(1) : 0u);
}

constexpr std::uint32_t macConfig(Activation activation) {
    const auto base = macSelector.value(1) | macAccumulatorBank0.value(1) |
                      macAccumulatorBank1.value(1) |
                      macAccumulatorBank2.value(1) | macEnable.value(1);
    return applyMacActivation(base, activation);
}

// The proven linked finite-clip emitter selects the descriptor through
// MacCfg.  KernelCfg's LUT-enable bit stays clear; convolution-format,
// kernel-engine and sparse-phase fields are preserved exactly.
constexpr std::uint32_t applyNeActivation(std::uint32_t config,
                                          Activation activation) {
    return activation == Activation::FiniteClip
        ? config & ~neLutEnable.mask() : config;
}

// Header word 1 is an activation mode for finite clips, not a geometry cost.
// The accurate linked forms use descriptor mode for three-tap kernels and
// mode zero for one-tap kernels.  ReLU and no-post-operation tasks retain the
// scheduler word supplied by their emitter.
constexpr std::uint32_t applyTaskActivation(std::uint32_t schedulerWord,
                                            bool threeTapKernel,
                                            Activation activation) {
    if (activation != Activation::FiniteClip) return schedulerWord;
    return (schedulerWord & ~taskMode.mask()) |
           (threeTapKernel ? taskModeKernelClipDescriptor.value(1) : 0u);
}

constexpr std::uint32_t sourceDmaConfig(bool,
                                        std::uint32_t task,
                                        bool scratch,
                                        bool dependency = true) {
    const auto role = scratch ? dmaRoleScratchSurface.value(1)
                              : dmaRoleLinearSurface.value(1);
    const auto extension = dmaExtensionCoordinates.value(1) |
                           dmaExtensionStrides.value(1) |
                           dmaExtensionTask.value(1);
    return dmaEnable.value(1) | role | extension | dmaTaskTag.value(task) |
           (scratch && dependency ? dmaScratchDependency.value(1) : 0u);
}

constexpr std::uint32_t sourceDmaExtension(bool programInput) {
    return dmaExtensionSource.value(1) |
           (programInput ? dmaExtensionProgramInput.value(1) : 0u) |
           dmaExtensionLowCoordinates.value(1) |
           dmaExtensionLowStrides.value(1) |
           dmaExtensionLowTask.value(1);
}

constexpr std::uint32_t sourceDmaTile(bool programInput) {
    return dmaEnable.value(1) | dmaTileFp16.value(1) | dmaTileNchw.value(1) |
           dmaTilePairedPlane.value(1) |
           (programInput ? dmaTileProgramInput.value(1)
                         : dmaTileScratch.value(1));
}

constexpr std::uint32_t resultDmaConfig(bool scratch,
                                        bool stagedDirect,
                                        std::uint32_t task) {
    const auto role = dmaRoleResultWrite.value(1) |
        dmaRoleLinearSurface.value(1) |
        (scratch ? dmaRoleScratchSurface.value(1) |
                       dmaRoleInterleavedResult.value(1) : 0u);
    const auto extension = dmaExtensionCoordinates.value(1) |
                           dmaExtensionStrides.value(1) |
                           dmaExtensionTask.value(1);
    return dmaEnable.value(1) | role | extension | dmaTaskTag.value(task) |
           (scratch || !stagedDirect ? dmaExternalSurface.value(1) : 0u);
}

constexpr std::uint32_t resultDmaTile(bool scratch) {
    return dmaEnable.value(1) | dmaTileFp16.value(1) | dmaTileNchw.value(1) |
           dmaTilePairedPlane.value(1) |
           dmaTileDirect.value((std::uint32_t(1) << dmaTileDirect.width) - 1) |
           (scratch ? dmaTileScratch.value(1)
                    : dmaTileProgramInput.value(1));
}

constexpr std::uint32_t kernelDmaConfig(std::uint32_t nextTask,
                                        bool sparse, bool terminal) {
    return kernelNextTaskTag.value(nextTask) | kernelDescriptorMode.value(1) |
           kernelBaseMode.value(1) | (sparse ? kernelSparse.value(1) : 0u) |
           (terminal ? kernelTerminal.value(1) : 0u);
}

constexpr std::uint32_t kernelLaneConfig(std::uint32_t task) {
    return dmaTaskTag.value(task) | kernelLaneEnable.value(1) |
           kernelLaneCacheClass.value(1);
}

// The ignored result extension is deliberately recognizable in dumps.  It is
// a padding value, not a hardware field or semantic task payload.
inline constexpr std::uint32_t deadDestinationPadding = 0x0000deadu; // ignored destination-extension payload
// fp16 1.0, used in NE scale and clip-descriptor slots.
inline constexpr std::uint16_t fp16One = 0x3c00u; // IEEE fp16 1.0
// fp16 6.0, the MobileNet ReLU6 upper bound.
inline constexpr std::uint16_t fp16Six = 0x4600u; // IEEE fp16 6.0
// Clip descriptor's structural scale and enable fields.
inline constexpr std::uint16_t clipDescriptorScale = 0x0040u; // descriptor scale field
inline constexpr std::uint16_t clipDescriptorEnable = 0x0001u; // descriptor enable field
// A coefficient lane prepends one 128-byte clip descriptor: two lower/upper
// pairs lead it, followed by two unity scales and the structural controls.
inline constexpr std::size_t clipDescriptorBytes = 128;
inline constexpr std::size_t clipLower0Offset = 0;
inline constexpr std::size_t clipUpper0Offset = 2;
inline constexpr std::size_t clipLower1Offset = 4;
inline constexpr std::size_t clipUpper1Offset = 6;
inline constexpr std::size_t clipUnity0Offset = 74;
inline constexpr std::size_t clipUnity1Offset = 78;
inline constexpr std::size_t clipScaleOffset = 82;
inline constexpr std::size_t clipEnableOffset = 84;

} // namespace ane::h14::fields
