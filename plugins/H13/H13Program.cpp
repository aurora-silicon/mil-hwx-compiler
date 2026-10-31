#include "H13Program.h"
#include "../LayerGraph.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <stdexcept>

namespace ane::h13 {

namespace {

// Field values follow allbilly/ane e159e2d examples/elementwise.py and
// examples/gemm.py. Constants use the aligned 0x280 KDMA-base representation.

std::uint32_t loadLE32(const std::vector<std::uint8_t> &bytes,
                       std::size_t offset) {
    if (offset + sizeof(std::uint32_t) > bytes.size())
        throw std::invalid_argument("H13 task word is truncated");
    std::uint32_t value = 0;
    for (std::size_t byte = 0; byte != sizeof(std::uint32_t); ++byte)
        value |= static_cast<std::uint32_t>(bytes[offset + byte]) << (byte * 8);
    return value;
}

void storeLE32(std::vector<std::uint8_t> &bytes, std::size_t offset,
               std::uint32_t value) {
    for (std::size_t byte = 0; byte != sizeof(std::uint32_t); ++byte)
        bytes[offset + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
}

/// Byte offset of one 32-bit register word inside a decoded H13 task
/// image: ten header words, one extra word when header[9] bit 1 is set,
/// then register records headed by `((count - 1) << 26) | byte address`.
/// False when the image does not carry that register.
bool findH13Register(const std::vector<std::uint8_t> &task, std::size_t offset,
                     std::size_t bytes, std::uint32_t address,
                     std::size_t &word) {
    if (offset + 40 > task.size() || bytes < 40 || offset + bytes > task.size())
        throw std::invalid_argument("H13 task image is truncated");
    const std::size_t extra =
        (loadLE32(task, offset + 36) & 3) == 3 ? sizeof(std::uint32_t) : 0;
    std::size_t cursor = offset + 40 + extra;
    const std::size_t end = offset + bytes;
    while (cursor + sizeof(std::uint32_t) <= end) {
        const auto header = loadLE32(task, cursor);
        const auto count = (header >> 26) + 1;
        const auto base = header & 0x03ffffff;
        if (address >= base && address < base + count * sizeof(std::uint32_t)) {
            if ((address - base) % sizeof(std::uint32_t))
                return false;
            word = cursor + sizeof(std::uint32_t) + (address - base);
            return word + sizeof(std::uint32_t) <= end;
        }
        cursor += sizeof(std::uint32_t) + count * sizeof(std::uint32_t);
    }
    return false;
}

std::size_t h13RegisterOffset(const std::vector<std::uint8_t> &task,
                              std::size_t offset, std::size_t bytes,
                              std::uint32_t address) {
    std::size_t word = 0;
    if (!findH13Register(task, offset, bytes, address, word))
        throw std::invalid_argument(
            "h13.chain-unrepresentable-edge: task image does not carry the "
            "decoded register record");
    return word;
}

/// Channel 3 is the scratch surface. A task selects it in the same three
/// five-bit fields `bindTasks` rebinds, and each of those roles has its
/// own tile-DMA descriptor pairing a base offset with a surface depth.
struct ScratchSlot {
    unsigned selectorShift;
    std::uint32_t baseRegister;
    std::uint32_t depthRegister;
};
constexpr ScratchSlot scratchSlots[] = {{0, 0x13808, 0x13814},
                                        {6, 0x1381c, 0x13828},
                                        {12, 0x17804, 0x17810}};
constexpr std::uint32_t scratchChannel = 3;

/// The highest scratch byte one task's channel-3 descriptors reach. The
/// captures record no scratch size, so this is how the extent is
/// recovered; it reproduces Apple's own `__DATA/__bss` gap below the
/// resource addresses on every decoded boolean capture.
std::uint64_t taskScratchExtent(const std::vector<std::uint8_t> &task,
                                std::size_t offset, std::size_t bytes) {
    const auto selectors = loadLE32(task, offset + 32);
    std::uint64_t extent = 0;
    for (const auto &slot : scratchSlots) {
        if (((selectors >> slot.selectorShift) & 31) != scratchChannel)
            continue;
        std::size_t word = 0;
        std::uint64_t reach = 0;
        if (findH13Register(task, offset, bytes, slot.baseRegister, word))
            reach = loadLE32(task, word);
        if (findH13Register(task, offset, bytes, slot.depthRegister, word))
            reach += loadLE32(task, word);
        extent = std::max(extent, reach);
    }
    return extent;
}

namespace reg {
constexpr std::size_t taskWord0 = 0x00;
constexpr std::size_t executionCycles = 0x08;
constexpr std::size_t debugLogEvents = 0x10;
constexpr std::size_t flags = 0x18;
constexpr std::size_t baseEnable = 0x20;
constexpr std::size_t taskWord9 = 0x24;
constexpr std::size_t kernelDMA = 0x28;
constexpr std::size_t firmwareDMA = 0x2c;
constexpr std::size_t commonStream = 0x124;
constexpr std::size_t inputDimensions = 0x128;
constexpr std::size_t commonPad0 = 0x12c;
constexpr std::size_t channelConfig = 0x130;
constexpr std::size_t inputChannels = 0x134;
constexpr std::size_t outputChannels = 0x138;
constexpr std::size_t outputDimensions = 0x13c;
constexpr std::size_t commonPad1 = 0x140;
constexpr std::size_t convolutionConfig = 0x144;
constexpr std::size_t commonPad2 = 0x148;
constexpr std::size_t groupConvolutionConfig = 0x14c;
constexpr std::size_t tileConfig = 0x150;
constexpr std::size_t commonPad3 = 0x154;
constexpr std::size_t pipelineConfig = 0x15c;
constexpr std::size_t taskInfo = 0x160;
constexpr std::size_t sourceStream = 0x168;
constexpr std::size_t sourceDMAConfig = 0x16c;
constexpr std::size_t sourceDMAPad0 = 0x170;
constexpr std::size_t sourceRowStride = 0x178;
constexpr std::size_t sourcePlaneStride = 0x17c;
constexpr std::size_t sourceDepthStride = 0x180;
constexpr std::size_t sourcePad2 = 0x18c;
constexpr std::size_t sourcePad3 = 0x190;
constexpr std::size_t sourcePad4 = 0x194;
constexpr std::size_t sourceFormat = 0x1a4;
constexpr std::size_t sourcePad8 = 0x1a8;
constexpr std::size_t sourcePadStream = 0x1ac;
constexpr std::size_t l2Stream = 0x1dc;
constexpr std::size_t l2SourceConfig = 0x1e4;
constexpr std::size_t l2SourceChannelStride = 0x1ec;
constexpr std::size_t l2SourceRowStride = 0x1f0;
constexpr std::size_t l2Pad0 = 0x1f4;
constexpr std::size_t l2Pad1 = 0x1f8;
constexpr std::size_t l2Pad2 = 0x1fc;
constexpr std::size_t l2Pad3 = 0x200;
constexpr std::size_t l2Pad4 = 0x204;
constexpr std::size_t l2Pad5 = 0x208;
constexpr std::size_t l2Pad6 = 0x20c;
constexpr std::size_t l2ResultConfig = 0x210;
constexpr std::size_t l2ResultBase = 0x214;
constexpr std::size_t convolutionResultChannelStride = 0x218;
constexpr std::size_t convolutionResultRowStride = 0x21c;
constexpr std::size_t l2ResultPad0 = 0x220;
constexpr std::size_t l2ResultPad1 = 0x224;
constexpr std::size_t processingElementStream = 0x228;
constexpr std::size_t processingElementConfig = 0x22c;
constexpr std::size_t biasScale = 0x230;
constexpr std::size_t preScale = 0x234;
constexpr std::size_t finalScale = 0x238;
constexpr std::size_t neuralEngineStream = 0x23c;
constexpr std::size_t kernelConfig = 0x240;
constexpr std::size_t multiplyAccumulateConfig = 0x244;
constexpr std::size_t postScale = 0x250;
constexpr std::size_t destinationStream = 0x254;
constexpr std::size_t destinationDMAConfig = 0x258;
constexpr std::size_t destinationRowStride = 0x260;
constexpr std::size_t destinationPlaneStride = 0x264;
constexpr std::size_t destinationDepthStride = 0x268;
constexpr std::size_t destinationFormat = 0x270;
} // namespace reg

constexpr std::size_t outputRows = 512;
constexpr std::uint32_t matvecInputChannels = 512; // channels matvecTask moves from surface channel 5, whatever the reduction
constexpr std::size_t rowsPerBlock = 32;
constexpr std::size_t blockBytes = 0x8000;
constexpr std::size_t blockHalfwords = blockBytes / 2;
constexpr std::size_t dmaBlocks = outputRows / rowsPerBlock;
constexpr std::uint32_t halfOne = 0x3c00;

constexpr std::uint32_t streamHeader(std::uint32_t address,
                                     std::uint32_t words) {
    return ((words - 1) << 26) | address;
}

void putLE32(std::vector<std::uint8_t> &bytes, std::size_t offset,
             std::uint32_t value) {
    for (std::size_t i = 0; i != 4; ++i)
        bytes[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
}

void putBE32(std::vector<std::uint8_t> &bytes, std::size_t offset,
             std::uint32_t value) {
    for (std::size_t i = 0; i != 4; ++i)
        bytes[offset + i] = static_cast<std::uint8_t>(value >> (8 * (3 - i)));
}

void putTaskHeader(std::vector<std::uint8_t> &task, std::uint32_t baseEnable,
                   std::uint32_t word9) {
    putLE32(task, reg::taskWord0, (0x40u << 16) | (1u << 25));
    putLE32(task, reg::executionCycles, 1058);
    putLE32(task, reg::debugLogEvents, 0x00fff86a);
    putLE32(task, reg::flags, (38u << 10) | (3u << 28));
    putLE32(task, reg::baseEnable, baseEnable);
    putLE32(task, reg::taskWord9, word9);
    putLE32(task, reg::kernelDMA, streamHeader(0x1f800, 62));
}

void putStreamHeaders(std::vector<std::uint8_t> &task) {
    putLE32(task, reg::commonStream, streamHeader(0x00000, 16));
    putLE32(task, reg::sourceStream, streamHeader(0x13800, 28));
    putLE32(task, reg::l2Stream, streamHeader(0x04800, 18));
    putLE32(task, reg::processingElementStream, streamHeader(0x08800, 4));
    putLE32(task, reg::neuralEngineStream, streamHeader(0x0c800, 5));
    putLE32(task, reg::destinationStream, streamHeader(0x17800, 7));
}

void putFirmwareDMA(std::vector<std::uint8_t> &task) {
    std::size_t offset = reg::firmwareDMA;
    putBE32(task, offset, 0x40000000);
    offset += 8;
    for (std::size_t i = 0; i != dmaBlocks; ++i, offset += 4)
        putBE32(task, offset, 0x81000000);
    for (std::uint32_t i = 0; i != dmaBlocks; ++i, offset += 4)
        putLE32(task, offset, i * blockBytes);
    for (std::size_t i = 0; i != dmaBlocks; ++i, offset += 4)
        putLE32(task, offset, blockBytes);
    for (std::size_t i = 0; i != 4; ++i, offset += 4)
        putBE32(task, offset, 0x80000000);
}

TensorLayout tensor(std::uint32_t index, std::uint64_t channels,
                    std::uint64_t allocationBytes) {
    return {index, {1, channels, 1, 1, 64, 64}, allocationBytes};
}

std::vector<std::uint8_t> binaryTask(std::uint32_t operation) {
    std::vector<std::uint8_t> task(taskBytes, 0);
    putTaskHeader(task,
                  6u | (1u << 5) | (5u << 6) | (1u << 11) | (4u << 12) |
                      (1u << 17),
                  0);
    putStreamHeaders(task);

    putLE32(task, reg::inputDimensions, (1u << 16) | 1u);
    putLE32(task, reg::commonPad0, 1);
    putLE32(task, reg::channelConfig, 2u | (2u << 2) | (2u << 4));
    putLE32(task, reg::inputChannels, 64);
    putLE32(task, reg::outputChannels, 64);
    putLE32(task, reg::outputDimensions, (1u << 16) | 1u);
    putLE32(task, reg::commonPad1, 1);
    putLE32(task, reg::convolutionConfig,
            1u | (1u << 5) | (1u << 13) | (1u << 15) | (1u << 28) |
                (1u << 30));
    putLE32(task, reg::commonPad2, 0x2041);
    putLE32(task, reg::groupConvolutionConfig, 1u | (1u << 16));
    putLE32(task, reg::tileConfig, 1);
    putLE32(task, reg::commonPad3, 4);
    putLE32(task, reg::pipelineConfig, 3u | (6u << 3));

    putLE32(task, reg::sourceDMAConfig,
            1u | (8u << 4) | (8u << 8) | (3u << 12) | (3u << 16));
    putLE32(task, reg::sourceDMAPad0, 0x33880);
    putLE32(task, reg::sourceRowStride, 0x40);
    putLE32(task, reg::sourcePlaneStride, 0x40);
    putLE32(task, reg::sourceDepthStride, 0x1000);
    putLE32(task, reg::sourcePad2, 0x40);
    putLE32(task, reg::sourcePad3, 0x40);
    putLE32(task, reg::sourcePad4, 0x1000);
    putLE32(task, reg::sourceFormat,
            1u | (3u << 4) | (2u << 12) | (1u << 24));
    putLE32(task, reg::sourcePad8, 0x2030);

    putLE32(task, reg::l2SourceConfig,
            2u | (1u << 4) | (1u << 5) | (1u << 6) | (1u << 8) |
                (1u << 20) | (1u << 22) | (1u << 24));
    putLE32(task, reg::l2SourceChannelStride, 0x10);
    putLE32(task, reg::l2SourceRowStride, 0x420);
    putLE32(task, reg::l2Pad0, 0x400);
    putLE32(task, reg::l2Pad1, 0x400);
    putLE32(task, reg::l2Pad2, 0x440);
    putLE32(task, reg::l2Pad3, 0x10);
    putLE32(task, reg::l2Pad4, 0x420);
    putLE32(task, reg::l2Pad5, 0x400);
    putLE32(task, reg::l2Pad6, 0x400);
    putLE32(task, reg::l2ResultConfig,
            2u | (2u << 2) | (1u << 4) | (1u << 5) | (1u << 6) |
                (1u << 8) | (1u << 20) | (1u << 22));
    putLE32(task, reg::l2ResultBase, 0x860);

    putLE32(task, reg::processingElementConfig,
            (2u << 18) | (operation << 2));
    putLE32(task, reg::biasScale, halfOne << 16);
    putLE32(task, reg::preScale, halfOne << 16);
    putLE32(task, reg::finalScale, 0x3f800000);
    if (operation == 1)
        putLE32(task, reg::multiplyAccumulateConfig, 0x30);

    putLE32(task, reg::destinationDMAConfig,
            1u | (12u << 4) | (1u << 26));
    putLE32(task, reg::destinationRowStride, 0x40);
    putLE32(task, reg::destinationPlaneStride, 0x40);
    putLE32(task, reg::destinationDepthStride, 0x1000);
    putLE32(task, reg::destinationFormat,
            1u | (3u << 4) | (2u << 12) | (1u << 24));
    return task;
}

std::vector<std::uint8_t> matvecTask() {
    std::vector<std::uint8_t> task(taskBytes, 0);
    putTaskHeader(task,
                  5u | (1u << 5) | (36u << 12) | (1u << 24) |
                      (1u << 26),
                  0x21);
    putFirmwareDMA(task);
    putStreamHeaders(task);

    putLE32(task, reg::inputDimensions, (1u << 16) | 1u);
    putLE32(task, reg::commonPad0, 1);
    putLE32(task, reg::channelConfig, 2u | (2u << 4));
    putLE32(task, reg::inputChannels, matvecInputChannels);
    putLE32(task, reg::outputChannels, 512);
    putLE32(task, reg::outputDimensions, (1u << 16) | 1u);
    putLE32(task, reg::commonPad1, 1);
    putLE32(task, reg::convolutionConfig, 0x5000b421);
    putLE32(task, reg::commonPad2, 0x2041);
    putLE32(task, reg::groupConvolutionConfig, 0x00010001);
    putLE32(task, reg::tileConfig, 1);
    putLE32(task, reg::pipelineConfig, 0x00244405);
    putLE32(task, reg::taskInfo, 1u << 20);

    putLE32(task, reg::sourceDMAConfig,
            1u | (8u << 4) | (8u << 8) | (3u << 12) | (3u << 16));
    putLE32(task, reg::sourceDMAPad0, 0x8880);
    putLE32(task, reg::sourceRowStride, 0x40);
    putLE32(task, reg::sourcePlaneStride, 0x40);
    putLE32(task, reg::sourceDepthStride, 0x8000);
    putLE32(task, reg::sourceFormat,
            1u | (3u << 4) | (2u << 12) | (1u << 24));
    putLE32(task, reg::sourcePadStream, 0x100);

    putLE32(task, reg::l2SourceConfig, 0x00500172);
    putLE32(task, reg::l2SourceChannelStride, 0x10);
    putLE32(task, reg::l2SourceRowStride, 0x2030);
    putLE32(task, reg::l2Pad0, 0x2000);
    putLE32(task, reg::l2Pad1, 0x2000);
    putLE32(task, reg::l2ResultConfig, 0x00500172);
    putLE32(task, reg::l2ResultBase, 0x2030);
    putLE32(task, reg::convolutionResultChannelStride, 0x10);
    putLE32(task, reg::convolutionResultRowStride, 0x2020);
    putLE32(task, reg::l2ResultPad0, 0x2000);
    putLE32(task, reg::l2ResultPad1, 0x2000);

    putLE32(task, reg::kernelConfig, 0x82);
    putLE32(task, reg::multiplyAccumulateConfig, 0x00101c00);
    putLE32(task, reg::postScale, halfOne);

    putLE32(task, reg::destinationDMAConfig, 1u | (12u << 4));
    putLE32(task, reg::destinationRowStride, 0x40);
    putLE32(task, reg::destinationPlaneStride, 0x40);
    putLE32(task, reg::destinationDepthStride, 0x8000);
    putLE32(task, reg::destinationFormat,
            1u | (3u << 4) | (2u << 12) | (3u << 20) | (1u << 24));
    return task;
}

std::vector<std::uint8_t> packWeights(std::uint32_t reduction,
                                      const std::uint8_t *weights, bool transposeY) {
    std::vector<std::uint8_t> packed(dmaBlocks * blockBytes, 0);
    const std::size_t outputStride = transposeY ? reduction : 1;
    const std::size_t inputStride = transposeY ? 1 : outputRows;
    for (std::size_t block = 0; block != dmaBlocks; ++block) {
        const auto blockOffset = block * blockHalfwords;
        for (std::size_t column = 0; column != reduction; ++column) {
            const auto columnOffset = blockOffset + column * rowsPerBlock;
            for (std::size_t row = 0; row != rowsPerBlock; ++row) {
                const auto source = ((block * rowsPerBlock + row) * outputStride + column * inputStride) * 2;
                const auto destination = (columnOffset + row) * 2;
                packed[destination] = weights[source];
                packed[destination + 1] = weights[source + 1];
            }
        }
    }
    return packed;
}

enum class ElementwiseKind : std::uint8_t {
    BinaryRuntime,
    BinaryScalar,
    Unary,
    /// Apple's constant-blob twin of the runtime binary form: one runtime
    /// surface plus a second source the task reads straight from the
    /// constant section (source 2 names the kernel BAR channel). Decoded
    /// from the pinned exported fixtures.
    BinaryConstant,
};

struct OracleTaskTemplate {
    ElementwiseKind kind;
    std::uint8_t operation;
    ElementwiseShape shape;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantBytes;
};

struct OracleMatvecTemplate {
    std::uint32_t rows;
    std::uint32_t reduction;
    std::uint32_t columns;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::uint64_t scratchAllocationBytes;
};

/// Which LUT layout a decoded normalization program's constant section holds.
/// The generator derives this from the recorded section hash, so the encoder
/// rebuilds the same bytes instead of guessing them from the operation.
enum class NormConstants : std::uint8_t {
    Zero,
    Exponential,
    ExponentialReciprocal,
};

struct OracleNormTemplate {
    NormOperation operation;
    ElementwiseShape input;
    ElementwiseShape output;
    std::uint32_t axisMask;
    bool keepDims;
    NormConstants constants;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
};

/// One decoded Apple convolution program, keyed by everything the compiler
/// knows before it picks one: the kernel, stride, group count, whether a bias
/// is folded in, and the two CHW surfaces. The padding follows from those, so
/// a 1x1 convolution's `same` and `valid` spellings share a template.
struct OracleConvTemplate {
    std::uint32_t kernel;
    std::uint32_t kernelWidth;
    std::uint32_t stride;
    std::uint32_t groups;
    bool bias;
    ElementwiseShape input;
    ElementwiseShape output;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
    std::uint32_t padTop;
    std::uint32_t padLeft;
    bool nativeLayer;
};

/// One decoded Apple matmul program, keyed by everything the compiler knows
/// before it picks a program: the geometry, both transpose flags, and whether
/// the second operand is runtime.
struct OracleMatmulTemplate {
    std::uint32_t rows;
    std::uint32_t reduction;
    std::uint32_t columns;
    bool transposeX;
    bool transposeY;
    bool runtimeWeight;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
};

struct OracleBroadcastTemplate {
    std::uint8_t operation;
    BroadcastOperand operand;
    BatchedShape x;
    BatchedShape y;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
};

struct OracleUnaryTask {
    UnaryOperation operation;
    ElementwiseShape shape;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
};

#include "H13ElementwiseTemplates.inc"
#include "H13ElementwiseConstants.inc"
#include "H13EncoderUnaryTemplates.inc"
#include "H13MatvecTemplates.inc"

/// One decoded Apple rank-3 linear program: the geometry, how the bias rides
/// (absent/uniform folds to a scalar register, distinct halves ride as
/// appended per-column bias tiles), the task stream, and for block modes the
/// verified column-tile layout of the constant section — each tile holds one
/// raw bias group, `group` columns cut reduction-outer, zero padded.
struct H13LinearTileGroup {
    std::uint32_t group;
    std::uint32_t count;
    std::uint32_t strideBytes;
};

struct OracleLinearTemplate {
    std::uint32_t rows;
    std::uint32_t reduction;
    std::uint32_t columns;
    LinearBiasMode biasMode;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
    const H13LinearTileGroup *tiles;
    std::size_t tileGroupCount;
};

/// One decoded Apple FFN-chain program: the whole five-operation
/// matmul → bias add → silu → matmul → bias add spell as one 28-task stream
/// with a two-stage constant section (stage tiles may repeat the 128-byte
/// kernel header at their head).
struct OracleFFNChainTemplate {
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
};

/// One decoded Apple transpose program: a whole-surface permutation carried
/// as one strided copy task. The shapes are the Apple-normalized elementwise
/// triples of the MIL input and result; the surfaces are plain
/// `elementwiseTensor` layouts whose rows differ per side.
struct OracleTransposeTemplate {
    ElementwiseShape input;
    ElementwiseShape output;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
};

/// One decoded Apple slice_by_index program: a whole-surface strided
/// row-copy carried as one task. The shapes are the Apple-normalized
/// elementwise triples of the MIL input and result; the surfaces are plain
/// `elementwiseTensor` layouts whose rows differ per side.
struct OracleSliceTemplate {
    ElementwiseShape input;
    ElementwiseShape output;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
};

#include "H13TransposeTemplates.inc"
#include "H13SliceTemplates.inc"

#include "H13LinearTemplates.inc"
#include "H13NormTemplates.inc"
#include "H13EnvelopeTemplates.inc"
#include "H13ConvTemplates.inc"
#include "H13BatchedMatmulTemplates.inc"
#include "H13BooleanTemplates.inc"
#include "H13TileTemplates.inc"

bool sameShape(ElementwiseShape left, ElementwiseShape right) {
    return left.channels == right.channels && left.height == right.height &&
           left.width == right.width;
}

const OracleTaskTemplate *elementwiseTemplate(ElementwiseKind kind,
                                               std::uint8_t operation,
                                               ElementwiseShape shape) {
    for (const auto &candidate : kElementwiseTasks)
        if (candidate.kind == kind && candidate.operation == operation &&
            sameShape(candidate.shape, shape)) return &candidate;
    return nullptr;
}

/// The decoded rank-3 encoder spell: Apple normalizes [1, A, B] to the
/// [1, 1, A, B] surface, so the lookup key is that canonical triple.
const OracleUnaryTask *encoderUnaryTemplate(UnaryOperation operation,
                                             ElementwiseShape shape) {
    for (const auto &candidate : kEncoderUnaryTasks)
        if (candidate.operation == operation &&
            sameShape(candidate.shape, shape)) return &candidate;
    return nullptr;
}

/// The decoded matmul program for a geometry: the envelope tables first, which
/// carry both transpose flags, then the first campaign's `transpose_y=true`
/// constant-weight table, which only ever compiled untransposed x.
const OracleMatmulTemplate *matmulTemplate(MatmulShape shape) {
    for (const auto &candidate : kMatmulEnvelopeTasks)
        if (candidate.rows == shape.rows &&
            candidate.reduction == shape.reduction &&
            candidate.columns == shape.columns &&
            candidate.transposeX == shape.transposeX &&
            candidate.transposeY == shape.transposeY &&
            candidate.runtimeWeight == shape.runtimeWeight) return &candidate;
    return nullptr;
}

const OracleMatvecTemplate *matvecTemplate(MatmulShape shape) {
    if (shape.transposeX || !shape.transposeY || shape.runtimeWeight)
        return nullptr;
    for (const auto &candidate : kMatvecTasks)
        if (candidate.rows == shape.rows &&
            candidate.reduction == shape.reduction &&
            candidate.columns == shape.columns) return &candidate;
    return nullptr;
}

bool sameShape(BatchedShape left, BatchedShape right) {
    return left.batch == right.batch && left.channels == right.channels &&
           left.height == right.height && left.width == right.width;
}

const OracleBroadcastTemplate *broadcastTemplate(BinaryOperation operation,
                                                 BroadcastOperand operand,
                                                 BroadcastShape shape) {
    for (const auto &candidate : kBroadcastTasks)
        if (candidate.operation == static_cast<std::uint8_t>(operation) &&
            candidate.operand == operand && sameShape(candidate.x, shape.x) &&
            (operand == BroadcastOperand::Scalar ||
             sameShape(candidate.y, shape.y))) return &candidate;
    return nullptr;
}

const OracleNormTemplate *normTemplate(NormOperation operation,
                                       NormShape shape) {
    for (const auto &candidate : kNormTasks)
        if (candidate.operation == operation &&
            candidate.axisMask == shape.axisMask &&
            candidate.keepDims == shape.keepDims &&
            sameShape(candidate.input, shape.input) &&
            sameShape(candidate.output, shape.output)) return &candidate;
    return nullptr;
}

/// The decoded 1-task transpose program for this surface pair; the encoder
/// spell swaps one non-unit pair and keeps every other axis in place.
const OracleTransposeTemplate *transposeTemplate(ElementwiseShape input,
                                                 ElementwiseShape output) {
    for (const auto &candidate : kTransposeTasks)
        if (sameShape(candidate.input, input) &&
            sameShape(candidate.output, output)) return &candidate;
    return nullptr;
}

/// The decoded 1-task slice_by_index program for this surface pair; the
/// encoder spell narrows only the last axis, from element 0.
const OracleSliceTemplate *sliceTemplate(ElementwiseShape input,
                                         ElementwiseShape output) {
    for (const auto &candidate : kSliceTasks)
        if (sameShape(candidate.input, input) &&
            sameShape(candidate.output, output)) return &candidate;
    return nullptr;
}

std::vector<std::uint8_t> taskBytesFor(const std::uint32_t *words,
                                       std::size_t wordCount) {
    std::vector<std::uint8_t> bytes(wordCount * sizeof(std::uint32_t));
    for (std::size_t index = 0; index != wordCount; ++index) {
        const auto value = words[index];
        for (std::size_t byte = 0; byte != sizeof(value); ++byte)
            bytes[index * sizeof(value) + byte] =
                static_cast<std::uint8_t>(value >> (byte * 8));
    }
    return bytes;
}

std::uint64_t alignTile(std::uint64_t value) {
    return (value + tileBytes - 1) & ~(static_cast<std::uint64_t>(tileBytes) - 1);
}

/// Apple's elementwise surface: 64-byte padded rows, and an allocation that
/// covers every batch element even though the descriptor's declared size
/// covers only one.
TensorLayout elementwiseTensor(std::uint32_t index, BatchedShape shape) {
    // Apple pads every elementwise/norm row to the 64-byte DMA stride, not
    // merely up to it: a width of 375 takes a 768-byte row, which the
    // encoder-geometry oracles record. Widths below 32 elements keep the
    // 64-byte floor.
    const std::uint64_t row = (shape.width * 2 + 63) / 64 * 64;
    const std::uint64_t plane = row * shape.height;
    const std::uint64_t element = plane * shape.channels;
    return {index,
            {shape.batch, shape.channels, shape.height, shape.width, plane, row},
            alignTile(element * shape.batch)};
}

TensorLayout elementwiseTensor(std::uint32_t index, ElementwiseShape shape) {
    return elementwiseTensor(index, BatchedShape{1, shape.channels,
                                                 shape.height, shape.width});
}

/// The decoded elementwise TDs address their surfaces through one flat window
/// of 2·elements bytes (their DMA counts are the whole tensor, not one row),
/// so staging maps logical element i to tile byte 2·i: a single wide row.
/// The 64-byte row floor keeps sub-row scalars on Apple's padded surface.
TensorLayout flatElementwiseTensor(std::uint32_t index, ElementwiseShape shape) {
    const std::uint64_t elements = static_cast<std::uint64_t>(shape.channels) *
                                   shape.height * shape.width;
    const std::uint64_t row = std::max<std::uint64_t>(64, elements * 2);
    return {index, {1, 1, 1, elements, row, row}, alignTile(row)};
}

/// Apple's matvec surface: one dense [1, 1, rows, width] fp16 plane whose row
/// stride is the logical row, padded to the 64-byte DMA stride every H13
/// surface uses.
TensorLayout matvecTensor(std::uint32_t index, std::uint32_t rows,
                          std::uint32_t width) {
    const std::uint64_t row = (width * 2 + 63) / 64 * 64;
    const std::uint64_t plane = row * rows;
    return {index, {1, 1, rows, width, plane, row}, alignTile(plane)};
}

/// Writes a decoded LUT block, little-endian, at a byte offset in a section.
template <std::size_t N>
void putWords(std::vector<std::uint8_t> &bytes, std::size_t offset,
              const std::uint32_t (&words)[N]) {
    if (offset + N * sizeof(std::uint32_t) > bytes.size())
        throw std::logic_error("H13 constant table exceeds its decoded section");
    for (std::size_t index = 0; index != N; ++index)
        for (std::size_t byte = 0; byte != sizeof(std::uint32_t); ++byte)
            bytes[offset + index * sizeof(std::uint32_t) + byte] =
                static_cast<std::uint8_t>(words[index] >> (byte * 8));
}

template <std::size_t N>
std::vector<std::uint8_t> paddedConstants(const std::uint32_t (&words)[N],
                                          std::size_t size) {
    std::vector<std::uint8_t> bytes(size, 0);
    putWords(bytes, 0, words);
    return bytes;
}

void putHalf(std::vector<std::uint8_t> &bytes, std::size_t index,
             std::uint16_t value) {
    bytes.at(index * 2) = static_cast<std::uint8_t>(value);
    bytes.at(index * 2 + 1) = static_cast<std::uint8_t>(value >> 8);
}

std::vector<std::uint8_t> leakyReluConstants(std::size_t size) {
    std::vector<std::uint8_t> bytes(size, 0);
    putHalf(bytes, 0, 0xfc00); putHalf(bytes, 1, 0x7c00);
    putHalf(bytes, 2, 0xfc00); putHalf(bytes, 3, 0x7c00);
    putHalf(bytes, 37, 0x3000); putHalf(bytes, 39, 0x3c00);
    putHalf(bytes, 41, 0x0040); putHalf(bytes, 42, 0x0001);
    return bytes;
}

/// The exact fp16 reciprocal of a power-of-two divisor, the only real_div
/// divisor class H13 lowers without rounding.
std::uint16_t exactReciprocal(std::uint16_t bits) {
    const std::uint16_t exponent = (bits >> 10) & 0x1f;
    if (!exponent || exponent == 0x1f || (bits & 0x03ff))
        throw std::invalid_argument("H13 real_div requires a power-of-two fp16 divisor");
    const std::uint16_t sign = bits & 0x8000;
    return exponent == 30 ? static_cast<std::uint16_t>(sign | 0x0200)
                          : static_cast<std::uint16_t>(sign | ((30 - exponent) << 10));
}

std::vector<std::uint8_t> scalarConstants(BinaryOperation operation,
                                          std::uint16_t value,
                                          std::size_t size,
                                          std::size_t elements) {
    std::vector<std::uint8_t> bytes(size, 0);
    if (operation == BinaryOperation::Maximum) {
        for (std::size_t index = 0; index != 37; ++index) putHalf(bytes, index, value);
        putHalf(bytes, 1, 0x7c00); putHalf(bytes, 3, 0x7c00);
        putHalf(bytes, 37, 0x3c00); putHalf(bytes, 39, 0x3c00);
        putHalf(bytes, 41, 0x0040); putHalf(bytes, 42, 0x0001);
    } else if (operation == BinaryOperation::Minimum) {
        putHalf(bytes, 0, 0xfc00); putHalf(bytes, 1, value);
        putHalf(bytes, 2, 0xfc00); putHalf(bytes, 3, value);
        putHalf(bytes, 37, 0x3c00); putHalf(bytes, 39, 0x3c00);
        putHalf(bytes, 41, 0x0040); putHalf(bytes, 42, 0x0001);
    } else if (operation == BinaryOperation::RealDivide) {
        const auto reciprocal = exactReciprocal(value);
        for (std::size_t index = 0; index != elements; ++index)
            putHalf(bytes, elements + index, reciprocal);
    }
    return bytes;
}

std::vector<std::uint8_t> unaryConstants(UnaryOperation operation,
                                         std::size_t size) {
    switch (operation) {
    case UnaryOperation::Absolute:
    case UnaryOperation::Relu:
        return std::vector<std::uint8_t>(size, 0);
    case UnaryOperation::Exponential:
        return paddedConstants(kExpKERNWords, size);
    case UnaryOperation::Gelu:
        return paddedConstants(kGeluKERNWords, size);
    case UnaryOperation::LeakyRelu:
        return leakyReluConstants(size);
    case UnaryOperation::ReciprocalSquareRoot:
        return paddedConstants(kRsqrtKERNWords, size);
    case UnaryOperation::Sigmoid:
        return paddedConstants(kSigmoidKERNWords, size);
    case UnaryOperation::Silu:
        return paddedConstants(kSiluKERNWords, size);
    case UnaryOperation::SquareRoot:
        return paddedConstants(kSqrtKERNWords, size);
    case UnaryOperation::Tanh:
        return paddedConstants(kTanhKERNWords, size);
    }
    throw std::invalid_argument("unsupported H13 unary operation");
}

/// Apple's softmax sections hold the exponential table at offset 0 and, when
/// the reduced axis is not the last one, the reciprocal table in the final
/// 128 bytes; layer_norm and every reduction leave the section zero.
std::vector<std::uint8_t> normConstants(NormConstants kind, std::size_t size) {
    if (kind == NormConstants::Zero)
        return std::vector<std::uint8_t>(size, 0);
    auto bytes = paddedConstants(kExpKERNWords, size);
    if (kind == NormConstants::ExponentialReciprocal)
        putWords(bytes, size - sizeof(kRecipKERNWords), kRecipKERNWords);
    return bytes;
}

/// Apple's per-channel section: one fp16 bias per channel, then one fp16
/// scale per channel. `add` stores the constant as the bias and scales by
/// 1.0; `mul` leaves the bias zero and stores the constant as the scale.
std::vector<std::uint8_t> perChannelConstants(BinaryOperation operation,
                                              std::uint32_t channels,
                                              const std::uint8_t *constant,
                                              std::size_t size) {
    const std::size_t block = static_cast<std::size_t>(channels) * 2;
    if (size != block * 2)
        throw std::logic_error(
            "H13 per-channel section is not a bias and a scale block");
    std::vector<std::uint8_t> bytes(size, 0);
    if (operation == BinaryOperation::Add) {
        std::memcpy(bytes.data(), constant, block);
        for (std::uint32_t index = 0; index != channels; ++index)
            putHalf(bytes, channels + index, halfOne);
    } else if (operation == BinaryOperation::Multiply) {
        std::memcpy(bytes.data() + block, constant, block);
    } else {
        throw std::invalid_argument(
            "H13 per-channel constants cover add and mul");
    }
    return bytes;
}

/// The broadcast result surface: every axis is the larger of the two
/// operands', and a scalar or per-channel constant leaves `x` unchanged.
BatchedShape broadcastOutput(BroadcastOperand operand, BroadcastShape shape) {
    if (operand != BroadcastOperand::Runtime) return shape.x;
    return {std::max(shape.x.batch, shape.y.batch),
            std::max(shape.x.channels, shape.y.channels),
            std::max(shape.x.height, shape.y.height),
            std::max(shape.x.width, shape.y.width)};
}

Program oracleProgram(const OracleTaskTemplate &source,
                      std::vector<std::uint8_t> constants,
                      std::size_t inputCount) {
    std::vector<TensorLayout> inputs;
    inputs.reserve(inputCount);
    for (std::size_t index = 0; index != inputCount; ++index)
        inputs.push_back(elementwiseTensor(static_cast<std::uint32_t>(5 + index),
                                           source.shape));
    const auto constantOffsetBytes =
        (source.wordCount * sizeof(std::uint32_t) + 0x3f) & ~std::size_t(0x3f);
    Program program{taskBytesFor(source.words, source.wordCount), std::move(constants),
                    std::move(inputs), elementwiseTensor(4, source.shape),
                    source.firstTaskBytes, source.taskCount, constantOffsetBytes, {}};
    program.taskSurfaceChannels = {5, 4, 6, 7};
    return program;
}

} // namespace

Program encodeBinary(BinaryOperation operation) {
    std::uint32_t operationCode;
    switch (operation) {
    case BinaryOperation::Add:
        operationCode = 0;
        break;
    case BinaryOperation::Multiply:
        operationCode = 1;
        break;
    case BinaryOperation::Maximum:
        operationCode = 2;
        break;
    case BinaryOperation::Minimum:
        operationCode = 3;
        break;
    default:
        throw std::invalid_argument("unsupported H13 binary operation");
    }
    return {binaryTask(operationCode),
            {},
            {tensor(5, 64, tileBytes), tensor(6, 64, tileBytes)},
            tensor(4, 64, tileBytes), taskBytes, 1, constantOffset, {}};
}

bool supportsElementwise(BinaryOperation operation, ElementwiseShape shape,
                         bool scalarConstant) {
    const auto kind = scalarConstant ? ElementwiseKind::BinaryScalar
                                     : ElementwiseKind::BinaryRuntime;
    return elementwiseTemplate(kind, static_cast<std::uint8_t>(operation), shape);
}

bool supportsElementwise(UnaryOperation operation, ElementwiseShape shape) {
    return elementwiseTemplate(ElementwiseKind::Unary,
                               static_cast<std::uint8_t>(operation), shape) ||
           encoderUnaryTemplate(operation, shape);
}

bool supportsElementwiseConstant(BinaryOperation operation,
                                 ElementwiseShape shape) {
    return elementwiseTemplate(ElementwiseKind::BinaryConstant,
                               static_cast<std::uint8_t>(operation), shape);
}
Program encodeElementwise(BinaryOperation operation, ElementwiseShape shape,
                          bool scalarConstant, std::uint16_t scalarBits) {
    const auto kind = scalarConstant ? ElementwiseKind::BinaryScalar
                                     : ElementwiseKind::BinaryRuntime;
    const auto *source = elementwiseTemplate(
        kind, static_cast<std::uint8_t>(operation), shape);
    if (!source)
        throw std::invalid_argument("H13 binary operation is outside the decoded parity envelope");
    if (scalarConstant && scalarBits != 0x3800)
        throw std::invalid_argument("H13 scalar operation requires the decoded fp16 0.5 operand");
    const auto constants = scalarConstant
        ? scalarConstants(operation, scalarBits, source->constantBytes,
                          static_cast<std::size_t>(shape.channels) * shape.height *
                              shape.width)
        : std::vector<std::uint8_t>(source->constantBytes, 0);
    return oracleProgram(*source, constants, scalarConstant ? 1 : 2);
}

Program encodeElementwiseConstant(BinaryOperation operation,
                                  ElementwiseShape shape,
                                  const std::uint8_t *constant,
                                  std::size_t constantBytes) {
    const auto *source = elementwiseTemplate(
        ElementwiseKind::BinaryConstant,
        static_cast<std::uint8_t>(operation), shape);
    if (!source)
        throw std::invalid_argument(
            "H13 constant-blob binary is outside the decoded parity envelope");
    const auto elements = static_cast<std::size_t>(shape.channels) *
                          shape.height * shape.width;
    if (!constant || constantBytes != elements * 2)
        throw std::invalid_argument(
            "H13 constant-blob binary needs the whole fp16 blob");
    if (source->constantBytes != constantBytes)
        throw std::logic_error("H13 constant-blob constant size mismatch");
    // The decoded constant-blob TDs cover one 1024-byte span; staging the
    // surface one-element-per-row reaches only the first 16 elements on
    // (M1 test host, 2026-09-15). Pack the tensor contiguously instead: the
    // TD's whole span then covers every element.
    auto program = oracleProgram(
        *source, std::vector<std::uint8_t>(constant, constant + constantBytes),
        1);
    program.inputs.front() = flatElementwiseTensor(5, shape);
    program.output = flatElementwiseTensor(4, shape);
    return program;
}
Program encodeElementwise(UnaryOperation operation, ElementwiseShape shape) {
    const auto *source = elementwiseTemplate(
        ElementwiseKind::Unary, static_cast<std::uint8_t>(operation), shape);
    if (source)
        return oracleProgram(*source, unaryConstants(operation, source->constantBytes), 1);
    const auto *encoder = encoderUnaryTemplate(operation, shape);
    if (!encoder)
        throw std::invalid_argument("H13 unary operation is outside the decoded parity envelope");
    Program program;
    program.taskSurfaceChannels = {5, 4, 6, 7};
    program.task = taskBytesFor(encoder->words, encoder->wordCount);
    program.constants = unaryConstants(operation, encoder->constantBytes);
    program.inputs = {elementwiseTensor(5, shape)};
    program.output = elementwiseTensor(4, shape);
    program.firstTaskBytes = encoder->firstTaskBytes;
    program.taskCount = encoder->taskCount;
    program.constantOffsetBytes = encoder->constantOffsetBytes;
    program.scratchAllocationBytes = encoder->scratchAllocationBytes;
    return program;
}

bool supportsTransposeParity(ElementwiseShape input, ElementwiseShape output) {
    return transposeTemplate(input, output) != nullptr;
}

Program encodeTransposeParity(ElementwiseShape input, ElementwiseShape output) {
    const auto *source = transposeTemplate(input, output);
    if (!source)
        throw std::invalid_argument(
            "H13 transpose is outside the decoded parity envelope");
    Program program;
    program.taskSurfaceChannels = {5, 4, 6, 7};
    program.task = taskBytesFor(source->words, source->wordCount);
    program.constants = std::vector<std::uint8_t>(source->constantBytes, 0);
    program.inputs = {elementwiseTensor(5, input)};
    program.output = elementwiseTensor(4, output);
    program.firstTaskBytes = source->firstTaskBytes;
    program.taskCount = source->taskCount;
    program.constantOffsetBytes = source->constantOffsetBytes;
    program.scratchAllocationBytes = source->scratchAllocationBytes;
    return program;
}

bool supportsSliceParity(ElementwiseShape input, ElementwiseShape output) {
    return sliceTemplate(input, output) != nullptr;
}

Program encodeSliceParity(ElementwiseShape input, ElementwiseShape output) {
    const auto *source = sliceTemplate(input, output);
    if (!source)
        throw std::invalid_argument(
            "H13 slice_by_index is outside the decoded parity envelope");
    Program program;
    program.taskSurfaceChannels = {5, 4, 6, 7};
    program.task = taskBytesFor(source->words, source->wordCount);
    program.constants = std::vector<std::uint8_t>(source->constantBytes, 0);
    program.inputs = {elementwiseTensor(5, input)};
    program.output = elementwiseTensor(4, output);
    program.firstTaskBytes = source->firstTaskBytes;
    program.taskCount = source->taskCount;
    program.constantOffsetBytes = source->constantOffsetBytes;
    program.scratchAllocationBytes = source->scratchAllocationBytes;
    return program;
}

Program encodeMatvec(std::uint32_t reduction, const std::uint8_t *weights,
                     std::size_t weightBytes, bool transposeY) {
    if (reduction != 256 && reduction != 512)
        throw std::invalid_argument("H13 matvec reduction must be 256 or 512");
    if (weightBytes != outputRows * reduction * 2)
        throw std::invalid_argument("H13 matvec requires 512 * reduction fp16 weights");
    if (!weights)
        throw std::invalid_argument("H13 matvec weights must not be null");
    std::vector<std::size_t> relocations(16);
    for (std::size_t index = 0; index != relocations.size(); ++index)
        relocations[index] = 0x74 + index * sizeof(std::uint32_t);
    return {matvecTask(),
            packWeights(reduction, weights, transposeY),
            // The task always moves 512 input channels (matvecTask: 32768 bytes from channel 5), so the physical surface is 512 channels at
            // both reductions. At reduction 256 the logical tensor is the first 256 channels and the padding lanes are zero.
            {tensor(5, matvecInputChannels, matvecInputChannels * 64)},
            tensor(4, outputRows, 0x8000),
            taskBytes,
            1,
            constantOffset,
            std::move(relocations)};
}

bool supportsMatmulParity(MatmulShape shape) {
    return matmulTemplate(shape) || matvecTemplate(shape);
}

/// Apple's interleave itself, shared by the matmul and rank-3 linear
/// encoders; the matmul path adds the rank-2 geometry gate on top.
static std::vector<std::uint8_t> packMatvecInterleave(
    std::uint32_t reduction, std::uint32_t columns, std::uint32_t rows,
    const std::uint8_t *weights, std::size_t weightBytes) {
    // The row count (Apple partitions above 128 x rows) gates the group cap.
    std::uint32_t group = std::min<std::uint32_t>(16, columns / 16);
    if (rows > 128)
        group = std::min<std::uint32_t>(group, 32768 / reduction);
    group = std::max<std::uint32_t>(1, group);
    const std::uint32_t groups = columns / group;
    std::vector<std::uint8_t> packed(weightBytes);
    for (std::uint32_t column = 0; column != columns; ++column) {
        const std::uint32_t plane = column / group;
        const std::uint32_t destinationPlane =
            (plane % 16) * (groups / 16) + plane / 16;
        std::size_t destination =
            (static_cast<std::size_t>(destinationPlane) * reduction *
                 group + column % group) * 2;
        std::size_t source = static_cast<std::size_t>(column) * reduction * 2;
        for (std::uint32_t index = 0; index != reduction; ++index) {
            packed[destination] = weights[source];
            packed[destination + 1] = weights[source + 1];
            destination += static_cast<std::size_t>(group) * 2;
            source += 2;
        }
    }
    return packed;
}

std::vector<std::uint8_t> packMatvecWeights(MatmulShape shape,
                                            const std::uint8_t *weights,
                                            std::size_t weightBytes) {
    if (!weights)
        throw std::invalid_argument("H13 matvec weights must not be null");
    if (!shape.reduction || shape.columns < 16 || shape.columns % 16 ||
        (shape.columns > 256 && shape.columns % 256))
        throw std::invalid_argument(
            "H13 matvec packing needs a positive reduction and 16 columns per "
            "row group, 256 per plane group above 256 columns");
    if (weightBytes !=
        static_cast<std::size_t>(shape.reduction) * shape.columns * 2)
        throw std::invalid_argument(
            "H13 matvec requires columns * reduction fp16 weights");
    return packMatvecInterleave(shape.reduction, shape.columns, shape.rows,
                                weights, weightBytes);
}

Program encodeMatmulParity(MatmulShape shape, const std::uint8_t *weights,
                           std::size_t weightBytes) {
    const auto *envelope = matmulTemplate(shape);
    const auto *legacy = envelope ? nullptr : matvecTemplate(shape);
    if (!envelope && !legacy)
        throw std::invalid_argument(
            "H13 matmul geometry is outside the decoded parity envelope");
    Program program;
    // Apple's matmul objects lay the output surface out first, then the second
    // operand, then x, and declare the operands in the opposite order.
    program.outputBindingIndex = 0;
    if (shape.runtimeWeight) {
        if (weights)
            throw std::invalid_argument(
                "H13 runtime-operand matmul takes no constant weight");
        program.constants.assign(envelope->constantBytes, 0);
        program.inputs = {
            shape.transposeY ? matvecTensor(5, shape.columns, shape.reduction)
                             : matvecTensor(5, shape.reduction, shape.columns),
            shape.transposeX ? matvecTensor(6, shape.reduction, shape.rows)
                             : matvecTensor(6, shape.rows, shape.reduction)};
    } else {
        program.constants = packMatvecWeights(shape, weights, weightBytes);
        program.inputs = {
            shape.transposeX ? matvecTensor(5, shape.reduction, shape.rows)
                             : matvecTensor(5, shape.rows, shape.reduction)};
    }
    program.output = matvecTensor(4, shape.rows, shape.columns);
    program.task = envelope
        ? taskBytesFor(envelope->words, envelope->wordCount)
        : taskBytesFor(legacy->words, legacy->wordCount);
    program.firstTaskBytes = envelope ? envelope->firstTaskBytes
                                      : legacy->firstTaskBytes;
    program.taskCount = envelope ? envelope->taskCount : legacy->taskCount;
    program.constantOffsetBytes = envelope ? envelope->constantOffsetBytes
                                           : legacy->constantOffsetBytes;
    program.scratchAllocationBytes = envelope
        ? envelope->scratchAllocationBytes : legacy->scratchAllocationBytes;
    return program;
}

/// The decoded rank-3 linear program for a geometry and bias mode: the
/// none/uniform forms pack weights exactly like the matvec table; the block
/// form appends the captured per-column bias tiles.
const OracleLinearTemplate *linearTemplate(std::uint32_t rows,
                                           std::uint32_t reduction,
                                           std::uint32_t columns,
                                           LinearBiasMode biasMode) {
    for (const auto &candidate : kLinearTasks)
        if (candidate.rows == rows && candidate.reduction == reduction &&
            candidate.columns == columns &&
            candidate.biasMode == biasMode)
            return &candidate;
    return nullptr;
}

bool supportsLinearParity(std::uint32_t rows, std::uint32_t reduction,
                          std::uint32_t columns, LinearBiasMode biasMode) {
    return linearTemplate(rows, reduction, columns, biasMode) != nullptr;
}

std::vector<std::uint8_t> packLinearBlockTiles(
    const OracleLinearTemplate &source, const std::uint8_t *weights,
    const std::uint8_t *bias) {
    std::vector<std::uint8_t> packed(source.constantBytes, 0);
    const auto *half = reinterpret_cast<const std::uint16_t *>(weights);
    const auto *biasHalf = reinterpret_cast<const std::uint16_t *>(bias);
    std::size_t at = 0;
    std::uint32_t column = 0;
    for (std::size_t index = 0; index != source.tileGroupCount; ++index) {
        const auto &group = source.tiles[index];
        for (std::uint32_t tile = 0; tile != group.count; ++tile) {
            std::memcpy(packed.data() + at, biasHalf + column,
                        group.group * 2);
            std::size_t cursor = at + group.group * 2;
            for (std::uint32_t red = 0; red != source.reduction; ++red)
                for (std::uint32_t c = 0; c != group.group; ++c) {
                    std::memcpy(packed.data() + cursor,
                                half + static_cast<std::size_t>(column + c) *
                                           source.reduction + red,
                                2);
                    cursor += 2;
                }
            // The stride's remainder stays zero, exactly as decoded.
            at += group.strideBytes;
            column += group.group;
        }
    }
    if (at != packed.size() || column != source.columns)
        throw std::logic_error("H13 linear tile model misses the section");
    return packed;
}

Program encodeLinearParity(std::uint32_t rows, std::uint32_t reduction,
                           std::uint32_t columns, LinearBiasMode biasMode,
                           const std::uint8_t *weights,
                           std::size_t weightBytes, const std::uint8_t *bias,
                           std::size_t biasBytes,
                           std::uint16_t uniformBiasHalves) {
    const auto *source = linearTemplate(rows, reduction, columns, biasMode);
    if (!source)
        throw std::invalid_argument(
            "H13 rank-3 linear geometry is outside the decoded parity "
            "envelope");
    if (weightBytes != static_cast<std::size_t>(columns) * reduction * 2)
        throw std::invalid_argument(
            "H13 linear requires columns * reduction fp16 weights");
    Program program;
    if (biasMode == LinearBiasMode::Block) {
        if (!bias || biasBytes != columns * 2)
            throw std::invalid_argument(
                "H13 block-mode linear requires one fp16 bias per column");
        program.constants = packLinearBlockTiles(*source, weights, bias);
    } else {
        if (bias && biasBytes)
            throw std::invalid_argument(
                "H13 folded linear takes no constant bias block");
        // H13 reads each 16-output-feature group at a permutation index
        // not equal to its identity index. The decode measured it
        // directly for the 64-plane (375, 1024, 1024) case: for output
        // group g = j / 16 the engine reads packer plane index
        //   P(g) = ((g & 0xF) << 1) | ((g >> 4) & 1) | ((g >> 5) << 5)
        // so the inverse (group g lives at packer plane P where
        // P(g) = P) is
        //   g = ((P >> 5) << 5) | ((P & 1) << 4) | ((P >> 1) & 0xF).
        // The within-plane layout already matches the engine's read
        // stride (lane=16, col=1) because the inner-loop ordering
        // (red outer, c inner) writes data[c + L*16] within the plane;
        // only the plane slot for each group is wrong. Re-mix by placing
        // group g_inv(P) at packer plane P. The oracles were uniform-
        // payload captures where every half is identical, so this bug
        // was structurally invisible to byte-parity vs the oracles.
        // The smaller geometries (n128, n640, n4096) have not been
        // measured on device; keep their identity packing so the parity
        // gates against the captured oracles keep passing.
        // 2026-09-17 CRT (T6001 test host, receipts/2026-09-17-ffn-mm2-plane.md):
        // the (375, 1024, 4096) uniform linear reads the SAME inverse
        // permutation, generalized to 256 planes of 16384 halves
        // (plane P holds group g_inv(P), lane stride 16, no wrap at
        // k=1024) — the g_inv formula below is bit-exact for 256
        // planes too. The (375, 4096, 1024) uniform linear instead
        // reads groups at IDENTITY plane order with a per-group 8/8
        // column split: outputs c<8 live in plane 2g at lane stride 8,
        // outputs c>=8 in plane 2g+1 at 8L + (c-8). Other geometries
        // are still unmeasured on device and keep identity packing so
        // the parity gates against the captured oracles keep passing.
        const std::uint32_t group = std::min<std::uint32_t>(16, columns);
        const auto *half = reinterpret_cast<const std::uint16_t *>(weights);
        const bool mm1Permutation =
            group == 16 && reduction == 1024 && columns == 1024;
        const bool mm1Permutation256 =
            group == 16 && reduction == 1024 && columns == 4096;
        const bool mm2ColumnSplit =
            group == 16 && reduction == 4096 && columns == 1024;
        if (mm2ColumnSplit) {
            program.constants.assign(weightBytes, 0);
            for (std::uint32_t g = 0; g != columns / group; ++g) {
                for (std::uint32_t half2 = 0; half2 != 2; ++half2) {
                    std::size_t at = (2 * g + half2) *
                        static_cast<std::size_t>(group / 2) * reduction * 2;
                    for (std::uint32_t red = 0; red != reduction; ++red)
                        for (std::uint32_t c = 0; c != group / 2; ++c) {
                            std::memcpy(
                                program.constants.data() + at,
                                half + (g * group + half2 * (group / 2) + c) *
                                           static_cast<std::size_t>(reduction) + red,
                                2);
                            at += 2;
                        }
                }
            }
        } else {
            program.constants.assign(weightBytes, 0);
            const std::size_t planeBytes =
                static_cast<std::size_t>(group) * reduction * 2;
            for (std::uint32_t plane = 0; plane != columns / group; ++plane) {
                const std::uint32_t g =
                    (mm1Permutation || mm1Permutation256)
                    ? (((plane >> 5) << 5) | ((plane & 1) << 4) |
                       ((plane >> 1) & 0xF))
                    : plane;
                std::size_t at = plane * planeBytes;
                for (std::uint32_t red = 0; red != reduction; ++red)
                    for (std::uint32_t c = 0; c != group; ++c) {
                        std::memcpy(program.constants.data() + at,
                                    half + (g * group + c) *
                                               static_cast<std::size_t>(reduction) + red,
                                    2);
                        at += 2;
                    }
            }
        }
    }
    if (program.constants.size() != source->constantBytes)
        throw std::logic_error(
            "H13 packed linear section differs from the decoded size");
    program.task = taskBytesFor(source->words, source->wordCount);
    if (biasMode == LinearBiasMode::Uniform) {
        // The decoded task templates carry the captured uniform-bias
        // immediate as 0x00003401 in the NE scalar-register word
        // (one occurrence per task in the linear; kLinearTask1..kLinearTask5
        // all stamp the same pattern). Replace with the real bias to
        // avoid +0.2502441 silently offsetting every output. The
        // uniform-payload oracles baked 0x3401 in because every half of
        // their constant section was 0x3400; with no captured byte-
        // parity reference for a different bias, the gate previously
        // compared a uniform-bias output against itself.
        constexpr std::uint8_t needle[2] = {0x01, 0x34};
        const std::uint8_t replacement[2] = {
            static_cast<std::uint8_t>(uniformBiasHalves & 0xFF),
            static_cast<std::uint8_t>(uniformBiasHalves >> 8)};
        std::size_t replacements = 0;
        std::size_t scan = 0;
        while (scan + 1 < program.task.size()) {
            if (program.task[scan] == needle[0] &&
                program.task[scan + 1] == needle[1]) {
                program.task[scan + 0] = replacement[0];
                program.task[scan + 1] = replacement[1];
                replacements++;
                scan += 2;
            } else {
                scan++;
            }
        }
        if (replacements == 0)
            throw std::logic_error(
                "H13 uniform linear template carries no 0x3401 bias "
                "immediate; uniformBiasHalves stamping has nothing to "
                "replace");
    }
    // The decoded stream binds the input surface at channel 4 and the
    // output at 5 (input-first); the manifest must name the same channels
    // or the strict bundle gate refuses the mismatch.
    program.inputs = {matvecTensor(4, rows, reduction)};
    program.output = matvecTensor(5, rows, columns);
    program.firstTaskBytes = source->firstTaskBytes;
    program.taskCount = source->taskCount;
    program.constantOffsetBytes = source->constantOffsetBytes;
    program.scratchAllocationBytes = source->scratchAllocationBytes;
    return program;
}

bool supportsFFNChain(std::uint32_t rows, std::uint32_t inner1,
                      std::uint32_t middle, std::uint32_t inner2,
                      std::uint32_t columns) {
    return rows == 375 && inner1 == 1024 && middle == 4096 &&
           inner2 == 4096 && columns == 1024;
}

Program encodeFFNChain(std::uint32_t rows, std::uint32_t inner1,
                       std::uint32_t middle, std::uint32_t inner2,
                       std::uint32_t columns, const std::uint8_t *weights1,
                       const std::uint8_t *bias1, const std::uint8_t *weights2,
                       const std::uint8_t *bias2) {
    if (!supportsFFNChain(rows, inner1, middle, inner2, columns))
        throw std::invalid_argument(
            "H13 FFN chain geometry is outside the decoded parity envelope");
    const auto *source = kFFNChainTasks;
    if (!weights1 || !bias1 || !weights2 || !bias2)
        throw std::invalid_argument(
            "H13 FFN chain requires all four constants");
    // Stage one repeats the decoded 128-byte kernel header at every tile's
    // head; stage two tiles carry no header. Each tile holds one raw bias
    // group, then its weight columns cut reduction-outer; the stride's
    // remainder stays zero.
    Program program;
    program.constants.assign(source->constantBytes, 0);
    const auto *w1 = reinterpret_cast<const std::uint16_t *>(weights1);
    const auto *b1 = reinterpret_cast<const std::uint16_t *>(bias1);
    const auto *w2 = reinterpret_cast<const std::uint16_t *>(weights2);
    const auto *b2 = reinterpret_cast<const std::uint16_t *>(bias2);
    std::size_t at = 0;
    std::uint32_t column = 0;
    for (const auto &group : kFFNChainMM1Tiles) {
        for (std::uint32_t tile = 0; tile != group.count; ++tile) {
            std::memcpy(program.constants.data() + at,
                        kFFNChainKernelHeader, sizeof(kFFNChainKernelHeader));
            std::size_t cursor = at + sizeof(kFFNChainKernelHeader);
            std::memcpy(program.constants.data() + cursor,
                        b1 + column, group.group * 2);
            cursor += group.group * 2;
            for (std::uint32_t red = 0; red != inner1; ++red)
                for (std::uint32_t c = 0; c != group.group; ++c) {
                    std::memcpy(program.constants.data() + cursor,
                                w1 + static_cast<std::size_t>(column + c) *
                                         inner1 + red,
                                2);
                    cursor += 2;
                }
            at += group.strideBytes;
            column += group.group;
        }
    }
    column = 0;
    for (const auto &group : kFFNChainMM2Tiles) {
        for (std::uint32_t tile = 0; tile != group.count; ++tile) {
            std::memcpy(program.constants.data() + at,
                        b2 + column, group.group * 2);
            std::size_t cursor = at + group.group * 2;
            for (std::uint32_t red = 0; red != middle; ++red)
                for (std::uint32_t c = 0; c != group.group; ++c) {
                    std::memcpy(program.constants.data() + cursor,
                                w2 + static_cast<std::size_t>(column + c) *
                                         middle + red,
                                2);
                    cursor += 2;
                }
            at += group.strideBytes;
            column += group.group;
        }
    }
    if (at != source->constantBytes || column != columns)
        throw std::logic_error("H13 FFN chain tile model misses the section");
    program.task = taskBytesFor(source->words, source->wordCount);
    // Input-first decoded binding, mirroring encodeLinearParity: the
    // manifest names the channels the task stream actually selects.
    program.inputs = {matvecTensor(4, rows, inner1)};
    program.output = matvecTensor(5, rows, columns);
    program.firstTaskBytes = source->firstTaskBytes;
    program.taskCount = source->taskCount;
    program.constantOffsetBytes = source->constantOffsetBytes;
    program.scratchAllocationBytes = source->scratchAllocationBytes;
    return program;
}

const OracleBatchedMatmulTemplate *batchedTemplate(BatchedMatmulShape shape) {
    for (const auto &candidate : kBatchedTasks)
        if (candidate.rows == shape.rows &&
            candidate.reduction == shape.reduction &&
            candidate.columns == shape.columns &&
            candidate.transposeX == shape.transposeX &&
            candidate.transposeY == shape.transposeY &&
            candidate.batch == shape.batch &&
            candidate.storage == (shape.runtimeWeight
                                      ? BatchedWeight::Runtime
                                      : BatchedWeight::Packed))
            return &candidate;
    return nullptr;
}

bool supportsBatchedMatmul(BatchedMatmulShape shape) {
    return batchedTemplate(shape) != nullptr;
}

/// The batched surface: one [1, B, rows, width] descriptor whose rows pad
/// to the 64-byte stride every H13 surface uses.
TensorLayout batchedTensor(std::uint32_t index, std::uint32_t batch,
                           std::uint32_t rows, std::uint32_t width) {
    const std::uint64_t row = (width * 2 + 63) / 64 * 64;
    const std::uint64_t plane = row * rows;
    return {index,
            {1, batch, rows, width, plane, row},
            alignTile(plane * batch)};
}

std::vector<std::uint8_t> packBatchedWeights(BatchedMatmulShape shape,
                                             const std::uint8_t *weights,
                                             std::size_t weightBytes) {
    const auto *source = batchedTemplate(shape);
    if (!source || source->storage != BatchedWeight::Packed)
        throw std::invalid_argument(
            "H13 batched packing needs a decoded packed template");
    if (weightBytes != static_cast<std::size_t>(source->batch) *
            source->packRows * source->packCols * 2)
        throw std::invalid_argument(
            "H13 batched weight must be the dense B-plane blob");
    if (source->identityPacking) {
        // The head-projection capture stores the weight blob verbatim.
        return std::vector<std::uint8_t>(weights, weights + weightBytes);
    }
    const std::uint32_t padded = (source->packCols + 31) / 32 * 32;
    std::vector<std::uint8_t> packed(source->constantBytes, 0);
    constexpr std::size_t packHeaderBytes = 128;
    constexpr std::size_t packPhaseHalfwords = 64;
    std::memcpy(packed.data(), source->packHeader, packHeaderBytes);
    const std::uint32_t rows = source->batch * source->packRows;
    for (std::uint32_t row = 0; row < rows; ++row) {
        std::size_t lo, at;
        if (row == 0) {
            lo = 0;
            at = packHeaderBytes;
        } else {
            lo = static_cast<std::size_t>(row) * source->packCols -
                packPhaseHalfwords;
            at = static_cast<std::size_t>(row) * padded * 2;
        }
        const std::size_t count = source->packCols -
            (row == 0 ? packPhaseHalfwords : 0);
        std::memcpy(packed.data() + at, weights + lo * 2, count * 2);
    }
    return packed;
}

/// The batched-matmul staging tasks read both runtime surfaces, but Apple's
/// decoded runtime template names only source 1: the source-2 selector keeps
/// channel 0 and the source-2 DMA configuration word keeps the disabled
/// encoding, even though those same tasks carry the source-2 transfer
/// registers. A task stream that under-names the surfaces its header
/// declares falls back to the positional channel map, which the strict
/// bundle gate refuses, so stamp the second source onto the descriptors
/// that read it: selector bits 6:10 name channel 6 and the configuration
/// word takes the live encoding Apple's own two-source tasks carry.
void nameSecondSource(std::vector<std::uint8_t> &task,
                      std::size_t firstTaskBytes, std::uint32_t taskCount) {
    constexpr std::uint32_t sourceOneConfig = 0x13800;
    constexpr std::uint32_t sourceTwoConfig = 0x13804;
    constexpr std::uint32_t disabled = 0x00008880;
    constexpr std::uint32_t live = 0x00033880;
    std::size_t offset = 0, bytes = firstTaskBytes;
    for (std::uint32_t index = 0; index != taskCount; ++index) {
        if (bytes < 40 || offset > task.size() || bytes > task.size() - offset)
            throw std::logic_error("H13 batched task is truncated");
        std::size_t word = 0;
        if (findH13Register(task, offset, bytes, sourceOneConfig, word) &&
            loadLE32(task, word) != disabled) {
            const std::size_t selector = offset + 32;
            storeLE32(task, selector,
                      (loadLE32(task, selector) & ~(31u << 6)) | (6u << 6));
            if (!findH13Register(task, offset, bytes, sourceTwoConfig, word))
                throw std::logic_error(
                    "H13 batched staging task carries no source-2 record");
            storeLE32(task, word, live);
        }
        const auto next = loadLE32(task, offset + 28);
        if (!next) break;
        bytes = (((loadLE32(task, offset + 4) >> 16) & 0x1ff) + 1) * 4;
        offset = next;
    }
}

Program encodeBatchedMatmul(BatchedMatmulShape shape,
                            const std::uint8_t *packed,
                            std::size_t packedBytes) {
    constexpr std::size_t tasksPerBatch = 26;
    const auto *source = batchedTemplate(shape);
    if (!source)
        throw std::invalid_argument(
            "H13 batched matmul is outside the decoded envelope");
    if (source->storage == BatchedWeight::Runtime) {
        if (packed || packedBytes)
            throw std::invalid_argument(
                "H13 runtime-operand batched matmul takes no constant weight");
    } else if (!packed || packedBytes != source->constantBytes) {
        throw std::invalid_argument(
            "H13 batched constant weight must be packed for the geometry");
    }
    if (source->groupWordCount % tasksPerBatch)
        throw std::logic_error(
            "H13 batched template group is not a whole task set");
    const std::size_t taskWords = source->groupWordCount / tasksPerBatch;
    Program program;
    // Emit the optional prefix task (the fold-flag forms), then stamp batch
    // zero's task group once per batch with the per-batch word rules,
    // following each task's patched link pointer for placement; the gaps
    // between tasks stay zero, exactly as decoded.
    std::size_t offset = 0;
    auto emit = [&](const std::uint32_t *words, std::size_t count) {
        if (program.task.size() < offset + count * 4)
            program.task.resize(offset + count * 4);
        std::memcpy(program.task.data() + offset, words, count * 4);
        offset = words[7];
    };
    if (source->prefixWordCount) {
        emit(source->prefixWords, source->prefixWordCount);
        if (!offset)
            throw std::logic_error(
                "H13 batched prefix does not link into the stream");
    }
    for (std::uint32_t batch = 0; batch < source->batch; ++batch) {
        for (std::size_t index = 0; index < tasksPerBatch; ++index) {
            std::vector<std::uint32_t> words(
                source->groupWords + index * taskWords,
                source->groupWords + (index + 1) * taskWords);
            for (std::size_t rule = 0; rule < source->taskRuleCount; ++rule) {
                const auto &taskRule = source->taskRules[rule];
                if (taskRule.task != index) continue;
                for (std::size_t word = 0; word < taskRule.ruleCount; ++word) {
                    const auto &rule = taskRule.rules[word];
                    if (rule.word >= words.size())
                        throw std::logic_error(
                            "H13 batched template patch is outside its task");
                    if (rule.literalCount) {
                        if (batch)
                            words[rule.word] = rule.literals[batch - 1];
                    } else {
                        words[rule.word] =
                            static_cast<std::uint32_t>(
                                static_cast<std::int64_t>(words[rule.word]) +
                                static_cast<std::int64_t>(batch) *
                                    rule.delta);
                    }
                }
                break;
            }
            if (batch == source->batch - 1 && index + 1 == tasksPerBatch &&
                words[7])
                throw std::logic_error(
                    "H13 batched template final task does not end the stream");
            emit(words.data(), words.size());
        }
    }
    if (offset != 0)
        throw std::logic_error(
            "H13 batched template link chain does not terminate");
    program.scratchAllocationBytes = source->scratchBytes;
    // The packed head-projection object binds its input surface first.
    program.outputBindingIndex = source->inputFirst ? 1 : 0;
    program.taskCount = static_cast<std::uint32_t>(
        tasksPerBatch * source->batch + (source->prefixWordCount ? 1 : 0));
    program.firstTaskBytes = (source->prefixWordCount
        ? source->prefixWordCount : taskWords) * 4;
    program.constants.assign(source->constantBytes, 0);
    if (packed)
        std::memcpy(program.constants.data(), packed, packedBytes);
    program.constantOffsetBytes = (program.task.size() + 127) / 128 * 128;
    if (shape.runtimeWeight) {
        program.inputs = {batchedTensor(5, shape.batch, shape.reduction,
                                        shape.columns),
                          batchedTensor(6, shape.batch, shape.rows,
                                        shape.reduction)};
    } else {
        program.inputs = {batchedTensor(5, shape.batch, shape.rows,
                                        shape.reduction)};
    }
    if (shape.runtimeWeight)
        nameSecondSource(program.task, program.firstTaskBytes,
                         program.taskCount);
    program.output = batchedTensor(4, shape.batch, shape.rows, shape.columns);
    return program;
}

const H13BooleanTemplate *booleanTemplate(H13BooleanShape shape) {
    for (const auto &candidate : kBooleanTasks)
        if (candidate.kind == shape.kind &&
            candidate.constInput == shape.constInput &&
            candidate.channels == shape.channels &&
            candidate.height == shape.height &&
            candidate.width == shape.width)
            return &candidate;
    return nullptr;
}

bool supportsBooleanOp(H13BooleanShape shape) {
    return booleanTemplate(shape) != nullptr;
}

/// The boolean surface: [1, C, H, W] with the row padded to the 64-byte
/// stride over the family's element size — fp16 operands, bool results.
TensorLayout booleanTensor(std::uint32_t index, H13BooleanShape shape,
                           bool boolElements) {
    const std::uint32_t element = boolElements ? 1 : 2;
    const std::uint64_t row =
        (shape.width * element + 63) / 64 * 64;
    const std::uint64_t plane = row * shape.height;
    return {index,
            {1, shape.channels, shape.height, shape.width, plane, row},
            alignTile(plane * shape.channels), element};
}

Program encodeBooleanOp(H13BooleanShape shape,
                            const std::uint8_t *scalarInput,
                            std::size_t scalarBytes) {
    const auto *source = booleanTemplate(shape);
    if (!source)
        throw std::invalid_argument(
            "H13 boolean op is outside the decoded envelope");
    if (scalarInput && scalarBytes != 2)
        throw std::invalid_argument(
            "H13 boolean scalar input must be one fp16 lane");
    Program program;
    // Replay the captured words at their linked offsets: the generator
    // concatenates the tasks densely, so walk the first task's size words,
    // place, follow its link, and continue — the gaps stay zero.
    std::size_t cursor = 0;
    std::size_t offset = 0;
    // Channel 3 passes bindTasks through untouched (only channels 4..7
    // remap), so any template task selecting it stages through the
    // scratch surface and the program must back it — with the extent the
    // template's own channel-3 descriptors reach, not a guess.
    bool stagesThroughScratch = false;
    std::uint64_t scratchExtent = 0;
    for (std::size_t index = 0; index < source->taskCount; ++index) {
        // The decoded header's size field can disagree with the record
        // size the capture walked, so the template carries the record
        // sizes and the links carry the placement.
        const std::uint32_t sizeWords = source->taskWords[index];
        if (cursor + sizeWords > source->wordCount)
            throw std::logic_error("H13 boolean template is truncated");
        const std::uint32_t *words = source->words + cursor;
        if (program.task.size() < offset + sizeWords * 4)
            program.task.resize(offset + sizeWords * 4);
        std::memcpy(program.task.data() + offset, words, sizeWords * 4);
        const std::uint32_t selectors = words[8];
        for (unsigned shift : {0u, 6u, 12u})
            stagesThroughScratch =
                stagesThroughScratch || ((selectors >> shift) & 31) == 3;
        scratchExtent = std::max(
            scratchExtent,
            taskScratchExtent(program.task, offset, sizeWords * 4));
        const std::size_t next = words[7];
        cursor += sizeWords;
        if (index + 1 == source->taskCount) {
            if (next)
                throw std::logic_error(
                    "H13 boolean template final task links on");
            break;
        }
        if (next <= offset + sizeWords * 4 || next % 4)
            throw std::logic_error("H13 boolean template link is invalid");
        offset = next;
    }
    program.taskCount = static_cast<std::uint32_t>(source->taskCount);
    program.firstTaskBytes = source->taskWords[0] * 4;
    program.constants.assign(source->constants, source->constants +
                                               source->constantBytes);
    if (scalarInput && !program.constants.empty())
        std::memcpy(program.constants.data(), scalarInput, 2);
    program.constantOffsetBytes = (program.task.size() + 127) / 128 * 128;
    switch (shape.kind) {
    case H13BooleanKind::Less:
        program.inputs = {booleanTensor(5, shape, false),
                          booleanTensor(6, shape, false)};
        program.output = booleanTensor(4, shape, true);
        // Words select x on 4 and y on 5 with the result on 6: allocated
        // channel 4 is the output, 5 and 6 the operands.
        program.taskSurfaceChannels = {6, 4, 5, 7};
        break;
    case H13BooleanKind::Floor:
        // The blob-x twin reads its operand from the constant section
        // head, so it binds no runtime surface. The capture writes the
        // result through channel 5, so the blob twin's output takes the
        // first-input slot; the runtime-operand form keeps it on 4.
        program.inputs =
            shape.constInput
                ? std::vector<TensorLayout>{}
                : std::vector<TensorLayout>{booleanTensor(5, shape, false)};
        program.output =
            booleanTensor(shape.constInput ? 5 : 4, shape, false);
        // Words select x on 4 and the result on 5: with a runtime x the
        // remap lands x on slot 5 and the result on slot 4, while the
        // blob twin's write through template 4 lands on slot 5.
        program.taskSurfaceChannels = {5, 4, 6, 7};
        break;
    case H13BooleanKind::FloorDiv:
        program.inputs = shape.constInput
            ? std::vector<TensorLayout>{booleanTensor(5, shape, false)}
            : std::vector<TensorLayout>{booleanTensor(5, shape, false),
                                        booleanTensor(6, shape, false)};
        program.output = booleanTensor(4, shape, false);
        // Words select x on 4 and the result on 5 (the s2 divisor rides
        // the constant section; the rr form selects y like floor's input).
        program.taskSurfaceChannels = {5, 4, 6, 7};
        break;
    case H13BooleanKind::Select:
        program.inputs = {booleanTensor(5, shape, false),
                          booleanTensor(6, shape, false),
                          booleanTensor(7, shape, true)};
        program.output = booleanTensor(4, shape, false);
        // Words select cond-true on 4, cond-false on 5, bool cond on 6,
        // result on 7. MIL select(a, b, cond) is cond?a:b, so a binds to
        // 4. a-on-5 was inverted: {64,1,1} matched only after host invert.
        // 375-wide DMA row 384 is the 1-byte cond; row 768 is fp16.
        program.taskSurfaceChannels = {7, 4, 5, 6};
        break;
    }
    if (stagesThroughScratch) {
        // The scratch is an arena, not one surface: the 375-wide select
        // stages the expanded cond mask at 2256000 and one interleaved
        // blend half at 0 and 4560000, reaching 6816000 bytes — three
        // times the output allocation the earlier guess used. Under that
        // guess the cond-true half's write, and task 4's read of it, ran
        // past the end of the surface the ANEC header declares.
        if (!scratchExtent)
            throw std::logic_error(
                "H13 boolean task stages through the scratch surface with "
                "no tile-DMA descriptor to size it");
        program.scratchAllocationBytes = alignTile(scratchExtent);
    }
    return program;
}

const H13TileTemplate *tileTemplate(H13TileShape shape) {
    for (const auto &candidate : kTileTasks)
        if (candidate.inChannels == shape.inChannels &&
            candidate.inHeight == shape.inHeight &&
            candidate.inWidth == shape.inWidth &&
            candidate.repChannels == shape.repChannels &&
            candidate.repHeight == shape.repHeight &&
            candidate.repWidth == shape.repWidth &&
            candidate.constInput == !shape.runtimeInput)
            return &candidate;
    return nullptr;
}

bool supportsTileOp(H13TileShape shape) {
    return tileTemplate(shape) != nullptr;
}

Program encodeTileOp(H13TileShape shape) {
    const auto *source = tileTemplate(shape);
    if (!source)
        throw std::invalid_argument(
            "H13 tile is outside the decoded envelope");
    Program program;
    std::size_t cursor = 0;
    std::size_t offset = 0;
    for (std::size_t index = 0; index < source->taskCount; ++index) {
        const std::uint32_t sizeWords = source->taskWords[index];
        if (cursor + sizeWords > source->wordCount)
            throw std::logic_error("H13 tile template is truncated");
        if (program.task.size() < offset + sizeWords * 4)
            program.task.resize(offset + sizeWords * 4);
        std::memcpy(program.task.data() + offset,
                    source->words + cursor, sizeWords * 4);
        const std::size_t next =
            (source->words + cursor)[7];
        cursor += sizeWords;
        if (index + 1 == source->taskCount) {
            if (next)
                throw std::logic_error(
                    "H13 tile template final task links on");
            break;
        }
        if (next <= offset + sizeWords * 4 || next % 4)
            throw std::logic_error("H13 tile template link is invalid");
        offset = next;
    }
    program.taskCount = static_cast<std::uint32_t>(source->taskCount);
    program.firstTaskBytes = source->taskWords[0] * 4;
    program.constants.assign(source->constants, source->constants +
                                               source->constantBytes);
    program.constantOffsetBytes = (program.task.size() + 127) / 128 * 128;
    const std::uint32_t inChannels = source->inChannels == 1 &&
            shape.inChannels != 1 ? shape.inChannels : source->inChannels;
    program.inputs = shape.runtimeInput
        ? std::vector<TensorLayout>{booleanTensor(5, {H13BooleanKind::Floor, false,
                                                      inChannels, source->inHeight,
                                                      source->inWidth}, false)}
        : std::vector<TensorLayout>{};
    program.output = booleanTensor(shape.runtimeInput ? 4 : 5,
                                   {H13BooleanKind::Floor, false,
                                    source->outChannels, source->outHeight,
                                    source->outWidth}, false);
    // The runtime input selects 4 and the result 5; the remap lands the
    // input on slot 5 and the result on slot 4. The blob twin reads its
    // operand from the constant section and writes through template 4,
    // which lands on slot 5 — the same first-input slot the blob floor
    // twin uses.
    program.taskSurfaceChannels = {5, 4, 6, 7};
    return program;
}

bool supportsBroadcast(BinaryOperation operation, BroadcastOperand operand,
                       BroadcastShape shape) {
    return broadcastTemplate(operation, operand, shape);
}

Program encodeBroadcast(BinaryOperation operation, BroadcastOperand operand,
                        BroadcastShape shape, const std::uint8_t *constant,
                        std::size_t constantBytes, std::uint16_t scalarBits) {
    const auto *source = broadcastTemplate(operation, operand, shape);
    if (!source)
        throw std::invalid_argument(
            "H13 broadcast is outside the decoded parity envelope");
    if (operand == BroadcastOperand::Scalar && scalarBits != 0x3800)
        throw std::invalid_argument(
            "H13 scalar broadcast requires the decoded fp16 0.5 operand");
    Program program;
    program.taskSurfaceChannels = {5, 4, 6, 7};
    program.task = taskBytesFor(source->words, source->wordCount);
    if (operand == BroadcastOperand::Constant) {
        const std::size_t elements =
            static_cast<std::size_t>(shape.y.channels) * shape.y.height *
            shape.y.width;
        if (!constant || constantBytes != elements * 2)
            throw std::invalid_argument(
                "H13 per-channel broadcast requires one fp16 constant per channel");
        if (source->constantBytes == constantBytes) {
            // The wide-constant rows (>= 1024 channels) carry one dense copy
            // of the resolved constant, blob header halfwords included.
            program.constants.assign(constant, constant + constantBytes);
        } else if (source->constantBytes == constantBytes * 2) {
            program.constants = perChannelConstants(operation, elements,
                                                    constant,
                                                    source->constantBytes);
        } else {
            throw std::invalid_argument(
                "H13 per-channel section matches neither decoded layout");
        }
    } else {
        program.constants.assign(source->constantBytes, 0);
    }
    program.inputs = {elementwiseTensor(5, shape.x)};
    if (operand == BroadcastOperand::Runtime) {
        program.inputs.push_back(elementwiseTensor(6, shape.y));
        // Identical operands keep declaration order; any broadcast puts the
        // output surface between the two operands.
        if (!sameShape(shape.x, shape.y)) program.outputBindingIndex = 1;
    }
    program.output = elementwiseTensor(4, broadcastOutput(operand, shape));
    program.firstTaskBytes = source->firstTaskBytes;
    program.taskCount = source->taskCount;
    program.constantOffsetBytes = source->constantOffsetBytes;
    program.scratchAllocationBytes = source->scratchAllocationBytes;
    return program;
}

bool supportsNormParity(NormOperation operation, NormShape shape) {
    return normTemplate(operation, shape);
}

Program encodeNormParity(NormOperation operation, NormShape shape) {
    const auto *source = normTemplate(operation, shape);
    if (!source)
        throw std::invalid_argument(
            "H13 normalization geometry is outside the decoded parity envelope");
    Program program;
    program.taskSurfaceChannels = {5, 4, 6, 7};
    program.task = taskBytesFor(source->words, source->wordCount);
    program.constants = normConstants(source->constants, source->constantBytes);
    program.inputs = {elementwiseTensor(5, source->input)};
    program.output = elementwiseTensor(4, source->output);
    program.firstTaskBytes = source->firstTaskBytes;
    program.taskCount = source->taskCount;
    program.constantOffsetBytes = source->constantOffsetBytes;
    program.scratchAllocationBytes = source->scratchAllocationBytes;
    return program;
}

namespace {

const OracleConvTemplate *convTemplate(ConvShape shape) {
    for (const auto &candidate : kConvTasks)
        if (candidate.kernel == shape.kernel &&
            candidate.kernelWidth == shape.kernelWidth &&
            candidate.stride == shape.stride &&
            candidate.groups == shape.groups && candidate.bias == shape.bias &&
            sameShape(candidate.input, shape.input) &&
            sameShape(candidate.output, shape.output)) return &candidate;
    return nullptr;
}

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

/// Apple's convolution surface. The row is the width padded to 64 bytes, not
/// merely floored at it: a `pad_type="valid"` result 62 columns wide takes a
/// 128-byte row, which is what the decoded oracles record.
TensorLayout convTensor(std::uint32_t index, ElementwiseShape shape) {
    const std::uint64_t row =
        alignUp(static_cast<std::uint64_t>(shape.width) * 2, 64);
    const std::uint64_t plane = row * shape.height;
    const std::uint64_t element = plane * shape.channels;
    return {index, {1, shape.channels, shape.height, shape.width, plane, row},
            alignTile(element)};
}

/// How Apple splits the output channels into halfword interleave widths.
///
/// Every plane group holds 16 planes, so the channels are consumed in chunks
/// of `16 * lanes`, where `lanes` is the largest power of two at or below
/// `outputs / 16` and at most the cap: 32 for an input surface of 8 or fewer
/// pixels per side and 16 above it. 768 outputs at cap 32 is the case that
/// proves the split -- 512 channels interleave 32 wide, the remaining 256
/// interleave 16 wide -- and known-weight probes pin the cap's boundary
/// between 8 and 9 pixels.
std::vector<std::uint32_t> laneChunks(std::uint32_t outputs, std::uint32_t cap) {
    std::vector<std::uint32_t> chunks;
    std::uint32_t remaining = outputs;
    while (remaining) {
        std::uint32_t lanes = 1;
        if (remaining >= 16) {
            lanes = cap;
            while (lanes > remaining / 16) lanes /= 2;
        }
        chunks.push_back(lanes);
        remaining -= std::min(remaining, 16 * lanes);
    }
    return chunks;
}

std::uint32_t laneCap(ElementwiseShape input) {
    return std::max(input.height, input.width) <= 8 ? 32 : 16;
}

void putHalfword(std::vector<std::uint8_t> &bytes, std::size_t offset,
                 const std::uint8_t *source) {
    bytes[offset] = source[0];
    bytes[offset + 1] = source[1];
}

/// Native small-output packing (c3/c8/c24): distribute adjacent outputs
/// evenly over up to 16 cores, padding each core's interleaved plane.
std::vector<std::uint8_t> packConvSmallDense(std::uint32_t reduction,
    std::uint32_t outputs, const std::uint8_t *weights, const std::uint8_t *bias) {
    const auto cores = std::min(16u, outputs);
    const std::uint64_t rows = reduction + (bias ? 1 : 0);
    std::vector<std::uint8_t> packed;
    unsigned first = 0;
    for (unsigned core = 0; core != cores; ++core) {
        const unsigned lanes = outputs / cores + (core < outputs % cores);
        const auto base = packed.size();
        packed.resize(base + alignUp(rows * lanes * 2, 64), 0);
        for (unsigned lane = 0; lane != lanes; ++lane) {
            auto cursor = base + lane * 2;
            if (bias) {
                putHalfword(packed, cursor, bias + (first + lane) * 2);
                cursor += lanes * 2;
            }
            for (unsigned row = 0; row != reduction; ++row) {
                putHalfword(packed, cursor, weights + (static_cast<std::size_t>(first + lane) * reduction + row) * 2);
                cursor += lanes * 2;
            }
        }
        first += lanes;
    }
    return packed;
}

/// Apple's dense convolution section. A plane interleaves `lanes` output
/// channels at halfword granularity over `reduction` rows, preceded by one
/// bias row when the convolution has a bias; 16 plane groups follow, each
/// holding one plane per chunk. A grouped convolution pads every plane to 64
/// bytes and a groups-1 convolution pads only the plane group.
std::vector<std::uint8_t> packConvDense(std::uint32_t reduction,
                                        const std::vector<std::uint32_t> &chunks,
                                        const std::uint8_t *weights,
                                        const std::uint8_t *bias,
                                        bool planePadding) {
    const std::uint64_t rows = reduction + (bias ? 1 : 0);
    std::vector<std::uint64_t> planeBytes;
    planeBytes.reserve(chunks.size());
    for (const auto lanes : chunks) {
        const std::uint64_t span = rows * lanes * 2;
        planeBytes.push_back(planePadding ? alignUp(span, 64) : span);
    }
    std::uint64_t total = 0;
    for (const auto span : planeBytes) total += span;
    const std::uint64_t groupBytes = alignUp(total, 64);
    std::vector<std::uint8_t> packed(groupBytes * 16, 0);
    std::uint32_t start = 0;
    for (std::size_t chunk = 0; chunk != chunks.size(); ++chunk) {
        const std::uint32_t lanes = chunks[chunk];
        std::uint64_t offset = 0;
        for (std::size_t earlier = 0; earlier != chunk; ++earlier)
            offset += planeBytes[earlier];
        for (std::uint32_t group = 0; group != 16; ++group) {
            for (std::uint32_t lane = 0; lane != lanes; ++lane) {
                const std::uint32_t column = start + group * lanes + lane;
                std::size_t cursor = static_cast<std::size_t>(
                    group * groupBytes + offset + lane * 2);
                if (bias) {
                    putHalfword(packed, cursor, bias + column * 2);
                    cursor += lanes * 2;
                }
                const std::uint8_t *source =
                    weights + static_cast<std::size_t>(column) * reduction * 2;
                for (std::uint32_t row = 0; row != reduction; ++row) {
                    putHalfword(packed, cursor, source);
                    cursor += lanes * 2;
                    source += 2;
                }
            }
        }
        start += 16 * lanes;
    }
    return packed;
}

/// Apple's F1 in-projection dense section (k1x1 g1 valid, 2048 outputs over
/// 1024 reduction, bias-free): 16 regions x 8 slots, slot (region, slotIdx)
/// a 16-lane plane over `reduction` consecutive rows holding colgroup
///
///     G = 8 * (4 * (3 - region / 4) + ((region / 2) % 2) + 2 * (slot % 2))
///         + 4 * (region % 2) + slot / 2
///
/// Colgroups 64 apart are invisible to any uint16 payload (e -> e + 64*1024
/// preserves every stored halfword); this order was decoded from the
/// 32-bit-pair remint and is byte-exact against both that capture and the
/// round-2 uint16 capture (receipts/2026-09-17-encoder-conv-lowering).
std::vector<std::uint8_t> packConvDenseColgroups(std::uint32_t reduction,
                                                 std::uint32_t outputs,
                                                 const std::uint8_t *weights) {
    if (reduction != 1024 || outputs != 2048)
        throw std::invalid_argument(
            "H13 colgroup-order dense section is decoded only for the F1 "
            "in-projection geometry (1024 -> 2048)");
    constexpr std::uint32_t regions = 16, slots = 8, lanes = 16;
    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(regions) * slots * reduction * lanes * 2);
    for (std::uint32_t region = 0; region != regions; ++region) {
        for (std::uint32_t slot = 0; slot != slots; ++slot) {
            const std::uint32_t block =
                4 * (3 - region / 4) + ((region / 2) % 2) + 2 * (slot % 2);
            const std::uint32_t colgroup =
                8 * block + 4 * (region % 2) + slot / 2;
            const std::size_t base =
                (static_cast<std::size_t>(region) * slots * reduction +
                 slot * reduction) * lanes * 2;
            for (std::uint32_t lane = 0; lane != lanes; ++lane) {
                const std::uint8_t *source = weights +
                    static_cast<std::size_t>(16 * colgroup + lane) *
                        reduction * 2;
                std::size_t cursor = base + lane * 2;
                for (std::uint32_t row = 0; row != reduction; ++row) {
                    putHalfword(packed, cursor, source);
                    cursor += lanes * 2;
                    source += 2;
                }
            }
        }
    }
    return packed;
}

/// Apple's depthwise section: 16 lanes, each holding `outputs / 16` channels
/// back to back with no padding between them and the lane padded to 64 bytes.
/// A bias precedes each channel's taps.
std::vector<std::uint8_t> packConvDepthwise(std::uint32_t taps,
                                            std::uint32_t outputs,
                                            const std::uint8_t *weights,
                                            const std::uint8_t *bias) {
    const std::uint32_t lanes = std::min<std::uint32_t>(16, outputs);
    const std::uint64_t rows = taps + (bias ? 1 : 0);
    // A non-multiple of 16 has unequal core chunks: c24 gives two channels
    // to cores 0..7 and one to 8..15. Pad each core's actual span separately.
    std::vector<std::uint64_t> laneBases(lanes);
    std::uint64_t total = 0;
    for (std::uint32_t lane = 0; lane != lanes; ++lane) {
        laneBases[lane] = total;
        const auto slots = (outputs - lane + lanes - 1) / lanes;
        total += alignUp(slots * rows * 2, 64);
    }
    std::vector<std::uint8_t> packed(total, 0);
    for (std::uint32_t column = 0; column != outputs; ++column) {
        std::size_t cursor = static_cast<std::size_t>(
            laneBases[column % lanes] + (column / lanes) * rows * 2);
        if (bias) {
            putHalfword(packed, cursor, bias + column * 2);
            cursor += 2;
        }
        const std::uint8_t *source =
            weights + static_cast<std::size_t>(column) * taps * 2;
        for (std::uint32_t tap = 0; tap != taps; ++tap)
            putHalfword(packed, cursor + tap * 2, source + tap * 2);
    }
    return packed;
}

/// Apple's stride-2 section, which skips zero weights.
///
/// A plane holds the plane's `lanes` bias halfwords when the convolution has
/// a bias, a 16-bit count of the body bytes, then one row per reduction step.
/// A row leads with a zero byte and then carries, per group of eight lanes, a
/// mask byte whose bit `l` marks a lane whose weight is not zero followed by
/// those lanes' halfwords, and closes with `lanes / 2 - 2` zero bytes. Both
/// signed zeros count as zero, so the section's size depends on the values.
///
/// With `skipZeros` false every mask bit is set and zero weights are stored
/// like any other value. That is the only form the device has run: random
/// weights have no zeros, and the first real-weight MobileNetV1 (whose
/// stride-2 layers hold exact zeros) wedged the M1 Pro engine on its first
/// submit. Apple's captures skip zeros, so oracle parity keeps `skipZeros`.
std::vector<std::uint8_t> packConvStrided(std::uint32_t reduction,
                                          const std::vector<std::uint32_t> &chunks,
                                          const std::uint8_t *weights,
                                          const std::uint8_t *bias,
                                          bool skipZeros = true) {
    std::vector<std::uint8_t> packed;
    std::uint32_t start = 0;
    for (const auto lanes : chunks) {
        const std::uint32_t padding = lanes >= 4 ? lanes / 2 - 2 : 0;
        for (std::uint32_t group = 0; group != 16; ++group) {
            const std::uint32_t first = start + group * lanes;
            std::vector<std::uint8_t> body;
            for (std::uint32_t row = 0; row != reduction; ++row) {
                body.push_back(0);
                for (std::uint32_t base = 0; base < lanes; base += 8) {
                    const std::uint32_t end = std::min(base + 8, lanes);
                    std::uint8_t mask = 0;
                    std::vector<std::uint8_t> values;
                    for (std::uint32_t lane = base; lane != end; ++lane) {
                        const std::uint8_t *value = weights +
                            (static_cast<std::size_t>(first + lane) * reduction +
                             row) * 2;
                        if (!skipZeros || value[0] || (value[1] & 0x7f)) {
                            mask |= static_cast<std::uint8_t>(1u << (lane - base));
                            values.push_back(value[0]);
                            values.push_back(value[1]);
                        }
                    }
                    body.push_back(mask);
                    body.insert(body.end(), values.begin(), values.end());
                }
                body.insert(body.end(), padding, 0);
            }
            if (body.size() > 0xffff)
                throw std::invalid_argument(
                    "H13 strided convolution plane exceeds a 16-bit body count");
            std::vector<std::uint8_t> plane;
            if (bias)
                plane.insert(plane.end(), bias + first * 2,
                             bias + (first + lanes) * 2);
            plane.push_back(static_cast<std::uint8_t>(body.size()));
            plane.push_back(static_cast<std::uint8_t>(body.size() >> 8));
            plane.insert(plane.end(), body.begin(), body.end());
            plane.resize(alignUp(plane.size(), 64), 0);
            packed.insert(packed.end(), plane.begin(), plane.end());
        }
        start += 16 * lanes;
    }
    return packed;
}

} // namespace

/// Apple's custom-pad depthwise section (k1x1 grouped, asymmetric W pad):
/// three 512-byte chunks of eight 64-byte lanes. Chunks one and two repeat
/// the structural constants — even lanes 0x3c00 (fp16 1.0), odd lanes
/// 0x0001 then 0x0100 — and chunk three carries one raw weight halfword per
/// lane in channel order. Verified against
/// encoder_conv_pad_c8_n8_k1x1_s1_g8_bias0_p0010_f4 and ..._tiny plus three
/// same-format payload runs
/// (receipts/2026-09-17-encoder-padconv-respell/): ascending payloads
/// (1..8, 2..16) prove the identity lane order and that the structural
/// chunks never carry weight bytes.
std::vector<std::uint8_t> packConvPadconvColumns(
    const std::uint8_t *weights, std::size_t weightBytes) {
    constexpr std::uint32_t lanes = 8;
    constexpr std::size_t laneBytes = 64;
    constexpr std::size_t chunk = lanes * laneBytes;
    if (!weights || weightBytes != lanes * 2)
        throw std::invalid_argument(
            "H13 custom-pad depthwise section is decoded only for the "
            "8-channel grouped 1x1 padconv");
    std::vector<std::uint8_t> packed(chunk * 3, 0);
    for (std::uint32_t lane = 0; lane != lanes; ++lane) {
        const std::size_t base = lane * laneBytes;
        if (lane % 2 == 0) {
            packed[base] = 0x00;
            packed[base + 1] = 0x3c;
            packed[chunk + base] = 0x00;
            packed[chunk + base + 1] = 0x3c;
        } else {
            packed[base] = 0x01;
            packed[base + 1] = 0x00;
            packed[base + 2] = 0x00;
            packed[base + 3] = 0x01;
            packed[chunk + base] = 0x01;
            packed[chunk + base + 1] = 0x00;
            packed[chunk + base + 2] = 0x00;
            packed[chunk + base + 3] = 0x01;
        }
        const std::uint8_t *weight = weights + lane * 2;
        std::memcpy(packed.data() + 2 * chunk + base, weight, 2);
    }
    return packed;
}

bool supportsConvParity(ConvShape shape) { return convTemplate(shape); }

std::size_t convParitySectionBytes(ConvShape shape) {
    const auto *source = convTemplate(shape);
    return source ? source->constantBytes : 0;
}

/// The captured W-major depthwise section (k1x9 `same`, bias present): 64
/// lanes x `outputs / 64` slots, slot `s` carrying channel
/// `16 * (s % 64) + s / 64` as 22 halfwords — the bias twice, then the nine
/// taps in the decoded two-pass lane pairing with two structural zeros.
/// Verified against
/// encoder_conv_idx_c1024_n1024_k1x9_s1_g1024_bias1_same_wmaj.
std::vector<std::uint8_t> packConvDepthwiseSlots(std::uint32_t taps,
                                                 std::uint32_t outputs,
                                                 const std::uint8_t *weights,
                                                 const std::uint8_t *bias) {
    constexpr std::uint32_t lanes = 64;
    constexpr std::size_t slotHalves = 22;
    static const int order[slotHalves] = {-1, -1, 0, -1, 2, 1, 4, 3, 6, 5, 8, 7,
                                          1, 0, 3, 2, 5, 4, 7, 6, -1, 8};
    if (taps != 9)
        throw std::invalid_argument(
            "H13 slotted depthwise section is decoded only for the 9-tap "
            "kernel");
    const std::uint32_t slots = outputs / lanes;
    if (slots * lanes != outputs)
        throw std::invalid_argument(
            "H13 slotted depthwise section packs 64-lane channel groups");
    std::vector<std::uint8_t> packed(slots * lanes * slotHalves * 2);
    auto *halfs = reinterpret_cast<std::uint16_t *>(packed.data());
    for (std::uint32_t slot = 0; slot < slots * lanes; ++slot) {
        const std::uint32_t column =
            16 * (slot % lanes) + slot / lanes;
        const auto biasWord = static_cast<std::uint16_t>(
            bias[2 * column] | (bias[2 * column + 1] << 8));
        const std::size_t base = slot * slotHalves;
        halfs[base] = biasWord;
        halfs[base + 1] = biasWord;
        for (std::size_t offset = 0; offset < slotHalves; ++offset) {
            const int tap = order[offset];
            if (tap < 0) continue;
            std::uint16_t word = 0;
            std::memcpy(&word,
                        weights + (static_cast<std::size_t>(column) * taps +
                                   tap) * 2,
                        2);
            halfs[base + offset] = word;
        }
    }
    return packed;
}

/// The F3 out-projection dense section (k1x1 g1 valid, 1024 outputs over
/// 1024 reduction, bias-free, [1, 1024, 1, 375] surfaces) with the engine
/// write-out inverse decoded in. The device emits the 1024 output planes
/// in 64 runs of 16 under a fixed run permutation R: dev run k displays
/// ref run
///
///     R(k) = ((k >> 1) & 7) | (k & 0x10) | ((((~k) >> 5) & 1) << 3)
///            | ((k & 1) << 5)
///
/// (T6001 test host measurement, receipts/2026-09-18-encoder-conv-device-gate/
/// f3-writeout-permutation.json). Exact numerics under the bijection mean
/// the engine computes output run k from the weights packed at run slot
/// R(k), so dev run k shows ref run k exactly when slot R(k) holds ref
/// run k: slot p carries ref run
///
///     g(p) = ((p >> 5) & 1) | ((p & 1) << 1) | (((p >> 1) & 1) << 2)
///            | (((p >> 2) & 1) << 3) | (((p >> 4) & 1) << 4)
///            | ((1 - ((p >> 3) & 1)) << 5)
///
/// with within-run plane order preserved. The same engine class is what
/// the m375 k1024 n1024 linear lowering already compensates (67dfcf1).
/// The record's captured section carries a uniform per-constant payload,
/// so its bytes are invariant under the run placement and byte parity
/// against the capture keeps holding; this is gated to the one measured
/// geometry, and device execution is the oracle.
std::vector<std::uint8_t> packConvDenseWriteoutInverse(
        std::uint32_t reduction, std::uint32_t outputs,
        const std::vector<std::uint32_t> &chunks,
        const std::uint8_t *weights) {
    if (reduction != 1024 || outputs != 1024)
        throw std::invalid_argument(
            "H13 write-out inverse is decoded only for the F3 out-projection "
            "geometry (1024 -> 1024)");
    std::vector<std::uint8_t> permuted(
        static_cast<std::size_t>(outputs) * reduction * 2);
    const std::size_t runBytes =
        static_cast<std::size_t>(16) * reduction * 2;
    for (std::uint32_t slot = 0; slot != outputs / 16; ++slot) {
        const std::uint32_t run =
            ((slot >> 5) & 1) | ((slot & 1) << 1) |
            (((slot >> 1) & 1) << 2) | (((slot >> 2) & 1) << 3) |
            (((slot >> 4) & 1) << 4) | ((1u - ((slot >> 3) & 1u)) << 5);
        std::memcpy(permuted.data() + static_cast<std::size_t>(slot) *
                                        runBytes,
                    weights + static_cast<std::size_t>(run) * runBytes,
                    runBytes);
    }
    return packConvDense(reduction, chunks, permuted.data(), nullptr, false);
}

std::vector<std::uint8_t> packConvWeights(ConvShape shape,
                                          const std::uint8_t *weights,
                                          std::size_t weightBytes,
                                          const std::uint8_t *bias,
                                          std::size_t biasBytes) {
    const std::uint32_t taps = shape.kernel * shape.kernelWidth;
    if (!shape.groups || shape.input.channels % shape.groups ||
        shape.output.channels % shape.groups)
        throw std::invalid_argument(
            "H13 convolution groups must divide both channel counts");
    const std::uint32_t reduction = shape.input.channels / shape.groups * taps;
    if (!weights || weightBytes !=
            static_cast<std::size_t>(shape.output.channels) * reduction * 2)
        throw std::invalid_argument(
            "H13 convolution requires Cout * Cin / groups * kh * kw fp16 weights");
    if (shape.bias != (bias != nullptr) ||
        (bias && biasBytes != static_cast<std::size_t>(shape.output.channels) * 2))
        throw std::invalid_argument(
            "H13 convolution bias must hold one fp16 value per output channel");
    if (shape.groups == shape.input.channels &&
        shape.groups == shape.output.channels) {
        if (bias && shape.kernel == 1 && shape.kernelWidth > 1)
            return packConvDepthwiseSlots(shape.kernelWidth,
                                          shape.output.channels, weights,
                                          bias);
        if (!bias && shape.kernel == 1 && shape.kernelWidth == 1 &&
            shape.input.width != shape.output.width)
            // The W-padded padconv form: a width-changing k1x1 grouped
            // depthwise is exactly the asymmetric custom-pad respell, and
            // its captured section is the three-chunk columns layout.
            return packConvPadconvColumns(weights, weightBytes);
        return packConvDepthwise(taps, shape.output.channels, weights, bias);
    }
    if (shape.kernel == 3 && shape.kernelWidth == 3 && shape.stride == 1 &&
        shape.groups == 1 && ((shape.input.channels == 128 && shape.output.channels == 512) ||
                             (shape.input.channels == 512 && shape.output.channels == 128)) &&
        shape.input.height == 35 && shape.input.width == 35 &&
        shape.output.height == 35 && shape.output.width == 35) {
        // The native 35x35 schedule writes the upper output slice first.
        // Indexed-weight hashes prove the two contiguous coefficient groups.
        std::vector<std::uint8_t> packed;
        const unsigned split = shape.output.channels == 512 ? 256 : 96;
        for (const unsigned first : {split, 0u}) {
            const unsigned count = first ? shape.output.channels - split : split;
            const auto part = packConvSmallDense(reduction, count,
                weights + static_cast<std::size_t>(first) * reduction * 2,
                bias ? bias + first * 2 : nullptr);
            packed.insert(packed.end(), part.begin(), part.end());
        }
        return packed;
    }
    if (shape.output.channels % 16) {
        if (shape.groups != 1 || shape.stride != 1)
            throw std::invalid_argument("small dense outputs are qualified only for stride 1, groups 1");
        return packConvSmallDense(reduction, shape.output.channels, weights, bias);
    }
    const auto chunks = laneChunks(shape.output.channels, laneCap(shape.input));
    if (shape.stride > 1) {
        if (shape.groups != 1 || taps != 1 ||
            *std::max_element(chunks.begin(), chunks.end()) > 8)
            throw std::invalid_argument(
                "H13 has no derived strided packing for grouped, multi-tap, or "
                "16-lane convolutions");
        return packConvStrided(reduction, chunks, weights, bias);
    }
    if (shape.groups > 1) {
        const std::uint32_t lanes =
            std::max<std::uint32_t>(1, shape.output.channels / 64);
        return packConvDense(reduction,
                             std::vector<std::uint32_t>(
                                 shape.output.channels / lanes / 16, lanes),
                             weights, bias, true);
    }
    if (!bias && shape.stride == 1 && taps == 1 && reduction == 1024 &&
        shape.output.channels == 2048)
        return packConvDenseColgroups(reduction, shape.output.channels,
                                      weights);
    if (!bias && shape.stride == 1 && taps == 1 && reduction == 1024 &&
        shape.output.channels == 1024 && chunks.size() == 4 &&
        shape.input.height == 1 && shape.input.width == 375)
        return packConvDenseWriteoutInverse(reduction,
                                            shape.output.channels, chunks,
                                            weights);
    return packConvDense(reduction, chunks, weights, bias, false);
}

bool convParityLeadingPaddingMatches(ConvShape shape,std::uint32_t top,std::uint32_t left) {
    const auto *source=convTemplate(shape);
    return source && source->padTop==top && source->padLeft==left;
}

Program encodeConvParity(ConvShape shape, const std::uint8_t *weights,
                         std::size_t weightBytes, const std::uint8_t *bias,
                         std::size_t biasBytes) {
    const auto *source = convTemplate(shape);
    if (!source)
        throw std::invalid_argument(
            "H13 convolution geometry is outside the decoded parity envelope");
    Program program;
    program.taskSurfaceChannels = {5, 4, 6, 7};
    program.task = taskBytesFor(source->words, source->wordCount);
    program.constants =
        packConvWeights(shape, weights, weightBytes, bias, biasBytes);
    if (program.constants.size() != source->constantBytes)
        throw std::logic_error(
            "H13 packed convolution section differs from the decoded size");
    program.inputs = {convTensor(5, shape.input)};
    program.output = convTensor(4, shape.output);
    program.firstTaskBytes = source->firstTaskBytes;
    program.taskCount = source->taskCount;
    program.constantOffsetBytes = source->constantOffsetBytes;
    program.scratchAllocationBytes = source->scratchAllocationBytes;
    return program;
}

namespace {

/// The L2 budget a generic convolution tile may fill: input rows plus the
/// result rows. The largest decoded single-task tile fills 0x3c0000 bytes
/// (encoder_conv_idx_c256_n256_k1x1_s1_g1_bias1_valid_pw375 and _pw750);
/// staying under it keeps every generic tile inside a measured envelope.
constexpr std::uint64_t kGenericL2Budget = 0x380000;
/// Kernel bytes one task streams through its 16 kernel-DMA chunks. Every
/// decoded task stays at or under 1 MiB (conv_probe_k1_c1024_n1024 splits
/// into two 1 MiB tasks; 3x3 at 1024 inputs into 576 KiB tasks).
constexpr std::uint64_t kGenericKernelBudget = 0x100000;

enum class ConvKind { Dense, Depthwise, Other };

ConvKind convKind(const ConvShape &shape) {
    if (shape.groups == 1) return ConvKind::Dense;
    if (shape.groups == shape.input.channels &&
        shape.groups == shape.output.channels) return ConvKind::Depthwise;
    return ConvKind::Other;
}

/// Symmetric padding the output surface implies, or -1 for an unsupported pad.
int convPadding(const ConvShape &shape) {
    const std::uint32_t k = shape.kernel, s = shape.stride;
    const std::uint32_t h = shape.input.height, w = shape.input.width;
    const std::uint32_t ho = shape.output.height, wo = shape.output.width;
    const int same = static_cast<int>((k - 1) / 2);
    auto outFor = [&](std::uint32_t extent, int pad) -> std::int64_t {
        const std::int64_t padded = static_cast<std::int64_t>(extent) + 2 * pad;
        if (padded < static_cast<std::int64_t>(k)) return -1;
        return (padded - k) / s + 1;
    };
    if (outFor(h, same) == ho && outFor(w, same) == wo) return same;
    if (outFor(h, 0) == ho && outFor(w, 0) == wo) return 0;
    return -1;
}

/// The decoded single-task template this family borrows its invariant words
/// from: same kernel, stride, group form and bias, smallest surface first.
const OracleConvTemplate *convFamilyBase(const ConvShape &shape) {
    const ConvKind kind = convKind(shape);
    const OracleConvTemplate *best = nullptr;
    for (const auto &candidate : kConvTasks) {
        if (candidate.nativeLayer) continue;
        if (candidate.taskCount != 1 || candidate.kernel != shape.kernel ||
            candidate.kernelWidth != shape.kernelWidth ||
            candidate.stride != shape.stride || candidate.bias != shape.bias)
            continue;
        if (candidate.input.channels < 64 || candidate.input.height > 64 ||
            candidate.input.width > 64)
            continue;
        ConvShape probe{candidate.kernel, candidate.kernelWidth,
                        candidate.stride, candidate.groups, candidate.bias,
                        candidate.input, candidate.output};
        if (convKind(probe) != kind) continue;
        // Same padding form, so the base's ConvCfg pad bits are the family's.
        if (shape.kernel > 1 &&
            (convPadding(probe) == 0) != (convPadding(shape) == 0)) continue;
        if (!best || candidate.wordCount < best->wordCount ||
            (candidate.wordCount == best->wordCount &&
             candidate.output.channels < best->output.channels))
            best = &candidate;
    }
    return best;
}

void putRegister(std::vector<std::uint8_t> &task, std::size_t offset,
                 std::size_t bytes, std::uint32_t address, std::uint32_t value) {
    storeLE32(task, h13RegisterOffset(task, offset, bytes, address), value);
}

std::uint32_t floorLog2(std::uint32_t value) {
    std::uint32_t log = 0;
    while (value > 1) { value >>= 1; ++log; }
    return log;
}

struct ConvGenericPlan {
    ConvKind kind;
    std::uint32_t inputPerTask;      // Common Cin
    std::uint32_t rowL2;             // L2 row bytes of the input
    std::uint32_t sourceRow;         // L2 input row stride incl. bank skew
    std::uint32_t tileHeight;
    std::uint32_t tileRows;          // input rows resident per tile
    std::vector<std::uint32_t> slices;  // output channels per task
};

/// The bank skew Apple adds to the L2 input row stride; reproduces all 175
/// decoded single-task templates (stride 1: 0x20/0x30 up to W 16, 0x50 to
/// W 32, none from W 64; stride 2: 0x20, 0x30 from W 64).
std::uint32_t l2Skew(std::uint32_t stride, std::uint32_t width,
                     std::uint32_t inputs) {
    if (stride == 2) return width >= 64 ? 0x30 : 0x20;
    if (width >= 64) return 0;
    if (width > 16) return 0x50;
    return inputs >= 32 ? 0x30 : 0x20;
}

std::string planConvGeneric(const ConvShape &shape, ConvGenericPlan &plan) {
    plan.kind = convKind(shape);
    if (plan.kind == ConvKind::Other)
        return "grouped convolutions other than depthwise have no generic form";
    if (shape.kernel != shape.kernelWidth ||
        (shape.kernel != 1 && shape.kernel != 3))
        return "the generic form covers square 1x1 and 3x3 kernels";
    if (shape.stride != 1 && shape.stride != 2)
        return "the generic form covers stride 1 and 2";
    if (shape.stride == 2 && (shape.kernel != 1 || plan.kind != ConvKind::Dense))
        return "stride 2 is generic only for dense 1x1 kernels";
    if (convPadding(shape) < 0)
        return "the output surface implies neither same nor valid padding";
    const auto &in = shape.input;
    const auto &out = shape.output;
    if (!in.channels || !out.channels || !in.height || !in.width ||
        !out.height || !out.width)
        return "convolution surfaces must be nonempty";
    if (in.channels % 16 || out.channels % 16)
        return "the generic form needs channel counts that are multiples of 16";
    if (in.channels & (in.channels - 1))
        return "non-power-of-two inputs need a qualified decoded native layout";
    if (in.width > 4096 || in.height > 16384)
        return "the generic form caps rows at 4096 columns";
    if (shape.groups == 1 && shape.kernel == 3 &&
        ((in.channels == 128 && out.channels >= 256) || (in.channels == 512 && out.channels == 128)) &&
        out.width > 32 && out.width < 64)
        return "unaligned dense 3x3 writeout needs a qualified decoded layout";
    if (!convFamilyBase(shape))
        return "no decoded template of this family supplies the invariant words";
    plan.inputPerTask = in.channels;
    plan.rowL2 = static_cast<std::uint32_t>(
        alignUp(std::max<std::uint32_t>(16, in.width * 2), 16));
    plan.sourceRow = plan.inputPerTask * plan.rowL2 +
                     l2Skew(shape.stride, in.width, plan.inputPerTask);
    const std::uint32_t taps = shape.kernel * shape.kernelWidth;
    const std::uint64_t reduction =
        plan.kind == ConvKind::Depthwise ? taps : in.channels * taps;
    // Output-channel tasks: each task's kernel bytes must fit the budget.
    const std::uint64_t perChannel = (reduction + (shape.bias ? 1 : 0)) * 2;
    std::uint32_t slice = out.channels;
    if (shape.stride == 2) {
        // The zero-skipping stride-2 section is byte-proven only up to 8
        // interleaved lanes, and a 16-lane program (128 -> 256 at 56 -> 28)
        // hung the M1 Pro engine. A wider stride-2 layer therefore runs as
        // output-channel tasks of 128 channels: each task is the proven
        // one-chunk 8-lane form, writing its own slice of the output.
        const auto chunks = laneChunks(out.channels, laneCap(in));
        if (chunks.size() != 1 || chunks.front() > 8) {
            if (out.channels % 128)
                return "stride 2 above 128 output channels needs a multiple of 128 outputs";
            slice = 128;
        }
        if (slice * perChannel > kGenericKernelBudget)
            return "a strided 128-channel task exceeds the per-task kernel budget";
    }
    if (plan.kind == ConvKind::Dense && slice * perChannel > kGenericKernelBudget) {
        if (shape.stride != 1)
            return "a strided convolution above the per-task kernel budget has no generic split";
        std::uint64_t fit = kGenericKernelBudget / perChannel;
        if (fit >= 256) slice = static_cast<std::uint32_t>(fit / 256 * 256);
        else {
            // Every task needs one uniform interleave. Rounding 113 fitting
            // channels to 112 makes 4+2+1-lane planes under an OCG=2 header.
            // Pick a power-of-two channel slice: 64 here, then another 64.
            slice = 16;
            while (slice <= fit / 2) slice *= 2;
            if (slice > fit) slice = 0;
        }
        if (!slice) return "one output channel exceeds the per-task kernel budget";
    }
    if (plan.kind == ConvKind::Dense) {
        // Uniform lane chunks only: a mixed split (768 outputs at cap 32 ->
        // 32 + 16 lanes) is decoded for one template but never ran from this
        // encoder, so the generic form keeps every chunk the same width.
        const auto chunks = laneChunks(out.channels, laneCap(in));
        for (const auto lanes : chunks)
            if (lanes != chunks.front())
                return "the output channels split into lane chunks of different widths";
    }
    if (shape.stride == 2) {
        // Device-verified only up to 8 interleaved lanes per plane (one lane
        // chunk): a 16-lane zero-skipping section (128 -> 256 at 56 -> 28)
        // hung the engine on the M1 Pro, matching the decoded record that the
        // 16-lane body count is two bytes off the model.
        const auto chunks = laneChunks(slice, laneCap(in));
        if (chunks.size() != 1 || chunks.front() > 8)
            return "stride 2 is verified only up to 8 interleaved lanes "
                   "(at most 128 output channels per task)";
    }
    if (plan.kind == ConvKind::Depthwise &&
        out.channels * perChannel > kGenericKernelBudget)
        return "depthwise kernel exceeds the per-task kernel budget";
    plan.slices.clear();
    for (std::uint32_t done = 0; done < out.channels; done += slice)
        plan.slices.push_back(std::min(slice, out.channels - done));
    const std::uint32_t widestSlice = plan.slices.front();
    const std::uint64_t outRowL2 =
        alignUp(std::max<std::uint32_t>(16, out.width * 2), 16);
    plan.tileHeight = 0;
    for (std::uint32_t tile = out.height; tile >= 1; --tile) {
        const std::uint32_t rows = std::min<std::uint32_t>(
            in.height, tile * shape.stride + shape.kernel - 1);
        const std::uint64_t used =
            static_cast<std::uint64_t>(rows) * plan.sourceRow +
            static_cast<std::uint64_t>(tile) * widestSlice * outRowL2;
        if (used <= kGenericL2Budget) {
            plan.tileHeight = tile;
            plan.tileRows = rows;
            break;
        }
    }
    if (!plan.tileHeight)
        return "one output row does not fit the L2 tiling budget";
    return {};
}

/// Byte offset and size of each of the 16 kernel-DMA chunks of one packed
/// section. Dense and depthwise sections are 16 equal groups; the stride-2
/// zero-skipping section is 16 variable planes, each a bias row, a 16-bit
/// body count and the body, padded to 64 bytes.
void kernelChunks(const ConvShape &shape, const std::vector<std::uint8_t> &packed,
                  std::uint32_t lanes, std::array<std::uint32_t, 16> &offsets,
                  std::array<std::uint32_t, 16> &sizes) {
    if (shape.stride == 1) {
        if (packed.size() % 16)
            throw std::logic_error("H13 generic conv section is not 16 chunks");
        const auto chunk = static_cast<std::uint32_t>(packed.size() / 16);
        for (std::uint32_t i = 0; i != 16; ++i) {
            offsets[i] = i * chunk;
            sizes[i] = chunk;
        }
        return;
    }
    std::size_t cursor = 0;
    const std::size_t biasBytes = shape.bias ? lanes * 2 : 0;
    for (std::uint32_t i = 0; i != 16; ++i) {
        if (cursor + biasBytes + 2 > packed.size())
            throw std::logic_error("H13 generic strided section is truncated");
        const std::size_t body = packed[cursor + biasBytes] |
                                 (packed[cursor + biasBytes + 1] << 8);
        const std::size_t plane = alignUp(biasBytes + 2 + body, 64);
        offsets[i] = static_cast<std::uint32_t>(cursor);
        sizes[i] = static_cast<std::uint32_t>(plane);
        cursor += plane;
    }
    if (cursor != packed.size())
        throw std::logic_error("H13 generic strided section has trailing bytes");
}

} // namespace

namespace {
const OracleConvTemplate *qualifiedNativeConv(const ConvShape &shape) {
    const auto *source=convTemplate(shape);
    return source && source->nativeLayer && !source->scratchAllocationBytes ? source : nullptr;
}
} // namespace

std::string convGenericRefusal(ConvShape shape) {
    if (qualifiedNativeConv(shape)) return {};
    ConvGenericPlan plan{};
    return planConvGeneric(shape, plan);
}

Program encodeConvGeneric(ConvShape shape, const std::uint8_t *weights,
                          std::size_t weightBytes, const std::uint8_t *bias,
                          std::size_t biasBytes) {
    if (qualifiedNativeConv(shape)) {
        Program program = encodeConvParity(shape, weights, weightBytes, bias, biasBytes);
        program.nativeDramLayer = true;
        program.depthwiseConv = shape.groups==shape.input.channels && shape.groups==shape.output.channels;
        return program;
    }
    ConvGenericPlan plan{};
    const std::string refusal = planConvGeneric(shape, plan);
    if (!refusal.empty())
        throw std::invalid_argument("h13.conv-outside-envelope: " + refusal);
    const auto *base = convFamilyBase(shape);
    const auto &in = shape.input;
    const auto &out = shape.output;
    const std::uint32_t taps = shape.kernel * shape.kernelWidth;
    const std::uint32_t reduction =
        plan.kind == ConvKind::Depthwise ? taps : in.channels * taps;
    if (!weights || weightBytes != static_cast<std::size_t>(out.channels) *
                                       reduction * 2)
        throw std::invalid_argument(
            "H13 convolution requires Cout * Cin / groups * kh * kw fp16 weights");
    if (shape.bias != (bias != nullptr) ||
        (bias && biasBytes != static_cast<std::size_t>(out.channels) * 2))
        throw std::invalid_argument(
            "H13 convolution bias must hold one fp16 value per output channel");

    const TensorLayout input = convTensor(5, in);
    const TensorLayout output = convTensor(4, out);
    const std::uint64_t inPlane = input.nchw[4], inRow = input.nchw[5];
    const std::uint64_t outPlane = output.nchw[4], outRow = output.nchw[5];
    const int pad = convPadding(shape);
    const std::uint32_t outputWidth = out.width;
    const std::uint32_t cfgClass = outputWidth <= 8 ? 4
        : std::min<std::uint32_t>(4, 7 - std::min<std::uint32_t>(7,
              floorLog2(outputWidth - 1) + 1));
    // Common 0x34: 0x20 in bits 23:16 plus a width class (4 up to 8 output
    // columns, then 3, 2, 1 per doubling, 0 from 128). The low half repeats
    // the class in two nibbles in the forms Apple picks for them: always at
    // class 4, for dense 3x3 kernels and for 1x1 kernels with 512 or more
    // channels on both sides at classes 2 and 3, never for depthwise or at
    // class 1. That rule reproduces every decoded single-task template.
    const bool repeatClass = cfgClass == 4 ||
        ((cfgClass == 2 || cfgClass == 3) && plan.kind == ConvKind::Dense &&
         (shape.kernel == 3 || (in.channels >= 512 && out.channels >= 512)));
    const std::uint32_t common34 = (0x20u << 16) | (cfgClass << 16) |
        (repeatClass ? (cfgClass << 12) | (cfgClass << 8) |
                       (cfgClass == 4 ? 0x05u : 0x01u)
                     : 0x0001u);

    const auto baseTask = taskBytesFor(base->words, base->wordCount);
    const std::size_t baseBytes = baseTask.size();
    const std::size_t stride = alignUp(baseBytes, 0x100);
    Program program;
    program.taskSurfaceChannels = {5, 4, 6, 7};
    program.taskCount = static_cast<std::uint32_t>(plan.slices.size());
    program.firstTaskBytes = base->firstTaskBytes;
    program.task.assign(stride * (plan.slices.size() - 1) + baseBytes, 0);

    std::uint32_t firstChannel = 0;
    for (std::size_t index = 0; index != plan.slices.size(); ++index) {
        const std::uint32_t slice = plan.slices[index];
        ConvShape part = shape;
        part.output.channels = slice;
        std::vector<std::uint8_t> packed;
        std::uint32_t lanes = 1;
        if (plan.kind == ConvKind::Depthwise) {
            packed = packConvDepthwise(taps, slice, weights, bias);
        } else {
            const auto chunks = laneChunks(slice, laneCap(in));
            lanes = chunks.front();
            const std::uint8_t *w = weights +
                static_cast<std::size_t>(firstChannel) * reduction * 2;
            const std::uint8_t *b = bias ? bias + firstChannel * 2 : nullptr;
            if (shape.stride == 2) {
                if (chunks.size() != 1)
                    throw std::invalid_argument(
                        "h13.conv-outside-envelope: the generic strided "
                        "section needs one lane chunk");
                packed = packConvStrided(reduction, chunks, w, b, false);
            } else {
                packed = packConvDense(reduction, chunks, w, b, false);
            }
        }
        const std::uint32_t sectionBase =
            static_cast<std::uint32_t>(program.constants.size());
        std::array<std::uint32_t, 16> offsets{}, sizes{};
        kernelChunks(part, packed, lanes, offsets, sizes);
        program.constants.insert(program.constants.end(), packed.begin(),
                                 packed.end());
        program.constants.resize(alignUp(program.constants.size(), 64), 0);

        const std::size_t offset = index * stride;
        std::copy(baseTask.begin(), baseTask.end(),
                  program.task.begin() + static_cast<std::ptrdiff_t>(offset));
        auto &task = program.task;
        const std::size_t bytes = baseBytes;
        // Task header: id, last-task flags, and the link to the next task.
        const bool last = index + 1 == plan.slices.size();
        if (plan.slices.size() > 1) {
            std::uint32_t word0 = static_cast<std::uint32_t>(index);
            if (index) word0 |= 1u << 24;
            if (last) word0 |= 1u << 24 | 1u << 25;
            storeLE32(task, offset, word0);
            const std::uint32_t header1 = loadLE32(task, offset + 4) & 0xffffu;
            storeLE32(task, offset + 4,
                      last ? header1
                           : header1 | ((static_cast<std::uint32_t>(baseBytes / 4) - 1) << 16));
            storeLE32(task, offset + 28,
                      last ? 0 : static_cast<std::uint32_t>(offset + stride));
            // Header flags of a linked stream, from the decoded multi-task
            // convolutions (conv_probe_k3_c1024_n128 and kin): the end-of-
            // stream bits (header[2] 0x22, header[4] 0x2, header[6] 0x9000)
            // only on the last task, header[8] bit 23 on every task with a
            // successor. Every generic task reloads the input surface, so
            // each keeps the input-side bits Apple's first task carries
            // (header[2] 0x8, header[4] 0x68, header[6] 0x10000800, the
            // header[8] source selectors).
            storeLE32(task, offset + 8, 0x400u | 0x08u | (last ? 0x22u : 0u));
            storeLE32(task, offset + 16, 0x00fff868u | (last ? 0x2u : 0u));
            storeLE32(task, offset + 24, 0x30000800u | (last ? 0x9000u : 0u));
            storeLE32(task, offset + 32,
                      (loadLE32(task, offset + 32) & ~(1u << 23)) |
                      (last ? 0u : 1u << 23));
        }
        const std::uint32_t commonIn = plan.kind == ConvKind::Depthwise ? in.channels : plan.inputPerTask;
        const std::uint32_t commonOut = plan.kind == ConvKind::Depthwise ? out.channels : slice;
        putRegister(task, offset, bytes, 0x00000, (in.height << 16) | in.width);
        putRegister(task, offset, bytes, 0x0000c, commonIn);
        putRegister(task, offset, bytes, 0x00010, commonOut);
        putRegister(task, offset, bytes, 0x00014, (out.height << 16) | out.width);
        const std::uint32_t baseCfg = loadLE32(task,
            h13RegisterOffset(task, offset, bytes, 0x0001c));
        const std::uint32_t ocg =
            plan.kind == ConvKind::Depthwise ? 0 : floorLog2(lanes);
        const std::uint32_t convCfg = (baseCfg & 0xf8000000u) |
            shape.kernelWidth | (shape.kernel << 5) | (ocg << 10) |
            (shape.stride << 13) | (shape.stride << 15) |
            (static_cast<std::uint32_t>(pad) << 17) |
            (static_cast<std::uint32_t>(pad) << 22);
        putRegister(task, offset, bytes, 0x0001c, convCfg);
        putRegister(task, offset, bytes, 0x00028, plan.tileHeight);
        putRegister(task, offset, bytes, 0x00034, common34);
        putRegister(task, offset, bytes, 0x0480c, plan.rowL2);
        putRegister(task, offset, bytes, 0x04810, plan.sourceRow);
        putRegister(task, offset, bytes, 0x04814, commonIn * plan.rowL2);
        putRegister(task, offset, bytes, 0x04818, commonIn * plan.rowL2);
        putRegister(task, offset, bytes, 0x04834, plan.tileRows * plan.sourceRow);
        putRegister(task, offset, bytes, 0x1380c, static_cast<std::uint32_t>(inRow));
        putRegister(task, offset, bytes, 0x13810, static_cast<std::uint32_t>(inPlane));
        putRegister(task, offset, bytes, 0x13814,
                    static_cast<std::uint32_t>(inPlane * in.channels));
        putRegister(task, offset, bytes, 0x13818,
                    static_cast<std::uint32_t>(inPlane * commonIn));
        putRegister(task, offset, bytes, 0x17804,
                    static_cast<std::uint32_t>(outPlane * firstChannel));
        putRegister(task, offset, bytes, 0x17808, static_cast<std::uint32_t>(outRow));
        putRegister(task, offset, bytes, 0x1780c, static_cast<std::uint32_t>(outPlane));
        putRegister(task, offset, bytes, 0x17810,
                    static_cast<std::uint32_t>(outPlane * out.channels));
        putRegister(task, offset, bytes, 0x17814,
                    static_cast<std::uint32_t>(outPlane * commonOut));
        for (std::uint32_t i = 0; i != 16; ++i) {
            putRegister(task, offset, bytes, 0x1f848 + 4 * i, sectionBase + offsets[i]);
            putRegister(task, offset, bytes, 0x1f888 + 4 * i, sizes[i]);
        }
        firstChannel += slice;
    }
    program.inputs = {input};
    program.output = output;
    program.formulaBuilt = true;
    program.depthwiseConv = plan.kind == ConvKind::Depthwise;
    program.constantOffsetBytes = alignUp(program.task.size(), 0x40);
    program.scratchAllocationBytes = 0;
    if (program.constants.empty()) program.constants.assign(64, 0);
    return program;
}

namespace {

/// One task of a program's linked stream: where it sits and how long it is.
struct TaskSpan {
    std::size_t offset;
    std::size_t bytes;
};

std::vector<TaskSpan> taskSpans(const Program &program) {
    std::vector<TaskSpan> spans;
    std::size_t offset = 0, bytes = program.firstTaskBytes;
    for (std::uint32_t index = 0; index != program.taskCount; ++index) {
        if (bytes < 40 || offset + bytes > program.task.size())
            throw std::invalid_argument(
                "h13.linked-outside-envelope: layer task stream is truncated");
        spans.push_back({offset, bytes});
        if (index + 1 == program.taskCount) break;
        const auto next = loadLE32(program.task, offset + 28);
        const auto header1 = loadLE32(program.task, offset + 4);
        if (next < offset + bytes || next % 4)
            throw std::invalid_argument(
                "h13.linked-outside-envelope: layer task link is invalid");
        offset = next;
        bytes = (((header1 >> 16) & 0x1ff) + 1) * sizeof(std::uint32_t);
    }
    return spans;
}

bool sameLayout(const TensorLayout &left, const TensorLayout &right) {
    return left.nchw == right.nchw && left.elementSize == right.elementSize;
}

} // namespace

namespace {

/// H13 L2 address space: 17-bit base and stride fields at 16-byte
/// granularity, so every L2 byte a task touches sits below 2 MiB.
constexpr std::uint64_t kL2Bytes = 0x200000;

enum class EdgeMode { Dram, Alias, Dependent };

/// What one single-task layer occupies in L2, read back from its words.
struct LayerGeometry {
    bool depthwise;
    std::uint32_t inHeight, inWidth, outHeight, outWidth, outChannels, tileHeight, kernel;
    std::uint32_t sourceRow, sourceChannel, sourceDepth;   // 0x4810 / 0x480c / 0x4814
    std::uint64_t stagedInput;   // rows the input DMA stages * source row
    std::uint64_t staging;       // the DRAM-output staging region
};

LayerGeometry layerGeometry(const Program &layer) {
    const auto &t = layer.task;
    const std::size_t bytes = layer.firstTaskBytes;
    auto reg = [&](std::uint32_t a) { return loadLE32(t, h13RegisterOffset(t, 0, bytes, a)); };
    LayerGeometry g{};
    g.depthwise = layer.depthwiseConv;
    g.kernel = reg(0x1c) & 0x1f;
    g.inHeight = reg(0x0) >> 16; g.inWidth = reg(0x0) & 0xffff;
    g.outHeight = reg(0x14) >> 16; g.outWidth = reg(0x14) & 0xffff;
    g.outChannels = reg(0x10);
    g.tileHeight = reg(0x28);
    g.sourceRow = reg(0x4810); g.sourceChannel = reg(0x480c); g.sourceDepth = reg(0x4814);
    g.stagedInput = reg(0x4834);
    const std::uint64_t outRow = std::max<std::uint64_t>(16, (2ull * g.outWidth + 15) / 16 * 16);
    g.staging = static_cast<std::uint64_t>(g.tileHeight) * g.outChannels * outRow;
    return g;
}

std::uint64_t align16(std::uint64_t v) { return (v + 15) & ~15ull; }

/// Chooses which edges stay in L2 and places every L2 region, following the
/// two edge forms Apple's compiler emits (t-3891): the alias edge after a
/// dense producer and the dependent edge from a depthwise producer into a
/// 1x1 consumer at spatial 8..64. Only single-task layers whose whole output
/// is one tile qualify, and an edge is taken only when every region the two
/// tasks touch fits in 2 MiB without overlap; everything else stays in DRAM.
/// Placement is Apple's: a producer that stages its input from DRAM leaves
/// its result at 0 and stages the input above it; a task whose input is
/// resident puts its own result (or output staging) right above that input,
/// or at 0 when that does not fit and the input starts high enough.
struct L2Plan {
    std::vector<EdgeMode> edge;              // edge[i]: layer i -> i + 1
    std::vector<std::uint64_t> resultBase;   // where layer i's L2 result sits
    std::vector<std::uint64_t> resultBytes;
    std::vector<std::uint64_t> sourceBase;   // layer i's Source1Base
    std::vector<std::uint64_t> stagingBase;  // layer i's DRAM-output staging base
};

L2Plan planL2(const std::vector<Program> &layers) {
    const std::size_t n = layers.size();
    L2Plan plan;
    plan.edge.assign(n - 1, EdgeMode::Dram);
    plan.resultBase.assign(n, 0); plan.resultBytes.assign(n, 0);
    plan.sourceBase.assign(n, 0); plan.stagingBase.assign(n, 0);
    std::vector<LayerGeometry> g;
    for (const auto &layer : layers)
        g.push_back(layer.nativeDramLayer ? LayerGeometry{} : layerGeometry(layer));
    for (std::size_t i = 0; i < n; ++i) plan.stagingBase[i] = g[i].stagedInput;
    // Only the task forms Apple's compiler is observed to emit (t-3891):
    //   dense (DRAM in) -alias-> depthwise -dependent-> dense 1x1 (DRAM out)
    //   dense (DRAM in) -alias-> dense (DRAM out)
    // at spatial 8..64, every layer one task with its whole output in one
    // tile. A task that both reads and writes L2 is only ever the depthwise
    // middle of the first form. Everything else stays in DRAM.
    auto single = [&](std::size_t k) {
        return !layers[k].nativeDramLayer && layers[k].taskCount == 1 && g[k].tileHeight == g[k].outHeight &&
               g[k].outHeight >= 8 && g[k].outHeight <= 64 &&
               g[k].outWidth >= 8 && g[k].outWidth <= 64 &&
               g[k].inHeight == g[k].outHeight && g[k].inWidth == g[k].outWidth;
    };
    auto edgeBytes = [&](std::size_t consumer) {
        return static_cast<std::uint64_t>(g[consumer].inHeight) * g[consumer].sourceRow;
    };
    std::size_t i = 0;
    while (i + 1 < n) {
        const bool producerOk = single(i) && !g[i].depthwise &&
            (i == 0 || plan.edge[i - 1] == EdgeMode::Dram);
        const bool triple = producerOk && i + 2 < n && single(i + 1) && single(i + 2) &&
            g[i + 1].depthwise && !g[i + 2].depthwise && g[i + 2].kernel == 1;
        const bool pair = producerOk && !triple && single(i + 1) && !g[i + 1].depthwise;
        if (!triple && !pair) { ++i; continue; }
        // Producer: result at 0, its DRAM-staged input above it.
        const std::uint64_t e1 = edgeBytes(i + 1);
        const std::uint64_t src0 = align16(e1);
        bool ok = src0 + g[i].stagedInput <= kL2Bytes;
        // Middle (or final) consumer: input at 0, its result/staging above.
        const std::uint64_t at1 = align16(e1);
        std::uint64_t e2 = 0, at2 = 0;
        if (triple) {
            e2 = edgeBytes(i + 2);
            at2 = align16(at1 + e2);
            ok = ok && at1 + e2 <= kL2Bytes && at2 + g[i + 2].staging <= kL2Bytes;
        } else {
            ok = ok && at1 + g[i + 1].staging <= kL2Bytes;
        }
        if (!ok) { ++i; continue; }
        plan.edge[i] = EdgeMode::Alias;
        plan.resultBase[i] = 0; plan.resultBytes[i] = e1; plan.sourceBase[i] = src0;
        plan.sourceBase[i + 1] = 0;
        if (triple) {
            plan.edge[i + 1] = EdgeMode::Dependent;
            plan.resultBase[i + 1] = at1; plan.resultBytes[i + 1] = e2;
            plan.sourceBase[i + 2] = at1;
            plan.stagingBase[i + 2] = at2;
            i += 3;
        } else {
            plan.stagingBase[i + 1] = at1;
            i += 2;
        }
    }
    return plan;
}

} // namespace

Program linkConvChain(const std::vector<Program> &layers,
                      std::vector<LinkedLayer> *layout, bool l2Resident) {
    if (layers.size() < 2)
        throw std::invalid_argument(
            "h13.linked-outside-envelope: a linked program needs two or more layers");
    constexpr std::array<std::uint32_t, 4> convChannels{5, 4, 6, 7};
    for (std::size_t index = 0; index != layers.size(); ++index) {
        const auto &layer = layers[index];
        const bool firstConcat = index == 0 && layer.nativeDramLayer && layer.inputs.size() == 2;
        if ((!layer.formulaBuilt && !layer.nativeDramLayer) || (layer.inputs.size() != 1 && !firstConcat) ||
            layer.inputs[0].index != 5 || (firstConcat && layer.inputs[1].index != 6) || layer.output.index != 4 ||
            layer.taskSurfaceChannels != convChannels ||
            layer.scratchAllocationBytes || !layer.taskCount)
            throw std::invalid_argument(
                "h13.linked-outside-envelope: every layer must be a one-input "
                "formula-built convolution");
        if (index && !sameLayout(layers[index - 1].output, layer.inputs[0]))
            throw std::invalid_argument(
                "h13.linked-outside-envelope: a layer's input surface differs "
                "from the previous layer's output surface");
    }

    // Scratch arena: one region per intermediate, never reused, so the only
    // ordering the stream relies on is producer before consumer.
    L2Plan l2;
    l2.edge.assign(layers.size() - 1, EdgeMode::Dram);
    if (l2Resident) l2=planL2(layers); // Native layers retain their qualified DRAM edges.
    std::vector<std::uint64_t> regions;
    std::uint64_t arena = 0;
    for (std::size_t index = 0; index + 1 != layers.size(); ++index) {
        regions.push_back(arena);
        if (l2.edge[index] == EdgeMode::Dram)
            arena += alignTile(layers[index].output.allocationBytes);
    }
    if (arena > 0xffffffffull)
        throw std::invalid_argument(
            "h13.linked-outside-envelope: the scratch arena exceeds the 32-bit "
            "tile-DMA base");

    Program program;
    program.taskSurfaceChannels = convChannels;
    program.inputs = layers.front().inputs;
    program.output = layers.back().output;
    program.formulaBuilt = true;
    program.scratchAllocationBytes = arena;

    std::uint32_t totalTasks = 0;
    std::vector<std::vector<TaskSpan>> spans;
    for (const auto &layer : layers) {
        spans.push_back(taskSpans(layer));
        totalTasks += layer.taskCount;
    }
    if (totalTasks > 0xffff)
        throw std::invalid_argument(
            "h13.linked-outside-envelope: more than 65535 tasks");

    std::uint32_t global = 0;
    std::size_t cursor = 0;
    if (layout) layout->clear();
    for (std::size_t index = 0; index != layers.size(); ++index) {
        const auto &layer = layers[index];
        const std::uint32_t constantBase =
            static_cast<std::uint32_t>(program.constants.size());
        const EdgeMode inEdge = index ? l2.edge[index - 1] : EdgeMode::Dram;
        const EdgeMode outEdge = index + 1 != layers.size() ? l2.edge[index] : EdgeMode::Dram;
        const bool inL2 = inEdge != EdgeMode::Dram;
        const bool outL2 = outEdge != EdgeMode::Dram;
        const bool readsScratch = index != 0 && !inL2;
        const bool writesScratch = index + 1 != layers.size() && !outL2;
        if (layout) {
            LinkedLayer entry{global, layer.taskCount, readsScratch,
                              readsScratch ? regions[index - 1] : 0,
                              writesScratch,
                              writesScratch ? regions[index] : 0};
            entry.inputEdge = static_cast<int>(inEdge);
            entry.outputEdge = static_cast<int>(outEdge);
            entry.l2Source = inL2 ? l2.sourceBase[index] : 0;
            entry.l2Result = outL2 ? l2.resultBase[index] : 0;
            entry.l2ResultBytes = outL2 ? l2.resultBytes[index] : 0;
            layout->push_back(entry);
        }
        bool nativeBoundaryRead = false;
        for (const auto &span : spans[index]) {
            const bool last = global + 1 == totalTasks;
            const std::size_t offset = cursor;
            program.task.resize(offset + span.bytes, 0);
            std::copy(layer.task.begin() + static_cast<std::ptrdiff_t>(span.offset),
                      layer.task.begin() + static_cast<std::ptrdiff_t>(span.offset + span.bytes),
                      program.task.begin() + static_cast<std::ptrdiff_t>(offset));
            auto &task = program.task;
            const std::size_t bytes = span.bytes;
            // Linked-stream header, the form the generic multi-task
            // convolution already runs on the M1 Pro: task id with the
            // continuation bit after the first task and the end bit on the
            // last; every task keeps the DRAM source and destination bits
            // (header[6] 0x30000800) because every task loads its input
            // surface from DRAM; the end-of-stream event bits go on the last
            // task only; TBE (header[8] bit 23) on every task with a
            // successor.
            std::uint32_t word0 = global;
            if (global) word0 |= 1u << 24;
            if (last) word0 |= 1u << 24 | 1u << 25;
            storeLE32(task, offset, word0);
            const std::uint32_t exeCycles = loadLE32(task, offset + 4) & 0xffffu;
            std::size_t nextOffset = 0, nextBytes = 0;
            if (!last) {
                nextOffset = alignUp(offset + bytes, 0x100);
                const auto &nextSpan = index + 1 != layers.size() &&
                    &span == &spans[index].back()
                    ? spans[index + 1].front()
                    : *(&span + 1);
                nextBytes = nextSpan.bytes;
                if (nextBytes % 4 || nextBytes / 4 - 1 > 0x1ff)
                    throw std::invalid_argument(
                        "h13.linked-outside-envelope: task exceeds the 9-bit NextSize field");
            }
            storeLE32(task, offset + 4,
                      last ? exeCycles
                           : exeCycles | ((static_cast<std::uint32_t>(nextBytes / 4) - 1) << 16));
            if (layer.nativeDramLayer) {
                // Preserve a native layer's internal L2 tasks. Some tasks
                // stage a per-channel scale from the constant BAR; only the
                // task that actually selects input channel 4 reads the edge.
                auto regWord = [&](std::uint32_t address) {
                    return h13RegisterOffset(task, offset, bytes, address);
                };
                auto regValue = [&](std::uint32_t address) { return loadLE32(task, regWord(address)); };
                auto selectors = loadLE32(task, offset + 32);
                const auto srcConfig = regValue(0x13800);
                const auto dstConfig = regValue(0x17800);
                const auto readChannel = selectors & 31;
                if (srcConfig & 1) {
                    const auto base = regWord(0x13808);
                    if (readChannel == 4 && (selectors & (1u << 5))) {
                        if (readsScratch) {
                            if (loadLE32(task, base) || regValue(0x13814) > layer.inputs[0].allocationBytes)
                                throw std::invalid_argument("h13.linked-outside-envelope: native input exceeds its region");
                            selectors = (selectors & ~31u) | 3u;
                            storeLE32(task, base, static_cast<std::uint32_t>(regions[index - 1]));
                            if (srcConfig != 0x33881)
                                throw std::invalid_argument("h13.linked-outside-envelope: unknown native source DMA config");
                            storeLE32(task, regWord(0x13800),
                                !nativeBoundaryRead && &span == &spans[index].front() ? 0x7bb81 : 0x3bb81);
                        }
                        nativeBoundaryRead = true;
                    } else if (index == 0 && layer.inputs.size() == 2 && readChannel == 6 && (selectors & (1u << 5))) {
                        if (loadLE32(task, base) || regValue(0x13814) > layer.inputs[1].allocationBytes)
                            throw std::invalid_argument("h13.linked-outside-envelope: second concat input exceeds its surface");
                    } else if (readChannel == 1 && (selectors & (1u << 5))) {
                        if (static_cast<std::uint64_t>(loadLE32(task, base)) + regValue(0x13814) > layer.constants.size())
                            throw std::invalid_argument("h13.linked-outside-envelope: native constant read exceeds its section");
                        storeLE32(task, base, loadLE32(task, base) + constantBase);
                    } else throw std::invalid_argument("h13.linked-outside-envelope: unknown native source channel");
                }
                if (dstConfig & 1) {
                    if (((selectors >> 12) & 31) != 5 || !(selectors & (1u << 17)))
                        throw std::invalid_argument("h13.linked-outside-envelope: unknown native destination channel");
                    if (writesScratch) {
                        const auto base = regWord(0x17804);
                        if (static_cast<std::uint64_t>(loadLE32(task, base)) +
                            layer.output.nchw[4] * regValue(0x10) > layer.output.allocationBytes)
                            throw std::invalid_argument("h13.linked-outside-envelope: native output exceeds its region");
                        selectors = (selectors & ~(31u << 12)) | (3u << 12);
                        storeLE32(task, base, loadLE32(task, base) + static_cast<std::uint32_t>(regions[index]));
                        if (dstConfig != 0x40000c1)
                            throw std::invalid_argument("h13.linked-outside-envelope: unknown native destination DMA config");
                        storeLE32(task, regWord(0x17800), 0x40003c1);
                    }
                }
                // Internal task headers and selectors keep their decoded
                // engine/L2 roles; only stream termination and links move.
                storeLE32(task, offset + 8, (loadLE32(task, offset + 8) & ~0x22u) | (last ? 0x22u : 0u));
                storeLE32(task, offset + 16, (loadLE32(task, offset + 16) & ~2u) | (last ? 2u : 0u));
                if (&span == &spans[index].back())
                    storeLE32(task, offset + 24, loadLE32(task, offset + 24) | 0x9000u);
                storeLE32(task, offset + 28, static_cast<std::uint32_t>(nextOffset));
                selectors = (selectors & ~(1u << 23)) | (last ? 0u : 1u << 23);
                storeLE32(task, offset + 32, selectors);
                for (unsigned core = 0; core != 16; ++core)
                    if (regValue(0x1f808 + core * 4) & 1) {
                        const auto word = regWord(0x1f848 + core * 4);
                        const auto first = loadLE32(task, word);
                        if (static_cast<std::uint64_t>(first) + regValue(0x1f888 + core * 4) > layer.constants.size())
                            throw std::invalid_argument("h13.linked-outside-envelope: native kernel read exceeds its section");
                        storeLE32(task, word, first + constantBase);
                    }
                cursor = last ? offset + bytes : nextOffset;
                ++global;
                continue;
            }
            // Event words: the input-DMA bits only when the task loads its
            // input from DRAM (t-3891: L2-reading tasks use 0x400 / 0xfff800).
            storeLE32(task, offset + 8, 0x400u | (inL2 ? 0u : 0x08u) | (last ? 0x22u : 0u));
            storeLE32(task, offset + 16,
                      last ? 0x00fff86au : (inL2 ? 0x00fff800u : 0x00fff868u));
            // TSE + TSR (header[6] bits 15 and 12) on the last task of every
            // layer. Apple's decoded streams that stage through scratch set
            // them on the last producer task before a consumer reads the
            // region (encoder_conv_c1024_n2048 task 4, chain_ffn_d1024_s375
            // task 16). Without them a consumer's input DMA can run ahead of
            // the producer's output DMA: the M1 Pro read stale scratch on a
            // cold run (dw 3x3 512 @14x14, prefix 16 of MobileNetV1).
            const bool layerEnd = &span == &spans[index].back();
            // L2 edges (t-3891): SrcLoc/DstLoc only for DRAM ends; SPL unless
            // the input arrives over a dependent edge; SPC on an alias
            // consumer, DPC on an alias producer; TSE+TSR closes every layer
            // except a dependent producer.
            std::uint32_t control = 0;
            if (!inL2) control |= 0x10000000u;
            if (!outL2) control |= 0x20000000u;
            if (inEdge != EdgeMode::Dependent) control |= 0x800u;
            if (inEdge == EdgeMode::Alias) control |= 0x2000u;
            if (outEdge == EdgeMode::Alias) control |= 0x4000u;
            if (layerEnd && (outEdge != EdgeMode::Dependent || last)) control |= 0x9000u;
            storeLE32(task, offset + 24, control);
            storeLE32(task, offset + 28, static_cast<std::uint32_t>(nextOffset));
            std::uint32_t selectors = loadLE32(task, offset + 32);
            if (((selectors >> 0) & 31) != 4 || !(selectors & (1u << 5)) ||
                ((selectors >> 12) & 31) != 5 || !(selectors & (1u << 17)))
                throw std::invalid_argument(
                    "h13.linked-outside-envelope: layer task does not read "
                    "channel 4 and write channel 5");
            if (inL2) {
                // Resident input: L2Read source at the producer's result base,
                // input DMA disabled. Alias form keeps the DMA geometry and
                // RBase0=3/RBE; dependent form zeroes both (t-3891).
                const bool alias = inEdge == EdgeMode::Alias;
                putRegister(task, offset, bytes, 0x04804, alias ? 0x160u : 0x104u);
                putRegister(task, offset, bytes, 0x04808,
                            static_cast<std::uint32_t>(l2.sourceBase[index]));
                putRegister(task, offset, bytes, 0x04818, 0);
                putRegister(task, offset, bytes, 0x13800, alias ? 0x48880u : 0x8880u);
                if (alias) {
                    selectors = (selectors & ~31u) | 3u;
                } else {
                    selectors &= ~(31u | (1u << 5));
                    for (std::uint32_t a : {0x13808u, 0x1380cu, 0x13810u, 0x13814u, 0x13818u})
                        putRegister(task, offset, bytes, a, 0);
                    putRegister(task, offset, bytes, 0x13838, 0x01000000u);
                }
                if (!outL2) {
                    putRegister(task, offset, bytes, 0x04830, 0x14a);
                    putRegister(task, offset, bytes, 0x04834,
                                static_cast<std::uint32_t>(l2.stagingBase[index]));
                }
            } else if (outL2) {
                // DRAM-staged input of an L2 producer sits above its result.
                putRegister(task, offset, bytes, 0x04808,
                            static_cast<std::uint32_t>(l2.sourceBase[index]));
            }
            if (outL2) {
                // Resident result in the consumer's source layout.
                const bool alias = outEdge == EdgeMode::Alias;
                const auto &next = layers[index + 1];
                auto nextReg = [&](std::uint32_t a) {
                    return loadLE32(next.task, h13RegisterOffset(next.task, 0, next.firstTaskBytes, a));
                };
                putRegister(task, offset, bytes, 0x04830, alias ? 0x500170u : 0x140u);
                putRegister(task, offset, bytes, 0x04834, static_cast<std::uint32_t>(l2.resultBase[index]));
                putRegister(task, offset, bytes, 0x04838, nextReg(0x0480c));
                putRegister(task, offset, bytes, 0x0483c, nextReg(0x04810));
                putRegister(task, offset, bytes, 0x04840, nextReg(0x04814));
                putRegister(task, offset, bytes, 0x04844, 0);
                putRegister(task, offset, bytes, 0x17800, 0xc0);
                if (alias) {
                    selectors = (selectors & ~(31u << 12)) | (3u << 12);
                } else {
                    selectors &= ~((31u << 12) | (1u << 17));
                    for (std::uint32_t a : {0x17804u, 0x17808u, 0x1780cu, 0x17810u, 0x17814u})
                        putRegister(task, offset, bytes, a, 0);
                    putRegister(task, offset, bytes, 0x17818, 0x01000000u);
                    const auto c24 = loadLE32(task, h13RegisterOffset(task, offset, bytes, 0x00024));
                    putRegister(task, offset, bytes, 0x00024, c24 | 0x4000u);
                }
            }
            if (readsScratch) {
                const std::uint64_t depth = loadLE32(task,
                    h13RegisterOffset(task, offset, bytes, 0x13814));
                if (loadLE32(task, h13RegisterOffset(task, offset, bytes, 0x13808)) ||
                    depth > layer.inputs[0].allocationBytes)
                    throw std::invalid_argument(
                        "h13.linked-outside-envelope: a task reads past its scratch region");
                selectors = (selectors & ~31u) | 3u;
                putRegister(task, offset, bytes, 0x13808,
                            static_cast<std::uint32_t>(regions[index - 1]));
                // TileDMASrc DMAConfig bits 19:16 are DependencyMode (public
                // H13 register map). The template's program-input value is 3;
                // Apple's first scratch read after a TSE/TSR producer uses 7
                // (chain_ffn_d1024_s375 task 17), the later reads of the same
                // region 3 again. Bit 2 is the wait on the producer's release.
                // Cache hints too: every decoded task that reads scratch uses
                // CacheHintReuse = CacheHintNoReuse = 0xb (0x3bb81 on 1076
                // tasks); none uses the program-input value 0x33881.
                {
                    const auto word = h13RegisterOffset(task, offset, bytes, 0x13800);
                    const auto cfg = loadLE32(task, word);
                    if (cfg != 0x00033881u)
                        throw std::invalid_argument(
                            "h13.linked-outside-envelope: unexpected input DMA config");
                    storeLE32(task, word, &span == &spans[index].front()
                                              ? 0x0007bb81u : 0x0003bb81u);
                }
            }
            if (writesScratch) {
                selectors = (selectors & ~(31u << 12)) | (3u << 12);
                const auto word = h13RegisterOffset(task, offset, bytes, 0x17804);
                const std::uint64_t first = loadLE32(task, word);
                const std::uint64_t depth = loadLE32(task,
                    h13RegisterOffset(task, offset, bytes, 0x17814));
                if (first + depth > layer.output.allocationBytes)
                    throw std::invalid_argument(
                        "h13.linked-outside-envelope: a task writes past its scratch region");
                storeLE32(task, word, static_cast<std::uint32_t>(first + regions[index]));
                // Every decoded task that writes scratch sets bits 9:8 of its
                // output DMA config (0x40003c1); program outputs use 0x40000c1.
                const auto dst = h13RegisterOffset(task, offset, bytes, 0x17800);
                if (loadLE32(task, dst) != 0x040000c1u)
                    throw std::invalid_argument(
                        "h13.linked-outside-envelope: unexpected output DMA config");
                storeLE32(task, dst, 0x040003c1u);
            }
            selectors = (selectors & ~(1u << 23)) | (last ? 0u : 1u << 23);
            storeLE32(task, offset + 32, selectors);
            for (std::uint32_t chunk = 0; chunk != 16; ++chunk) {
                const auto word = h13RegisterOffset(task, offset, bytes,
                                                    0x1f848 + 4 * chunk);
                const std::uint64_t moved =
                    static_cast<std::uint64_t>(loadLE32(task, word)) + constantBase;
                if (moved > 0xffffffffull)
                    throw std::invalid_argument(
                        "h13.linked-outside-envelope: kernel offset overflows");
                storeLE32(task, word, static_cast<std::uint32_t>(moved));
            }
            cursor = last ? offset + bytes : alignUp(offset + bytes, 0x100);
            ++global;
        }
        program.constants.insert(program.constants.end(), layer.constants.begin(),
                                 layer.constants.end());
        // Preserve the alignment of the native 128-byte activation tables.
        program.constants.resize(alignUp(program.constants.size(), 128), 0);
    }
    program.taskCount = totalTasks;
    program.firstTaskBytes = spans.front().front().bytes;
    program.constantOffsetBytes = alignUp(program.task.size(), 0x40);
    // The driver places the kernel base at the task length rounded to 16
    // bytes; the ANEC places constants at the 64-byte rounding. They agree
    // only when the last task starts 64-byte aligned, which the 0x100 task
    // placement guarantees; check it rather than assume it.
    if (alignUp(program.task.size(), 0x10) != program.constantOffsetBytes)
        throw std::logic_error(
            "h13.linked-outside-envelope: kernel base would not match the constant offset");
    return program;
}

Program linkLayerGraph(const std::vector<Program> &layers,
    const std::vector<ane::layers::Connection> &connections,
    const std::vector<std::string> &boundaries,const std::string &result,
    ane::layers::GraphPlan *placement,std::vector<LinkedLayer> *layout) {
    auto fail=[](const char *why)->void {
        throw std::invalid_argument(std::string("h13.linked-outside-envelope: ")+why);
    };
    auto plan=ane::layers::planGraph(layers,connections,boundaries,result,"h13.linked-outside-envelope");
    std::vector<std::vector<TaskSpan>> spans;
    std::vector<std::size_t> sizes;
    for (std::size_t i=0;i<layers.size();++i) {
        const auto &p=layers[i];
        if ((!p.formulaBuilt && !p.nativeDramLayer) || p.scratchAllocationBytes ||
            p.taskSurfaceChannels!=std::array<std::uint32_t,4>{5,4,6,7} || p.output.index!=4)
            fail("graph layer has no qualified DRAM routing");
        for (std::size_t j=0;j<p.inputs.size();++j)
            if (p.inputs[j].index!=5+j || p.inputs[j].elementSize!=2) fail("invalid graph input role");
        spans.push_back(taskSpans(p));
        if (!plan.zeroCopy[i]) for (const auto &span:spans.back()) sizes.push_back(span.bytes);
    }
    if (sizes.empty() || sizes.size()>65535) fail("native graph task budget exceeded");
    Program out;
    out.taskSurfaceChannels={5,4,6,7};
    out.inputs.resize(boundaries.size());
    for (std::size_t i=0;i<layers.size();++i)
        for (std::size_t j=0;j<plan.inputs[i].size();++j)
            if (plan.inputs[i][j].boundary) {
                const auto b=plan.inputs[i][j].index;
                out.inputs[b]=layers[i].inputs[j];out.inputs[b].index=static_cast<unsigned>(5+b);
            }
    out.output=layers[plan.result].output;
    out.scratchAllocationBytes=plan.scratchBytes;
    std::vector<std::int64_t> writer(layers.size(),-1);
    unsigned global=0;
    std::size_t cursor=0;
    if (layout) layout->clear();
    for (std::size_t i=0;i<layers.size();++i) {
        const auto &layer=layers[i];
        const auto target=plan.outputs[i];
        const auto constantBase=out.constants.size();
        if (constantBase>0xffffffffull-layer.constants.size()) fail("graph constants exceed 32 bits");
        bool readsScratch=false;
        std::uint64_t firstRead=0;
        for (const auto ref:plan.inputs[i])
            if (!ref.boundary) { readsScratch=true;firstRead=plan.outputs[ref.index].offset;break; }
        if (layout) layout->push_back({global,plan.zeroCopy[i]?0:layer.taskCount,readsScratch,firstRead,
                                      !target.output,target.offset});
        if (plan.zeroCopy[i]) continue;
        bool wrote=false;
        for (const auto span:spans[i]) {
            const auto offset=cursor,bytes=span.bytes;
            const bool last=global+1==sizes.size();
            out.task.resize(offset+bytes,0);
            std::copy(layer.task.begin()+span.offset,layer.task.begin()+span.offset+bytes,out.task.begin()+offset);
            auto &task=out.task;
            auto pos=[&](unsigned address){ return h13RegisterOffset(task,offset,bytes,address); };
            auto value=[&](unsigned address){ return loadLE32(task,pos(address)); };
            const auto next=last?0:alignUp(offset+bytes,0x100);
            if (next>0xffffffffu || (!last && sizes[global+1]/4-1>0x1ff)) fail("graph task link exceeds its field");
            unsigned id=global|(global?1u<<24:0u)|(last?(1u<<24)|(1u<<25):0u);
            storeLE32(task,offset,id);
            storeLE32(task,offset+4,(loadLE32(task,offset+4)&0xffffu)|
                (last?0:static_cast<unsigned>(sizes[global+1]/4-1)<<16));
            storeLE32(task,offset+8,(loadLE32(task,offset+8)&~0x22u)|(last?0x22u:0u));
            storeLE32(task,offset+16,(loadLE32(task,offset+16)&~2u)|(last?2u:0u));
            if (span.offset==spans[i].back().offset)
                storeLE32(task,offset+24,loadLE32(task,offset+24)|0x9000u);
            storeLE32(task,offset+28,static_cast<unsigned>(next));
            auto selectors=loadLE32(task,offset+32);
            for (unsigned source=0;source!=2;++source) {
                const auto cfg=0x13800u+source*4,base=0x13808u+source*0x34;
                if (!(value(cfg)&1)) continue;
                const auto shift=source*6,channel=(selectors>>shift)&31;
                if (!(selectors&(1u<<(shift+5)))) fail("active source has no declared channel");
                if (channel==1) {
                    if (static_cast<std::uint64_t>(value(base))+value(0x13814+source*0x34)>layer.constants.size())
                        fail("graph constant DMA exceeds its layer section");
                    storeLE32(task,pos(base),value(base)+static_cast<unsigned>(constantBase));
                    continue;
                }
                const unsigned input=channel==4?0:channel==6?1:3;
                if (input>=plan.inputs[i].size() || value(base) || value(cfg)!=0x33881)
                    fail("unqualified graph runtime source form");
                const auto ref=plan.inputs[i][input];
                unsigned bound;
                if (ref.boundary) bound=ref.index?6:4;
                else {
                    const auto route=plan.outputs[ref.index];
                    if (route.output || writer[route.owner]<0) fail("graph reads an unavailable surface");
                    bound=3;
                    storeLE32(task,pos(base),static_cast<unsigned>(route.offset));
                    storeLE32(task,pos(cfg),writer[route.owner]==static_cast<std::int64_t>(global)-1?0x7bb81:0x3bb81);
                }
                selectors=(selectors&~(31u<<shift))|(bound<<shift);
            }
            if (value(0x17800)&1) {
                if (((selectors>>12)&31)!=5 || !(selectors&(1u<<17)) || value(0x17800)!=0x40000c1)
                    fail("unqualified graph destination form");
                const auto first=value(0x17804);
                if (static_cast<std::uint64_t>(first)+layer.output.nchw[4]*value(0x10)>layer.output.allocationBytes ||
                    first+target.offset>0xffffffffu) fail("graph output exceeds its owning surface");
                selectors=(selectors&~(31u<<12))|((target.output?5u:3u)<<12);
                storeLE32(task,pos(0x17804),static_cast<unsigned>(first+target.offset));
                if (target.owner!=i) {
                    const auto &owner=layers[target.owner].output;
                    const auto depth=owner.nchw[1]*owner.nchw[4];
                    if (depth>0xffffffffu) fail("concat owner stride exceeds 32 bits");
                    storeLE32(task,pos(0x17810),static_cast<unsigned>(depth));
                }
                if (!target.output) storeLE32(task,pos(0x17800),0x40003c1);
                writer[target.owner]=global;wrote=true;
            }
            selectors=(selectors&~(1u<<23))|(last?0u:1u<<23);
            storeLE32(task,offset+32,selectors);
            for (unsigned core=0;core!=16;++core)
                if (value(0x1f808+core*4)&1) {
                    const auto first=value(0x1f848+core*4);
                    if (static_cast<std::uint64_t>(first)+value(0x1f888+core*4)>layer.constants.size())
                        fail("graph kernel DMA exceeds its layer section");
                    storeLE32(task,pos(0x1f848+core*4),first+static_cast<unsigned>(constantBase));
                }
            cursor=last?offset+bytes:next;++global;
        }
        if (!wrote) fail("native graph layer does not write its declared output");
        out.constants.insert(out.constants.end(),layer.constants.begin(),layer.constants.end());
        out.constants.resize(alignUp(out.constants.size(),128),0);
    }
    out.taskCount=global;out.firstTaskBytes=sizes.front();out.constantOffsetBytes=alignUp(out.task.size(),64);
    if (alignUp(out.task.size(),16)!=out.constantOffsetBytes) fail("graph kernel base differs from its constant offset");
    if (placement) *placement=std::move(plan);
    return out;
}

Program composePrograms(const std::vector<Program> &programs) {
    if (programs.empty())
        throw std::invalid_argument("H13 chain requires at least one program");
    if (programs.size() == 1) return programs.front();
    throw std::invalid_argument(
        "h13.chain-unrepresentable-edge: relinking standalone programs keeps "
        "every task's own surface routing; the decoded corpus wires task DMA to "
        "the fixed boundary channels (reads through Src1/Src2 to the input "
        "channels, writes through Dst to the output channel or L2) and provides "
        "no addressing for declared intermediate surfaces, while Apple's own "
        "chains keep intermediates L2-resident through the 0x04800-0x04844 "
        "words, which have no resolved formula, so no multi-program relink can "
        "be emitted correctly");
}

void fuseMatmulPostOperation(Program &program, PostOperation operation) {
    if (operation != PostOperation::Relu)
        throw std::invalid_argument(
            "h13.chain-unrepresentable-edge: only the relu post-operation has a "
            "decoded matmul epilogue; lookup-table and bias epilogues extend the "
            "per-lane kernel-DMA layout with section bytes the decoded records "
            "do not retain");
    if (program.taskCount != 2)
        throw std::invalid_argument(
            "h13.chain-unrepresentable-edge: matmul post-operation fusion is "
            "decoded only for the two-task parity form");
    const auto offset = loadLE32(program.task, 28);
    if (offset <= 28 || offset >= program.task.size() ||
        offset % sizeof(std::uint32_t))
        throw std::invalid_argument(
            "h13.chain-unrepresentable-edge: matmul compute task link is invalid");
    const auto bytes = program.task.size() - offset;
    const auto word =
        h13RegisterOffset(program.task, offset, bytes, 0x0c804);
    const auto value = loadLE32(program.task, word);
    if (value != 0x00101c00)
        throw std::invalid_argument(
            "h13.chain-unrepresentable-edge: matmul compute task is not the "
            "decoded no-post-operation form 0x00101c00");
    storeLE32(program.task, word, value | 0x00010000u);
}

void fuseConvPostOperation(Program &program, PostOperation operation) {
    if (operation != PostOperation::Relu)
        throw std::invalid_argument(
            "h13.chain-unrepresentable-edge: only the relu post-operation has a "
            "decoded convolution epilogue");
    std::size_t offset = 0;
    std::size_t bytes = program.firstTaskBytes;
    for (std::uint32_t index = 0; index != program.taskCount; ++index) {
        if (!bytes || offset + bytes > program.task.size())
            throw std::invalid_argument(
                "h13.chain-unrepresentable-edge: convolution task stream is truncated");
        const auto word = h13RegisterOffset(program.task, offset, bytes, 0x0c804);
        const auto value = loadLE32(program.task, word);
        if (value != 0x00101c00 && value != 0x00101c10)
            throw std::invalid_argument(
                "h13.chain-unrepresentable-edge: convolution NE word is not a "
                "decoded no-post-operation form");
        storeLE32(program.task, word, value | 0x00010000u);
        const auto header1 = loadLE32(program.task, offset + 4);
        const auto next = loadLE32(program.task, offset + 28);
        if (index + 1 == program.taskCount) break;
        if (!next)
            throw std::invalid_argument(
                "h13.chain-unrepresentable-edge: convolution task link is missing");
        offset = next;
        bytes = (((header1 >> 16) & 0x1ff) + 1) * sizeof(std::uint32_t);
    }
}

void fuseElementwisePostOperation(Program &program, PostOperation operation) {
    if (operation != PostOperation::Relu)
        throw std::invalid_argument(
            "h13.chain-unrepresentable-edge: only the relu post-operation has a "
            "decoded elementwise epilogue");
    if (program.taskCount != 1)
        throw std::invalid_argument(
            "h13.chain-unrepresentable-edge: elementwise post-operation fusion "
            "is decoded only for one-task programs");
    const auto word = h13RegisterOffset(program.task, 0,
                                        program.firstTaskBytes, 0x08800);
    const auto value = loadLE32(program.task, word);
    if (value != 0x00080000)
        throw std::invalid_argument(
            "h13.chain-unrepresentable-edge: elementwise PE word is not the "
            "decoded no-post-operation form 0x00080000");
    storeLE32(program.task, word, value | 0x20u);
}

} // namespace ane::h13
