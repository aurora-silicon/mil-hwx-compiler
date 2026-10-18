#include "H13Program.h"

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
        remaining -= 16 * lanes;
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
    const std::uint32_t slots = outputs / lanes;
    const std::uint64_t rows = taps + (bias ? 1 : 0);
    const std::uint64_t laneBytes = alignUp(slots * rows * 2, 64);
    std::vector<std::uint8_t> packed(laneBytes * lanes, 0);
    for (std::uint32_t column = 0; column != outputs; ++column) {
        std::size_t cursor = static_cast<std::size_t>(
            (column % lanes) * laneBytes + (column / lanes) * rows * 2);
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
std::vector<std::uint8_t> packConvStrided(std::uint32_t reduction,
                                          const std::vector<std::uint32_t> &chunks,
                                          const std::uint8_t *weights,
                                          const std::uint8_t *bias) {
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
                        if (value[0] || (value[1] & 0x7f)) {
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
