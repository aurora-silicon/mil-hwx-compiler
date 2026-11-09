#include "H13Program.h"
#include "../H16G/Encoding/H16ConvProgram.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ane::h13 {

namespace {

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment);
bool nonzeroHalfword(const std::uint8_t *value);
void appendHalfword(std::vector<std::uint8_t> &out,
                    const std::uint8_t *value);

using ResidualH13TapGroup = std::vector<unsigned>;
const std::array<ResidualH13TapGroup, 8> residualH133x3Phases = {{
    {0,2}, {6,8}, {1}, {7}, {3,5}, {}, {4}, {},
}};
const std::array<ResidualH13TapGroup, 16> residualH137x7Phases = {{
    {0,2,4,6}, {14,16,18,20}, {28,30,32,34}, {42,44,46,48},
    {1,3,5}, {15,17,19}, {29,31,33}, {43,45,47},
    {7,9,11,13}, {21,23,25,27}, {35,37,39,41}, {},
    {8,10,12}, {22,24,26}, {36,38,40}, {},
}};

std::vector<unsigned> residualH13LaneChunks(unsigned outputs, unsigned cap) {
    unsigned lanes = outputs / 16;
    std::vector<unsigned> chunks;
    while (lanes > cap) { chunks.push_back(cap); lanes -= cap; }
    if (lanes) chunks.push_back(lanes);
    return chunks;
}

template <std::size_t N>
void appendResidualH13PhasePlane(
    std::vector<std::uint8_t> &out,
    const std::array<ResidualH13TapGroup, N> &groups, unsigned cin,
    unsigned lanes, unsigned first, unsigned taps,
    const std::uint8_t *weights, const std::uint8_t *bias,
    unsigned perInputPadding) {
    if (bias)
        out.insert(out.end(), bias + 2 * first, bias + 2 * (first + lanes));
    const std::size_t lengthAt = out.size();
    out.insert(out.end(), 3, 0);
    const unsigned maxTaps = std::max_element(
        groups.begin(), groups.end(), [](const auto &left, const auto &right) {
            return left.size() < right.size();
        })->size();
    const std::size_t bodyAt = out.size();
    for (unsigned input = 0; input != cin; ++input) {
        unsigned structural = 0;
        if constexpr (N == 1) {
            // H13's k1 sparse form, like its phase form, places each mask
            // immediately before the eight lanes it selects.
            for (unsigned begin = 0; begin < lanes; begin += 8) {
                const unsigned width = std::min(8u, lanes - begin);
                std::uint8_t mask = 0;
                std::vector<std::uint8_t> values;
                for (unsigned lane = 0; lane != width; ++lane) {
                    const auto *value = weights + 2 *
                        ((static_cast<std::size_t>(first + begin + lane) * cin +
                          input) * taps);
                    if (!nonzeroHalfword(value)) continue;
                    mask |= static_cast<std::uint8_t>(1u << lane);
                    appendHalfword(values, value);
                }
                out.push_back(mask);
                out.insert(out.end(), values.begin(), values.end());
            }
        } else {
            const unsigned evenLanes = (lanes + 1) & ~1u;
            std::vector<const std::uint8_t *> slots;
            for (const auto &group : groups)
                for (unsigned tapIndex = 0; tapIndex != maxTaps; ++tapIndex)
                    for (unsigned lane = 0; lane != evenLanes; ++lane)
                        slots.push_back(tapIndex < group.size() && lane < lanes
                            ? weights + 2 *
                                ((static_cast<std::size_t>(first + lane) * cin +
                                  input) * taps + group[tapIndex])
                            : nullptr);
            if (slots.size() % 16)
                throw std::logic_error(
                    "H13 ResNet phase slots are not a whole mask vector");
            for (std::size_t at = 0; at != slots.size(); at += 16) {
                // H13 serializes each 8-bit mask immediately before the
                // values it selects; H16G writes the two masks first.
                for (unsigned half = 0; half != 2; ++half) {
                    std::uint8_t mask = 0;
                    std::vector<std::uint8_t> values;
                    for (unsigned bit = 0; bit != 8; ++bit) {
                        const auto *value = slots[at + half * 8 + bit];
                        if (!value || !nonzeroHalfword(value)) continue;
                        mask |= static_cast<std::uint8_t>(1u << bit);
                        appendHalfword(values, value);
                    }
                    out.push_back(mask);
                    out.insert(out.end(), values.begin(), values.end());
                }
            }
        }
        unsigned trailing = perInputPadding;
        if constexpr (N == 1)
            if (structural) --trailing;
        out.insert(out.end(), trailing, 0);
    }
    const std::size_t bytes = out.size() - bodyAt;
    if (bytes > 0xffffff)
        throw std::invalid_argument(
            "H13 ResNet sparse plane exceeds a 24-bit body count");
    out[lengthAt] = static_cast<std::uint8_t>(bytes);
    out[lengthAt + 1] = static_cast<std::uint8_t>(bytes >> 8);
    out[lengthAt + 2] = static_cast<std::uint8_t>(bytes >> 16);
}

std::vector<std::uint8_t> packResidualH13Stride2(
    ConvShape shape, const std::uint8_t *weights, const std::uint8_t *bias,
    std::array<std::uint32_t, 16> &chunkBytes) {
    if (shape.groups != 1 || shape.stride != 2 || !shape.bias ||
        !weights || !bias || !shape.output.channels ||
        shape.output.channels % 16 ||
        (shape.kernel != 1 && shape.kernel != 3 && shape.kernel != 7))
        throw std::invalid_argument("invalid H13 ResNet stride-2 convolution");
    const unsigned cap = std::max(shape.output.width, shape.output.height) <= 8
        ? 32 : 16;
    const auto chunks = residualH13LaneChunks(shape.output.channels, cap);
    std::vector<std::uint8_t> out;
    for (unsigned region = 0; region != 16; ++region) {
        const auto start = out.size();
        unsigned base = 0;
        for (unsigned lanes : chunks) {
            const unsigned first = base + region * lanes;
            if (shape.kernel == 7) {
                appendResidualH13PhasePlane(out, residualH137x7Phases,
                    shape.input.channels, lanes, first, 49, weights, bias, 0);
            } else if (shape.kernel == 3) {
                appendResidualH13PhasePlane(out, residualH133x3Phases,
                    shape.input.channels, lanes, first, 9, weights, bias, 0);
            } else {
                const std::array<ResidualH13TapGroup, 1> one = {{{0}}};
                const unsigned masks = (lanes + 7) / 8;
                appendResidualH13PhasePlane(out, one, shape.input.channels,
                    lanes, first, 1, weights, bias,
                    lanes / 2 > masks ? lanes / 2 - masks : 0);
            }
            base += 16 * lanes;
        }
        out.resize(alignUp(out.size(), 64), 0);
        chunkBytes[region] = static_cast<std::uint32_t>(out.size() - start);
    }
    return out;
}


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

// Header BaseEnable: kernel-carrying tasks need both the kernel table stream
// and kernel execution paths enabled.  PE-only tasks deliberately carry
// neither bit.
constexpr std::uint32_t kHeaderKernelTableEnable = 1u << 24;
constexpr std::uint32_t kHeaderKernelExecutionEnable = 1u << 26;
constexpr std::uint32_t kHeaderKernelEnableMask =
    kHeaderKernelTableEnable | kHeaderKernelExecutionEnable;

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
                  5u | (1u << 5) | (36u << 12) |
                      kHeaderKernelEnableMask,
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

// Native MobileNet 3x3/s2 uses the phase-sparse grammar, rather than the
// linear-reduction sparse grammar above. A mask bit names a (tap, lane) pair;
// the two phase rows cover the four stride phases in source-tap order.
using SparseMaskByte = std::vector<std::pair<unsigned, unsigned>>;
using SparsePhaseRow = std::array<SparseMaskByte, 2>;

const std::vector<SparsePhaseRow> kMobileDepthwisePhaseRows = {
    {SparseMaskByte{{0,0},{2,2},{4,6},{6,8}},
     SparseMaskByte{{0,1},{4,7}}},
    {SparseMaskByte{{0,3},{2,5}}, SparseMaskByte{{0,4}}},
};

const std::vector<SparsePhaseRow> kMobileDepthwisePhaseRows32 = {
    {SparseMaskByte{{0,0},{4,2}}, SparseMaskByte{{0,6},{4,8}}},
    {SparseMaskByte{{0,1}}, SparseMaskByte{{0,7}}},
    {SparseMaskByte{{0,3},{4,5}}, SparseMaskByte{}},
    {SparseMaskByte{{0,4}}, SparseMaskByte{}},
};

bool nonzeroHalfword(const std::uint8_t *value) {
    return value[0] || (value[1] & 0x7f);
}

void appendHalfword(std::vector<std::uint8_t> &out,
                    const std::uint8_t *value) {
    out.push_back(value[0]);
    out.push_back(value[1]);
}

template <class Weight>
void appendSparsePlane(std::vector<std::uint8_t> &out,
                       const std::vector<SparsePhaseRow> &rows,
                       unsigned repeats, unsigned lanes, Weight weight) {
    const std::size_t start = out.size();
    // u16 body length followed by the structural zero byte. The body length
    // covers masks and selected coefficients only.
    out.insert(out.end(), 3, 0);
    for (unsigned repeat = 0; repeat != repeats; ++repeat) {
        for (const auto &row : rows) {
            for (unsigned mask = 0; mask != 2; ++mask) {
                std::uint8_t bits = 0;
                std::vector<std::uint8_t> values;
                for (const auto &[bit, tap] : row[mask]) {
                    for (unsigned lane = 0; lane != lanes; ++lane) {
                        const auto *value = weight(repeat, lane, tap);
                        if (!nonzeroHalfword(value)) continue;
                        bits |= static_cast<std::uint8_t>(1u << (bit + lane));
                        appendHalfword(values, value);
                    }
                }
                // H13 serializes each mask immediately before the values it
                // selects. (Later generations put both masks first.)
                out.push_back(bits);
                out.insert(out.end(), values.begin(), values.end());
            }
        }
    }
    const std::size_t bodyBytes = out.size() - start - 3;
    if (bodyBytes > 0xffff)
        throw std::invalid_argument(
            "H13 native stride-2 sparse plane exceeds a 16-bit body count");
    out[start] = static_cast<std::uint8_t>(bodyBytes);
    out[start + 1] = static_cast<std::uint8_t>(bodyBytes >> 8);
}

std::vector<std::uint8_t> packMobileSparseStride2(
    const ConvShape &shape, const std::uint8_t *weights,
    const std::uint8_t *bias, std::array<std::uint32_t, 16> &offsets,
    std::array<std::uint32_t, 16> &sizes) {
    const bool depthwise = shape.groups == shape.input.channels &&
        shape.groups == shape.output.channels;
    const bool stem = shape.groups == 1 && shape.input.channels == 3 &&
        shape.output.channels == 32;
    if (shape.kernel != 3 || shape.kernelWidth != 3 || shape.stride != 2 ||
        (!stem && !depthwise))
        throw std::invalid_argument(
            "H13 native sparse packing covers the MobileNet stem and depthwise 3x3/s2 forms");

    std::vector<std::uint8_t> packed;
    for (unsigned chunk = 0; chunk != 16; ++chunk) {
        const std::size_t start = packed.size();
        offsets[chunk] = static_cast<std::uint32_t>(start);
        if (depthwise) {
            const unsigned perChunk = shape.input.channels / 16;
            const auto &rows = perChunk >= 32
                ? kMobileDepthwisePhaseRows32 : kMobileDepthwisePhaseRows;
            for (unsigned channel = chunk; channel < shape.input.channels;
                 channel += 16) {
                if (bias) appendHalfword(packed, bias + 2 * channel);
                appendSparsePlane(packed, rows, 1, 1,
                    [&](unsigned, unsigned, unsigned tap) {
                        return weights + 2 *
                            (static_cast<std::size_t>(channel) * 9 + tap);
                    });
            }
        } else {
            const unsigned output = 2 * chunk;
            if (bias) {
                appendHalfword(packed, bias + 2 * output);
                appendHalfword(packed, bias + 2 * (output + 1));
            }
            appendSparsePlane(packed, kMobileDepthwisePhaseRows,
                shape.input.channels, 2,
                [&](unsigned input, unsigned lane, unsigned tap) {
                    return weights + 2 *
                        ((static_cast<std::size_t>(output + lane) *
                          shape.input.channels + input) * 9 + tap);
                });
        }
        packed.resize(alignUp(packed.size(), 64), 0);
        sizes[chunk] = static_cast<std::uint32_t>(packed.size() - start);
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
const OracleConvTemplate *convFamilyBase(const ConvShape &shape,
                                         bool linkedRing = false) {
    const ConvKind kind = convKind(shape);
    const OracleConvTemplate *best = nullptr;
    for (const auto &candidate : kConvTasks) {
        if (candidate.taskCount != 1 || candidate.bias != shape.bias)
            continue;
        const bool exactFamily = candidate.kernel == shape.kernel &&
            candidate.kernelWidth == shape.kernelWidth &&
            candidate.stride == shape.stride;
        const bool ringStemBase = linkedRing && convKind(shape) == ConvKind::Dense &&
            (shape.kernel == 3 || shape.kernel == 7) && shape.stride == 2 &&
            candidate.kernel == 1 && candidate.kernelWidth == 1 &&
            candidate.stride == 2;
        const bool ringDepthwiseBase = linkedRing &&
            convKind(shape) == ConvKind::Depthwise && shape.kernel == 3 &&
            shape.stride == 2 && candidate.kernel == 3 &&
            candidate.kernelWidth == 3 && candidate.stride == 1;
        if (!exactFamily && !ringStemBase && !ringDepthwiseBase) continue;
        if (candidate.input.channels < 64 || candidate.input.height > 64 ||
            candidate.input.width > 64)
            continue;
        ConvShape probe{candidate.kernel, candidate.kernelWidth,
                        candidate.stride, candidate.groups, candidate.bias,
                        candidate.input, candidate.output};
        if (convKind(probe) != kind) continue;
        // Same padding form, so the base's ConvCfg pad bits are the family's.
        if (exactFamily && shape.kernel > 1 &&
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

std::uint32_t ceilLog2(std::uint32_t value) {
    if (!value) return 0;
    const auto floor = floorLog2(value);
    return value == (1u << floor) ? floor : floor + 1;
}

std::uint32_t commonWidthClass(std::uint32_t outputWidth) {
    if (!outputWidth)
        throw std::invalid_argument(
            "h13.common-schedule: output width must be nonzero");
    return outputWidth <= 8 ? 4
        : std::min<std::uint32_t>(4, 7 - std::min<std::uint32_t>(
              7, floorLog2(outputWidth - 1) + 1));
}

std::uint32_t commonColumnSchedule(std::uint32_t outputWidth,
                                   bool repeatWidthClass) {
    const auto widthClass = commonWidthClass(outputWidth);
    // Common ColumnSchedule: bits 23:16 carry the column-piece width class.
    // Width class 4 also repeats in the two low piece fields and selects its
    // short-surface form; wider depthwise and PE surfaces leave those piece
    // fields clear.  This is the shared geometry rule for Common +0x34.
    return (0x20u << 16) | (widthClass << 16) |
        (repeatWidthClass ? (widthClass << 12) | (widthClass << 8) |
                                (widthClass == 4 ? 0x05u : 0x01u)
                          : 0x0001u);
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

std::string planConvGeneric(const ConvShape &shape, ConvGenericPlan &plan,
                            bool linkedRing = false) {
    plan.kind = convKind(shape);
    if (plan.kind == ConvKind::Other)
        return "grouped convolutions other than depthwise have no generic form";
    const bool nativeSevenStem = linkedRing && plan.kind == ConvKind::Dense &&
        shape.kernel == 7 && shape.kernelWidth == 7 && shape.stride == 2 &&
        shape.input.channels == 3 && shape.output.channels == 64;
    if (shape.kernel != shape.kernelWidth ||
        (shape.kernel != 1 && shape.kernel != 3 && !nativeSevenStem))
        return "the generic form covers square 1x1 and 3x3 kernels";
    if (shape.stride != 1 && shape.stride != 2)
        return "the generic form covers stride 1 and 2";
    if (!linkedRing && shape.stride == 2 &&
        (shape.kernel != 1 || plan.kind != ConvKind::Dense))
        return "stride 2 is generic only for dense 1x1 kernels";
    if (convPadding(shape) < 0)
        return "the output surface implies neither same nor valid padding";
    const auto &in = shape.input;
    const auto &out = shape.output;
    if (!in.channels || !out.channels || !in.height || !in.width ||
        !out.height || !out.width)
        return "convolution surfaces must be nonempty";
    const bool nativeStem = linkedRing && shape.stride == 2 &&
        plan.kind == ConvKind::Dense && in.channels == 3 &&
        ((shape.kernel == 3 && out.channels == 32) || nativeSevenStem);
    if ((!nativeStem && in.channels % 16) || out.channels % 16)
        return "the generic form needs channel counts that are multiples of 16";
    if (in.width > 4096 || in.height > 16384)
        return "the generic form caps rows at 4096 columns";
    if (!convFamilyBase(shape, linkedRing))
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
    if (shape.stride == 2 && !linkedRing) {
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
    const bool largeKernelNeedsSplit = linkedRing &&
        (out.height > 7 || out.width > 7);
    if (plan.kind == ConvKind::Dense &&
        slice * perChannel > kGenericKernelBudget &&
        (!linkedRing || largeKernelNeedsSplit)) {
        if (shape.stride != 1 && !linkedRing)
            return "a strided convolution above the per-task kernel budget has no generic split";
        std::uint64_t fit = kGenericKernelBudget / perChannel;
        slice = static_cast<std::uint32_t>(
            fit >= 256 ? fit / 256 * 256 : fit / 16 * 16);
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
    if (shape.stride == 2 && !linkedRing) {
        // Device-verified only up to 8 interleaved lanes per plane (one lane
        // chunk): a 16-lane zero-skipping section (128 -> 256 at 56 -> 28)
        // hung the engine on the M1 Pro, matching the decoded record that the
        // 16-lane body count is two bytes off the model.
        const auto chunks = laneChunks(slice, laneCap(in));
        if (chunks.size() != 1 || chunks.front() > 8)
            return "stride 2 is verified only up to 8 interleaved lanes "
                   "(at most 128 output channels per task)";
    }
    if (!linkedRing && plan.kind == ConvKind::Depthwise &&
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
    if (linkedRing && plan.slices.size() == 1 &&
        out.channels * perChannel > kGenericKernelBudget &&
        plan.tileHeight != out.height)
        return "a coefficient stream above 1 MiB requires one whole-output tile";
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

std::string convGenericRefusal(ConvShape shape) {
    ConvGenericPlan plan{};
    return planConvGeneric(shape, plan);
}

std::string convLinkedRingRefusal(ConvShape shape) {
    ConvGenericPlan plan{};
    return planConvGeneric(shape, plan, true);
}

static Program encodeConvFormula(ConvShape shape, const std::uint8_t *weights,
                                 std::size_t weightBytes,
                                 const std::uint8_t *bias,
                                 std::size_t biasBytes, bool linkedRing) {
    ConvGenericPlan plan{};
    const std::string refusal = planConvGeneric(shape, plan, linkedRing);
    if (!refusal.empty())
        throw std::invalid_argument("h13.conv-outside-envelope: " + refusal);
    const auto *base = convFamilyBase(shape, linkedRing);
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
    const std::uint32_t cfgClass = commonWidthClass(outputWidth);
    // Common 0x34: 0x20 in bits 23:16 plus a width class (4 up to 8 output
    // columns, then 3, 2, 1 per doubling, 0 from 128). The low half repeats
    // the class in two nibbles in the forms Apple picks for them: always at
    // class 4, for dense 3x3 kernels and for 1x1 kernels with 512 or more
    // channels on both sides at classes 2 and 3, never for depthwise or at
    // class 1. That rule reproduces every decoded single-task template.
    const bool repeatClass = cfgClass == 4 ||
        ((cfgClass == 2 || cfgClass == 3) && plan.kind == ConvKind::Dense &&
         (shape.kernel == 3 || (in.channels >= 512 && out.channels >= 512)));
    const std::uint32_t common34 =
        commonColumnSchedule(outputWidth, repeatClass);

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
        std::array<std::uint32_t, 16> offsets{}, sizes{};
        const bool nativeSparse = linkedRing && shape.kernel == 3 &&
            shape.kernelWidth == 3 && shape.stride == 2;
        if (nativeSparse) {
            if (plan.kind == ConvKind::Dense) lanes = 2;
            const std::uint8_t *w = weights +
                static_cast<std::size_t>(firstChannel) * reduction * 2;
            const std::uint8_t *b = bias ? bias + firstChannel * 2 : nullptr;
            packed = packMobileSparseStride2(part, w, b, offsets, sizes);
        } else if (plan.kind == ConvKind::Depthwise && shape.stride == 1) {
            packed = packConvDepthwise(taps, slice, weights, bias);
        } else {
            const auto chunks = linkedRing && shape.stride == 2
                ? std::vector<std::uint32_t>{slice / 16}
                : laneChunks(slice, laneCap(in));
            lanes = chunks.front();
            const std::uint8_t *w = weights +
                static_cast<std::size_t>(firstChannel) * reduction * 2;
            const std::uint8_t *b = bias ? bias + firstChannel * 2 : nullptr;
            if (shape.stride == 2) {
                if (!linkedRing && chunks.size() != 1)
                    throw std::invalid_argument(
                        "h13.conv-outside-envelope: the generic strided "
                        "section needs one lane chunk");
                packed = packConvStrided(reduction, chunks, w, b,
                                          linkedRing);
            } else {
                packed = packConvDense(reduction, chunks, w, b, false);
            }
        }
        const std::uint32_t sectionBase =
            static_cast<std::uint32_t>(program.constants.size());
        if (!nativeSparse)
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
        const std::uint32_t encodedPad = nativeSparse ? 0u :
            static_cast<std::uint32_t>(pad);
        const std::uint32_t convCfg = (baseCfg & 0xf8000000u) |
            shape.kernelWidth | (shape.kernel << 5) | (ocg << 10) |
            (shape.stride << 13) | (shape.stride << 15) |
            (encodedPad << 17) | (encodedPad << 22);
        putRegister(task, offset, bytes, 0x0001c, convCfg);
        if (shape.stride == 2)
            putRegister(task, offset, bytes, 0x0c800, 0x182u);
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


Program encodeConvGeneric(ConvShape shape, const std::uint8_t *weights,
                          std::size_t weightBytes, const std::uint8_t *bias,
                          std::size_t biasBytes) {
    return encodeConvFormula(shape, weights, weightBytes, bias, biasBytes,
                             false);
}

Program encodeConvLinkedRing(ConvShape shape, const std::uint8_t *weights,
                             std::size_t weightBytes, const std::uint8_t *bias,
                             std::size_t biasBytes) {
    return encodeConvFormula(shape, weights, weightBytes, bias, biasBytes,
                             true);
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

/// The inter-layer edge forms Apple's compiler emits (t-3891, t-4020):
///   AliasInit    producer ResultCfg 0x500170 -> consumer SourceCfg 0x160
///   AliasChained producer ResultCfg 0x140    -> consumer SourceCfg 0x144
///   Dep104       producer ResultCfg 0x140    -> consumer SourceCfg 0x104
///   Dep120       producer ResultCfg 0x140    -> consumer SourceCfg 0x120
/// The alias forms carry the DPC/SPC handoff, the channel-3 selectors and the
/// populated-but-disabled DMA geometry; the dependent forms carry none of it.
enum class EdgeMode { Dram, AliasInit, AliasChained, Dep104, Dep120 };

bool isAlias(EdgeMode m) { return m == EdgeMode::AliasInit || m == EdgeMode::AliasChained; }
bool isDependent(EdgeMode m) { return m == EdgeMode::Dep104 || m == EdgeMode::Dep120; }

constexpr std::uint32_t kL2ResultResident = 1u << 8;
constexpr std::uint32_t kL2ResultChainedTraversal = 4u << 4;
constexpr std::uint32_t kL2ResultInitialAliasTraversal = 7u << 4;
constexpr std::uint32_t kL2ResultInitialAliasDMAClass = 5u << 20;
constexpr std::uint32_t kL2SourceInitialAliasTraversal = 6u << 4;
constexpr std::uint32_t kL2SourceResident = 1u << 8;
constexpr std::uint32_t kL2SourceDependentClass = 1u << 2;

/// L2 ResultCfg fields shared by linked-ring and residual resident results.
/// The first producer in a resident run carries the initial-alias traversal
/// and DMA class; later resident producers use the chained traversal only.
constexpr std::uint32_t l2ResidentResultConfig(bool initialAlias) {
    return kL2ResultResident |
        (initialAlias ? kL2ResultInitialAliasTraversal |
                            kL2ResultInitialAliasDMAClass
                      : kL2ResultChainedTraversal);
}

constexpr std::uint32_t l2InitialAliasSourceConfig() {
    return kL2SourceInitialAliasTraversal | kL2SourceResident;
}

constexpr std::uint32_t l2DependentSourceConfig() {
    return kL2SourceResident | kL2SourceDependentClass;
}

/// What one layer occupies in L2, read back from its first task's words.
struct LayerGeometry {
    bool depthwise;
    std::uint32_t tasks;
    std::uint32_t inHeight, inWidth, outHeight, outWidth, outChannels, sliceChannels, tileHeight, kernel;
    std::uint32_t sourceRow, sourceChannel, sourceDepth;   // 0x4810 / 0x480c / 0x4814
    std::uint64_t stagedInput;   // rows the input DMA stages * source row
    std::uint64_t staging;       // the DRAM-output staging region of one task
    std::uint32_t outRowL2;
};

LayerGeometry layerGeometry(const Program &layer) {
    const auto &t = layer.task;
    const std::size_t bytes = layer.firstTaskBytes;
    auto reg = [&](std::uint32_t a) { return loadLE32(t, h13RegisterOffset(t, 0, bytes, a)); };
    LayerGeometry g{};
    g.depthwise = layer.depthwiseConv;
    g.tasks = layer.taskCount;
    g.kernel = reg(0x1c) & 0x1f;
    g.inHeight = reg(0x0) >> 16; g.inWidth = reg(0x0) & 0xffff;
    g.outHeight = reg(0x14) >> 16; g.outWidth = reg(0x14) & 0xffff;
    g.sliceChannels = reg(0x10);
    g.outChannels = layer.output.nchw[1];
    g.tileHeight = reg(0x28);
    g.sourceRow = reg(0x4810); g.sourceChannel = reg(0x480c); g.sourceDepth = reg(0x4814);
    g.stagedInput = reg(0x4834);
    g.outRowL2 = static_cast<std::uint32_t>(std::max<std::uint64_t>(16, (2ull * g.outWidth + 15) / 16 * 16));
    g.staging = static_cast<std::uint64_t>(g.tileHeight) * g.sliceChannels * g.outRowL2;
    return g;
}

std::uint64_t align16(std::uint64_t v) { return (v + 15) & ~15ull; }

/// Two ranges on the 2 MiB L2 ring overlap.
bool ringOverlap(std::uint64_t a, std::uint64_t an, std::uint64_t b, std::uint64_t bn) {
    if (an + bn > kL2Bytes) return true;
    for (std::uint64_t ka : {a % kL2Bytes, a % kL2Bytes + kL2Bytes})
        for (std::uint64_t kb : {b % kL2Bytes, b % kL2Bytes + kL2Bytes})
            if (ka < kb + bn && kb < ka + an) return true;
    return false;
}

/// Edge choice and placement (t-4020 "minimal planner rules"):
/// - a resident run starts with an initial alias edge out of a producer that
///   reads DRAM, then alternates a dependent and a chained-alias phase;
/// - a dependent edge out of a single-task producer uses SourceCfg 0x104 up
///   to the largest edge Apple was seen to give it (0x80800 bytes) and
///   spills above that;
/// - a split (multi-task) producer can only feed the dependent phase, as one
///   aggregate result read with 0x120 by a depthwise consumer;
/// - every edge spans the consumer's full input (rows * its source row),
///   addresses advance around the 2 MiB ring right after the live input, and
///   a task's live input never overlaps its outgoing result or staging;
/// - anything else, or anything that does not fit, stays in DRAM.
struct L2Plan {
    std::vector<EdgeMode> edge;              // edge[i]: layer i -> i + 1
    std::vector<std::uint64_t> resultBase;   // aggregate base of layer i's L2 result
    std::vector<std::uint64_t> resultBytes;
    std::vector<std::uint64_t> sourceBase;   // layer i's Source1Base
    std::vector<std::uint64_t> stagingBase;  // layer i's DRAM-output staging base
};

constexpr std::uint64_t kDep104MaxEdge = 0x80800;

L2Plan planL2(const std::vector<Program> &layers) {
    const std::size_t n = layers.size();
    L2Plan plan;
    plan.edge.assign(n - 1, EdgeMode::Dram);
    plan.resultBase.assign(n, 0); plan.resultBytes.assign(n, 0);
    plan.sourceBase.assign(n, 0); plan.stagingBase.assign(n, 0);
    std::vector<LayerGeometry> g;
    for (const auto &layer : layers) g.push_back(layerGeometry(layer));
    for (std::size_t i = 0; i < n; ++i) plan.stagingBase[i] = g[i].stagedInput;
    bool inL2 = false;            // layer i's input is resident
    bool nextIsDependent = false; // phase of the next edge in a resident run
    std::uint64_t inBase = 0, inBytes = 0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const auto &p = g[i];
        const auto &c = g[i + 1];
        const std::uint64_t edge = static_cast<std::uint64_t>(c.inHeight) * c.sourceRow;
        EdgeMode mode = EdgeMode::Dram;
        // Every L2 task computes its whole output in one tile here: tiled
        // tasks next to a resident edge differ in Apple's result config
        // (0x142), which has not been mapped.
        const bool tilesOk = p.tileHeight == p.outHeight && c.tileHeight == c.outHeight &&
            c.inHeight == p.outHeight && c.sourceRow && c.sourceRow % 16 == 0;
        if (tilesOk) {
            if (!inL2) {
                if (p.tasks == 1) mode = EdgeMode::AliasInit;
            } else if (nextIsDependent) {
                // A single-task producer's wide (0x120) edge only appears at
                // 56x56 with a result config (0x142) and result strides on
                // the final task that are not mapped yet, so it spills.
                if (p.tasks == 1 && edge <= kDep104MaxEdge) mode = EdgeMode::Dep104;
                else if (p.tasks > 1 && c.depthwise &&
                         p.sliceChannels * p.tasks == p.outChannels)
                    mode = EdgeMode::Dep120;
            } else {
                if (p.tasks == 1) mode = EdgeMode::AliasChained;
            }
        }
        std::uint64_t result = 0, source = plan.sourceBase[i];
        if (mode != EdgeMode::Dram) {
            if (!inL2) {
                // Apple's first producer: result at 0, staged input above it.
                result = 0;
                source = align16(edge);
                if (source + p.stagedInput > kL2Bytes ||
                    ringOverlap(result, edge, source, p.stagedInput)) mode = EdgeMode::Dram;
            } else {
                result = align16(inBase + inBytes) % kL2Bytes;
                if (ringOverlap(result, edge, inBase, inBytes)) mode = EdgeMode::Dram;
            }
        }
        if (mode != EdgeMode::Dram) {
            // The consumer's own outgoing region (DRAM staging; a later
            // resident edge re-checks itself) goes right after its input.
            const std::uint64_t after = align16(result + edge) % kL2Bytes;
            if (ringOverlap(after, c.staging, result, edge)) mode = EdgeMode::Dram;
        }
        if (mode == EdgeMode::Dram) {
            inL2 = false;
            nextIsDependent = false;
            continue;
        }
        plan.edge[i] = mode;
        plan.resultBase[i] = result;
        plan.resultBytes[i] = edge;
        plan.sourceBase[i] = source;
        plan.sourceBase[i + 1] = result;
        plan.stagingBase[i + 1] = align16(result + edge) % kL2Bytes;
        inL2 = true;
        nextIsDependent = isAlias(mode);
        inBase = result;
        inBytes = edge;
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
        if (!layer.formulaBuilt || layer.inputs.size() != 1 ||
            layer.inputs[0].index != 5 || layer.output.index != 4 ||
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
    if (l2Resident) l2 = planL2(layers);
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
    program.inputs = {layers.front().inputs[0]};
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
            // L2 edges (t-3891, t-4020). The incoming handoff belongs to a
            // layer's first task only: later output-channel slices reread the
            // same resident input as a plain L2 read with no handoff bits.
            // SrcLoc/DstLoc mark DRAM ends; SPL unless the input arrives over
            // a dependent edge (or is a reread); SPC on an alias consumer, DPC
            // on an alias producer; TSE+TSR closes a layer unless it feeds a
            // dependent edge.
            const bool firstSlice = &span == &spans[index].front();
            const bool reread = inL2 && !firstSlice;
            std::uint32_t control = 0;
            if (!inL2) control |= 0x10000000u;
            if (!outL2) control |= 0x20000000u;
            if (!reread && !isDependent(inEdge)) control |= 0x800u;
            if (!reread && isAlias(inEdge)) control |= 0x2000u;
            if (isAlias(outEdge)) control |= 0x4000u;
            if (layerEnd && (!isDependent(outEdge) || last)) control |= 0x9000u;
            storeLE32(task, offset + 24, control);
            storeLE32(task, offset + 28, static_cast<std::uint32_t>(nextOffset));
            std::uint32_t selectors = loadLE32(task, offset + 32);
            if (((selectors >> 0) & 31) != 4 || !(selectors & (1u << 5)) ||
                ((selectors >> 12) & 31) != 5 || !(selectors & (1u << 17)))
                throw std::invalid_argument(
                    "h13.linked-outside-envelope: layer task does not read "
                    "channel 4 and write channel 5");
            if (inL2) {
                // Resident input at the producer's (aggregate) result base,
                // input DMA disabled. The first slice takes the edge's own
                // form; later slices are plain L2 rereads (SourceCfg 0x100).
                const bool alias = !reread && isAlias(inEdge);
                const std::uint32_t sourceCfg = reread ? 0x100u
                    : inEdge == EdgeMode::AliasInit ? 0x160u
                    : inEdge == EdgeMode::AliasChained ? 0x144u
                    : inEdge == EdgeMode::Dep104 ? 0x104u : 0x120u;
                putRegister(task, offset, bytes, 0x04804, sourceCfg);
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
                // Resident result in the consumer's source layout. A split
                // producer writes one aggregate, each slice with the full
                // aggregate strides (t-4020). Slice j (output channels
                // j * sliceChannels onward) sits at base + j * sliceChannels *
                // rowL2. Apple's decoded slices descend instead, which matches
                // only if its tasks run the channel slices in reverse order;
                // the descending form was wrong on the M1 Pro for this
                // encoder's ascending slices (cold rel. L2 1.15), the
                // ascending form exact.
                const auto &next = layers[index + 1];
                auto nextReg = [&](std::uint32_t a) {
                    return loadLE32(next.task, h13RegisterOffset(next.task, 0, next.firstTaskBytes, a));
                };
                const auto g = layerGeometry(layer);
                const std::size_t slice = static_cast<std::size_t>(&span - &spans[index].front());
                const std::uint64_t sliceBase = l2.resultBase[index] +
                    static_cast<std::uint64_t>(slice) * g.sliceChannels * nextReg(0x0480c);
                putRegister(task, offset, bytes, 0x04830,
                            outEdge == EdgeMode::AliasInit ? 0x500170u : 0x140u);
                putRegister(task, offset, bytes, 0x04834,
                            static_cast<std::uint32_t>(sliceBase % kL2Bytes));
                putRegister(task, offset, bytes, 0x04838, nextReg(0x0480c));
                putRegister(task, offset, bytes, 0x0483c, nextReg(0x04810));
                putRegister(task, offset, bytes, 0x04840, nextReg(0x04814));
                putRegister(task, offset, bytes, 0x04844, 0);
                putRegister(task, offset, bytes, 0x17800, 0xc0);
                if (isAlias(outEdge)) {
                    selectors = (selectors & ~(31u << 12)) | (3u << 12);
                } else {
                    selectors &= ~((31u << 12) | (1u << 17));
                    for (std::uint32_t a : {0x17804u, 0x17808u, 0x1780cu, 0x17810u, 0x17814u})
                        putRegister(task, offset, bytes, a, 0);
                    putRegister(task, offset, bytes, 0x17818, 0x01000000u);
                }
            }
            // Common 0x24 bit 14 tracks a depthwise task next to a resident
            // edge in every decoded chain (t-4020: it is not an edge mode).
            if (layer.depthwiseConv && (inL2 || outL2)) {
                const auto c24 = loadLE32(task, h13RegisterOffset(task, offset, bytes, 0x00024));
                putRegister(task, offset, bytes, 0x00024, c24 | 0x4000u);
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
        program.constants.resize(alignUp(program.constants.size(), 64), 0);
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

namespace {

struct MobileRingGeometry {
    std::uint32_t ci, co, hi, wi, ho, wo, kernel, stride;
    bool depthwise;
    std::uint32_t sourceChannel, sourceRow;
    std::uint64_t sourceSpan;
};

std::uint32_t mobileRingCommon34(const MobileRingGeometry &g,
                                 bool outputResident,
                                 bool shortPrefixBridge) {
    // The linked ring's Common class follows the resident result phase.  A
    // terminal DRAM result instead uses the operation's ordinary shape class.
    if (shortPrefixBridge) return 0x00233301u;
    if (outputResident) {
        if (g.wo == 112)
            return g.kernel == 1 ? 0x00231101u : 0x00230001u;
        if (g.wo == 56)
            return g.stride == 2 ? 0x00211101u : 0x00243301u;
        if (g.wo == 28) return 0x00222201u;
        if (g.wo == 14) return 0x00233301u;
        if (g.wo == 7) return 0x00244405u;
    }
    if (g.ci == 3 && g.co == 32) return 0x00230001u;
    if (g.wo == 112)
        return g.kernel == 1 ? 0x00231101u : 0x00230001u;
    if (g.wo == 56) {
        if (g.stride == 2 || g.depthwise) return 0x00210001u;
        return 0x00241101u;
    }
    if (g.wo == 28)
        return (g.stride == 2 || (!g.depthwise && g.ci == g.co))
            ? 0x00222201u : 0x00220001u;
    if (g.wo == 14)
        return (g.stride == 2 || (!g.depthwise && g.ci == g.co))
            ? 0x00233301u : 0x00230001u;
    if (g.wo == 7) return 0x00244405u;
    throw std::invalid_argument(
        "h13.linked-ring-outside-envelope: unsupported Common class");
}

std::uint32_t mobileRingCostHint(const MobileRingGeometry &g, bool terminal) {
    // H13's scheduler hint is a shape/edge-role class.  It does not affect
    // arithmetic, but matching it keeps native forms byte-comparable.
    if (g.ci == 3 && g.co == 32) return terminal ? 0x23u : 0x08u;
    if (g.depthwise) {
        if (g.stride == 2) {
            if (g.wi == 112) return 0x0fu;
            if (g.wi == 56) return 0x07u;
            if (g.wi == 28) return terminal ? 0x06u : 0x03u;
            if (g.wi == 14) return terminal ? 0x07u : 0x01u;
        } else {
            if (g.wi == 112) return terminal ? 0x1au : 0x17u;
            if (g.wi == 56) return terminal ? 0x1au : 0x09u;
            if (g.wi == 28) return terminal ? 0x0du : 0x03u;
            if (g.wi == 14) return terminal ? 0x0du : 0x01u;
            if (g.wi == 7) return terminal ? 0x0eu : 0x02u;
        }
    } else if (g.kernel == 1) {
        if (g.wi == 112) return terminal ? 0x35u : 0x17u;
        if (g.wi == 56) {
            if (terminal) return 0x1bu;
            return g.ci == g.co ? 0x08u : 0x04u;
        }
        if (g.wi == 28) {
            if (terminal) return g.ci == g.co ? 0x11u : 0x0fu;
            return g.ci == g.co ? 0x0au : 0x05u;
        }
        if (g.wi == 14) {
            if (terminal) return g.ci == g.co ? 0x1cu : 0x15u;
            return g.ci == g.co ? 0x0fu : 0x07u;
        }
        if (g.wi == 7) {
            if (terminal) return g.ci == 512 ? 0x2cu : 0x4au;
            return g.ci == 512 ? 0x1eu : 0x3du;
        }
    }
    throw std::invalid_argument(
        "h13.linked-ring-outside-envelope: unsupported scheduler hint class");
}

std::uint32_t mobileRingSkew(const MobileRingGeometry &g) {
    // H13's bank rotation is a function of the native MobileNet surface and
    // consumer role.  These cases are shape classes, not layer numbers: the
    // same rule therefore rebuilds every prefix independently.
    if (g.wi == 112) return g.stride == 2 ? 0x20u : 0x30u;
    if (g.wi == 56) return g.ci == 64 ? 0x80u : 0x30u;
    if (g.wi == 28) return g.stride == 2 ? 0x40u : 0x50u;
    if (g.wi == 14) return g.stride == 2 ? 0x20u : 0x30u;
    if (g.wi == 7) return 0x30u;
    throw std::invalid_argument(
        "h13.linked-ring-outside-envelope: unsupported MobileNet surface");
}

bool mobileRingOverlap(std::uint64_t a, std::uint64_t an,
                       std::uint64_t b, std::uint64_t bn) {
    return ringOverlap(a, an, b, bn);
}

} // namespace

Program linkMobileNetRing(const std::vector<Program> &layers,
                          std::vector<LinkedLayer> *layout) {
    if (layers.empty() || layers.size() > 28)
        throw std::invalid_argument(
            "h13.linked-ring-outside-envelope: expected a MobileNet prefix of 1..28 convolutions");
    for (const auto &layer : layers)
        if (!layer.formulaBuilt || layer.taskCount != 1 || layer.inputs.size() != 1)
            throw std::invalid_argument(
                "h13.linked-ring-outside-envelope: every layer must be one formula-built convolution task");

    std::vector<MobileRingGeometry> g;
    g.reserve(layers.size());
    for (const auto &layer : layers) {
        const auto reg = [&](std::uint32_t a) {
            return loadLE32(layer.task,
                h13RegisterOffset(layer.task, 0, layer.firstTaskBytes, a));
        };
        MobileRingGeometry x{};
        x.hi = reg(0x00000) >> 16; x.wi = reg(0x00000) & 0xffff;
        x.ci = reg(0x0000c); x.co = reg(0x00010);
        x.ho = reg(0x00014) >> 16; x.wo = reg(0x00014) & 0xffff;
        x.kernel = reg(0x0001c) & 0x1f;
        x.stride = (reg(0x0001c) >> 13) & 3;
        x.depthwise = layer.depthwiseConv;
        if (g.empty()) {
            const bool stem = x.ci == 3 && x.co == 32 && x.hi == 224 &&
                x.wi == 224 && x.ho == 112 && x.wo == 112 &&
                x.kernel == 3 && x.stride == 2;
            const bool standaloneDepthwise = layers.size() == 1 && x.depthwise &&
                x.ci == x.co && x.kernel == 3 && x.stride == 2;
            if (!stem && !standaloneDepthwise)
                throw std::invalid_argument(
                    "h13.linked-ring-outside-envelope: prefix does not start "
                    "with the native 3x3-s2 MobileNet stem");
            if (stem) {
                x.sourceChannel = 2 * x.wi;
                x.sourceRow = x.ci * x.sourceChannel;
            } else {
                x.sourceChannel = static_cast<std::uint32_t>(align16(2ull * x.wi));
                x.sourceRow = x.ci * x.sourceChannel + mobileRingSkew(x);
            }
        } else if (g.size() == 2 && layers.size() >= 4) {
            // The sole DRAM-to-L2 bridge stages 64 source pixels while its
            // whole 112-wide result remains resident.
            x.sourceChannel = 2 * 64;
            x.sourceRow = x.ci * x.sourceChannel + 0x50;
        } else {
            x.sourceChannel = static_cast<std::uint32_t>(align16(2ull * x.wi));
            x.sourceRow = x.ci * x.sourceChannel + mobileRingSkew(x);
        }
        x.sourceSpan = static_cast<std::uint64_t>(x.hi) * x.sourceRow;
        if (!g.empty() && (g.back().co != x.ci || g.back().ho != x.hi ||
                           g.back().wo != x.wi))
            throw std::invalid_argument(
                "h13.linked-ring-outside-envelope: MobileNet layer surfaces are not a straight native chain");
        g.push_back(x);
    }

    Program program = layers.size() == 1 ? layers.front()
                                         : linkConvChain(layers, nullptr, false);
    auto spans = taskSpans(program);
    if (spans.size() != layers.size())
        throw std::logic_error("h13.linked-ring: task/layer count changed while linking");

    // A 628-byte task may not straddle a 16 KiB text page.  Repack the one
    // boundary where 0x100 alignment alone would place a task at 0x3f00.
    if (layers.size() > 21) {
        std::vector<std::uint8_t> repacked;
        std::vector<std::size_t> offsets;
        for (const auto &span : spans) {
            std::size_t at = static_cast<std::size_t>(alignUp(repacked.size(), 0x100));
            if ((at & 0x3fffu) + span.bytes > 0x4000u)
                at = static_cast<std::size_t>(alignUp(at, 0x4000));
            repacked.resize(at + span.bytes, 0);
            std::copy(program.task.begin() + static_cast<std::ptrdiff_t>(span.offset),
                      program.task.begin() + static_cast<std::ptrdiff_t>(span.offset + span.bytes),
                      repacked.begin() + static_cast<std::ptrdiff_t>(at));
            offsets.push_back(at);
        }
        for (std::size_t i = 0; i + 1 < offsets.size(); ++i)
            storeLE32(repacked, offsets[i] + 28,
                      static_cast<std::uint32_t>(offsets[i + 1]));
        program.task = std::move(repacked);
        program.constantOffsetBytes = alignUp(program.task.size(), 0x40);
        spans = taskSpans(program);
    }

    const std::size_t n = layers.size();
    std::vector<std::uint32_t> sourceCfg(n, 0x500172u), resultCfg(n, 0x14au);
    std::vector<std::uint64_t> sourceBase(n, 0), resultBase(n, 0), resultSpan(n, 0);
    std::vector<bool> readsScratch(n, false), writesScratch(n, false), outResident(n, false);
    if (n == 1) {
        resultCfg[0] = 0x500172u;
        resultBase[0] = g[0].sourceSpan;
    }
    if (n >= 2) {
        sourceCfg[1] = 0x160u;
        outResident[0] = true;
        resultCfg[0] = l2ResidentResultConfig(true);
        resultSpan[0] = g[1].sourceSpan;
        resultBase[0] = 0;
        sourceBase[0] = resultSpan[0];
        sourceBase[1] = 0;
    }
    if (n >= 4) {
        // The one intentional spill separates conv_1 and conv_2.
        writesScratch[1] = true;
        readsScratch[2] = true;
        resultCfg[1] = 0x84au;
        resultBase[1] = g[1].sourceSpan;
        sourceCfg[2] = 0x842u;
        outResident[2] = true;
        resultCfg[2] = l2ResidentResultConfig(false);
        resultBase[2] = 0;
        resultSpan[2] = g[3].sourceSpan;
        sourceBase[2] = resultSpan[2];
        sourceBase[3] = 0;
        sourceCfg[3] = 0x160u;
        for (std::size_t i = 3; i + 1 < n; ++i) {
            const auto &consumer = g[i + 1];
            const bool alias = consumer.depthwise;
            if (alias) {
                const bool phaseStart = !g[i].depthwise &&
                    g[i].co > g[i].ci && consumer.wi >= 28;
                sourceCfg[i + 1] = phaseStart ? 0x160u : 0x144u;
            } else {
                const bool pointwiseAlias = !g[i].depthwise &&
                    g[i].kernel == 1 && consumer.kernel == 1 &&
                    g[i].co == consumer.ci && consumer.co == consumer.ci;
                sourceCfg[i + 1] = pointwiseAlias ? 0x144u
                    : consumer.wo >= 56 ||
                        (consumer.wo == 28 && consumer.co > consumer.ci)
                    ? 0x120u : 0x104u;
            }
            outResident[i] = true;
            resultCfg[i] = l2ResidentResultConfig(false);
            resultSpan[i] = consumer.sourceSpan;
            std::uint64_t candidate = align16(sourceBase[i] + g[i].sourceSpan) % kL2Bytes;
            if (!g[i].depthwise && g[i].co > g[i].ci) {
                candidate = 0;
            } else if (g[i].depthwise && i + 2 < n &&
                       !g[i + 1].depthwise && g[i + 1].co > g[i + 1].ci) {
                // Reserve the expanding pointwise result at ring base zero.
                const std::uint64_t future = g[i + 2].sourceSpan;
                if (mobileRingOverlap(candidate, resultSpan[i], 0, future))
                    candidate = align16(future);
            }
            resultBase[i] = candidate;
            sourceBase[i + 1] = candidate;
        }
    } else if (n == 2) {
        // Apple's short prefixes write the terminal surface directly rather
        // than introducing the full-network spill.
        resultCfg[1] = 0x142u;
        resultBase[1] = g[1].sourceSpan;
    } else if (n == 3) {
        sourceCfg[2] = 0x120u;
        resultCfg[1] = l2ResidentResultConfig(false);
        outResident[1] = true;
        resultSpan[1] = g[2].sourceSpan;
        resultBase[1] = g[1].sourceSpan;
        sourceBase[2] = resultBase[1];
        resultCfg[2] = 0x142u;
        resultBase[2] = align16(sourceBase[2] + g[2].sourceSpan) % kL2Bytes;
    }

    if (n >= 4) {
        const std::size_t last = n - 1;
        if (g[last].depthwise && g[last].wi == 56 &&
            sourceCfg[last] == 0x160u)
            sourceCfg[last] = 0x144u;
        if (!g[last].depthwise && g[last].wi <= 28 &&
            sourceCfg[last] == 0x120u)
            sourceCfg[last] = 0x104u;
        outResident[last] = false;
        const std::uint64_t externalBytes = layers[last].output.allocationBytes;
        resultCfg[last] = sourceCfg[last] == 0x120u && externalBytes > kDep104MaxEdge
            ? 0x142u : 0x14au;
        resultBase[last] = align16(sourceBase[last] + g[last].sourceSpan) % kL2Bytes;
    }

    // Scratch is derived from the interleave-8 bridge: four channel groups,
    // 16 halfwords per pixel, plus the two ordinary padded surface slots the
    // H13 program descriptor reserves around it.
    std::uint64_t scratch = 0, spillBytes = 0;
    if (n >= 4) {
        const std::uint64_t spillRow = 16ull * g[1].wo;
        const std::uint64_t spillPlane = spillRow * g[1].ho;
        spillBytes = spillPlane * (g[1].co / 8);
        scratch = 2 * layers[1].output.allocationBytes + spillBytes;
    }
    program.scratchAllocationBytes = scratch;

    if (layout) layout->clear();
    for (std::size_t i = 0; i < n; ++i) {
        const auto span = spans[i];
        auto &task = program.task;
        const std::size_t off = span.offset, bytes = span.bytes;
        const bool last = i + 1 == n;
        const bool inAlias = sourceCfg[i] == 0x160u || sourceCfg[i] == 0x144u;
        const bool inDependent = sourceCfg[i] == 0x120u || sourceCfg[i] == 0x104u;
        const bool outAlias = outResident[i] && i + 1 < n &&
            (sourceCfg[i + 1] == 0x160u || sourceCfg[i + 1] == 0x144u);
        const bool outDependent = outResident[i] && i + 1 < n &&
            (sourceCfg[i + 1] == 0x120u || sourceCfg[i + 1] == 0x104u);

        std::uint32_t control = 0;
        if (!inAlias && !inDependent) control |= 0x10000000u;
        if (!outResident[i]) control |= 0x20000000u;
        if (!inDependent && !readsScratch[i]) control |= 0x800u;
        if (inAlias) control |= 0x2000u;
        if (outAlias) control |= 0x4000u;
        if ((!outDependent || last) && !writesScratch[i]) control |= 0x9000u;
        storeLE32(task, off + 8, 0x400u |
            ((!inAlias && !inDependent && !readsScratch[i]) ? 0x08u : 0u) |
            (last ? 0x22u : 0u));
        storeLE32(task, off + 16, last ? 0x00fff86au
            : ((inAlias || inDependent || readsScratch[i])
                ? 0x00fff800u : 0x00fff868u));
        storeLE32(task, off + 24, control);
        storeLE32(task, off, loadLE32(task, off) & ~(1u << 22));
        storeLE32(task, off + 4,
                  (loadLE32(task, off + 4) & 0xffff0000u) |
                  ((n == 3 && i == 1) ? 0x08u
                                      : mobileRingCostHint(g[i], last)));

        putRegister(task, off, bytes, 0x00034,
                    mobileRingCommon34(g[i], outResident[i],
                                       n == 3 && i == 1));

        putRegister(task, off, bytes, 0x04804, sourceCfg[i]);
        putRegister(task, off, bytes, 0x04808, static_cast<std::uint32_t>(sourceBase[i]));
        putRegister(task, off, bytes, 0x0480c, g[i].sourceChannel);
        putRegister(task, off, bytes, 0x04810, g[i].sourceRow);
        putRegister(task, off, bytes, 0x04814, g[i].ci * g[i].sourceChannel);
        putRegister(task, off, bytes, 0x04818,
                    (inAlias || inDependent) ? 0u : g[i].ci * g[i].sourceChannel);
        putRegister(task, off, bytes, 0x04830, resultCfg[i]);
        putRegister(task, off, bytes, 0x04834, static_cast<std::uint32_t>(resultBase[i]));
        if (outResident[i]) {
            putRegister(task, off, bytes, 0x04838, g[i + 1].sourceChannel);
            putRegister(task, off, bytes, 0x0483c, g[i + 1].sourceRow);
            putRegister(task, off, bytes, 0x04840, g[i].co * g[i + 1].sourceChannel);
            putRegister(task, off, bytes, 0x04844, 0);
        } else {
            if (n == 1 || resultCfg[i] == 0x142u) {
                const std::uint32_t channel = resultCfg[i] == 0x142u &&
                    sourceCfg[i] == 0x120u ? 0x80u
                    : static_cast<std::uint32_t>(layers[i].output.nchw[5]);
                const std::uint32_t skew = resultCfg[i] == 0x142u &&
                    sourceCfg[i] == 0x120u ? 0x2000u / g[i].co : 0x80u;
                putRegister(task, off, bytes, 0x04838, channel);
                putRegister(task, off, bytes, 0x0483c, g[i].co * channel + skew);
                putRegister(task, off, bytes, 0x04840, g[i].co * channel);
                putRegister(task, off, bytes, 0x04844, g[i].co * channel);
            } else {
                for (std::uint32_t a : {0x04838u, 0x0483cu, 0x04840u, 0x04844u})
                    putRegister(task, off, bytes, a, 0);
            }
        }

        std::uint32_t selectors = loadLE32(task, off + 32);
        if (inAlias) {
            selectors = (selectors & ~31u) | 3u;
            putRegister(task, off, bytes, 0x13800, 0x48880u);
            putRegister(task, off, bytes, 0x13808,
                        (i == 1 || i == 3) && n >= 4
                            ? static_cast<std::uint32_t>(spillBytes) : 0u);
        } else if (inDependent) {
            selectors &= ~(31u | (1u << 5));
            putRegister(task, off, bytes, 0x13800, 0x8880u);
            for (std::uint32_t a : {0x13808u,0x1380cu,0x13810u,0x13814u,0x13818u})
                putRegister(task, off, bytes, a, 0);
            putRegister(task, off, bytes, 0x13838, 0x01000000u);
        } else if (readsScratch[i]) {
            selectors = (selectors & ~31u) | 3u;
            putRegister(task, off, bytes, 0x13800, 0x53901u);
            putRegister(task, off, bytes, 0x13808, 0);
            putRegister(task, off, bytes, 0x1380c, 16u * g[1].wo);
            putRegister(task, off, bytes, 0x13810, 16u * g[1].wo * g[1].ho);
            putRegister(task, off, bytes, 0x13814, static_cast<std::uint32_t>(spillBytes));
            putRegister(task, off, bytes, 0x13818, static_cast<std::uint32_t>(spillBytes));
            putRegister(task, off, bytes, 0x13838, 0x08002031u);
        } else {
            selectors = (selectors & ~31u) | 5u;
            putRegister(task, off, bytes, 0x13800, 0x33881u);
        }

        if (outResident[i]) {
            putRegister(task, off, bytes, 0x17800, 0xc0u);
            if (outAlias) {
                selectors = (selectors & ~(31u << 12)) | (3u << 12);
                putRegister(task, off, bytes, 0x17804,
                            (i == 0 || i == 2) && n >= 4
                                ? static_cast<std::uint32_t>(spillBytes) : 0u);
            } else {
                selectors &= ~((31u << 12) | (1u << 17));
                for (std::uint32_t a : {0x17804u,0x17808u,0x1780cu,0x17810u,0x17814u})
                    putRegister(task, off, bytes, a, 0);
                putRegister(task, off, bytes, 0x17818, 0x01000000u);
            }
        } else if (writesScratch[i]) {
            selectors = (selectors & ~(31u << 12)) | (3u << 12);
            putRegister(task, off, bytes, 0x17800, 0x040003c1u);
            putRegister(task, off, bytes, 0x17804, 0);
            putRegister(task, off, bytes, 0x17808, 16u * g[i].wo);
            putRegister(task, off, bytes, 0x1780c, 16u * g[i].wo * g[i].ho);
            putRegister(task, off, bytes, 0x17810, static_cast<std::uint32_t>(spillBytes));
            putRegister(task, off, bytes, 0x17814, static_cast<std::uint32_t>(spillBytes));
            putRegister(task, off, bytes, 0x17818, 0x08302031u);
        } else {
            selectors = (selectors & ~(31u << 12)) | (4u << 12);
            putRegister(task, off, bytes, 0x17800,
                        resultCfg[i] == 0x142u || n == 1 ? 0xc1u : 0x040000c1u);
        }
        selectors = (selectors & ~(1u << 23)) | (last ? 0u : 1u << 23);
        storeLE32(task, off + 32, selectors);

        if (g[i].depthwise && (inAlias || inDependent || outResident[i])) {
            const auto c24 = loadLE32(task,
                h13RegisterOffset(task, off, bytes, 0x00024));
            putRegister(task, off, bytes, 0x00024, c24 | 0x4000u);
        }
        if (g[i].stride == 2)
            putRegister(task, off, bytes, 0x0c800, 0x182u);
        if (g[i].stride == 2)
            putRegister(task, off, bytes, 0x1f800, 0x60u);

        if (layout) {
            LinkedLayer item{static_cast<std::uint32_t>(i), 1,
                readsScratch[i], 0, writesScratch[i], 0};
            item.inputEdge = (inAlias ? (sourceCfg[i] == 0x160u ? 1 : 2)
                                      : (inDependent ? (sourceCfg[i] == 0x104u ? 3 : 4) : 0));
            item.outputEdge = outAlias ? (sourceCfg[i + 1] == 0x160u ? 1 : 2)
                : (outDependent ? (sourceCfg[i + 1] == 0x104u ? 3 : 4) : 0);
            item.l2Source = (inAlias || inDependent) ? sourceBase[i] : 0;
            item.l2Result = outResident[i] ? resultBase[i] : 0;
            item.l2ResultBytes = outResident[i] ? resultSpan[i] : 0;
            layout->push_back(item);
        }
    }

    // `encodeConvGeneric` sizes the boundary layouts from their logical
    // NCHW strides.  The native MobileNet task forms can advertise a larger
    // source/destination transfer window in TileDMA: the stride-2 stem, for
    // example, keeps the external [3,224,224] descriptor but programs a
    // 917504-byte source extent.  libane deliberately treats 0x13814/18 and
    // 0x17810/14 as conservative BO bounds and refuses an ANEC whose tiles[]
    // allocation is smaller.  Grow only the header allocations to cover the
    // live boundary selectors; the NCHW layout and every device-visible task
    // word remain unchanged.
    const auto growChannel = [&](std::uint32_t channel,
                                 std::uint64_t extent) {
        if (program.output.index == channel)
            program.output.allocationBytes = std::max(
                program.output.allocationBytes, extent);
        for (auto &input : program.inputs)
            if (input.index == channel)
                input.allocationBytes = std::max(input.allocationBytes,
                                                 extent);
    };
    const auto boundChannel = [&](std::uint32_t channel) {
        if (channel < 4) return channel;
        const auto role = std::find(program.taskSurfaceChannels.begin(),
                                    program.taskSurfaceChannels.end(),
                                    channel);
        if (role == program.taskSurfaceChannels.end())
            throw std::logic_error(
                "h13.linked-ring: task selects an undeclared surface");
        return static_cast<std::uint32_t>(
            4 + (role - program.taskSurfaceChannels.begin()));
    };
    for (const auto &span : spans) {
        const auto selectors = loadLE32(program.task, span.offset + 32);
        const auto sourceChannel = boundChannel(selectors & 31u);
        const auto resultChannel = boundChannel((selectors >> 12) & 31u);
        if (sourceChannel >= 4 &&
            loadLE32(program.task, h13RegisterOffset(
                program.task, span.offset, span.bytes, 0x13800)) != 0x8880u) {
            const auto sourceExtent = std::max(
                loadLE32(program.task, h13RegisterOffset(
                    program.task, span.offset, span.bytes, 0x13814)),
                loadLE32(program.task, h13RegisterOffset(
                    program.task, span.offset, span.bytes, 0x13818)));
            growChannel(sourceChannel, sourceExtent);
        }
        if (resultChannel >= 4 &&
            loadLE32(program.task, h13RegisterOffset(
                program.task, span.offset, span.bytes, 0x17800)) != 0x8880u) {
            const auto resultExtent = std::max(
                loadLE32(program.task, h13RegisterOffset(
                    program.task, span.offset, span.bytes, 0x17810)),
                loadLE32(program.task, h13RegisterOffset(
                    program.task, span.offset, span.bytes, 0x17814)));
            growChannel(resultChannel, resultExtent);
        }
    }
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

namespace {

std::array<std::uint8_t, 128> clipBoundsDescriptor(std::uint16_t lower,
                                                   std::uint16_t upper) {
    std::array<std::uint8_t, 128> descriptor{};
    const auto slot = [&](std::size_t index, std::uint16_t value) {
        descriptor[index * 2] = static_cast<std::uint8_t>(value);
        descriptor[index * 2 + 1] = static_cast<std::uint8_t>(value >> 8);
    };
    slot(0, lower);
    slot(1, upper);
    slot(2, lower);
    slot(3, upper);
    slot(37, 0x3c00);
    slot(39, 0x3c00);
    slot(41, 0x0040);
    slot(42, 0x0001);
    return descriptor;
}

} // namespace

void fuseConvClip(Program &program, std::uint16_t lower,
                  std::uint16_t upper) {
    if (!program.formulaBuilt)
        throw std::invalid_argument(
            "h13.conv-clip-outside-envelope: clip fusion requires a "
            "formula-built convolution");
    const auto descriptor = clipBoundsDescriptor(lower, upper);
    std::vector<std::uint8_t> constants;
    constants.reserve(program.constants.size() +
                      static_cast<std::size_t>(program.taskCount) * 16 *
                          descriptor.size());
    std::size_t offset = 0;
    std::size_t bytes = program.firstTaskBytes;
    for (std::uint32_t index = 0; index != program.taskCount; ++index) {
        if (!bytes || offset + bytes > program.task.size())
            throw std::invalid_argument(
                "h13.conv-clip-outside-envelope: convolution task stream is "
                "truncated");
        const auto selector = h13RegisterOffset(program.task, offset, bytes,
                                                0x0c804);
        const auto value = loadLE32(program.task, selector);
        if (value != 0x00101c00 && value != 0x00101c10)
            throw std::invalid_argument(
                "h13.conv-clip-outside-envelope: convolution NE word is not "
                "a decoded no-post-operation form");
        storeLE32(program.task, selector,
                  (value & ~0x00010000u) | 0x00020000u);

        for (std::uint32_t lane = 0; lane != 16; ++lane) {
            const auto offsetWord = h13RegisterOffset(
                program.task, offset, bytes, 0x1f848 + 4 * lane);
            const auto sizeWord = h13RegisterOffset(
                program.task, offset, bytes, 0x1f888 + 4 * lane);
            const std::size_t oldOffset = loadLE32(program.task, offsetWord);
            const std::size_t oldSize = loadLE32(program.task, sizeWord);
            if (oldOffset + oldSize > program.constants.size())
                throw std::invalid_argument(
                    "h13.conv-clip-outside-envelope: kernel lane exceeds the "
                    "constant section");
            if (constants.size() > UINT32_MAX || oldSize > UINT32_MAX - 128)
                throw std::invalid_argument(
                    "h13.conv-clip-outside-envelope: kernel lane offset or "
                    "size overflows");
            storeLE32(program.task, offsetWord,
                      static_cast<std::uint32_t>(constants.size()));
            storeLE32(program.task, sizeWord,
                      static_cast<std::uint32_t>(oldSize + 128));
            constants.insert(constants.end(), descriptor.begin(),
                             descriptor.end());
            constants.insert(constants.end(),
                program.constants.begin() + static_cast<std::ptrdiff_t>(oldOffset),
                program.constants.begin() +
                    static_cast<std::ptrdiff_t>(oldOffset + oldSize));
        }
        if (program.depthwiseConv) {
            const auto header1 = loadLE32(program.task, offset + 4);
            storeLE32(program.task, offset + 4,
                      (header1 & 0xffff0000u) | 2u);
        }
        const auto header1 = loadLE32(program.task, offset + 4);
        const auto next = loadLE32(program.task, offset + 28);
        if (index + 1 == program.taskCount) break;
        if (!next)
            throw std::invalid_argument(
                "h13.conv-clip-outside-envelope: convolution task link is "
                "missing");
        offset = next;
        bytes = (((header1 >> 16) & 0x1ff) + 1) * sizeof(std::uint32_t);
    }
    program.constants = std::move(constants);
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

namespace {

std::uint64_t residualStride2RegionBytes(const ConvShape &shape,
                                         std::uint32_t outputs) {
    const std::uint32_t laneCap = shape.output.width <= 8 ? 32 : 16;
    std::uint32_t lanes = outputs / 16;
    std::uint64_t bytes = 0;
    while (lanes) {
        const std::uint32_t width = std::min(lanes, laneCap);
        const std::uint32_t even = (width + 1) & ~1u;
        std::uint64_t perInput = 0;
        if (shape.kernel == 1) {
            const auto masks = (width + 7) / 8;
            perInput = masks + 2 * width +
                (width / 2 > masks ? width / 2 - masks : 0);
        } else if (shape.kernel == 3) {
            perInput = 2 * even + 2 * width * 9;
        } else {
            perInput = 8 * even + 2 * width * 49;
        }
        bytes += 2 * width + 3 + perInput * shape.input.channels;
        lanes -= width;
    }
    return alignUp(bytes, 64);
}

std::vector<std::pair<std::uint32_t, std::uint32_t>>
residualSlices(const ConvShape &shape) {
    // A task has a 1 MiB kernel-DMA window and sixteen output lanes.  Dense
    // reductions are also bounded by the 256 KiB per-input-channel window.
    // Deriving the split from those two hardware capacities makes the public
    // 224 ResNet-50 land on four split layers without naming any layer.
    constexpr std::uint64_t baseKernelWindow = 1u << 20;
    constexpr std::uint64_t channelWindow = 1u << 18;
    const std::uint64_t reduction = static_cast<std::uint64_t>(
        shape.input.channels) * shape.kernel * shape.kernelWidth;
    if (!reduction || shape.output.channels < 16)
        return {{0, shape.output.channels}};
    const std::uint64_t smallSpatialScale = shape.output.width < 14
        ? std::uint64_t(14 / shape.output.width) *
              (14 / shape.output.width) * (14 / shape.output.width)
        : 1;
    std::uint64_t byKernel = baseKernelWindow * smallSpatialScale /
                             (2 * reduction);
    std::uint64_t byChannel = UINT32_MAX;
    if (shape.stride == 2) {
        // Each lane's KDMA region has a 0xfe00-byte usable window at the
        // 14-wide class. Smaller spatial classes expose proportionally more
        // region capacity. Compute the largest whole 16-output group whose
        // worst-case nonzero sparse planes fit; this yields the split without
        // naming a layer or channel count.
        const std::uint64_t regionWindow = 0xfe00ull * smallSpatialScale;
        std::uint32_t sparseCap = 16;
        for (std::uint32_t outputs = 16;
             outputs <= shape.output.channels; outputs += 16) {
            if (residualStride2RegionBytes(shape, outputs) > regionWindow)
                break;
            sparseCap = outputs;
        }
        byKernel = sparseCap;
    } else if (shape.kernel == 1 && shape.output.width <= 14) {
        const std::uint64_t spatialScale =
            (14ull * 14ull) /
            (static_cast<std::uint64_t>(shape.output.width) *
             shape.output.width);
        byChannel = channelWindow * std::max<std::uint64_t>(1, spatialScale) /
                    shape.input.channels;
    }
    std::uint32_t cap = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(16, std::min(byKernel, byChannel)));
    cap = std::max(16u, cap & ~15u);
    if (shape.output.channels <= cap)
        return {{0, shape.output.channels}};

    std::vector<std::pair<std::uint32_t, std::uint32_t>> slices;
    const std::uint32_t tail = shape.output.channels % cap;
    std::uint32_t high = shape.output.channels;
    if (tail) {
        high -= tail;
        slices.push_back({high, tail});
    }
    while (high) {
        const std::uint32_t count = std::min(cap, high);
        high -= count;
        slices.push_back({high, count});
    }
    return slices;
}

void appendWord(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
    const auto at = bytes.size();
    bytes.resize(at + 4);
    storeLE32(bytes, at, value);
}

void appendRecord(std::vector<std::uint8_t> &bytes, std::uint32_t address,
                  const std::vector<std::uint32_t> &values) {
    appendWord(bytes, streamHeader(address,
        static_cast<std::uint32_t>(values.size())));
    for (const auto value : values) appendWord(bytes, value);
}

enum class ResidualRole {
    Stem,
    Pool,
    Projection,
    Main1,
    Main2,
    Main3,
    Basic1,
    Basic2,
    Add,
    Mean,
    Linear,
};

ResidualRole residualMainRole(std::size_t count, std::size_t position) {
    if (count == 2)
        return position ? ResidualRole::Basic2 : ResidualRole::Basic1;
    if (position == 0) return ResidualRole::Main1;
    if (position == 1) return ResidualRole::Main2;
    return ResidualRole::Main3;
}

struct ResidualTask {
    ResidualRole role;
    std::uint32_t slice = 0;
    std::uint32_t sliceCount = 1;
    std::uint32_t firstOutput = 0;
    std::uint32_t fullOutputs = 0;
    ConvShape shape{};
    Program encoded;
};

constexpr std::uint32_t residualSparseKernelConfig() {
    // NE KernelCfg: fp16 format 2, kernel engine enable and sparse phases.
    return 2u | (1u << 7) | (1u << 8);
}

constexpr std::uint32_t residualSparseKernelDMAConfig() {
    // KernelDMA Cfg: sparse phase-stream mode in the mode field at bit 5.
    return 3u << 5;
}

Program residualConvTask(const ResNet50ConvWeights &conv,
                         std::uint32_t first, std::uint32_t count,
                         bool relu) {
    ConvShape shape = conv.shape;
    shape.output.channels = count;
    const auto reduction = static_cast<std::size_t>(shape.input.channels) *
        shape.kernel * shape.kernelWidth;
    const auto *weights = conv.weights + 2 * first * reduction;
    const auto *bias = conv.bias + 2 * first;
    ConvShape baseShape = shape;
    std::vector<std::uint8_t> fakeWeights;
    std::vector<std::uint8_t> fakeBias;
    const auto denseChunks = laneChunks(count, 16);
    const bool mixedDense = shape.stride == 1 &&
        std::any_of(denseChunks.begin(), denseChunks.end(),
                    [&](auto lanes) { return lanes != denseChunks.front(); });
    if (shape.stride == 2 || mixedDense) {
        if (shape.stride == 2)
            baseShape.kernel = baseShape.kernelWidth = 1;
        if (shape.stride == 2 && baseShape.input.channels % 16)
            baseShape.input.channels = 16;
        baseShape.output.channels = 16;
        fakeWeights.assign(static_cast<std::size_t>(16) *
                           baseShape.input.channels * baseShape.kernel *
                           baseShape.kernelWidth * 2, 1);
        fakeBias.assign(32, 1);
    }
    Program task = encodeConvLinkedRing(
        baseShape,
        fakeWeights.empty() ? weights : fakeWeights.data(),
        fakeWeights.empty() ? count * reduction * 2 : fakeWeights.size(),
        fakeBias.empty() ? bias : fakeBias.data(),
        fakeBias.empty() ? count * 2 : fakeBias.size());
    if (task.taskCount != 1)
        throw std::logic_error(
            "H13 residual split rule did not produce one descriptor");
    if (shape.stride == 2) {
        std::array<std::uint32_t, 16> chunkBytes{};
        task.constants = packResidualH13Stride2(shape, weights, bias,
                                                chunkBytes);
        std::uint32_t chunkAt = 0;
        for (std::uint32_t lane = 0; lane != 16; ++lane) {
            putRegister(task.task, 0, task.firstTaskBytes,
                        0x1f848 + 4 * lane, chunkAt);
            putRegister(task.task, 0, task.firstTaskBytes,
                        0x1f888 + 4 * lane, chunkBytes[lane]);
            chunkAt += chunkBytes[lane];
        }
        auto cfg = loadLE32(task.task,
            h13RegisterOffset(task.task, 0, task.firstTaskBytes, 0x0001c));
        cfg &= ~((31u << 0) | (31u << 5) | (7u << 10) |
                 (3u << 13) | (3u << 15) |
                 (31u << 17) | (31u << 22));
        const auto cap = shape.output.width <= 8 ? 32u : 16u;
        const auto lanes = std::min(cap, conv.shape.output.channels / 16);
        cfg |= shape.kernelWidth | (shape.kernel << 5) |
               (floorLog2(lanes) << 10) |
               (shape.stride << 13) | (shape.stride << 15);
        putRegister(task.task, 0, task.firstTaskBytes, 0x0001c, cfg);
        putRegister(task.task, 0, task.firstTaskBytes, 0x00000,
                    (shape.input.height << 16) | shape.input.width);
        putRegister(task.task, 0, task.firstTaskBytes, 0x0000c,
                    shape.input.channels);
        putRegister(task.task, 0, task.firstTaskBytes, 0x00010, count);
        putRegister(task.task, 0, task.firstTaskBytes, 0x00014,
                    (shape.output.height << 16) | shape.output.width);
        putRegister(task.task, 0, task.firstTaskBytes, 0x00028,
                    shape.output.height);
        putRegister(task.task, 0, task.firstTaskBytes, 0x0c800,
                    residualSparseKernelConfig());
        putRegister(task.task, 0, task.firstTaskBytes, 0x1f800,
                    residualSparseKernelDMAConfig());
    } else if (mixedDense) {
        task.constants = packConvDense(static_cast<std::uint32_t>(reduction),
                                      denseChunks, weights, bias, false);
        if (task.constants.size() % 16)
            throw std::logic_error(
                "H13 residual dense section is not sixteen regions");
        const auto region = static_cast<std::uint32_t>(task.constants.size() / 16);
        for (std::uint32_t lane = 0; lane != 16; ++lane) {
            putRegister(task.task, 0, task.firstTaskBytes,
                        0x1f848 + 4 * lane, lane * region);
            putRegister(task.task, 0, task.firstTaskBytes,
                        0x1f888 + 4 * lane, region);
        }
        auto cfg = loadLE32(task.task,
            h13RegisterOffset(task.task, 0, task.firstTaskBytes, 0x0001c));
        // OCG encodes the ceiling log2 of the active output-lane group.  A
        // non-power-of-two tail therefore selects the next hardware class.
        const auto ocg = ceilLog2(denseChunks.front());
        cfg = (cfg & ~(7u << 10)) | (ocg << 10);
        putRegister(task.task, 0, task.firstTaskBytes, 0x0001c, cfg);
        putRegister(task.task, 0, task.firstTaskBytes, 0x00010, count);
    }
    if (relu) fuseConvPostOperation(task, PostOperation::Relu);
    return task;
}

constexpr std::uint32_t residualChannelConfig(bool hasSecondSource) {
    // Common ChannelCfg: fp16 source 0, optional fp16 source 1, fp16 result.
    return 2u | ((hasSecondSource ? 2u : 0u) << 2) | (2u << 4);
}

constexpr std::uint32_t residualConvolutionConfig(std::uint32_t kernel,
                                                  std::uint32_t stride) {
    // Common ConvCfg: Kw, Kh, X/Y stride, fp16 accumulate and PE traversal.
    return kernel | (kernel << 5) | (stride << 13) | (stride << 15) |
           (1u << 28) | (1u << 30);
}

constexpr std::uint32_t residualCommonTraversal() {
    // Common traversal: one source group, bias lane enabled, row-major tile.
    return 1u | (1u << 6) | (1u << 13);
}

constexpr std::uint32_t residualGroupConfig() {
    // Common GroupCfg: one input group and one output group.
    return 1u | (1u << 16);
}

constexpr std::uint32_t residualTileClass(ResidualRole role) {
    // Common TileClass: four-lane issue plus the PE operation's tile class.
    const std::uint32_t operationClass = role == ResidualRole::Mean ? 3u : 4u;
    return 4u | (operationClass << 4);
}

std::uint32_t residualColumnSchedule(ResidualRole role,
                                     ElementwiseShape input,
                                     ElementwiseShape output) {
    // Common +0x34 is ColumnSchedule, not a PE pipeline-stage mask.  Pool and
    // add traverse their output width; mean traverses the input reduction
    // surface.  All PE roles use the depthwise-style width form.  The shared
    // short-surface class repeats only when the width class itself requires
    // it, independent of a model or layer identity.
    const auto width = role == ResidualRole::Mean ? input.width : output.width;
    return commonColumnSchedule(width, commonWidthClass(width) == 4);
}

constexpr std::uint32_t residualPEConfig(ResidualRole role) {
    // PE Cfg: pool opcode 2, or binary-add enable with fused ReLU bit 5.
    return role == ResidualRole::Pool ? 2u
        : role == ResidualRole::Add ? (2u << 18) | (1u << 5) : 0u;
}

constexpr std::uint32_t residualKernelConfig(ResidualRole role) {
    // NE KernelCfg: base fp16 kernel format plus spatial-reduction enable.
    return 1u << 7 | (role == ResidualRole::Mean ? 1u << 14 : 0u);
}

constexpr std::uint32_t residualMacConfig() {
    // NE MacCfg: one accumulator group; PE-only tasks do not execute a MAC.
    return 1u << 20;
}

constexpr std::uint32_t residualHalfOneScale() {
    // PE scale register: fp16 1.0 in the high halfword lane.
    return 0x3c00u << 16;
}

std::uint32_t fp32Word(float value) {
    static_assert(sizeof(value) == sizeof(std::uint32_t));
    std::uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    return word;
}

std::uint32_t residualFinalScale(ResidualRole role, ElementwiseShape input) {
    if (role != ResidualRole::Mean) return fp32Word(1.0f);
    const auto elements = static_cast<std::uint64_t>(input.height) *
                          input.width;
    if (!elements)
        throw std::invalid_argument(
            "h13.residual-shape: mean input has an empty spatial extent");
    return fp32Word(1.0f / static_cast<float>(elements));
}

constexpr std::uint32_t residualDisabledDMAConfig() {
    // TileDMA Cfg: disabled transfer with cache hint 8 in all three nibbles.
    return (8u << 4) | (8u << 8) | (8u << 12);
}

constexpr std::uint32_t residualAliasSourceDMAConfig() {
    // TileDMA source Cfg: disabled transfer, cache hint 8 in all three
    // nibbles, and dependency class 4 for an initial resident alias.
    return residualDisabledDMAConfig() | (4u << 16);
}

constexpr std::uint32_t residualNeutralFormat() {
    // TileDMA format: neutral fp16 surface class with no layout transform.
    return 1u << 24;
}

constexpr std::uint32_t residualSourceRecordTail() {
    // TileDMA source tail: source-record valid bit.
    return 1u << 8;
}

constexpr std::uint32_t residualDestinationDisabledConfig() {
    // TileDMA destination Cfg: disabled transfer with cache hint 12.
    return 12u << 4;
}

constexpr std::uint32_t residualL2ExternalSourceConfig() {
    // L2 SourceCfg: fp16, external-source traversal, DMA-fed source class.
    return 2u | (7u << 4) | (1u << 8) | (5u << 20);
}

constexpr std::uint32_t residualL2ScratchSourceConfig() {
    // L2 SourceCfg: fp16, scratch traversal, interleave-8 source class.
    return 2u | (4u << 4) | (8u << 8);
}

constexpr std::uint32_t residualL2ExternalResultConfig() {
    // L2 ResultCfg: fp16 result, external traversal and output-DMA class.
    return 2u | (1u << 3) | (4u << 4) | (1u << 8);
}

constexpr std::uint32_t residualL2ScratchResultConfig() {
    // L2 ResultCfg: fp16 result, scratch traversal and interleave-8 class.
    return 2u | (1u << 3) | (4u << 4) | (8u << 8);
}

constexpr std::uint32_t residualSourceDMAConfig(bool external) {
    // TileDMA source Cfg: enable, cache hints, and dependency class 3 or 5.
    return 1u | ((external ? 8u : 0u) << 4) |
           ((external ? 8u : 9u) << 8) |
           (3u << 12) | ((external ? 3u : 5u) << 16);
}

constexpr std::uint32_t residualSourceFormat(bool external) {
    // TileDMA source format: fp16 NCHW, layout class 1 or interleave-8.
    return 1u | (3u << 4) | (2u << 12) |
           ((external ? 1u : 8u) << 24);
}

constexpr std::uint32_t residualDestinationDMAConfig(bool external) {
    // TileDMA destination Cfg: enable, cache hint 12, scratch data-set 3,
    // and the external/scratch write-path enable at bit 26.
    return 1u | (12u << 4) | (external ? 0u : 3u << 8) | (1u << 26);
}

constexpr std::uint32_t residualDestinationFormat(bool external) {
    // TileDMA destination format: fp16 NCHW, depth-major result layout,
    // and external class 1 or scratch interleave class 8.
    return 1u | (3u << 4) | (2u << 12) | (3u << 20) |
           ((external ? 1u : 8u) << 24);
}

constexpr std::uint32_t residualSecondSourceDMAConfig() {
    // TileDMA second-source Cfg: dependency class 1, scratch hint class 9.
    return (9u << 8) | (3u << 12) | (1u << 16);
}

constexpr std::uint32_t residualSecondSourceFormat() {
    // TileDMA second-source format: fp16 with scratch interleave class 8.
    return 1u | (8u << 24);
}

constexpr std::uint32_t residualSecondSourceExtension() {
    // TileDMA second-source extension: fp16 element and NCHW stride class.
    return (3u << 4) | (2u << 12);
}

constexpr std::uint32_t residualDebugEvents(bool first, bool last) {
    // Header debug events: all engine events plus source endpoint events on
    // the first task and the completion event on the terminal task.
    std::uint32_t events = (0xfffu << 12) | (1u << 11);
    if (first) events |= (1u << 3) | (1u << 5) | (1u << 6);
    if (last) events |= (1u << 1) | (1u << 3) | (1u << 5) | (1u << 6);
    return events;
}

constexpr std::uint32_t residualHeaderEvents(bool first, bool last) {
    // Header event word: compute event, source-DMA event on the first task,
    // and both terminal completion events on the final task.
    std::uint32_t events = 1u << 10;
    if (first) events |= 1u << 3;
    if (last) events |= (1u << 1) | (1u << 5);
    return events;
}

constexpr std::uint32_t residualStemConvFlags() {
    // Common ConvCfg stem flags: direct sparse input and native phase mode.
    return (1u << 18) | (1u << 23);
}

constexpr std::uint32_t kResidualHeaderSrcLoc = 1u << 28;
constexpr std::uint32_t kResidualHeaderDstLoc = 1u << 29;
constexpr std::uint32_t kResidualHeaderSPL = 1u << 11;
constexpr std::uint32_t kResidualHeaderTaskSignalEnable = 1u << 15;
constexpr std::uint32_t kResidualHeaderTaskSignalReset = 1u << 12;
constexpr std::uint32_t kResidualHeaderProducerControl = 1u << 14;
constexpr std::uint32_t kResidualHeaderConsumerControl = 1u << 13;
constexpr std::uint32_t kResidualTaskBufferEnable = 1u << 23;
constexpr std::uint32_t kResidualHeaderSourceRouteMask = 63u;
constexpr std::uint32_t kResidualHeaderSecondSourceRouteMask = 31u << 6;
constexpr std::uint32_t kResidualHeaderResultRouteMask = 63u << 12;
constexpr std::uint32_t kResidualHeaderRouteMask =
    kResidualHeaderSourceRouteMask |
    kResidualHeaderSecondSourceRouteMask |
    kResidualHeaderResultRouteMask |
    kResidualTaskBufferEnable;

Program residualPETask(ResidualRole role, ElementwiseShape input,
                       ElementwiseShape output) {
    const bool pool = role == ResidualRole::Pool;
    const bool add = role == ResidualRole::Add;
    const bool mean = role == ResidualRole::Mean;
    if (!pool && !add && !mean)
        throw std::logic_error("invalid H13 residual PE task role");
    Program task;
    task.task.resize(40, 0);
    // PE-only descriptors have both KernelDMA paths disabled in BaseEnable
    // and carry no constants.  Do not emit an unused kernel record.
    std::vector<std::uint32_t> common(16, 0);
    common[0] = (input.height << 16) | input.width;
    common[1] = 1;
    common[2] = residualChannelConfig(add);
    common[3] = input.channels;
    common[4] = output.channels;
    common[5] = ((mean ? input.height : output.height) << 16) |
                (mean ? input.width : output.width);
    common[6] = 1;
    common[7] = residualConvolutionConfig(pool ? 3u : 1u,
                                          pool ? 2u : 1u);
    common[8] = residualCommonTraversal();
    common[9] = residualGroupConfig();
    common[10] = mean ? input.height : output.height;
    common[11] = residualTileClass(role);
    common[13] = residualColumnSchedule(role, input, output);
    appendRecord(task.task, 0x00000, common);
    std::vector<std::uint32_t> source(28, 0);
    source[0] = source[1] = residualDisabledDMAConfig();
    source[14] = residualNeutralFormat();
    source[16] = residualSourceRecordTail();
    appendRecord(task.task, 0x13800, source);
    std::vector<std::uint32_t> l2(18, 0);
    appendRecord(task.task, 0x04800, l2);
    appendRecord(task.task, 0x08800,
        {residualPEConfig(role), residualHalfOneScale(),
         residualHalfOneScale(), residualFinalScale(role, input)});
    appendRecord(task.task, 0x0c800,
        {residualKernelConfig(role), residualMacConfig(), 0, 0, 0});
    std::vector<std::uint32_t> destination(7, 0);
    destination[0] = residualDestinationDisabledConfig();
    destination[6] = residualNeutralFormat();
    appendRecord(task.task, 0x17800, destination);
    if (task.task.size() != 376)
        throw std::logic_error("H13 compact PE task is not 376 bytes");
    task.firstTaskBytes = task.task.size();
    task.taskCount = 1;
    task.formulaBuilt = true;
    return task;
}

struct ResidualL2Layout {
    std::uint32_t channel, row, plane, depth;
};

void residualL2Source(std::vector<std::uint8_t> &stream, std::size_t at,
                      std::size_t bytes, std::uint32_t cfg,
                      std::uint32_t base, ResidualL2Layout layout) {
    putRegister(stream, at, bytes, 0x04804, cfg);
    putRegister(stream, at, bytes, 0x04808, base);
    putRegister(stream, at, bytes, 0x0480c, layout.channel);
    putRegister(stream, at, bytes, 0x04810, layout.row);
    putRegister(stream, at, bytes, 0x04814, layout.plane);
    putRegister(stream, at, bytes, 0x04818, layout.depth);
}

void residualL2Second(std::vector<std::uint8_t> &stream, std::size_t at,
                      std::size_t bytes, std::uint32_t base,
                      ResidualL2Layout layout) {
    putRegister(stream, at, bytes, 0x0481c, base);
    putRegister(stream, at, bytes, 0x04820, layout.channel);
    putRegister(stream, at, bytes, 0x04824, layout.row);
    putRegister(stream, at, bytes, 0x04828, layout.plane);
    putRegister(stream, at, bytes, 0x0482c, layout.depth);
}

void residualL2Result(std::vector<std::uint8_t> &stream, std::size_t at,
                      std::size_t bytes, std::uint32_t cfg,
                      std::uint32_t base, ResidualL2Layout layout) {
    putRegister(stream, at, bytes, 0x04830, cfg);
    putRegister(stream, at, bytes, 0x04834, base);
    putRegister(stream, at, bytes, 0x04838, layout.channel);
    putRegister(stream, at, bytes, 0x0483c, layout.row);
    putRegister(stream, at, bytes, 0x04840, layout.plane);
    putRegister(stream, at, bytes, 0x04844, layout.depth);
}

void residualTileSource(std::vector<std::uint8_t> &stream, std::size_t at,
                        std::size_t bytes, std::uint32_t cfg,
                        std::uint32_t base, std::uint32_t row,
                        std::uint32_t plane, std::uint32_t depth,
                        std::uint32_t format) {
    putRegister(stream, at, bytes, 0x13800, cfg);
    putRegister(stream, at, bytes, 0x13804, residualDisabledDMAConfig());
    putRegister(stream, at, bytes, 0x13808, base);
    putRegister(stream, at, bytes, 0x1380c, row);
    putRegister(stream, at, bytes, 0x13810, plane);
    putRegister(stream, at, bytes, 0x13814, depth);
    putRegister(stream, at, bytes, 0x13818, depth);
    for (std::uint32_t address = 0x1381c; address <= 0x13834; address += 4)
        putRegister(stream, at, bytes, address, 0);
    putRegister(stream, at, bytes, 0x13838, format);
    putRegister(stream, at, bytes, 0x1383c, 0);
    putRegister(stream, at, bytes, 0x13840, residualSourceRecordTail());
}

[[maybe_unused]] void residualTileSecond(std::vector<std::uint8_t> &stream, std::size_t at,
                        std::size_t bytes, std::uint32_t cfg,
                        std::uint32_t row, std::uint32_t plane,
                        std::uint32_t depth, std::uint32_t format,
                        std::uint32_t extension) {
    putRegister(stream, at, bytes, 0x13804, cfg);
    putRegister(stream, at, bytes, 0x13820, row);
    putRegister(stream, at, bytes, 0x13824, plane);
    putRegister(stream, at, bytes, 0x13828, depth);
    putRegister(stream, at, bytes, 0x1382c, depth);
    putRegister(stream, at, bytes, 0x13838, format);
    putRegister(stream, at, bytes, 0x1383c, extension);
}

void residualTileDestination(std::vector<std::uint8_t> &stream,
                             std::size_t at, std::size_t bytes,
                             std::uint32_t cfg, std::uint32_t base,
                             std::uint32_t row, std::uint32_t plane,
                             std::uint32_t depth, std::uint32_t format) {
    putRegister(stream, at, bytes, 0x17800, cfg);
    putRegister(stream, at, bytes, 0x17804, base);
    putRegister(stream, at, bytes, 0x17808, row);
    putRegister(stream, at, bytes, 0x1780c, plane);
    putRegister(stream, at, bytes, 0x17810, depth);
    putRegister(stream, at, bytes, 0x17814, depth);
    putRegister(stream, at, bytes, 0x17818, format);
}

} // namespace


namespace {

constexpr std::size_t kNoConvolution = std::numeric_limits<std::size_t>::max();

struct ResidualBlockPlan {
    std::vector<std::size_t> main;
    std::size_t projection = kNoConvolution;
};

bool sameResidualTensor(ElementwiseShape a, ElementwiseShape b) {
    return a.channels == b.channels && a.height == b.height &&
           a.width == b.width;
}

bool residualMainChain(const std::vector<ResNet50ConvWeights> &convolutions,
                       const std::vector<std::size_t> &indices,
                       ElementwiseShape input) {
    if (indices.size() != 2 && indices.size() != 3) return false;
    if (!sameResidualTensor(convolutions[indices.front()].shape.input, input))
        return false;
    if (indices.size() == 2) {
        if (convolutions[indices[0]].shape.kernel != 3 ||
            convolutions[indices[1]].shape.kernel != 3)
            return false;
    } else if (convolutions[indices[0]].shape.kernel != 1 ||
               convolutions[indices[1]].shape.kernel != 3 ||
               convolutions[indices[2]].shape.kernel != 1) {
        return false;
    }
    for (std::size_t index = 1; index != indices.size(); ++index)
        if (!sameResidualTensor(
                convolutions[indices[index - 1]].shape.output,
                convolutions[indices[index]].shape.input))
            return false;
    return true;
}

bool residualProjection(const ConvShape &shape, ElementwiseShape input,
                        ElementwiseShape output) {
    return shape.kernel == 1 && shape.kernelWidth == 1 &&
           sameResidualTensor(shape.input, input) &&
           sameResidualTensor(shape.output, output);
}

std::vector<ResidualBlockPlan> discoverResidualBlocks(
    const std::vector<ResNet50ConvWeights> &convolutions,
    ElementwiseShape input) {
    std::vector<ResidualBlockPlan> blocks;
    std::size_t cursor = 1;
    while (cursor < convolutions.size()) {
        bool found = false;
        ResidualBlockPlan chosen;
        std::size_t consumed = 0;
        for (std::size_t count = 4; count >= 2 && !found; --count) {
            if (cursor + count > convolutions.size()) continue;
            std::vector<std::size_t> order(count);
            std::iota(order.begin(), order.end(), cursor);
            do {
                for (const std::size_t mainCount : {std::size_t(3),
                                                    std::size_t(2)}) {
                    if (mainCount > count) continue;
                    std::vector<std::size_t> main(order.begin(),
                                                  order.begin() + mainCount);
                    if (!residualMainChain(convolutions, main, input))
                        continue;
                    const auto output =
                        convolutions[main.back()].shape.output;
                    const bool changed = !sameResidualTensor(input, output);
                    if (count == mainCount) {
                        if (changed) continue;
                        chosen = {main, kNoConvolution};
                    } else if (count == mainCount + 1 &&
                               residualProjection(
                                   convolutions[order.back()].shape,
                                   input, output)) {
                        chosen = {main, order.back()};
                    } else {
                        continue;
                    }
                    consumed = count;
                    found = true;
                    break;
                }
                if (found) break;
            } while (std::next_permutation(order.begin(), order.end()));
        }
        if (!found)
            throw std::invalid_argument(
                "h13.residual-shape: convolution sequence is not a basic "
                "or bottleneck block");
        input = convolutions[chosen.main.back()].shape.output;
        blocks.push_back(std::move(chosen));
        cursor += consumed;
    }
    return blocks;
}

struct ResidualTensorLayout {
    ResidualL2Layout words;
    std::uint32_t bytes;
};

ResidualTensorLayout residualTensorLayout(ElementwiseShape shape,
                                          std::uint32_t consumerStride = 1) {
    const std::uint64_t channel = align16(2ull * shape.width);
    const std::uint64_t plane = channel * shape.channels;
    // L2 advances one spatial row across all channels, then rotates banks by
    // the same geometry-derived skew as a formula-built convolution source.
    // Adding a whole channel row here used to turn the 64x56 pool result's
    // 0x1c00 plane into a 0x1c70 row.  The consumer expects the stride-class
    // skew, not another pixel row; the oversized stride made resident PE
    // results differ from their following convolution's source layout.
    const std::uint64_t row = plane +
        l2Skew(consumerStride, shape.width, shape.channels);
    const std::uint64_t bytes = row * shape.height;
    if (!shape.channels || !shape.height || !shape.width ||
        bytes >= kL2Bytes || row > UINT32_MAX || plane > UINT32_MAX)
        throw std::invalid_argument(
            "h13.residual-allocation: tensor does not fit one 2 MiB L2 range");
    return {ResidualL2Layout{static_cast<std::uint32_t>(channel),
                             static_cast<std::uint32_t>(row),
                             static_cast<std::uint32_t>(plane), 0},
            static_cast<std::uint32_t>(bytes)};
}

struct ResidualRange {
    std::uint32_t base;
    std::uint32_t bytes;
};

bool residualOverlap(ResidualRange a, ResidualRange b) {
    return a.base < static_cast<std::uint64_t>(b.base) + b.bytes &&
           b.base < static_cast<std::uint64_t>(a.base) + a.bytes;
}

class ResidualRingAllocator {
public:
    bool allocate(std::uint32_t bytes, const std::vector<ResidualRange> &live,
                  std::uint32_t *result) {
        const std::uint32_t aligned = static_cast<std::uint32_t>(align16(bytes));
        if (!aligned || aligned >= kL2Bytes) return false;
        // First-fit from the circular cursor.  The second pass implements the
        // wrap; live ranges are linear because no allocated tensor straddles
        // the physical end of the ring.
        for (unsigned pass = 0; pass != 2; ++pass) {
            std::uint32_t base = pass ? 0 : cursor_;
            const std::uint32_t limit = pass ? cursor_ :
                static_cast<std::uint32_t>(kL2Bytes);
            for (; static_cast<std::uint64_t>(base) + aligned <= limit;
                 base += 16) {
                const ResidualRange candidate{base, aligned};
                if (std::none_of(live.begin(), live.end(),
                    [&](ResidualRange range) {
                        return residualOverlap(candidate, range);
                    })) {
                    *result = base;
                    cursor_ = static_cast<std::uint32_t>(
                        (static_cast<std::uint64_t>(base) + aligned) %
                        kL2Bytes);
                    return true;
                }
            }
        }
        // Cursor can sit inside a live range after that range is reused for
        // an in-place result.  A complete linear fallback keeps first-fit
        // total without making placement depend on task identity.
        for (std::uint32_t base = 0;
             static_cast<std::uint64_t>(base) + aligned <= kL2Bytes;
             base += 16) {
            const ResidualRange candidate{base, aligned};
            if (std::none_of(live.begin(), live.end(),
                [&](ResidualRange range) {
                    return residualOverlap(candidate, range);
                })) {
                *result = base;
                cursor_ = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(base) + aligned) % kL2Bytes);
                return true;
            }
        }
        return false;
    }

private:
    std::uint32_t cursor_ = 0;
};

enum class ResidualStorage { Ring, Scratch, External };

struct ResidualValue {
    ElementwiseShape shape{};
    ResidualTensorLayout layout{};
    ResidualStorage storage = ResidualStorage::Ring;
    std::uint32_t base = 0;
    std::uint32_t scratchOffset = 0;
    std::uint32_t externalRow = 0;
    std::uint32_t externalPlane = 0;
    std::uint32_t externalDepth = 0;
};

struct ResidualRoute {
    ResidualValue source;
    ResidualValue result;
    ResidualValue second;
    bool hasSecond = false;
    std::uint32_t sourceStage = 0;
    std::uint32_t resultStage = 0;
    std::uint32_t secondStage = 0;
};

struct PlannedResidualTask {
    ResidualTask task;
    ResidualRoute route;
    std::string layer;
    std::string role;
    std::string boundary;
};

std::uint32_t residualScratchBytes(ElementwiseShape shape) {
    const std::uint64_t row = align16(16ull * shape.width);
    const std::uint64_t plane = row * shape.height;
    const std::uint64_t bytes = plane * ((shape.channels + 7) / 8);
    if (bytes > UINT32_MAX)
        throw std::invalid_argument(
            "h13.residual-allocation: scratch tensor exceeds 32-bit DMA extent");
    return static_cast<std::uint32_t>(bytes);
}

ResidualRange ringRange(const ResidualValue &value) {
    return {value.base, value.layout.bytes};
}

std::vector<ResidualRange> ringRanges(
    std::initializer_list<const ResidualValue *> values) {
    std::vector<ResidualRange> ranges;
    for (const auto *value : values)
        if (value && value->storage == ResidualStorage::Ring)
            ranges.push_back(ringRange(*value));
    return ranges;
}

ResidualValue allocateResidualValue(ElementwiseShape shape,
                                    ResidualRingAllocator &ring,
                                    std::uint32_t &scratchCursor,
                                    const std::vector<ResidualRange> &live,
                                    bool allowScratch = true,
                                    std::uint32_t consumerStride = 1) {
    ResidualValue value;
    value.shape = shape;
    value.layout = residualTensorLayout(shape, consumerStride);
    if (ring.allocate(value.layout.bytes, live, &value.base)) return value;
    if (!allowScratch)
        throw std::invalid_argument(
            "h13.residual-allocation: live tensors exceed the 2 MiB L2 ring");
    value.storage = ResidualStorage::Scratch;
    value.scratchOffset = static_cast<std::uint32_t>(
        alignUp(scratchCursor, 0x4000));
    const auto bytes = residualScratchBytes(shape);
    if (value.scratchOffset > UINT32_MAX - bytes)
        throw std::invalid_argument(
            "h13.residual-allocation: scratch arena exceeds 32 bits");
    scratchCursor = static_cast<std::uint32_t>(
        alignUp(static_cast<std::uint64_t>(value.scratchOffset) + bytes,
                0x4000));
    return value;
}

std::uint32_t allocateResidualStage(ResidualRingAllocator &ring,
                                    const ResidualValue &value,
                                    const std::vector<ResidualRange> &live) {
    if (value.storage == ResidualStorage::Ring) return value.base;
    // Tile DMA stages at most one quarter-ring at a time.  The task's tensor
    // geometry determines the smaller bound.
    auto bytes = std::min<std::uint32_t>(
        value.layout.bytes, static_cast<std::uint32_t>(kL2Bytes / 4));
    std::uint32_t base = 0;
    while (bytes >= 16) {
        if (ring.allocate(bytes, live, &base)) return base;
        bytes = static_cast<std::uint32_t>(align16(bytes / 2));
    }
    std::uint64_t occupied = 0;
    for (const auto range : live) occupied += range.bytes;
    throw std::invalid_argument(
        "h13.residual-allocation: no non-overlapping L2 DMA staging tile for " +
        std::to_string(value.shape.channels) + "x" +
        std::to_string(value.shape.height) + "x" +
        std::to_string(value.shape.width) + " with live bytes " +
        std::to_string(occupied));
}

std::uint32_t residualCost(const PlannedResidualTask &planned) {
    const auto &task = planned.task;
    std::uint64_t work = static_cast<std::uint64_t>(task.shape.output.height) *
        task.shape.output.width * std::max(1u, task.shape.output.channels);
    if (task.role != ResidualRole::Pool && task.role != ResidualRole::Add &&
        task.role != ResidualRole::Mean)
        work *= static_cast<std::uint64_t>(task.shape.input.channels) *
                task.shape.kernel * task.shape.kernelWidth;
    // CostHint is a saturated count of 2^23 work units.
    return static_cast<std::uint32_t>(std::max<std::uint64_t>(
        1, std::min<std::uint64_t>(0x7f, (work + (1u << 23) - 1) >> 23)));
}

std::uint32_t residualScheduleWord(const ResidualTask &task) {
    if (task.role == ResidualRole::Linear)
        return 0x10200001u; // linear schedule enable and one output tile
    const auto width = std::max(1u, task.shape.output.width);
    const auto scale = std::max(1u, width / 7);
    const auto digit = std::max(1u, 4u - std::min(3u, floorLog2(scale)));
    // Three 4-bit tile-group fields plus schedule/enable fields.
    std::uint32_t word = 0x00200001u | digit * 0x00011100u;
    if ((task.role == ResidualRole::Main3 ||
         task.role == ResidualRole::Projection) &&
        task.shape.output.channels >= 4 * task.shape.input.channels)
        word |= 0x00800000u; // expanded-output schedule class
    return word;
}

void disableResidualSourceDMA(std::vector<std::uint8_t> &stream,
                              std::size_t at, std::size_t bytes) {
    // TileDMA Disabled and neutral format fields.
    residualTileSource(stream, at, bytes, residualDisabledDMAConfig(),
                       0, 0, 0, 0, residualNeutralFormat());
}

void disableResidualDestinationDMA(std::vector<std::uint8_t> &stream,
                                   std::size_t at, std::size_t bytes) {
    // TileDMA Disabled and neutral format fields.
    residualTileDestination(stream, at, bytes,
                            residualDestinationDisabledConfig(),
                            0, 0, 0, 0, residualNeutralFormat());
}

void configureResidualSource(std::vector<std::uint8_t> &stream,
                             std::size_t at, std::size_t bytes,
                             const ResidualValue &value,
                             std::uint32_t stage,
                             bool initialAlias) {
    if (value.storage == ResidualStorage::Ring) {
        residualL2Source(stream, at, bytes,
                         initialAlias ? l2InitialAliasSourceConfig()
                                      : l2DependentSourceConfig(),
                         value.base,
                         value.layout.words); // L2 resident source
        if (initialAlias) {
            const auto row = static_cast<std::uint32_t>(
                align16(2ull * value.shape.width));
            const auto plane = row * value.shape.height;
            const auto depth = plane * value.shape.channels;
            residualTileSource(stream, at, bytes,
                residualAliasSourceDMAConfig(), 0, row, plane, depth,
                residualSourceFormat(true));
        } else {
            disableResidualSourceDMA(stream, at, bytes);
        }
        return;
    }
    const bool external = value.storage == ResidualStorage::External;
    auto l2 = value.layout.words;
    if (external) {
        // An external NCHW source is staged as one complete pixel row across
        // channels.  The device-proven linked-ring stem repeats that span in
        // all three extent fields; it does not append the resident bank skew
        // and it never leaves the final extent zero.
        l2.channel = static_cast<std::uint32_t>(
            align16(2ull * value.shape.width));
        l2.row = l2.channel * value.shape.channels;
        l2.plane = l2.row;
        l2.depth = l2.row;
    }
    residualL2Source(stream, at, bytes,
                     external ? residualL2ExternalSourceConfig()
                              : residualL2ScratchSourceConfig(),
                     stage, l2);
    const std::uint32_t row = external
        ? static_cast<std::uint32_t>(align16(2ull * value.shape.width))
        : static_cast<std::uint32_t>(align16(16ull * value.shape.width));
    const std::uint32_t plane = row * value.shape.height;
    const std::uint32_t depth = external
        ? plane * value.shape.channels
        : plane * ((value.shape.channels + 7) / 8);
    residualTileSource(stream, at, bytes,
        residualSourceDMAConfig(external), value.scratchOffset,
        row, plane, depth, residualSourceFormat(external));
}

void configureResidualResult(std::vector<std::uint8_t> &stream,
                             std::size_t at, std::size_t bytes,
                             const ResidualValue &value,
                             std::uint32_t stage,
                             std::uint32_t channelOffset,
                             bool initialAlias) {
    if (value.storage == ResidualStorage::Ring) {
        residualL2Result(stream, at, bytes,
                         l2ResidentResultConfig(initialAlias),
                         value.base + channelOffset * value.layout.words.channel,
                         value.layout.words); // L2 resident result
        if (initialAlias) {
            // Initial aliases retain the convolution template's populated
            // destination geometry; only the transfer itself is disabled.
            putRegister(stream, at, bytes, 0x17800,
                        residualDestinationDisabledConfig());
        } else {
            disableResidualDestinationDMA(stream, at, bytes);
        }
        return;
    }
    residualL2Result(stream, at, bytes,
                     value.storage == ResidualStorage::External
                         ? residualL2ExternalResultConfig()
                         : residualL2ScratchResultConfig(),
                     stage, {0, 0, 0, 0});
    const bool external = value.storage == ResidualStorage::External;
    const std::uint32_t row = external
        ? (value.externalRow ? value.externalRow
                             : static_cast<std::uint32_t>(
                                   align16(2ull * value.shape.channels)))
        : static_cast<std::uint32_t>(align16(16ull * value.shape.width));
    const std::uint32_t plane = external
        ? (value.externalPlane ? value.externalPlane : row)
        : row * value.shape.height;
    const std::uint32_t depth = external
        ? (value.externalDepth ? value.externalDepth : row)
        : plane * ((value.shape.channels + 7) / 8);
    residualTileDestination(stream, at, bytes,
        residualDestinationDMAConfig(external),
        value.scratchOffset, row, plane, depth,
        residualDestinationFormat(external));
}

void configureResidualSecond(std::vector<std::uint8_t> &stream,
                             std::size_t at, std::size_t bytes,
                             const ResidualValue &value,
                             std::uint32_t stage) {
    residualL2Second(stream, at, bytes,
                     value.storage == ResidualStorage::Ring ? value.base : stage,
                     value.layout.words);
    if (value.storage == ResidualStorage::Ring) return;
    const std::uint32_t row = static_cast<std::uint32_t>(
        align16(16ull * value.shape.width));
    const std::uint32_t plane = row * value.shape.height;
    const std::uint32_t depth = plane * ((value.shape.channels + 7) / 8);
    putRegister(stream, at, bytes, 0x13804,
                residualSecondSourceDMAConfig());
    putRegister(stream, at, bytes, 0x1381c, value.scratchOffset);
    putRegister(stream, at, bytes, 0x13820, row);
    putRegister(stream, at, bytes, 0x13824, plane);
    putRegister(stream, at, bytes, 0x13828, depth);
    putRegister(stream, at, bytes, 0x1382c, depth);
    putRegister(stream, at, bytes, 0x13838,
                residualSecondSourceFormat());
    putRegister(stream, at, bytes, 0x1383c,
                residualSecondSourceExtension());
}

void routeResidualTask(Program &program, PlannedResidualTask &planned,
                       std::size_t at, bool last) {
    auto &task = planned.task;
    const auto bytes = task.encoded.firstTaskBytes;
    const bool initialAliasProducer = task.role == ResidualRole::Stem &&
        planned.route.result.storage == ResidualStorage::Ring;
    const bool initialAliasConsumer = task.role == ResidualRole::Pool &&
        planned.route.source.storage == ResidualStorage::Ring;
    configureResidualSource(program.task, at, bytes, planned.route.source,
                            planned.route.sourceStage,
                            initialAliasConsumer);
    configureResidualResult(program.task, at, bytes, planned.route.result,
                            planned.route.resultStage, task.firstOutput,
                            initialAliasProducer);
    if (planned.route.hasSecond)
        configureResidualSecond(program.task, at, bytes, planned.route.second,
                                planned.route.secondStage);

    // Every task closes its work with TSE+TSR.  The following task's L2 or
    // TileDMA dependency consumes that signal, serialising resident reads as
    // well as spill reads.  Without the signal, even the two-task stem/pool
    // rung can let the pool read the ring before the stem has filled it.
    std::uint32_t control = kResidualHeaderTaskSignalEnable |
                            kResidualHeaderTaskSignalReset;
    if (planned.route.source.storage != ResidualStorage::Ring)
        control |= kResidualHeaderSrcLoc | kResidualHeaderSPL;
    if (planned.route.result.storage != ResidualStorage::Ring)
        control |= kResidualHeaderDstLoc;
    if (initialAliasProducer)
        control |= kResidualHeaderProducerControl;
    if (initialAliasConsumer)
        control |= kResidualHeaderConsumerControl | kResidualHeaderSPL;
    std::uint32_t selectors = last ? 0u : kResidualTaskBufferEnable;
    const auto sourceChannel = planned.route.source.storage ==
            ResidualStorage::External ? 5u : 3u;
    if (planned.route.source.storage != ResidualStorage::Ring)
        selectors |= sourceChannel | 1u << 5;
    else if (initialAliasConsumer)
        selectors |= 3u | 1u << 5;
    if (planned.route.hasSecond && planned.route.second.storage !=
            ResidualStorage::Ring)
        selectors |= 3u << 6; // scratch second-source channel
    const auto resultChannel = planned.route.result.storage ==
            ResidualStorage::External ? 4u : 3u;
    if (planned.route.result.storage != ResidualStorage::Ring)
        selectors |= (resultChannel | 1u << 5) << 12;
    else if (initialAliasProducer)
        selectors |= (3u | 1u << 5) << 12;
    storeLE32(program.task, at + 24, control);
    // Replace only the route-owned fields.  Kernel templates already set the
    // two BaseEnable kernel bits, while compact PE tasks leave them clear.
    // Keeping that base word is required: replacing it with `selectors`
    // disables the kernel path before the first residual convolution runs.
    const auto baseEnable = loadLE32(program.task, at + 32);
    storeLE32(program.task, at + 32,
              (baseEnable & ~kResidualHeaderRouteMask) | selectors);
}

} // namespace

Program encodeResidualNetwork(
    const std::vector<ResNet50ConvWeights> &convolutions,
    const std::uint8_t *headWeights, std::size_t headWeightBytes,
    const std::uint8_t *headBias, std::size_t headBiasBytes,
    const std::string &terminateAfter) {
    if (convolutions.size() < 3)
        throw std::invalid_argument(
            "h13.residual-shape: network needs a stem and one block");
    for (const auto &conv : convolutions) {
        const std::size_t reduction = static_cast<std::size_t>(
            conv.shape.input.channels) * conv.shape.kernel *
            conv.shape.kernelWidth;
        if (conv.shape.groups != 1 || !conv.shape.bias || !conv.weights ||
            !conv.bias || conv.weightBytes !=
                static_cast<std::size_t>(conv.shape.output.channels) *
                    reduction * 2 ||
            conv.biasBytes !=
                static_cast<std::size_t>(conv.shape.output.channels) * 2)
            throw std::invalid_argument(
                "h13.residual-shape: convolution constants disagree with geometry");
    }
    const auto stemOutput = convolutions.front().shape.output;
    const ElementwiseShape pooled{
        stemOutput.channels, (stemOutput.height + 1) / 2,
        (stemOutput.width + 1) / 2};
    const auto blocks = discoverResidualBlocks(convolutions, pooled);
    const auto finalShape =
        convolutions[blocks.back().main.back()].shape.output;
    const std::uint32_t headOutputs =
        static_cast<std::uint32_t>(headBiasBytes / 2);
    if (!headWeights || !headBias || !headOutputs ||
        headBiasBytes != static_cast<std::size_t>(headOutputs) * 2 ||
        headWeightBytes != static_cast<std::size_t>(finalShape.channels) *
                               headOutputs * 2)
        throw std::invalid_argument(
            "h13.residual-shape: classifier constants disagree with final channels");

    std::uint32_t taskCount = 3 + static_cast<std::uint32_t>(blocks.size());
    for (const auto &conv : convolutions)
        taskCount += static_cast<std::uint32_t>(
            residualSlices(conv.shape).size());

    ResidualRingAllocator ring;
    std::uint32_t scratchCursor = 0;
    std::vector<PlannedResidualTask> planned;
    planned.reserve(taskCount);

    auto stageRoute = [&](ResidualRoute &route,
                          const std::vector<ResidualRange> &live) {
        auto occupied = live;
        if (route.result.storage == ResidualStorage::Ring)
            occupied.push_back(ringRange(route.result));
        route.sourceStage = allocateResidualStage(ring, route.source,
                                                  occupied);
        if (route.result.storage != ResidualStorage::Ring)
            route.resultStage = allocateResidualStage(ring, route.result,
                                                       occupied);
        if (route.hasSecond &&
            route.second.storage != ResidualStorage::Ring)
            route.secondStage = allocateResidualStage(ring, route.second,
                                                       occupied);
    };

    auto emitConv = [&](std::size_t index, ResidualRole role,
                        const std::string &layer, const std::string &roleName,
                        const ResidualValue &source,
                        const ResidualValue &result, bool relu,
                        const std::vector<ResidualRange> &live) {
        const auto slices = residualSlices(convolutions[index].shape);
        for (std::size_t slice = 0; slice != slices.size(); ++slice) {
            const auto [first, count] = slices[slice];
            PlannedResidualTask entry;
            entry.task.role = role;
            entry.task.slice = static_cast<std::uint32_t>(slice);
            entry.task.sliceCount = static_cast<std::uint32_t>(slices.size());
            entry.task.firstOutput = first;
            entry.task.fullOutputs = convolutions[index].shape.output.channels;
            entry.task.shape = convolutions[index].shape;
            entry.task.shape.output.channels = count;
            entry.task.encoded = residualConvTask(convolutions[index], first,
                                                  count, relu);
            entry.layer = layer;
            entry.role = roleName;
            entry.route.source = source;
            entry.route.result = result;
            stageRoute(entry.route, live);
            planned.push_back(std::move(entry));
        }
    };

    ResidualValue externalInput;
    externalInput.shape = convolutions.front().shape.input;
    externalInput.layout = residualTensorLayout(externalInput.shape, 2);
    externalInput.storage = ResidualStorage::External;
    auto stem = allocateResidualValue(stemOutput, ring, scratchCursor, {},
                                      true, 2);
    const bool poolViaDRAM = terminateAfter == "pool-via-dram";
    if (poolViaDRAM) {
        stem.storage = ResidualStorage::Scratch;
        stem.scratchOffset = static_cast<std::uint32_t>(
            alignUp(scratchCursor, 0x4000));
        const auto bytes = residualScratchBytes(stem.shape);
        scratchCursor = static_cast<std::uint32_t>(alignUp(
            static_cast<std::uint64_t>(stem.scratchOffset) + bytes, 0x4000));
    }
    emitConv(0, ResidualRole::Stem, "stem", "stem", externalInput, stem, true,
             ringRanges({&stem}));
    planned.back().boundary = "stem";

    auto current = allocateResidualValue(pooled, ring, scratchCursor,
                                         ringRanges({&stem}));
    PlannedResidualTask pool;
    pool.task.role = ResidualRole::Pool;
    pool.task.shape = {3, 3, 2, 1, false, stemOutput, pooled};
    pool.task.encoded = residualPETask(ResidualRole::Pool, stemOutput, pooled);
    pool.layer = "pool";
    pool.role = "pool";
    pool.boundary = "pool";
    pool.route.source = stem;
    pool.route.result = current;
    stageRoute(pool.route, ringRanges({&stem, &current}));
    planned.push_back(std::move(pool));

    for (std::size_t blockIndex = 0; blockIndex != blocks.size(); ++blockIndex) {
        const auto &block = blocks[blockIndex];
        const std::string blockName = "block" +
            (blockIndex + 1 < 10 ? std::string("0") : std::string()) +
            std::to_string(blockIndex + 1);
        const auto blockInput = current;
        ResidualValue shortcut = blockInput;
        if (block.projection != kNoConvolution) {
            const auto output = convolutions[block.projection].shape.output;
            shortcut = allocateResidualValue(output, ring, scratchCursor,
                                              ringRanges({&blockInput}));
            emitConv(block.projection, ResidualRole::Projection,
                     blockName, "projection", blockInput,
                     shortcut, false, ringRanges({&blockInput, &shortcut}));
        }

        ResidualValue main = blockInput;
        for (std::size_t position = 0; position != block.main.size();
             ++position) {
            const auto index = block.main[position];
            const auto output = convolutions[index].shape.output;
            auto live = ringRanges({&shortcut, &main});
            const auto consumerStride = position + 1 < block.main.size()
                ? convolutions[block.main[position + 1]].shape.stride : 1u;
            auto next = allocateResidualValue(output, ring, scratchCursor,
                                              live, true, consumerStride);
            const auto role = residualMainRole(block.main.size(), position);
            const std::string roleName = block.main.size() == 2
                ? (position ? "basic2" : "basic1")
                : (position == 0 ? "main1" : position == 1 ? "main2" : "main3");
            emitConv(index, role, blockName, roleName, main, next,
                     position + 1 != block.main.size(),
                     ringRanges({&shortcut, &main, &next}));
            main = next;
        }

        ResidualValue sum = main;
        if (sum.storage != ResidualStorage::Ring) {
            sum = allocateResidualValue(main.shape, ring, scratchCursor,
                                        ringRanges({&shortcut}));
        }
        PlannedResidualTask add;
        add.task.role = ResidualRole::Add;
        add.task.shape = {1, 1, 1, 1, false, main.shape, main.shape};
        add.task.encoded = residualPETask(ResidualRole::Add, main.shape,
                                          main.shape);
        add.layer = blockName;
        add.role = "add+relu";
        add.boundary = blockName;
        add.route.source = main;
        add.route.second = shortcut;
        add.route.hasSecond = true;
        add.route.result = sum;
        stageRoute(add.route, ringRanges({&main, &shortcut, &sum}));
        planned.push_back(std::move(add));
        current = sum;
    }

    const ElementwiseShape meanShape{finalShape.channels, 1, 1};
    auto meanValue = allocateResidualValue(meanShape, ring, scratchCursor,
                                           ringRanges({&current}));
    PlannedResidualTask mean;
    mean.task.role = ResidualRole::Mean;
    mean.task.shape = {1, 1, 1, 1, false, finalShape, meanShape};
    mean.task.encoded = residualPETask(ResidualRole::Mean, finalShape,
                                       meanShape);
    mean.layer = "mean";
    mean.role = "reduce_mean";
    mean.boundary = "mean";
    mean.route.source = current;
    mean.route.result = meanValue;
    stageRoute(mean.route, ringRanges({&current, &meanValue}));
    planned.push_back(std::move(mean));

    ResidualValue externalOutput;
    externalOutput.shape = {headOutputs, 1, 1};
    externalOutput.layout = residualTensorLayout(externalOutput.shape);
    externalOutput.storage = ResidualStorage::External;
    ConvShape headShape{1, 1, 1, 1, true, meanShape,
                        externalOutput.shape};
    ConvShape encodedHeadShape = headShape;
    std::uint32_t encodedHeadOutputs = 16;
    while (encodedHeadOutputs < headOutputs) encodedHeadOutputs <<= 1;
    encodedHeadShape.output.channels = encodedHeadOutputs;
    std::vector<std::uint8_t> fakeWeights(
        static_cast<std::size_t>(finalShape.channels) * encodedHeadOutputs * 2,
        1);
    std::vector<std::uint8_t> fakeBias(encodedHeadOutputs * 2, 1);
    PlannedResidualTask linear;
    linear.task.role = ResidualRole::Linear;
    linear.task.firstOutput = 0;
    linear.task.fullOutputs = headOutputs;
    linear.task.shape = headShape;
    linear.task.encoded = encodeConvLinkedRing(
        encodedHeadShape, fakeWeights.data(), fakeWeights.size(),
        fakeBias.data(), fakeBias.size());
    linear.layer = "logits";
    linear.role = "linear";
    linear.boundary = "logits";
    if (linear.task.encoded.taskCount != 1)
        throw std::invalid_argument(
            "h13.residual-fields: classifier template split unexpectedly");
    std::array<std::uint32_t, 16> headChunks{};
    linear.task.encoded.constants = ane::h16::packResNetLinearHead(
        finalShape.channels, headOutputs, headWeights, headBias, headChunks);
    std::uint32_t headAt = 0;
    for (std::uint32_t lane = 0; lane != 16; ++lane) {
        putRegister(linear.task.encoded.task, 0,
                    linear.task.encoded.firstTaskBytes, 0x1f848 + 4 * lane,
                    headAt);
        putRegister(linear.task.encoded.task, 0,
                    linear.task.encoded.firstTaskBytes, 0x1f888 + 4 * lane,
                    headChunks[lane]);
        headAt += headChunks[lane];
    }
    putRegister(linear.task.encoded.task, 0,
                linear.task.encoded.firstTaskBytes, 0x00010, headOutputs);
    linear.route.source = meanValue;
    linear.route.result = externalOutput;
    stageRoute(linear.route, ringRanges({&meanValue}));
    planned.push_back(std::move(linear));

    if (planned.size() != taskCount)
        throw std::logic_error(
            "h13.residual-fields: topology task count changed while encoding");

    ElementwiseShape boundaryShape{};
    bool boundaryOutput = false;
    if (!terminateAfter.empty() && terminateAfter != "logits") {
        const std::string boundary = poolViaDRAM ? "pool" : terminateAfter;
        const auto target = std::find_if(planned.begin(), planned.end(),
            [&](const PlannedResidualTask &entry) {
                return entry.boundary == boundary;
            });
        if (target == planned.end())
            throw std::invalid_argument(
                "h13.residual-boundary: unknown boundary " + terminateAfter);
        boundaryShape = target->route.result.shape;
        ResidualValue external = target->route.result;
        if (external.storage == ResidualStorage::Ring)
            target->route.resultStage = external.base;
        external.storage = ResidualStorage::External;
        external.scratchOffset = 0;
        external.externalRow = static_cast<std::uint32_t>(
            alignUp(2ull * boundaryShape.width, 64));
        external.externalPlane = external.externalRow * boundaryShape.height;
        const std::uint64_t depth = static_cast<std::uint64_t>(
            external.externalPlane) * boundaryShape.channels;
        if (depth > UINT32_MAX)
            throw std::invalid_argument(
                "h13.residual-boundary: external result exceeds 32-bit DMA extent");
        external.externalDepth = static_cast<std::uint32_t>(depth);
        target->route.result = external;
        planned.erase(target + 1, planned.end());
        boundaryOutput = true;
    }

    Program program;
    program.taskSurfaceChannels = {4, 5, 6, 7};
    program.formulaBuilt = true;
    program.scratchAllocationBytes = scratchCursor;
    program.outputBoundary = boundaryOutput ? terminateAfter : "logits";
    for (const auto &entry : planned)
        program.taskRecords.push_back({entry.layer, entry.role,
                                       entry.boundary,
                                       entry.task.shape.output});
    std::vector<std::size_t> offsets(planned.size());
    for (std::size_t index = 0; index != planned.size(); ++index) {
        std::size_t at = static_cast<std::size_t>(
            alignUp(program.task.size(), 0x100));
        const auto bytes = planned[index].task.encoded.firstTaskBytes;
        if ((at & 0x3fffu) + bytes > 0x4000u)
            at = static_cast<std::size_t>(alignUp(at, 0x4000));
        program.task.resize(at + bytes, 0);
        offsets[index] = at;
        std::copy_n(planned[index].task.encoded.task.begin(), bytes,
                    program.task.begin() + static_cast<std::ptrdiff_t>(at));
    }

    for (std::size_t index = 0; index != planned.size(); ++index) {
        auto &entry = planned[index];
        const auto at = offsets[index];
        const bool first = index == 0;
        const bool last = index + 1 == planned.size();
        const auto nextBytes = last ? 0
            : planned[index + 1].task.encoded.firstTaskBytes;
        std::uint32_t word0 = static_cast<std::uint32_t>(index);
        if (index) word0 |= 1u << 24;
        if (last) word0 |= 1u << 25;
        storeLE32(program.task, at, word0);
        storeLE32(program.task, at + 4, residualCost(entry) |
            (last ? 0u :
             (static_cast<std::uint32_t>(nextBytes / 4 - 1) << 16)));
        storeLE32(program.task, at + 8, residualHeaderEvents(first, last));
        storeLE32(program.task, at + 16, residualDebugEvents(first, last));
        storeLE32(program.task, at + 28,
                  last ? 0u : static_cast<std::uint32_t>(offsets[index + 1]));
        if (entry.task.role != ResidualRole::Pool &&
            entry.task.role != ResidualRole::Add &&
            entry.task.role != ResidualRole::Mean) {
            putRegister(program.task, at, entry.task.encoded.firstTaskBytes,
                        0x00034, residualScheduleWord(entry.task));
            if (entry.task.role == ResidualRole::Stem) {
                const auto word = h13RegisterOffset(
                    program.task, at, entry.task.encoded.firstTaskBytes,
                    0x0001c);
                storeLE32(program.task, word,
                          loadLE32(program.task, word) |
                              residualStemConvFlags());
            }
        }
        routeResidualTask(program, entry, at, last);
        if (entry.task.role != ResidualRole::Pool &&
            entry.task.role != ResidualRole::Add &&
            entry.task.role != ResidualRole::Mean) {
            const auto constantBase = static_cast<std::uint32_t>(
                program.constants.size());
            for (std::uint32_t lane = 0; lane != 16; ++lane) {
                const auto word = h13RegisterOffset(
                    program.task, at, entry.task.encoded.firstTaskBytes,
                    0x1f848 + 4 * lane);
                storeLE32(program.task, word,
                          loadLE32(program.task, word) + constantBase);
            }
            program.constants.insert(program.constants.end(),
                entry.task.encoded.constants.begin(),
                entry.task.encoded.constants.end());
            program.constants.resize(alignUp(program.constants.size(), 64), 0);
        }
    }

    program.taskCount = static_cast<std::uint32_t>(planned.size());
    program.firstTaskBytes = planned.front().task.encoded.firstTaskBytes;
    program.constantOffsetBytes = alignUp(program.task.size(), 0x40);
    const auto input = convolutions.front().shape.input;
    const std::uint64_t inputRow = align16(2ull * input.width);
    const std::uint64_t inputPlane = inputRow * input.height;
    program.inputs = {{5, {1, input.channels, input.height, input.width,
                           inputPlane, inputRow},
                       alignUp(inputPlane * input.channels, 0x4000)}};
    if (boundaryOutput) {
        const std::uint64_t outputRow = alignUp(2ull * boundaryShape.width, 64);
        const std::uint64_t outputPlane = outputRow * boundaryShape.height;
        program.output = {4, {1, boundaryShape.channels,
                              boundaryShape.height, boundaryShape.width,
                              outputPlane, outputRow},
                          alignUp(outputPlane * boundaryShape.channels,
                                  tileBytes)};
    } else {
        const std::uint64_t outputRow = align16(2ull * headOutputs);
        program.output = {4, {1, 1, 1, headOutputs, outputRow, outputRow},
                          alignUp(2ull * headOutputs, tileBytes)};
    }
    return program;
}

Program encodeResNet50(
    const std::vector<ResNet50ConvWeights> &convolutions,
    const std::uint8_t *headWeights, std::size_t headWeightBytes,
    const std::uint8_t *headBias, std::size_t headBiasBytes) {
    return encodeResidualNetwork(convolutions, headWeights, headWeightBytes,
                                 headBias, headBiasBytes);
}

} // namespace ane::h13
