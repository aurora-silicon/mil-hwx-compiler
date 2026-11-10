#include "../plugins/H13/H13Program.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ane::h13::ConvShape;
using ane::h13::ElementwiseShape;

struct Shape { unsigned cin, cout, spatial, kernel, stride; };

static const Shape bottleneckShapes[] = {
    {3,64,224,7,2},{64,64,56,1,1},{64,64,56,3,1},{64,256,56,1,1},
    {64,256,56,1,1},{256,64,56,1,1},{64,64,56,3,1},{64,256,56,1,1},
    {256,64,56,1,1},{64,64,56,3,1},{64,256,56,1,1},{256,512,56,1,2},
    {256,128,56,1,1},{128,128,56,3,2},{128,512,28,1,1},{512,128,28,1,1},
    {128,128,28,3,1},{128,512,28,1,1},{512,128,28,1,1},{128,128,28,3,1},
    {128,512,28,1,1},{512,128,28,1,1},{128,128,28,3,1},{128,512,28,1,1},
    {512,1024,28,1,2},{512,256,28,1,1},{256,256,28,3,2},{256,1024,14,1,1},
    {1024,256,14,1,1},{256,256,14,3,1},{256,1024,14,1,1},{1024,256,14,1,1},
    {256,256,14,3,1},{256,1024,14,1,1},{1024,256,14,1,1},{256,256,14,3,1},
    {256,1024,14,1,1},{1024,256,14,1,1},{256,256,14,3,1},{256,1024,14,1,1},
    {1024,256,14,1,1},{256,256,14,3,1},{256,1024,14,1,1},{1024,2048,14,1,2},
    {1024,512,14,1,1},{512,512,14,3,2},{512,2048,7,1,1},{2048,512,7,1,1},
    {512,512,7,3,1},{512,2048,7,1,1},{2048,512,7,1,1},{512,512,7,3,1},
    {512,2048,7,1,1},
};

static const Shape basicShapes[] = {
    {3,64,224,7,2},
    {64,64,56,3,1},{64,64,56,3,1},
    {64,64,56,3,1},{64,64,56,3,1},
    {64,128,56,1,2},{64,128,56,3,2},{128,128,28,3,1},
    {128,128,28,3,1},{128,128,28,3,1},
    {128,256,28,1,2},{128,256,28,3,2},{256,256,14,3,1},
    {256,256,14,3,1},{256,256,14,3,1},
    {256,512,14,1,2},{256,512,14,3,2},{512,512,7,3,1},
    {512,512,7,3,1},{512,512,7,3,1},
};

static void putHalfOne(std::vector<std::uint8_t> &bytes, std::size_t index) {
    bytes[2 * index] = 0;
    bytes[2 * index + 1] = 0x3c;
}

struct OwnedNetwork {
    std::vector<std::vector<std::uint8_t>> weights;
    std::vector<std::vector<std::uint8_t>> biases;
    std::vector<ane::h13::ResNet50ConvWeights> convs;
};

template <std::size_t N>
static OwnedNetwork makeNetwork(const Shape (&shapes)[N], unsigned input) {
    OwnedNetwork network;
    network.weights.reserve(N);
    network.biases.reserve(N);
    network.convs.reserve(N);
    for (unsigned index = 0; index != N; ++index) {
        const auto &source = shapes[index];
        const unsigned spatial = source.spatial * input / 224;
        const unsigned output = (spatial + source.stride - 1) / source.stride;
        network.weights.emplace_back(
            2 * std::size_t(source.cout) * source.cin * source.kernel *
                source.kernel,
            0);
        network.biases.emplace_back(2 * source.cout, 0);
        // The synthetic float64 oracle uses a centre-tap channel map.  It is
        // spatially constant under TF-SAME stride 1/2, so the full 224/192
        // topology can be checked without a prohibitively large host tensor.
        for (unsigned outputChannel = 0; outputChannel != source.cout;
             ++outputChannel) {
            const auto inputChannel = outputChannel % source.cin;
            const auto tap = source.kernel / 2;
            const auto weightIndex =
                ((std::size_t(outputChannel) * source.cin + inputChannel) *
                     source.kernel + tap) * source.kernel + tap;
            putHalfOne(network.weights.back(), weightIndex);
        }
        ConvShape shape{source.kernel, source.kernel, source.stride, 1, true,
            ElementwiseShape{source.cin, spatial, spatial},
            ElementwiseShape{source.cout, output, output}};
        network.convs.push_back({shape, network.weights.back().data(),
            network.weights.back().size(), network.biases.back().data(),
            network.biases.back().size()});
    }
    return network;
}

static std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>>
makeHead(unsigned inputs, unsigned outputs) {
    std::vector<std::uint8_t> weights(2 * std::size_t(inputs) * outputs, 0);
    std::vector<std::uint8_t> bias(2 * outputs, 0);
    for (unsigned output = 0; output != outputs; ++output)
        putHalfOne(weights, std::size_t(output) * inputs + output % inputs);
    return {std::move(weights), std::move(bias)};
}

struct BlockSpec {
    std::array<unsigned, 3> main;
    unsigned mainCount;
    int projection;
};

static const BlockSpec bottleneckBlocks[] = {
    {{1,2,3},3,4}, {{5,6,7},3,-1}, {{8,9,10},3,-1},
    {{12,13,14},3,11}, {{15,16,17},3,-1}, {{18,19,20},3,-1},
    {{21,22,23},3,-1}, {{25,26,27},3,24}, {{28,29,30},3,-1},
    {{31,32,33},3,-1}, {{34,35,36},3,-1}, {{37,38,39},3,-1},
    {{40,41,42},3,-1}, {{44,45,46},3,43}, {{47,48,49},3,-1},
    {{50,51,52},3,-1},
};

static const BlockSpec basicBlocks[] = {
    {{1,2,0},2,-1}, {{3,4,0},2,-1}, {{6,7,0},2,5},
    {{8,9,0},2,-1}, {{11,12,0},2,10}, {{13,14,0},2,-1},
    {{16,17,0},2,15}, {{18,19,0},2,-1},
};

static std::vector<double> referenceConv(const std::vector<double> &input,
                                         unsigned outputs) {
    std::vector<double> result(outputs);
    for (unsigned output = 0; output != outputs; ++output)
        result[output] = input[output % input.size()];
    return result;
}

template <std::size_t N>
static std::vector<double> runFloat64Reference(
    const OwnedNetwork &network, const BlockSpec (&blocks)[N],
    unsigned headOutputs) {
    std::vector<double> current(network.convs.front().shape.input.channels);
    for (std::size_t channel = 0; channel != current.size(); ++channel)
        current[channel] = static_cast<double>(channel + 1);
    current = referenceConv(current,
                            network.convs.front().shape.output.channels);
    for (const auto &block : blocks) {
        const auto shortcut = block.projection < 0 ? current
            : referenceConv(current, network.convs[block.projection]
                                         .shape.output.channels);
        auto main = current;
        for (unsigned position = 0; position != block.mainCount; ++position)
            main = referenceConv(
                main, network.convs[block.main[position]].shape.output.channels);
        if (main.size() != shortcut.size())
            throw std::runtime_error("float64 reference residual shape mismatch");
        for (std::size_t channel = 0; channel != main.size(); ++channel)
            main[channel] = std::max(0.0, main[channel] + shortcut[channel]);
        current = std::move(main);
    }
    auto logits = referenceConv(current, headOutputs);
    for (const auto value : logits)
        if (!std::isfinite(value) || value <= 0.0)
            throw std::runtime_error("float64 reference produced bad logits");
    return logits;
}

struct TaskSpan { std::size_t offset, bytes; };

static std::uint32_t word(const std::vector<std::uint8_t> &bytes,
                          std::size_t offset) {
    if (offset + 4 > bytes.size())
        throw std::runtime_error("task word is out of range");
    return std::uint32_t(bytes[offset]) |
           (std::uint32_t(bytes[offset + 1]) << 8) |
           (std::uint32_t(bytes[offset + 2]) << 16) |
           (std::uint32_t(bytes[offset + 3]) << 24);
}

static std::vector<TaskSpan> taskSpans(const ane::h13::Program &program) {
    std::vector<TaskSpan> spans;
    std::size_t offset = 0, bytes = program.firstTaskBytes;
    for (std::uint32_t index = 0; index != program.taskCount; ++index) {
        if (!bytes || offset + bytes > program.task.size())
            throw std::runtime_error("residual task stream is truncated");
        spans.push_back({offset, bytes});
        if (index + 1 == program.taskCount) break;
        const auto nextWords = ((word(program.task, offset + 4) >> 16) &
                                0x1ffu) + 1;
        bytes = 4 * nextWords;
        offset = word(program.task, offset + 28);
    }
    return spans;
}

static std::uint32_t reg(const ane::h13::Program &program,
                         TaskSpan span, std::uint32_t address) {
    std::size_t cursor = span.offset + 40;
    while (cursor < span.offset + span.bytes) {
        const auto header = word(program.task, cursor);
        const auto count = (header >> 26) + 1;
        const auto base = header & 0x03ffffffu;
        if (base <= address && address < base + 4 * count)
            return word(program.task, cursor + 4 + address - base);
        cursor += 4 * (count + 1);
    }
    throw std::runtime_error("residual task register is absent");
}

static std::size_t regOffset(const ane::h13::Program &program,
                             TaskSpan span, std::uint32_t address) {
    std::size_t cursor = span.offset + 40;
    while (cursor < span.offset + span.bytes) {
        const auto header = word(program.task, cursor);
        const auto count = (header >> 26) + 1;
        const auto base = header & 0x03ffffffu;
        if (base <= address && address < base + 4 * count)
            return cursor + 4 + address - base;
        cursor += 4 * (count + 1);
    }
    throw std::runtime_error("residual task register is absent");
}

static bool hasReg(const ane::h13::Program &program,
                   TaskSpan span, std::uint32_t address) {
    std::size_t cursor = span.offset + 40;
    while (cursor < span.offset + span.bytes) {
        const auto header = word(program.task, cursor);
        const auto count = (header >> 26) + 1;
        const auto base = header & 0x03ffffffu;
        if (base <= address && address < base + 4 * count) return true;
        cursor += 4 * (count + 1);
    }
    return false;
}

static void expect(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr std::uint32_t kKernelTableEnable = 1u << 24;
constexpr std::uint32_t kKernelExecutionEnable = 1u << 26;
constexpr std::uint32_t kKernelEnableMask =
    kKernelTableEnable | kKernelExecutionEnable;
constexpr std::uint32_t kTaskBufferEnable = 1u << 23;
constexpr std::uint32_t kTaskSignalEnable = 1u << 15;
constexpr std::uint32_t kTaskSignalReset = 1u << 12;

static bool isPETask(const ane::h13::Program::TaskRecord &record) {
    return record.role == "pool" || record.role == "add+relu" ||
           record.role == "reduce_mean";
}

static void validateTaskHeaders(const ane::h13::Program &program) {
    const auto spans = taskSpans(program);
    expect(spans.size() == program.taskRecords.size(),
           "residual task records do not cover every decoded task");
    for (std::size_t index = 0; index != spans.size(); ++index) {
        const auto span = spans[index];
        const bool kernel = !isPETask(program.taskRecords[index]);
        const auto baseEnable = word(program.task, span.offset + 32);
        expect((baseEnable & kKernelEnableMask) ==
                   (kernel ? kKernelEnableMask : 0u),
               "residual kernel enables disagree with the task role");
        expect((word(program.task, span.offset + 36) != 0) == kernel,
               "residual header word 9 disagrees with the task role");
        expect(bool(baseEnable & kTaskBufferEnable) ==
                   (index + 1 != spans.size()),
               "residual task-buffer chain is not closed at the terminal task");
        const auto control = word(program.task, span.offset + 24);
        expect((control & (kTaskSignalEnable | kTaskSignalReset)) ==
                   (kTaskSignalEnable | kTaskSignalReset),
               "residual task does not signal and reset completion");
    }
}

static ane::h13::Program linkedRingStemReference() {
    const ConvShape stemShape{3, 3, 2, 1, true,
        ElementwiseShape{3, 224, 224}, ElementwiseShape{32, 112, 112}};
    const ConvShape depthwiseShape{3, 3, 1, 32, true,
        stemShape.output, stemShape.output};
    const auto encode = [](const ConvShape &shape) {
        const auto reduction = static_cast<std::size_t>(shape.input.channels) /
            shape.groups * shape.kernel * shape.kernelWidth;
        std::vector<std::uint8_t> weights(
            2 * static_cast<std::size_t>(shape.output.channels) * reduction,
            0);
        std::vector<std::uint8_t> bias(2 * shape.output.channels, 0);
        return ane::h13::encodeConvLinkedRing(
            shape, weights.data(), weights.size(), bias.data(), bias.size());
    };
    return ane::h13::linkMobileNetRing(
        {encode(stemShape), encode(depthwiseShape)});
}

static ane::h13::Program linkedRingPoolGeometryReference(
    ElementwiseShape input, ElementwiseShape output) {
    const ConvShape shape{3, 3, 2, input.channels, true, input, output};
    std::vector<std::uint8_t> weights(
        2 * static_cast<std::size_t>(output.channels) * 3 * 3, 0);
    std::vector<std::uint8_t> bias(2 * output.channels, 0);
    return ane::h13::encodeConvLinkedRing(
        shape, weights.data(), weights.size(), bias.data(), bias.size());
}

struct ColumnWrite {
    std::uint32_t source;
    std::uint32_t destinationByte;
};

static std::uint32_t ceilLog2(std::uint32_t value) {
    std::uint32_t result = 0;
    for (std::uint32_t covered = 1; covered < value; covered <<= 1)
        ++result;
    return result;
}

static std::vector<ColumnWrite> modelPoolColumnPieces(
    std::uint32_t width, bool scratchWriter) {
    constexpr std::uint32_t pieceColumns = 8;
    constexpr std::uint32_t halfwordBytes = 2;
    constexpr std::uint32_t pieceBytes = pieceColumns * halfwordBytes;
    constexpr std::uint32_t rowBytes = 128;
    const auto pieces = (width + pieceColumns - 1) / pieceColumns;
    std::vector<ColumnWrite> writes;
    const auto firstPiece = scratchWriter ? 4u : 0u;
    const auto activePieces = scratchWriter ? std::min(3u, pieces) : pieces;
    for (std::uint32_t index = 0; index != activePieces; ++index) {
        const auto sourcePiece = firstPiece + index;
        if (sourcePiece >= pieces) break;
        // The misclassified scratch writer emits two adjacent pieces, then
        // advances to the next 128-byte bank row.  The external writer emits
        // all seven logical pieces linearly into the padded physical row.
        const auto destinationByte = scratchWriter && index >= 2
            ? rowBytes + (index - 2) * pieceBytes
            : (firstPiece + index) * pieceBytes;
        for (std::uint32_t within = 0;
             within != pieceColumns && sourcePiece * pieceColumns + within < width;
             ++within)
            writes.push_back({sourcePiece * pieceColumns + within,
                              destinationByte + within * halfwordBytes});
    }
    return writes;
}

static void validateMeasuredPoolColumnModel() {
    constexpr std::uint32_t logicalWidth = 56;
    constexpr std::uint32_t physicalWidth = 64;
    const auto oldWrites = modelPoolColumnPieces(logicalWidth, true);
    std::vector<int> oldDestination(2 * physicalWidth, -1);
    for (const auto write : oldWrites) {
        const auto halfword = write.destinationByte / 2;
        oldDestination[halfword] = static_cast<int>(write.source);
    }
    for (std::uint32_t column = 8; column != 32; ++column)
        expect(oldDestination[column] < 0,
               "old pool model does not leave columns 8..31 unwritten");
    for (std::uint32_t column = 32; column != 48; ++column)
        expect(oldDestination[column] == static_cast<int>(column),
               "old pool model does not make columns 32..47 correct");
    for (std::uint32_t column = 48; column != 56; ++column)
        expect(oldDestination[column] < 0,
               "old pool model does not leave columns 48..55 unwritten");
    for (std::uint32_t column = 0; column != 8; ++column)
        expect(oldDestination[physicalWidth + column] ==
                   static_cast<int>(48 + column),
               "old pool model does not wrap the last piece one physical row");

    const auto fixedWrites = modelPoolColumnPieces(logicalWidth, false);
    std::vector<int> fixedDestination(logicalWidth, -1);
    for (const auto write : fixedWrites)
        if (write.destinationByte / 2 < fixedDestination.size())
            fixedDestination[write.destinationByte / 2] =
                static_cast<int>(write.source);
    for (std::uint32_t column = 0; column != logicalWidth; ++column)
        expect(fixedDestination[column] == static_cast<int>(column),
               "fixed pool model does not cover every logical output column");
}

static void validatePoolTilingFields(const ane::h13::Program &program) {
    const auto spans = taskSpans(program);
    expect(spans.size() >= 2, "residual stream has no pool task");
    const auto pool = spans[1];
    const auto inputWord = reg(program, pool, 0x00000);
    const auto outputWord = reg(program, pool, 0x00014);
    const ElementwiseShape input{reg(program, pool, 0x0000c),
                                 inputWord >> 16, inputWord & 0xffffu};
    const ElementwiseShape output{reg(program, pool, 0x00010),
                                  outputWord >> 16, outputWord & 0xffffu};
    const auto proven = linkedRingPoolGeometryReference(input, output);
    const auto provenTask = taskSpans(proven).front();

    const auto spatialMask = 31u | (31u << 5) | (3u << 13) | (3u << 15);
    expect((reg(program, pool, 0x0001c) & spatialMask) ==
               (reg(proven, provenTask, 0x0001c) & spatialMask),
           "pool kernel/stride fields differ from proven 3x3/s2 traversal");
    for (const auto address : {0x00020u, 0x00028u})
        expect(reg(program, pool, address) == reg(proven, provenTask, address),
               "pool traversal/tile field differs from proven geometry");
    const auto groupMask = 1u | (1u << 16);
    expect((reg(program, pool, 0x00024) & groupMask) ==
               (reg(proven, provenTask, 0x00024) & groupMask),
           "pool input/output group counts differ from proven geometry");
    constexpr std::uint32_t fourLaneIssue = 4;
    constexpr std::uint32_t columnClassShift = 4;
    const auto expectedTileConfig = fourLaneIssue |
        (ceilLog2(output.width) << columnClassShift);
    expect(reg(program, pool, 0x0002c) == expectedTileConfig &&
               reg(proven, provenTask, 0x0002c) == 0,
           "pool PE column class differs from the output-width rule");
    const auto poolPipeline = 1u | (2u << 3);
    expect(reg(program, pool, 0x00034) == poolPipeline,
           "pool Common Cfg is not the executing PE pipeline form");
    expect(reg(program, pool, 0x04818) == reg(program, pool, 0x04814) &&
               reg(program, pool, 0x04818) ==
                   reg(proven, provenTask, 0x04818),
           "pool L2 source group advance differs from proven 112-to-56 geometry");
    expect(pool.bytes == 504 && hasReg(program, pool, 0x1f800) &&
               hasReg(program, pool, 0x1f8c8),
           "pool does not retain the required KernelDMA descriptor grammar");
}

static void validateResidualPETasks(const ane::h13::Program &program) {
    const auto spans = taskSpans(program);
    expect(spans.size() == program.taskRecords.size(),
           "residual task metadata count differs from task stream");
    constexpr std::uint32_t fourLaneIssue = 4;
    constexpr std::uint32_t columnClassShift = 4;
    const auto disabledCoefficientPath = 2u << 6;
    const auto disabledPalettePath = 1u << 6;
    for (std::size_t index = 0; index != spans.size(); ++index) {
        const auto &role = program.taskRecords[index].role;
        if (role != "pool" && role != "add+relu" && role != "mean")
            continue;
        const auto span = spans[index];
        const auto inputWidth = reg(program, span, 0x00000) & 0xffffu;
        const auto outputWidth = reg(program, span, 0x00014) & 0xffffu;
        const auto activeWidth = role == "mean" ? inputWidth : outputWidth;
        expect(span.bytes == 504 && hasReg(program, span, 0x1f800) &&
                   hasReg(program, span, 0x1f8c8),
               "residual PE task lost its 504-byte KernelDMA grammar");
        expect(reg(program, span, 0x0002c) ==
                   (fourLaneIssue |
                    (ceilLog2(activeWidth) << columnClassShift)),
               "residual PE task has the wrong active-column class");
        const auto stage = role == "pool" ? 2u : role == "add+relu" ? 6u : 10u;
        expect(reg(program, span, 0x00034) == (1u | (stage << 3)),
               "residual PE task lost the executing pipeline config");
        expect(reg(program, span, 0x04818) == reg(program, span, 0x04814),
               "residual PE source group advance is not its depth advance");
        expect(reg(program, span, 0x1f800) == 0 &&
                   reg(program, span, 0x1f804) == 0,
               "residual PE coefficient record leading words changed");
        for (std::uint32_t channel = 0; channel != 16; ++channel)
            expect(reg(program, span, 0x1f808 + 4 * channel) ==
                       disabledCoefficientPath,
                   "residual PE coefficient path is not disabled");
        for (std::uint32_t wordIndex = 0; wordIndex != 4; ++wordIndex) {
            expect(reg(program, span, 0x1f8c8 + 4 * wordIndex) ==
                       disabledCoefficientPath,
                   "residual PE bias path policy changed");
            expect(reg(program, span, 0x1f8d8 + 4 * wordIndex) == 0,
                   "residual PE post-scale path is not neutral");
            expect(reg(program, span, 0x1f8e8 + 4 * wordIndex) ==
                       disabledPalettePath,
                   "residual PE palette path policy changed");
        }
    }
}

static void validateFullPoolCompletionChain(
    const ane::h13::Program &program) {
    const auto spans = taskSpans(program);
    expect(spans.size() > 2 && program.taskRecords[1].role == "pool",
           "task index 1 is not the first nonterminal pool");
    const auto pool = spans[1];
    const auto successor = spans[2];
    expect(word(program.task, pool.offset + 28) == successor.offset,
           "pool task-buffer link does not target task index 2");
    expect((((word(program.task, pool.offset + 4) >> 16) & 0x1ffu) + 1) * 4 ==
               successor.bytes,
           "pool task-buffer link does not describe task index 2");
    expect(word(program.task, pool.offset + 32) & kTaskBufferEnable,
           "nonterminal pool does not enable the task-buffer chain");
    expect((word(program.task, pool.offset + 24) &
               (kTaskSignalEnable | kTaskSignalReset)) ==
               (kTaskSignalEnable | kTaskSignalReset),
           "pool does not emit the completion signal consumed by q4");
}

static void validateStemPoolHandoff(const ane::h13::Program &program) {
    const auto spans = taskSpans(program);
    expect(spans.size() >= 2, "residual stream has no stem/pool pair");
    const auto stem = spans[0];
    const auto pool = spans[1];
    expect(reg(program, stem, 0x04834) == reg(program, pool, 0x04808) &&
               reg(program, stem, 0x04838) == reg(program, pool, 0x0480c) &&
               reg(program, stem, 0x0483c) == reg(program, pool, 0x04810) &&
               reg(program, stem, 0x04840) == reg(program, pool, 0x04814) &&
               reg(program, stem, 0x04844) == reg(program, pool, 0x04818),
           "stem result and pool source L2 layouts differ");

    const auto inputWidth = reg(program, stem, 0x00000) & 0xffffu;
    const auto inputChannels = reg(program, stem, 0x0000c);
    const auto inputChannelStride = (2 * inputWidth + 15) & ~15u;
    const auto inputSpan = inputChannelStride * inputChannels;
    expect(reg(program, stem, 0x0480c) == inputChannelStride &&
               reg(program, stem, 0x04810) == inputSpan &&
               reg(program, stem, 0x04814) == inputSpan &&
               reg(program, stem, 0x04818) == inputSpan,
           "stem external-source L2 extents differ from NCHW staging");

    const auto stemWidth = reg(program, stem, 0x00014) & 0xffffu;
    const auto stemChannels = reg(program, stem, 0x00010);
    const auto stemChannelStride = (2 * stemWidth + 15) & ~15u;
    const auto stemPlane = stemChannelStride * stemChannels;
    const auto stemBankSkew = stemWidth >= 64 ? 3u * 16u : 2u * 16u;
    expect(reg(program, stem, 0x04838) == stemChannelStride &&
               reg(program, stem, 0x0483c) == stemPlane + stemBankSkew &&
               reg(program, stem, 0x04840) == stemPlane,
           "stem resident result does not use the stride-2 bank-skew rule");

    const auto linked = linkedRingStemReference();
    const auto linkedStem = taskSpans(linked).front();
    expect(word(program.task, stem.offset + 32) ==
               word(linked.task, linkedStem.offset + 32) &&
               word(program.task, stem.offset + 36) ==
                   word(linked.task, linkedStem.offset + 36),
           "residual stem kernel header fields differ from linked-ring stem");
    expect(reg(program, stem, 0x04830) ==
               reg(linked, linkedStem, 0x04830),
           "residual stem ResultCfg differs from linked-ring initial alias");
    if (inputWidth == (reg(linked, linkedStem, 0x00000) & 0xffffu)) {
        for (const auto address : {0x04804u, 0x0480cu, 0x04810u,
                                   0x04814u, 0x04818u})
            expect(reg(program, stem, address) ==
                       reg(linked, linkedStem, address),
                   "residual stem source layout differs from proven ring stem");
    }
    expect(reg(program, stem, 0x0c800) ==
               reg(linked, linkedStem, 0x0c800) &&
               reg(program, stem, 0x1f800) ==
                   reg(linked, linkedStem, 0x1f800),
           "residual stem sparse kernel grammar differs from ring stem");
    const auto conv = reg(program, stem, 0x0001c);
    const auto linkedConv = reg(linked, linkedStem, 0x0001c);
    expect((conv & (3u << 13 | 3u << 15 | 15u << 28)) ==
               (linkedConv & (3u << 13 | 3u << 15 | 15u << 28)) &&
               (conv & 31u) == 7u && ((conv >> 5) & 31u) == 7u &&
               ((conv >> 17) & 31u) == 2u &&
               ((conv >> 22) & 31u) == 2u,
           "residual 7x7 stem stride or TF-SAME leading padding is wrong");
}

static void validateKernelDMASpans(const ane::h13::Program &program) {
    const auto stem = taskSpans(program).front();
    std::uint64_t end = 0;
    for (std::uint32_t lane = 0; lane != 16; ++lane) {
        const auto offset = reg(program, stem, 0x1f848 + 4 * lane);
        const auto bytes = reg(program, stem, 0x1f888 + 4 * lane);
        expect(offset == end && bytes && bytes % 64 == 0,
               "stem kernel-DMA spans are not contiguous 64-byte regions");
        end = static_cast<std::uint64_t>(offset) + bytes;
    }
    expect(end <= program.constants.size(),
           "stem kernel-DMA spans exceed the rebuilt constants");
}

static void validatePoolResidentLayout(const ane::h13::Program &program) {
    const auto spans = taskSpans(program);
    expect(spans.size() >= 3, "full stream has no pool consumer");
    const auto pool = spans[1];
    const auto consumer = spans[2];
    const auto width = reg(program, pool, 0x00014) & 0xffffu;
    const auto channels = reg(program, pool, 0x00010);
    const auto channel = (2 * width + 15) & ~15u;
    const auto plane = channel * channels;
    const auto skew = width > 16 ? 5u * 16u
        : channels >= 32 ? 3u * 16u : 2u * 16u;
    expect(reg(program, pool, 0x04838) == channel &&
               reg(program, pool, 0x0483c) == plane + skew &&
               reg(program, pool, 0x04840) == plane &&
               reg(program, pool, 0x04834) ==
                   reg(program, consumer, 0x04808) &&
               reg(program, pool, 0x0483c) ==
                   reg(program, consumer, 0x04810),
           "nonterminal pool result disagrees with its convolution consumer");
}

static void validatePoolViaDRAM(const ane::h13::Program &program) {
    const auto spans = taskSpans(program);
    expect(spans.size() == 2 && program.scratchAllocationBytes,
           "pool-via-DRAM rung does not contain one real spill");
    const auto scratchResult = 2u | (1u << 3) | (4u << 4) | (8u << 8);
    const auto scratchSource = 2u | (4u << 4) | (8u << 8);
    expect(reg(program, spans[0], 0x04830) == scratchResult &&
               reg(program, spans[1], 0x04804) == scratchSource,
           "pool-via-DRAM rung does not route stem through scratch");
    expect(((word(program.task, spans[0].offset + 32) >> 12) & 31u) == 3u &&
               (word(program.task, spans[1].offset + 32) & 31u) == 3u,
           "pool-via-DRAM header selectors do not bind the scratch channel");
}

static void validatePoolBoundaryLayout(const ane::h13::Program &program) {
    const auto terminal = taskSpans(program).back();
    const auto depth = program.output.nchw[4] * program.output.nchw[1];
    const auto externalNCHW = 1u | (3u << 4) | (2u << 12) |
                              (3u << 20) | (1u << 24);
    expect((reg(program, terminal, 0x17800) & (1u << 26)) == 0 &&
               program.output.nchw[5] == reg(program, terminal, 0x17808) &&
               program.output.nchw[4] == reg(program, terminal, 0x1780c) &&
               depth == reg(program, terminal, 0x17810) &&
               depth == reg(program, terminal, 0x17814) &&
               externalNCHW == reg(program, terminal, 0x17818) &&
               program.output.allocationBytes >= depth,
           "pool terminal DMA row/channel/depth/format differs from manifest NCHW");
}

static void expectOnlyRegisterChanges(
    const ane::h13::Program &control, const ane::h13::Program &variant,
    const std::vector<std::uint32_t> &addresses, const std::string &name) {
    const auto controlSpan = taskSpans(control).at(1);
    const auto variantSpan = taskSpans(variant).at(1);
    expect(controlSpan.bytes == variantSpan.bytes,
           "pool sweep changed the descriptor grammar");
    std::set<std::size_t> allowed;
    for (const auto address : addresses) {
        const auto offset = regOffset(control, controlSpan, address) -
                            controlSpan.offset;
        for (std::size_t byte = 0; byte != 4; ++byte)
            allowed.insert(offset + byte);
    }
    bool changed = false;
    for (std::size_t byte = 0; byte != controlSpan.bytes; ++byte) {
        if (control.task[controlSpan.offset + byte] ==
            variant.task[variantSpan.offset + byte])
            continue;
        changed = true;
        expect(allowed.count(byte),
               "pool sweep changed a field outside its named group");
    }
    if (!changed)
        throw std::runtime_error("pool sweep " + name +
                                 " is byte-identical to its control");
    expect(variantSpan.bytes == 504 &&
               hasReg(variant, variantSpan, 0x1f800) &&
               hasReg(variant, variantSpan, 0x1f8c8) &&
               reg(variant, variantSpan, 0x00034) == (1u | (2u << 3)),
           "pool sweep lost the round-10 PE grammar or pipeline guard");
}

static void validatePoolSweeps(
    const OwnedNetwork &network,
    const std::pair<std::vector<std::uint8_t>,
                    std::vector<std::uint8_t>> &head,
    const ane::h13::Program &fixed) {
    const auto encode = [&](const std::string &boundary) {
        return ane::h13::encodeResidualNetwork(
            network.convs, head.first.data(), head.first.size(),
            head.second.data(), head.second.size(), boundary);
    };
    const auto control = encode("pool-sweep-control");
    const auto controlPool = taskSpans(control).at(1);
    expect(reg(control, controlPool, 0x17800) & (1u << 26),
           "pool sweep control does not reproduce the round-10 writer path");
    expectOnlyRegisterChanges(control, fixed, {0x17800}, "main fix");

    struct Sweep {
        const char *name;
        std::vector<std::uint32_t> fields;
    };
    const std::vector<Sweep> sweeps = {
        {"destination-path", {0x17800}},
        {"destination-dma", {0x17800, 0x17804, 0x17808, 0x1780c,
                             0x17810, 0x17814, 0x17818}},
        {"source-dma", {0x13800, 0x13808, 0x1380c, 0x13810,
                        0x13814, 0x13818, 0x13838, 0x13840}},
        {"l2-source", {0x04804, 0x04808, 0x0480c, 0x04810,
                       0x04814, 0x04818}},
        {"common-traversal", {0x00020}},
        {"group-config", {0x00024}},
        {"column-tile", {0x0002c}},
        {"output-size", {0x00010, 0x00014, 0x00028}},
        {"halo-padding", {0x0001c}},
        {"pe-config", {0x08800}},
        {"width-28", {0x00014}},
        {"width-16", {0x00014}},
    };
    for (const auto &sweep : sweeps) {
        const auto variant = encode("pool-sweep-" + std::string(sweep.name));
        expectOnlyRegisterChanges(control, variant, sweep.fields, sweep.name);
    }
}

static std::uint32_t floatWord(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

template <std::size_t N>
static void validateGeometry(const ane::h13::Program &program,
                             const OwnedNetwork &network,
                             const BlockSpec (&blocks)[N],
                             unsigned headOutputs) {
    const auto spans = taskSpans(program);
    std::size_t cursor = 0;
    const auto consumeConv = [&](const ConvShape &shape) {
        unsigned channels = 0;
        do {
            expect(cursor < spans.size(), "missing convolution task");
            const auto span = spans[cursor++];
            const auto input = reg(program, span, 0x00000);
            const auto output = reg(program, span, 0x00014);
            expect((input >> 16) == shape.input.height &&
                       (input & 0xffffu) == shape.input.width &&
                       reg(program, span, 0x0000c) == shape.input.channels,
                   "compiled convolution input geometry differs from reference");
            expect((output >> 16) == shape.output.height &&
                       (output & 0xffffu) == shape.output.width,
                   "compiled convolution output geometry differs from reference");
            channels += reg(program, span, 0x00010);
            expect(channels <= shape.output.channels,
                   "compiled convolution slices exceed reference channels");
        } while (channels != shape.output.channels);
    };
    const auto consumePE = [&](std::uint32_t pe,
                               ElementwiseShape input,
                               ElementwiseShape output) {
        expect(cursor < spans.size(), "missing PE task");
        const auto span = spans[cursor++];
        const auto in = reg(program, span, 0x00000);
        const auto out = reg(program, span, 0x00014);
        expect(reg(program, span, 0x08800) == pe,
               "compiled PE opcode differs from reference");
        expect((in >> 16) == input.height && (in & 0xffffu) == input.width &&
                   reg(program, span, 0x0000c) == input.channels,
               "compiled PE input geometry differs from reference");
        expect((out >> 16) == output.height &&
                   (out & 0xffffu) == output.width &&
                   reg(program, span, 0x00010) == output.channels,
               "compiled PE output geometry differs from reference");
    };

    consumeConv(network.convs.front().shape);
    const auto stem = network.convs.front().shape.output;
    ElementwiseShape current{stem.channels, (stem.height + 1) / 2,
                             (stem.width + 1) / 2};
    consumePE(2u, stem, current);
    for (const auto &block : blocks) {
        const auto blockInput = current;
        if (block.projection >= 0)
            consumeConv(network.convs[block.projection].shape);
        for (unsigned position = 0; position != block.mainCount; ++position)
            consumeConv(network.convs[block.main[position]].shape);
        current = network.convs[block.main[block.mainCount - 1]].shape.output;
        consumePE((2u << 18) | (1u << 5), current, current);
        expect(current.channels ==
                   (block.projection < 0 ? blockInput.channels
                    : network.convs[block.projection].shape.output.channels),
               "reference shortcut channels differ from main branch");
    }

    expect(cursor < spans.size(), "missing mean task");
    const auto mean = spans[cursor++];
    const auto dimensions = reg(program, mean, 0x00000);
    expect((dimensions >> 16) == current.height &&
               (dimensions & 0xffffu) == current.width,
           "compiled final spatial size differs from float64 reference");
    const auto elements = static_cast<std::uint64_t>(current.height) *
                          current.width;
    expect(reg(program, mean, 0x0880c) ==
               floatWord(1.0f / static_cast<float>(elements)),
           "compiled mean scale is not 1/(H*W)");
    expect(reg(program, mean, 0x0c800) == ((1u << 7) | (1u << 14)),
           "compiled mean kernel mode differs from reference");

    ConvShape linear{1, 1, 1, 1, true,
                     ElementwiseShape{current.channels, 1, 1},
                     ElementwiseShape{headOutputs, 1, 1}};
    consumeConv(linear);
    expect(cursor == spans.size(), "compiled stream has unverified tasks");
}

struct EncodedCase {
    ane::h13::Program program;
    std::vector<double> logits;
};

template <std::size_t S, std::size_t B>
static EncodedCase encodeAndCheck(const Shape (&shapes)[S],
                                  const BlockSpec (&blocks)[B],
                                  unsigned input) {
    auto network = makeNetwork(shapes, input);
    const auto finalChannels = network.convs.back().shape.output.channels;
    auto head = makeHead(finalChannels, 1000);
    auto logits = runFloat64Reference(network, blocks, 1000);
    auto program = ane::h13::encodeResidualNetwork(
        network.convs, head.first.data(), head.first.size(),
        head.second.data(), head.second.size());
    validateGeometry(program, network, blocks, 1000);
    validateTaskHeaders(program);
    validateStemPoolHandoff(program);
    validateKernelDMASpans(program);
    validatePoolTilingFields(program);
    validateResidualPETasks(program);
    return {std::move(program), std::move(logits)};
}

static bool writeANEC(const std::string &path,
                      const ane::h13::Program &program) {
    const auto bytes = ane::h13::encodeANEC(program);
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(stream);
}

int main(int argc, char **argv) {
    validateMeasuredPoolColumnModel();
    auto fullCase = encodeAndCheck(bottleneckShapes, bottleneckBlocks, 224);
    auto &full = fullCase.program;
    if (full.taskCount != 80 || full.task.empty() || full.constants.empty() ||
        !full.formulaBuilt || full.inputs.size() != 1 ||
        full.inputs[0].nchw[2] != 224 || full.inputs[0].nchw[3] != 224 ||
        full.output.nchw[3] != 1000 || !full.scratchAllocationBytes) {
        std::cerr << "full tasks=" << full.taskCount
                  << " scratch=" << full.scratchAllocationBytes
                  << " input=" << full.inputs[0].nchw[2]
                  << " output=" << full.output.nchw[3] << "\n";
        return 1;
    }

    auto boundaryNetwork = makeNetwork(bottleneckShapes, 224);
    auto boundaryHead = makeHead(
        boundaryNetwork.convs.back().shape.output.channels, 1000);
    auto pool = ane::h13::encodeResidualNetwork(
        boundaryNetwork.convs, boundaryHead.first.data(),
        boundaryHead.first.size(), boundaryHead.second.data(),
        boundaryHead.second.size(), "pool");
    auto block = ane::h13::encodeResidualNetwork(
        boundaryNetwork.convs, boundaryHead.first.data(),
        boundaryHead.first.size(), boundaryHead.second.data(),
        boundaryHead.second.size(), "block01");
    auto mean = ane::h13::encodeResidualNetwork(
        boundaryNetwork.convs, boundaryHead.first.data(),
        boundaryHead.first.size(), boundaryHead.second.data(),
        boundaryHead.second.size(), "mean");
    auto stem = ane::h13::encodeResidualNetwork(
        boundaryNetwork.convs, boundaryHead.first.data(),
        boundaryHead.first.size(), boundaryHead.second.data(),
        boundaryHead.second.size(), "stem");
    auto poolViaDRAM = ane::h13::encodeResidualNetwork(
        boundaryNetwork.convs, boundaryHead.first.data(),
        boundaryHead.first.size(), boundaryHead.second.data(),
        boundaryHead.second.size(), "pool-via-dram");
    expect(stem.taskCount == 1 && stem.outputBoundary == "stem" &&
               stem.output.nchw[1] == 64 && stem.output.nchw[2] == 112 &&
               stem.output.nchw[3] == 112,
           "stem boundary metadata differs from the compiler plan");
    expect(pool.taskCount == 2 && pool.outputBoundary == "pool" &&
               pool.output.nchw[1] == 64 && pool.output.nchw[2] == 56 &&
               pool.output.nchw[3] == 56,
           "pool boundary metadata differs from the compiler plan");
    expect(full.taskRecords.size() > 2 &&
               full.taskRecords[2].role == "projection" &&
               full.taskRecords[2].boundary == "projection01",
           "first post-pool task is not the isolated projection boundary");
    expect(block.taskCount == 7 && block.outputBoundary == "block01" &&
               block.output.nchw[1] == 256,
           "first block boundary metadata differs from the compiler plan");
    expect(mean.taskCount + 1 == full.taskCount &&
               mean.outputBoundary == "mean" &&
               mean.output.nchw[1] == 2048 && mean.output.nchw[2] == 1 &&
               mean.output.nchw[3] == 1,
           "mean boundary metadata differs from the compiler plan");
    validateTaskHeaders(pool);
    validateTaskHeaders(stem);
    validateTaskHeaders(poolViaDRAM);
    validateTaskHeaders(block);
    validateResidualPETasks(pool);
    validateResidualPETasks(poolViaDRAM);
    validateResidualPETasks(block);
    validateStemPoolHandoff(pool);
    validateStemPoolHandoff(block);
    validatePoolBoundaryLayout(pool);
    validateKernelDMASpans(stem);
    validatePoolViaDRAM(poolViaDRAM);
    validatePoolSweeps(boundaryNetwork, boundaryHead, poolViaDRAM);
    validatePoolResidentLayout(full);
    validateFullPoolCompletionChain(full);
    const auto fullSpans = taskSpans(full);
    for (const auto *boundary : {&pool, &block, &mean}) {
        const auto spans = taskSpans(*boundary);
        const auto terminal = spans.back();
        const auto fullTerminal = fullSpans.back();
        expect(reg(*boundary, terminal, 0x04830) ==
                   reg(full, fullTerminal, 0x04830) &&
               reg(*boundary, terminal, 0x17800) ==
                   reg(full, fullTerminal, 0x17800) &&
               reg(*boundary, terminal, 0x17818) ==
                   reg(full, fullTerminal, 0x17818) &&
               word(boundary->task, terminal.offset + 8) ==
                   word(full.task, fullTerminal.offset + 8),
               "boundary terminal fields do not use the full-network constructors");
    }

    auto alternateCase = encodeAndCheck(bottleneckShapes, bottleneckBlocks, 192);
    auto &alternate = alternateCase.program;
    if (alternate.taskCount != 80 || alternate.inputs[0].nchw[2] != 192 ||
        alternate.inputs[0].nchw[3] != 192 || !alternate.formulaBuilt)
        return 2;
    if (fullCase.logits != alternateCase.logits ||
        fullCase.logits.front() != 65536.0)
        return 5;

    auto basicCase = encodeAndCheck(basicShapes, basicBlocks, 224);
    auto &basic = basicCase.program;
    if (basic.taskCount != 34 || basic.task.empty() || basic.constants.empty() ||
        !basic.formulaBuilt || basic.output.nchw[3] != 1000 ||
        basicCase.logits.front() != 256.0)
        return 3;

    if (argc == 2 &&
        (!writeANEC(std::string(argv[1]) + "-224.anec", full) ||
         !writeANEC(std::string(argv[1]) + "-192.anec", alternate) ||
         !writeANEC(std::string(argv[1]) + "-basic.anec", basic)))
        return 4;
    std::cout << "H13 rule-built ResNet-50 224/192 and ResNet-18 "
                 "float64 geometry PASS\n";
}
