#include "../plugins/H13/H13Program.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

static void expect(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
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
