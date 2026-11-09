#include "../plugins/H13/H13Program.h"

#include <cstdint>
#include <fstream>
#include <iostream>
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

static std::vector<std::uint8_t> halves(std::size_t count, unsigned seed) {
    std::vector<std::uint8_t> out(2 * count);
    for (std::size_t i = 0; i != count; ++i) {
        const auto value = static_cast<std::uint16_t>(
            0x2001 + (i + seed) % 0x1ffe);
        out[2 * i] = static_cast<std::uint8_t>(value);
        out[2 * i + 1] = static_cast<std::uint8_t>(value >> 8);
    }
    return out;
}

struct OwnedNetwork {
    std::vector<std::vector<std::uint8_t>> weights;
    std::vector<std::vector<std::uint8_t>> biases;
    std::vector<ane::h13::ResNet50ConvWeights> convs;
};

template <std::size_t N>
static OwnedNetwork makeNetwork(const Shape (&shapes)[N], unsigned input,
                                unsigned seed) {
    OwnedNetwork network;
    network.weights.reserve(N);
    network.biases.reserve(N);
    network.convs.reserve(N);
    for (unsigned index = 0; index != N; ++index) {
        const auto &source = shapes[index];
        const unsigned spatial = source.spatial * input / 224;
        const unsigned output = (spatial + source.stride - 1) / source.stride;
        network.weights.push_back(halves(
            std::size_t(source.cout) * source.cin * source.kernel *
                source.kernel,
            seed + index));
        network.biases.push_back(halves(source.cout, seed + 100 + index));
        ConvShape shape{source.kernel, source.kernel, source.stride, 1, true,
            ElementwiseShape{source.cin, spatial, spatial},
            ElementwiseShape{source.cout, output, output}};
        network.convs.push_back({shape, network.weights.back().data(),
            network.weights.back().size(), network.biases.back().data(),
            network.biases.back().size()});
    }
    return network;
}

static ane::h13::Program encodeBottleneck(unsigned input) {
    auto network = makeNetwork(bottleneckShapes, input, 1);
    auto headWeights = halves(2048 * 1000, 501);
    auto headBias = halves(1000, 502);
    return ane::h13::encodeResidualNetwork(
        network.convs, headWeights.data(), headWeights.size(),
        headBias.data(), headBias.size());
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
    auto full = encodeBottleneck(224);
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

    auto alternate = encodeBottleneck(192);
    if (alternate.taskCount != 80 || alternate.inputs[0].nchw[2] != 192 ||
        alternate.inputs[0].nchw[3] != 192 || !alternate.formulaBuilt)
        return 2;

    auto basicNetwork = makeNetwork(basicShapes, 224, 700);
    auto basicHeadWeights = halves(512 * 1000, 901);
    auto basicHeadBias = halves(1000, 902);
    auto basic = ane::h13::encodeResidualNetwork(
        basicNetwork.convs, basicHeadWeights.data(), basicHeadWeights.size(),
        basicHeadBias.data(), basicHeadBias.size());
    if (basic.taskCount != 34 || basic.task.empty() || basic.constants.empty() ||
        !basic.formulaBuilt || basic.output.nchw[3] != 1000)
        return 3;

    if (argc == 2 &&
        (!writeANEC(std::string(argv[1]) + "-224.anec", full) ||
         !writeANEC(std::string(argv[1]) + "-192.anec", alternate) ||
         !writeANEC(std::string(argv[1]) + "-basic.anec", basic)))
        return 4;
    std::cout << "H13 rule-built ResNet-50 224/192 and ResNet-18 PASS\n";
}
