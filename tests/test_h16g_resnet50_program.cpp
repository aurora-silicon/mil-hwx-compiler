#include "../plugins/H16G/Encoding/H16ConvProgram.h"

#include <cstdint>
#include <iostream>
#include <vector>

using ane::h13::ElementwiseShape;
using ane::h16::ConvShape;

struct Shape { unsigned cin,cout,spatial,kernel,stride; };

static const Shape shapes[] = {
    {3,64,224,7,2},{64,64,56,1,1},{64,64,56,3,1},{64,256,56,1,1},
    {64,256,56,1,1},{256,64,56,1,1},{64,64,56,3,1},{64,256,56,1,1},
    {256,64,56,1,1},{64,64,56,3,1},{64,256,56,1,1},{256,512,56,1,2},
    {256,128,56,1,1},{128,128,56,3,2},{128,512,28,1,1},{512,128,28,1,1},
    {128,128,28,3,1},{128,512,28,1,1},{512,128,28,1,1},{128,128,28,3,1},
    {128,512,28,1,1},{512,128,28,1,1},{128,128,28,3,1},{128,512,28,1,1},
    {512,256,28,1,1},{256,256,28,3,2},{512,1024,28,1,2},{256,1024,14,1,1},
    {1024,256,14,1,1},{256,256,14,3,1},{256,1024,14,1,1},{1024,256,14,1,1},
    {256,256,14,3,1},{256,1024,14,1,1},{1024,256,14,1,1},{256,256,14,3,1},
    {256,1024,14,1,1},{1024,256,14,1,1},{256,256,14,3,1},{256,1024,14,1,1},
    {1024,256,14,1,1},{256,256,14,3,1},{256,1024,14,1,1},{1024,2048,14,1,2},
    {1024,512,14,1,1},{512,512,14,3,2},{512,2048,7,1,1},{2048,512,7,1,1},
    {512,512,7,3,1},{512,2048,7,1,1},{2048,512,7,1,1},{512,512,7,3,1},
    {512,2048,7,1,1},
};

static std::vector<std::uint8_t> nonzeroHalves(std::size_t count,unsigned seed) {
    std::vector<std::uint8_t> out(2*count);
    for (std::size_t i=0;i<count;++i) {
        const std::uint16_t value=std::uint16_t(0x2001+(i+seed)%0x1ffe);
        out[2*i]=std::uint8_t(value); out[2*i+1]=std::uint8_t(value>>8);
    }
    return out;
}

int main() {
    std::vector<std::vector<std::uint8_t>> weights,biases;
    weights.reserve(53); biases.reserve(53);
    for (unsigned i=0;i<53;++i) {
        const auto &s=shapes[i];
        weights.push_back(nonzeroHalves(
            std::size_t(s.cout)*s.cin*s.kernel*s.kernel,i));
        biases.push_back(nonzeroHalves(s.cout,100+i));
    }
    std::vector<ane::h16::ResNet50ConvWeights> convs;
    for (unsigned i=0;i<53;++i) {
        const auto &s=shapes[i];
        const unsigned output=(s.spatial+s.stride-1)/s.stride;
        ConvShape shape;
        shape.input=ElementwiseShape{s.cin,s.spatial,s.spatial};
        shape.output=ElementwiseShape{s.cout,output,output};
        shape.kernel=shape.kernelWidth=s.kernel; shape.stride=s.stride;
        shape.groups=1; shape.bias=true;
        convs.push_back({shape,weights[i].data(),weights[i].size(),
                         biases[i].data(),biases[i].size()});
    }
    auto headWeights=nonzeroHalves(2048*1000,501);
    auto headBias=nonzeroHalves(1000,502);
    auto encoded=ane::h16::encodeResNet50(
        convs,headWeights.data(),headWeights.size(),
        headBias.data(),headBias.size());
    if (encoded.program.taskCount!=80 || encoded.program.task.empty() ||
        encoded.program.constants.size()!=53158912 ||
        encoded.program.kernelRelocations.size()!=62 ||
        encoded.program.scratchAllocationBytes!=4u*1024u*1024u ||
        encoded.programRecordCount!=161 || encoded.programFormatCode!=0x4c ||
        encoded.program.task[0]!=1 || !encoded.program.formulaBuilt) {
        std::cerr << "resnet50 tasks=" << encoded.program.taskCount
                  << " constants=" << encoded.program.constants.size()
                  << " relocations=" << encoded.program.kernelRelocations.size()
                  << " records=" << encoded.programRecordCount << "\n";
        return 1;
    }

    // A 192-pixel input uses the identical weights and rule constructors;
    // only tensor geometry changes (224/56/28/14/7 -> 192/48/24/12/6).
    std::vector<ane::h16::ResNet50ConvWeights> convs192=convs;
    for (auto &conv:convs192) {
        conv.shape.input.height=conv.shape.input.height*6/7;
        conv.shape.input.width=conv.shape.input.width*6/7;
        conv.shape.output.height=
            (conv.shape.input.height+conv.shape.stride-1)/conv.shape.stride;
        conv.shape.output.width=
            (conv.shape.input.width+conv.shape.stride-1)/conv.shape.stride;
    }
    auto encoded192=ane::h16::encodeResidualNetwork(
        convs192,headWeights.data(),headWeights.size(),
        headBias.data(),headBias.size());
    if (encoded192.program.taskCount!=80 ||
        encoded192.program.kernelRelocations.size()!=62 ||
        encoded192.program.inputs.front().nchw[2]!=192 ||
        !encoded192.program.formulaBuilt) return 3;

    // A basic-block ResNet-18 geometry takes the same topology discovery,
    // allocator, DMA, and descriptor constructors.  It deliberately has no
    // network-specific descriptor data.
    const Shape basicShapes[] = {
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
    std::vector<std::vector<std::uint8_t>> basicWeights,basicBiases;
    std::vector<ane::h16::ResNet50ConvWeights> basicConvs;
    for (unsigned i=0;i<sizeof(basicShapes)/sizeof(basicShapes[0]);++i) {
        const auto &s=basicShapes[i];
        basicWeights.push_back(nonzeroHalves(
            std::size_t(s.cout)*s.cin*s.kernel*s.kernel,i+700));
        basicBiases.push_back(nonzeroHalves(s.cout,i+800));
        ConvShape shape;
        shape.input=ElementwiseShape{s.cin,s.spatial,s.spatial};
        const unsigned output=(s.spatial+s.stride-1)/s.stride;
        shape.output=ElementwiseShape{s.cout,output,output};
        shape.kernel=shape.kernelWidth=s.kernel;shape.stride=s.stride;
        shape.groups=1;shape.bias=true;
        basicConvs.push_back({shape,basicWeights.back().data(),
            basicWeights.back().size(),basicBiases.back().data(),
            basicBiases.back().size()});
    }
    auto basicHeadWeights=nonzeroHalves(512*1000,901);
    auto basicHeadBias=nonzeroHalves(1000,902);
    auto basic=ane::h16::encodeResidualNetwork(
        basicConvs,basicHeadWeights.data(),basicHeadWeights.size(),
        basicHeadBias.data(),basicHeadBias.size());
    if (basic.program.taskCount!=34 || basic.program.task.empty() ||
        basic.program.kernelRelocations.size()!=24 ||
        !basic.program.formulaBuilt) {
        std::cerr << "basic tasks=" << basic.program.taskCount
                  << " relocations=" << basic.program.kernelRelocations.size()
                  << "\n";
        return 2;
    }
    std::cout << "rule-built ResNet-50@224/192 and ResNet-18 programs passed\n";
}
