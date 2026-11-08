#include "../plugins/H16G/Encoding/H16ConvProgram.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

using ane::h13::ElementwiseShape;
using ane::h16::ConvShape;

static std::vector<std::uint8_t> halves(std::size_t count) {
    std::vector<std::uint8_t> out(2*count);
    for (std::size_t i=0;i<count;++i) {
        // Finite, nonzero fp16 values; packing size must not depend on a zero.
        const std::uint16_t value=std::uint16_t(0x3000+(i%0xbff));
        out[2*i]=std::uint8_t(value); out[2*i+1]=std::uint8_t(value>>8);
    }
    return out;
}

static ConvShape shape(unsigned cin,unsigned cout,unsigned input,unsigned kernel) {
    ConvShape s;
    s.input=ElementwiseShape{cin,input,input};
    s.output=ElementwiseShape{cout,(input+1)/2,(input+1)/2};
    s.kernel=s.kernelWidth=kernel; s.stride=2; s.groups=1; s.bias=true;
    return s;
}

static bool equalChunks(const std::array<std::uint32_t,16> &chunks,
                        std::uint32_t expected) {
    for (auto value:chunks) if (value!=expected) return false;
    return true;
}

int main() {
    std::array<std::uint32_t,16> chunks{};
    auto stem=shape(3,64,224,7);
    auto stemWeights=halves(64*3*49),stemBias=halves(64);
    auto stemPacked=ane::h16::packResNetStem7x7(stem,stemWeights.data(),
                                                stemBias.data(),chunks);
    if (stemPacked.size()!=21504 || !equalChunks(chunks,1344)) return 1;

    auto regular=shape(128,128,56,3);
    auto regularWeights=halves(128*128*9),regularBias=halves(128);
    auto regularPacked=ane::h16::packResNetStride2(
        regular,regularWeights.data(),regularBias.data(),chunks);
    if (regularPacked.size()!=328704 || !equalChunks(chunks,20544)) return 2;

    auto projection=shape(512,672,28,1);
    auto projectionWeights=halves(512*672),projectionBias=halves(672);
    auto projectionPacked=ane::h16::packResNetStride2(
        projection,projectionWeights.data(),projectionBias.data(),chunks);
    if (projectionPacked.size()!=862208 || !equalChunks(chunks,53888)) return 3;

    auto wide3=shape(512,512,14,3);
    auto wide3Weights=halves(512*512*9),wide3Bias=halves(512);
    auto wide3Packed=ane::h16::packResNetStride2(
        wide3,wide3Weights.data(),wide3Bias.data(),chunks);
    if (wide3Packed.size()!=5244928 || !equalChunks(chunks,0x50080)) return 4;

    auto wide1=shape(1024,2048,14,1);
    auto wide1Weights=halves(1024*2048),wide1Bias=halves(2048);
    auto wide1Packed=ane::h16::packResNetStride2(
        wide1,wide1Weights.data(),wide1Bias.data(),chunks);
    if (wide1Packed.size()!=5248000 || !equalChunks(chunks,0x50140)) return 5;

    auto headWeights=halves(2048*1000),headBias=halves(1000);
    auto headPacked=ane::h16::packResNetLinearHead(
        2048,1000,headWeights.data(),headBias.data(),chunks);
    if (headPacked.size()!=4098048) return 6;
    for (unsigned i=0;i<16;++i)
        if (chunks[i]!=(i<8 ? 0x3f080u : 0x3e080u)) return 7;
    std::cout << "resnet50 kernel packing passed\n";
}
