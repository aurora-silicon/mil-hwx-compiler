#include "../plugins/H16G/Encoding/H16ConvProgram.h"

#include <cstdint>
#include <iostream>
#include <map>
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

static std::uint32_t read32(const std::vector<std::uint8_t> &bytes,
                            std::size_t at) {
    return std::uint32_t(bytes.at(at)) |
           std::uint32_t(bytes.at(at+1))<<8 |
           std::uint32_t(bytes.at(at+2))<<16 |
           std::uint32_t(bytes.at(at+3))<<24;
}

static std::size_t taskOffset(const std::vector<std::uint8_t> &stream,
                              unsigned wanted) {
    std::size_t task=16;
    for (unsigned index=0;index<wanted;++index) {
        task+=4*(read32(stream,task)>>16);
        task=(task+15)&~std::size_t(15);
    }
    return task;
}

static std::map<std::uint32_t,std::uint32_t> taskRegisters(
    const std::vector<std::uint8_t> &stream,unsigned index) {
    const std::size_t task=taskOffset(stream,index);
    const std::size_t words=read32(stream,task)>>16;
    std::size_t cursor=task+9*4;
    const std::size_t end=task+words*4;
    std::map<std::uint32_t,std::uint32_t> out;
    while (cursor<end) {
        const std::uint32_t packed=read32(stream,cursor);cursor+=4;
        const std::uint32_t base=packed&0x7fff;
        std::vector<std::uint32_t> indices;
        if (packed&(1u<<31)) {
            indices.push_back(base);
            for (unsigned delta=1;delta<=16;++delta)
                if (packed&(1u<<(delta+14))) indices.push_back(base+delta);
        } else {
            const unsigned count=((packed>>15)&63)+1;
            for (unsigned i=0;i<count;++i) indices.push_back(base+i);
        }
        for (std::uint32_t index:indices) {
            out[index*4]=read32(stream,cursor);cursor+=4;
        }
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
    if (encoded.program.taskCount!=82 || encoded.program.task.empty() ||
        encoded.program.constants.size()!=53177344 ||
        encoded.program.kernelRelocations.size()!=64 ||
        encoded.program.scratchAllocationBytes!=4u*1024u*1024u ||
        encoded.programRecordCount!=165 || encoded.programFormatCode!=0x4c ||
        encoded.program.task[0]!=1 || !encoded.program.formulaBuilt) {
        std::cerr << "resnet50 tasks=" << encoded.program.taskCount
                  << " constants=" << encoded.program.constants.size()
                  << " relocations=" << encoded.program.kernelRelocations.size()
                  << " records=" << encoded.programRecordCount << "\n";
        return 1;
    }
    const auto stemRegisters=taskRegisters(encoded.program.task,0);
    const std::size_t stemTask=16;
    constexpr std::uint32_t stemTaskWords=103,stemExecutionCost=12;
    constexpr std::uint32_t stemLiveBanks=(1u<<25)-1;
    constexpr std::uint32_t stemSchedule=(1u<<21)|(4u<<16)|(4u<<12)|(4u<<8);
    constexpr std::uint32_t stemSourceBase=0x18ae00;
    constexpr std::uint32_t stemSourceChannelStride=112;
    constexpr std::uint32_t stemSourceRowStride=3*stemSourceChannelStride;
    constexpr std::uint32_t stemResultConfig=(5u<<20)|(1u<<8)|(7u<<4);
    constexpr std::uint32_t stemResultRowStride=64*224+32;
    if ((read32(encoded.program.task,stemTask)>>16)!=stemTaskWords ||
        read32(encoded.program.task,stemTask+4)!=stemExecutionCost ||
        read32(encoded.program.task,stemTask+28)!=stemLiveBanks ||
        stemRegisters.at(0x3c)!=stemSchedule ||
        stemRegisters.at(0x4110)!=stemSourceBase ||
        stemRegisters.at(0x4114)!=stemSourceChannelStride ||
        stemRegisters.at(0x4118)!=stemSourceRowStride ||
        stemRegisters.at(0x411c)!=stemSourceRowStride ||
        stemRegisters.at(0x4120)!=stemSourceRowStride ||
        stemRegisters.at(0x4148)!=stemResultConfig ||
        stemRegisters.at(0x4154)!=stemResultRowStride ||
        stemRegisters.count(0x4158) || stemRegisters.count(0x415c)) {
        std::cerr << "rule-built stem does not use the native sparse-phase layout\n";
        return 6;
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

    // Qualification prefixes keep a small mean/linear probe head so every
    // rung has one ordinary output surface. Stem-only may omit max-pool;
    // stem+maxpool and block prefixes use the same path as the full network.
    auto stemHeadWeights=nonzeroHalves(64*1000,601);
    auto stemHeadBias=nonzeroHalves(1000,602);
    ane::h16::ResidualNetworkOptions stemOptions;stemOptions.maxPool=false;
    auto stemProbe=ane::h16::encodeResidualNetwork(
        {convs[0]},stemHeadWeights.data(),stemHeadWeights.size(),
        stemHeadBias.data(),stemHeadBias.size(),stemOptions);
    stemOptions.maxPool=true;
    auto poolProbe=ane::h16::encodeResidualNetwork(
        {convs[0]},stemHeadWeights.data(),stemHeadWeights.size(),
        stemHeadBias.data(),stemHeadBias.size(),stemOptions);
    if (stemProbe.program.taskCount!=3 || poolProbe.program.taskCount!=4 ||
        stemProbe.program.kernelRelocations.size()!=2 ||
        poolProbe.program.kernelRelocations.size()!=2) return 4;
    const auto stemMean=taskRegisters(stemProbe.program.task,1);
    const auto stemLinear=taskRegisters(stemProbe.program.task,2);
    const auto poolTask=taskRegisters(poolProbe.program.task,1);
    const auto acceptedPoolTask=taskRegisters(encoded.program.task,1);
    const std::size_t poolOffset=taskOffset(poolProbe.program.task,1);
    constexpr std::uint32_t poolResultBase=112*stemResultRowStride;
    constexpr std::uint32_t poolResultRow=128;
    constexpr std::uint32_t poolResultPlane=poolResultRow*56;
    constexpr std::uint32_t acceptedPoolResultRow=16*56;
    constexpr std::uint32_t acceptedPoolResultPlane=acceptedPoolResultRow*56;
    if (stemMean.at(0x4104)!=((1u<<22)|(1u<<8)) ||
        stemMean.at(0x4118)!=stemResultRowStride ||
        stemMean.count(0x4d00) || stemMean.count(0x411c) ||
        stemMean.count(0x4120) || stemMean.count(0x4148) ||
        stemMean.count(0x4154) || stemMean.count(0x4158) ||
        stemMean.count(0x415c) ||
        stemLinear.at(0x4104)!=((1u<<22)|(1u<<8)) ||
        stemLinear.count(0x4118) || stemLinear.count(0x411c) ||
        stemLinear.count(0x4120) || stemLinear.count(0x4150) ||
        stemLinear.count(0x4154) || stemLinear.count(0x4158) ||
        stemLinear.count(0x415c) || stemLinear.at(0x5138)!=0x01302031 ||
        read32(poolProbe.program.task,poolOffset+4)!=11 ||
        read32(poolProbe.program.task,poolOffset+28)!=0x01ffffff ||
        poolTask.at(0x414c)!=poolResultBase ||
        poolTask.at(0x5110)!=poolResultRow ||
        poolTask.at(0x5114)!=poolResultPlane ||
        acceptedPoolTask.at(0x414c)!=poolResultBase ||
        acceptedPoolTask.at(0x5110)!=acceptedPoolResultRow ||
        acceptedPoolTask.at(0x5114)!=acceptedPoolResultPlane ||
        poolTask.count(0x411c) || poolTask.count(0x4120) ||
        poolTask.count(0x4150) || poolTask.count(0x4154) ||
        poolTask.count(0x4158) || poolTask.count(0x415c)) {
        std::cerr << "residual probe head does not preserve its resident boundary\n";
        return 7;
    }
    auto stemBoundary=ane::h16::encodeResidualStemBoundary(convs[0]);
    const auto stemBoundaryTask=taskRegisters(stemBoundary.program.task,0);
    auto headOnly=ane::h16::encodeResidualHead(
        convs[0].shape.output,stemHeadWeights.data(),stemHeadWeights.size(),
        stemHeadBias.data(),stemHeadBias.size());
    if (stemBoundary.program.taskCount!=1 ||
        stemBoundary.program.kernelRelocations.size()!=1 ||
        stemBoundary.program.output.nchw[1]!=64 ||
        stemBoundaryTask.at(0x4148)!=stemResultConfig ||
        stemBoundaryTask.at(0x5110)!=256 ||
        stemBoundaryTask.at(0x5114)!=28672 ||
        headOnly.program.taskCount!=2 ||
        headOnly.program.kernelRelocations.size()!=1 ||
        headOnly.program.inputs.front().nchw[1]!=64 ||
        !taskRegisters(headOnly.program.task,0).count(0x4d00)) return 8;

    auto blockHeadWeights=nonzeroHalves(256*1000,603);
    auto blockHeadBias=nonzeroHalves(1000,604);
    std::vector<ane::h16::ResNet50ConvWeights> firstBlock(
        convs.begin(),convs.begin()+5);
    std::vector<ane::h16::ResNet50ConvWeights> firstStage(
        convs.begin(),convs.begin()+11);
    auto blockProbe=ane::h16::encodeResidualNetwork(
        firstBlock,blockHeadWeights.data(),
        blockHeadWeights.size(),blockHeadBias.data(),blockHeadBias.size());
    auto stageProbe=ane::h16::encodeResidualNetwork(
        firstStage,blockHeadWeights.data(),
        blockHeadWeights.size(),blockHeadBias.data(),blockHeadBias.size());
    if (blockProbe.program.taskCount!=9 || stageProbe.program.taskCount!=17 ||
        blockProbe.program.kernelRelocations.size()!=6 ||
        stageProbe.program.kernelRelocations.size()!=12) return 5;

    ane::h16::ResidualNetworkOptions boundaryOptions;
    boundaryOptions.boundaryOutput=true;
    boundaryOptions.maxPool=false;
    auto stemDump=ane::h16::encodeResidualNetwork(
        {convs[0]},nullptr,0,nullptr,0,boundaryOptions);
    boundaryOptions.maxPool=true;
    auto poolDump=ane::h16::encodeResidualNetwork(
        {convs[0]},nullptr,0,nullptr,0,boundaryOptions);
    auto blockDump=ane::h16::encodeResidualNetwork(
        firstBlock,nullptr,0,nullptr,0,boundaryOptions);
    auto stageDump=ane::h16::encodeResidualNetwork(
        firstStage,nullptr,0,nullptr,0,boundaryOptions);
    const auto poolDumpTask=taskRegisters(poolDump.program.task,1);
    const auto blockDumpTask=taskRegisters(blockDump.program.task,6);
    const auto stageDumpTask=taskRegisters(stageDump.program.task,14);
    if (stemDump.program.taskCount!=1 || poolDump.program.taskCount!=2 ||
        blockDump.program.taskCount!=7 || stageDump.program.taskCount!=15 ||
        poolDump.program.output.nchw[1]!=64 ||
        blockDump.program.output.nchw[1]!=256 ||
        stageDump.program.output.nchw[1]!=256 ||
        poolDumpTask.at(0x5110)!=128 || poolDumpTask.at(0x5114)!=7168 ||
        poolDumpTask.at(0x5138)!=0x01302031 ||
        blockDumpTask.at(0x5110)!=128 || blockDumpTask.at(0x5114)!=7168 ||
        blockDumpTask.at(0x5138)!=0x01302031 ||
        stageDumpTask.at(0x5110)!=128 || stageDumpTask.at(0x5114)!=7168 ||
        stageDumpTask.at(0x5138)!=0x01302031) return 9;

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
