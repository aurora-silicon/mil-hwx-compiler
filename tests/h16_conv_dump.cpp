// A small host-only adapter: parity exercises the production formula backend.
#include "../plugins/H16G/Encoding/H16ConvProgram.h"
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    try {
        if (argc<10 || (argc-3)%7) throw std::invalid_argument("mode target then cin cout width kernel stride groups relu per layer");
        const std::string mode=argv[1],target=argv[2];
        if (target!="H16G" && target!="H17") throw std::invalid_argument("target");
        auto gen=target=="H17" ? ane::h16::Generation::H17 : ane::h16::Generation::H16G;
        std::vector<ane::h16::Layer> layers;
        for (int at=3;at<argc;at+=7) {
            const auto number=[&](int i) {return static_cast<std::uint32_t>(std::stoul(argv[at+i]));};
            auto ci=number(0),co=number(1),w=number(2),k=number(3),s=number(4),g=number(5);
            if (!s) throw std::invalid_argument("zero stride");
            auto wo=(w+s-1)/s;
            // Flags: bit 0 fused ReLU, bit 1 bias.
            layers.push_back({{k,k,s,g,(number(6)&2)!=0,{ci,w,w},{co,wo,wo}},(number(6)&1)!=0});
        }
        std::vector<std::uint8_t> bytes;
        if (mode=="describe") bytes=ane::h16::describe(gen,layers).stream;
        else if (mode=="describe-dense") bytes=ane::h16::describe(gen,layers,false).stream;
        else if (mode.rfind("dram",0)==0) {
            const auto form=mode=="dram-measured" ? ane::h16::DramEdgeForm::Measured
                          : mode=="dram-dep" ? ane::h16::DramEdgeForm::DependencyOnly
                          : ane::h16::DramEdgeForm::None;
            bytes=ane::h16::describeDramChain(gen,layers,form,0x200000).stream;
        }
        else if (mode=="pack" && layers.size()==1) {
            auto s=layers[0].shape;
            const std::size_t size=std::size_t(s.output.channels)*s.input.channels/s.groups*s.kernel*s.kernel*2;
            const std::size_t biasSize=s.bias ? std::size_t(s.output.channels)*2 : 0;
            std::vector<std::uint8_t> weights(size+biasSize);
            std::cin.read(reinterpret_cast<char *>(weights.data()),weights.size());
            if (std::size_t(std::cin.gcount()) != weights.size() || std::cin.peek() != EOF)
                throw std::invalid_argument("pack mode needs the exact resolved fp16 weights (then bias) on stdin");
            auto p=ane::h16::encodeConv(gen,s,weights.data(),size,
                                        s.bias ? weights.data()+size : nullptr,biasSize);
            bytes=p.constants;
        } else throw std::invalid_argument("mode");
        std::cout.write(reinterpret_cast<const char *>(bytes.data()),bytes.size());
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
