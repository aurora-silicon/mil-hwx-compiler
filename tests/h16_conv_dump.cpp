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
            layers.push_back({{k,k,s,g,false,{ci,w,w},{co,wo,wo}},number(6)!=0});
        }
        std::vector<std::uint8_t> bytes;
        if (mode=="describe") bytes=ane::h16::describe(gen,layers).stream;
        else if (mode=="pack" && layers.size()==1) {
            auto s=layers[0].shape;
            const std::size_t size=std::size_t(s.output.channels)*s.input.channels/s.groups*s.kernel*s.kernel*2;
            std::vector<std::uint8_t> weights(size);
            std::cin.read(reinterpret_cast<char *>(weights.data()),weights.size());
            if (std::size_t(std::cin.gcount()) != size || std::cin.peek() != EOF)
                throw std::invalid_argument("pack mode needs the exact resolved fp16 weights on stdin");
            auto p=ane::h16::encodeConv(gen,s,weights.data(),weights.size());
            bytes=p.constants;
        } else throw std::invalid_argument("mode");
        std::cout.write(reinterpret_cast<const char *>(bytes.data()),bytes.size());
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
