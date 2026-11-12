#include "../plugins/H16G/Encoding/H16ConvProgram.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc,char **argv) {
    if (argc==5 && std::string(argv[1])=="fused") {
        const auto encoded=ane::h16::encodeFusedAttention(
            std::strtoul(argv[2],nullptr,0),std::strtoul(argv[3],nullptr,0),
            std::strtoul(argv[4],nullptr,0));
        std::cout.write(reinterpret_cast<const char *>(encoded.program.task.data()),
                        encoded.program.task.size());
        return 0;
    }
    if (argc==4 && std::string(argv[1])=="softmax") {
        const auto encoded=ane::h16::encodeAttentionSoftmax(
            std::strtoul(argv[2],nullptr,0),std::strtoul(argv[3],nullptr,0));
        std::cout.write(reinterpret_cast<const char *>(encoded.program.task.data()),
                        encoded.program.task.size());
        return 0;
    }
    if ((argc!=6 && argc!=7) || (std::string(argv[1])!="layout" &&
                    std::string(argv[1])!="matmul")) {
        std::cerr << "usage: h16_transformer_attention_dump "
                     "layout split|splitT|merge S H D | matmul qk|qkT|pv S H D\n";
        return 64;
    }
    const std::string name=argv[2];
    ane::h16::TransformerAttentionProgram encoded;
    if (std::string(argv[1])=="layout") {
        ane::h16::AttentionLayout operation;
        if (name=="split") operation=ane::h16::AttentionLayout::Split;
        else if (name=="splitT") operation=ane::h16::AttentionLayout::SplitTranspose;
        else if (name=="merge") operation=ane::h16::AttentionLayout::Merge;
        else throw std::invalid_argument("unknown attention layout operation");
        encoded=ane::h16::encodeAttentionLayout(
            std::strtoul(argv[3],nullptr,0),std::strtoul(argv[4],nullptr,0),
            std::strtoul(argv[5],nullptr,0),operation);
    } else {
        ane::h16::AttentionMatmul operation;
        if (name=="qk") operation=ane::h16::AttentionMatmul::QK;
        else if (name=="qkT") operation=ane::h16::AttentionMatmul::QKTransposeY;
        else if (name=="pv") operation=ane::h16::AttentionMatmul::PV;
        else throw std::invalid_argument("unknown attention matmul operation");
        const std::uint16_t outputScale=argc==7
            ? std::strtoul(argv[6],nullptr,0) : 0x3c00;
        encoded=ane::h16::encodeAttentionMatmul(
            std::strtoul(argv[3],nullptr,0),std::strtoul(argv[4],nullptr,0),
            std::strtoul(argv[5],nullptr,0),operation,outputScale);
    }
    std::cout.write(reinterpret_cast<const char *>(encoded.program.task.data()),
                    encoded.program.task.size());
}
