#include "../plugins/H16G/Encoding/H16ConvProgram.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc,char **argv) {
    if(argc!=3&&argc!=4) {
        std::cerr << "usage: h16_transformer_layer_norm_dump S C "
                     "[mean|centered|variance|rsqrt|normalized]\n";
        return 64;
    }
    const auto tokens=std::strtoul(argv[1],nullptr,0);
    const auto channels=std::strtoul(argv[2],nullptr,0);
    auto encoded=ane::h16::encodeTransformerLayerNorm(tokens,channels);
    if(argc==4) {
        const std::string stage=argv[3];
        using Stage=ane::h16::LayerNormDumpStage;
        const Stage selected=stage=="mean"?Stage::Mean:
            stage=="centered"?Stage::Centered:
            stage=="variance"?Stage::Variance:
            stage=="rsqrt"?Stage::ReciprocalStdDev:
            stage=="normalized"?Stage::Normalized:
            throw std::invalid_argument("unknown layer_norm dump stage");
        encoded=ane::h16::encodeTransformerLayerNormDump(
            tokens,channels,selected);
    }
    std::cout.write(reinterpret_cast<const char *>(encoded.program.task.data()),
                    encoded.program.task.size());
    return 0;
}
