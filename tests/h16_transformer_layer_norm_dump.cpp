#include "../plugins/H16G/Encoding/H16ConvProgram.h"

#include <cstdlib>
#include <iostream>

int main(int argc,char **argv) {
    if(argc!=3) {
        std::cerr << "usage: h16_transformer_layer_norm_dump S C\n";
        return 64;
    }
    const auto encoded=ane::h16::encodeTransformerLayerNorm(
        std::strtoul(argv[1],nullptr,0),std::strtoul(argv[2],nullptr,0));
    std::cout.write(reinterpret_cast<const char *>(encoded.program.task.data()),
                    encoded.program.task.size());
    return 0;
}
