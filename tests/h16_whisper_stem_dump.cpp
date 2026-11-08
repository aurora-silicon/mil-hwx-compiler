#include "../plugins/H16G/Encoding/H16ConvProgram.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc,char **argv) {
    try {
        if (argc!=5)
            throw std::invalid_argument(
                "usage: h16_whisper_stem_dump task|kernel T C gelu");
        const std::string mode=argv[1];
        const auto tokens=static_cast<std::uint32_t>(std::stoul(argv[2]));
        const auto channels=static_cast<std::uint32_t>(std::stoul(argv[3]));
        const bool gelu=std::stoul(argv[4])!=0;
        const std::size_t weightBytes=std::size_t(channels)*80*3*2;
        const std::size_t biasBytes=std::size_t(channels)*2;
        std::vector<std::uint8_t> input(weightBytes+biasBytes+(gelu ? 128 : 0));
        std::cin.read(reinterpret_cast<char *>(input.data()),input.size());
        if (std::size_t(std::cin.gcount())!=input.size() || std::cin.peek()!=EOF)
            throw std::invalid_argument("stdin must contain exact weights, bias, and table bytes");
        ane::h16::TransformerLinearOptions options;
        options.bias=true; options.gelu=gelu;
        options.geluTable=gelu ? input.data()+weightBytes+biasBytes : nullptr;
        options.geluTableBytes=gelu ? 128 : 0;
        auto encoded=ane::h16::encodeWhisperStem1(tokens,channels,input.data(),
            weightBytes,input.data()+weightBytes,biasBytes,options);
        const auto &bytes=mode=="task" ? encoded.program.task
                          : mode=="kernel" ? encoded.program.constants
                          : throw std::invalid_argument("mode must be task or kernel");
        std::cout.write(reinterpret_cast<const char *>(bytes.data()),bytes.size());
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
