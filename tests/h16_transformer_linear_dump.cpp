#include "../plugins/H16G/Encoding/H16ConvProgram.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    try {
        if (argc != 7)
            throw std::invalid_argument(
                "usage: h16_transformer_linear_dump task|kernel S Cin Cout flags scale_bits");
        const std::string mode=argv[1];
        const auto number=[&](int index) {
            return static_cast<std::uint32_t>(std::stoul(argv[index],nullptr,0));
        };
        const std::uint32_t tokens=number(2),cin=number(3),cout=number(4);
        const std::uint32_t flags=number(5),scale=number(6);
        const bool bias=flags&1,gelu=flags&2;
        const std::size_t weightBytes=std::size_t(cin)*cout*2;
        const std::size_t biasBytes=bias ? std::size_t(cout)*2 : 0;
        const std::size_t tableBytes=gelu ? 128 : 0;
        std::vector<std::uint8_t> input(weightBytes+biasBytes+tableBytes);
        std::cin.read(reinterpret_cast<char *>(input.data()),input.size());
        if (std::size_t(std::cin.gcount())!=input.size() || std::cin.peek()!=EOF)
            throw std::invalid_argument("stdin must contain exact weights, bias, and table bytes");
        ane::h16::TransformerLinearOptions options;
        options.bias=bias; options.gelu=gelu;
        options.outputScale=static_cast<std::uint16_t>(scale);
        options.geluTable=gelu ? input.data()+weightBytes+biasBytes : nullptr;
        options.geluTableBytes=tableBytes;
        auto encoded=ane::h16::encodeTransformerLinear(tokens,cin,cout,
            input.data(),weightBytes,bias ? input.data()+weightBytes : nullptr,
            biasBytes,options);
        const auto &bytes=mode=="task" ? encoded.program.task
                                      : mode=="kernel" ? encoded.program.constants
                                      : throw std::invalid_argument("mode must be task or kernel");
        std::cout.write(reinterpret_cast<const char *>(bytes.data()),bytes.size());
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
