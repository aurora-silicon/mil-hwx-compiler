// SPDX-License-Identifier: MIT
// Portable test driver for the H14 C++ encoding API (no MIL parser/device).
#include "H14Program.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
static std::vector<std::uint8_t> read(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open probe input");
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
int main(int argc, char **argv) {
    if (argc != 16) return 64;
    try {
        auto n = [&](int i) { return static_cast<std::uint32_t>(std::stoul(argv[i])); };
        ane::h14::ConvShape s{n(6),n(7),n(8),n(9),bool(n(10)),
                              {n(2),n(4),n(5)},{n(3),n(11),n(12)}};
        const auto w = read(argv[13]), b = read(argv[14]);
        const auto p = std::string(argv[1]) == "parity"
            ? ane::h14::encodeConvParity(s,w.data(),w.size(),s.bias?b.data():nullptr,b.size())
            : ane::h14::encodeConvGeneric(s,w.data(),w.size(),s.bias?b.data():nullptr,b.size());
        const auto a = ane::h14::encodeANEC(p);
        std::ofstream out(argv[15], std::ios::binary);
        out.write(reinterpret_cast<const char *>(a.data()), static_cast<std::streamsize>(a.size()));
        return out ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "h14.conv-outside-envelope: " << e.what() << '\n';
        return 1;
    }
}
