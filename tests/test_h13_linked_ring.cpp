#include "../plugins/H13/H13Program.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ane::h13;

static std::uint32_t word(const std::vector<std::uint8_t> &v, std::size_t at) {
    assert(at + 4 <= v.size());
    return v[at] | std::uint32_t(v[at + 1]) << 8 |
           std::uint32_t(v[at + 2]) << 16 | std::uint32_t(v[at + 3]) << 24;
}

static std::uint32_t reg(const Program &p, std::size_t off,
                         std::size_t bytes, std::uint32_t address) {
    for (std::size_t at = off + 40; at < off + bytes;) {
        const auto h = word(p.task, at);
        const std::uint32_t count = (h >> 26) + 1;
        const std::uint32_t base = h & 0x03ffffffu;
        if (address >= base && address < base + 4 * count)
            return word(p.task, at + 4 + address - base);
        at += 4 * (count + 1);
    }
    std::cerr << "register absent off=" << off << " bytes=" << bytes
              << " address=0x" << std::hex << address << std::dec << "\n";
    assert(false && "register absent");
    return 0;
}

static Program conv(std::uint32_t ci, std::uint32_t co, std::uint32_t h,
                    std::uint32_t k, std::uint32_t stride, bool depthwise,
                    bool relu) {
    const std::uint32_t ho = (h + stride - 1) / stride;
    ConvShape shape{k, k, stride, depthwise ? ci : 1, true,
                    {ci, h, h}, {co, ho, ho}};
    std::vector<std::uint8_t> weights(
        static_cast<std::size_t>(co) * (depthwise ? 1 : ci) * k * k * 2);
    std::vector<std::uint8_t> bias(co * 2);
    for (std::size_t i = 0; i < weights.size(); i += 2) {
        weights[i] = static_cast<std::uint8_t>(1 + (i / 2) % 251);
        weights[i + 1] = 0x34;
    }
    auto p = encodeConvLinkedRing(shape, weights.data(), weights.size(),
                                  bias.data(), bias.size());
    if (relu) fuseConvPostOperation(p, PostOperation::Relu);
    return p;
}

int main() {
    static const std::uint32_t co[] = {
        32,32,64,64,128,128,128,128,256,256,256,256,512,512,
        512,512,512,512,512,512,512,512,512,512,1024,1024,1024,1024};
    static const std::uint32_t stride[] = {
        2,1,1,2,1,1,1,2,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1};
    std::vector<Program> layers;
    std::uint32_t ci = 3, h = 224;
    for (std::size_t i = 0; i < 28; ++i) {
        const bool depthwise = i != 0 && i != 27 && (i & 1);
        const std::uint32_t k = i == 0 || depthwise ? 3 : 1;
        layers.push_back(conv(ci, co[i], h, k, stride[i], depthwise, i != 27));
        assert(layers.back().taskCount == 1);
        ci = co[i];
        h = (h + stride[i] - 1) / stride[i];
    }

    std::vector<LinkedLayer> layout;
    const auto ring = linkMobileNetRing(layers, &layout);
    assert(ring.taskCount == 28 && layout.size() == 28);
    assert(ring.scratchAllocationBytes == 0x284000);
    // Native task DMA extents are conservative hardware windows, larger
    // than the external NCHW physical spans.  The ANEC tiles[] allocations
    // must satisfy libane's bind-overrun check without changing those task
    // words or the NCHW strides.
    assert(ring.inputs[0].allocationBytes == 458752);
    assert(ring.output.allocationBytes == 458752);
    assert(layout[0].l2ResultBytes == 808192);
    assert(layout[1].writesScratch && layout[2].readsScratch);
    assert(layout[2].l2Result == 0 && layout[2].l2ResultBytes == 1609216);
    assert(layout[2].l2ResultBytes + 467712 == 2076928);
    for (std::size_t i = 2; i + 1 < layout.size(); ++i)
        assert(layout[i].outputEdge != 0);
    assert(layers[24].constants.size() > 0x100000);
    assert(layers[26].constants.size() >= 2099200);
    assert(layers[27].constants.size() >= 2099200);
    // Native 3x3/s2 uses 16 phase-sparse chunks, including the C512
    // depthwise form whose 32 channels/chunk need the four-row grammar.
    assert(layers[0].constants.size() == 2048);
    assert(layers[3].constants.size() == 2048);
    assert(layers[7].constants.size() == 4096);
    assert(layers[11].constants.size() == 7168);
    assert(layers[23].constants.size() == 16384);
    // ANECOMP fleet44: the 2.1 MB 7x7 form is safe, while the same kernel
    // attached to a multi-tile 14x14 task wedges.  The policy therefore
    // retains the ordinary 1 MiB split outside the whole-output 7x7 form.
    const auto unsafe14 = conv(1024, 1024, 14, 1, 1, false, true);
    assert(unsafe14.taskCount == 4);

    std::size_t off = 0, bytes = ring.firstTaskBytes;
    for (std::size_t i = 0; i < 28; ++i) {
        if (i == 0 || i == 3 || i == 7 || i == 11 || i == 23)
            assert(reg(ring, off, bytes, 0x0c800) == 0x182);
        else
            assert(reg(ring, off, bytes, 0x0c800) == 0x82);
        if (i == 0 || i == 3 || i == 7 || i == 11 || i == 23)
            assert(reg(ring, off, bytes, 0x1f800) == 0x60);
        assert(reg(ring, off, bytes, 0x00028) ==
               (reg(ring, off, bytes, 0x00014) >> 16));
        if (i + 1 < 28) {
            const std::size_t nextBytes =
                (((word(ring.task, off + 4) >> 16) & 0x1ff) + 1) * 4;
            off = word(ring.task, off + 28);
            bytes = nextBytes;
        }
    }
    assert(reg(ring, 0, ring.firstTaskBytes, 0x04804) == 0x500172);
    assert(reg(ring, 0, ring.firstTaskBytes, 0x04830) == 0x500170);

    for (std::size_t prefix = 1; prefix <= 28; ++prefix) {
        std::vector<Program> p(layers.begin(), layers.begin() + prefix);
        const auto q = linkMobileNetRing(p);
        assert(q.taskCount == prefix);
    }
    std::cout << "H13 linked-ring: 28 native tasks, one spill, 2 MiB ring PASS\n";
}
