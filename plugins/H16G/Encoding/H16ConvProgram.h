#pragma once

#include "../../H13/H13Program.h"
#include <array>
#include <map>

// Formula-built convolution descriptors, shared by the H16G and H17 targets.
// No decoded oracle or generated template is loaded by this implementation.
namespace ane::h16 {
enum class Generation { H16G, H17 };
using ConvShape = h13::ConvShape;
struct Layer {
    ConvShape shape;
    bool relu = false;
};
struct Task {
    std::array<std::uint32_t, 9> header{};
    std::map<std::uint32_t, std::uint32_t> registers;
    std::vector<std::uint8_t> bytes;
    // Byte offset of the relocatable 64-bit kernel base in bytes.
    std::size_t kernelRelocation = 0;
    // Relocation record flags (BAR slot) of the source and result bases.
    std::uint32_t sourceSlotFlags = 0x23000000;
    std::uint32_t resultSlotFlags = 0x23800000;
};
struct Description {
    std::vector<Task> tasks;
    // Includes the zero prefix and inter-task alignment, as in HWX __text.
    std::vector<std::uint8_t> stream;
    std::vector<std::size_t> kernelRelocations;
    std::size_t kernelBytes = 0;
};
// Empty only for one of the nine measured geometries. Bias, VALID, dilation
// and general scheduling are deliberately outside this API's envelope.
std::string convRefusal(ConvShape shape);
// A stricter boundary for executable packages: sparse H16/H17 kernel packing
// and DRAM task synchronization have no proven implementation in this corpus.
std::string executableRefusal(ConvShape shape);
// Single conv or the measured two/three-stage resident chain. Stride-2 chunk
// lengths here describe the campaign's legacy BLOBFILE view ONLY; they
// must never be used to package arbitrary sparse weights.
Description describe(Generation generation, const std::vector<Layer> &layers,
                     bool sparseStride2 = true);
// Packages only a measured stride-1 standalone form. Uses the established
// stride-1 packer whose output is verified against every applicable hash.
// A chain of measured standalone tasks whose intermediates go through the
// scratch BAR (slot 3) in DRAM, ping-ponging between two regions. Measured
// uses the r2 DRAM-chain producer/consumer fields; DependencyOnly sets only
// the DependencyMode bit and producer flag; None links with no dependency.
// Device verification pending.
enum class DramEdgeForm { Measured, DependencyOnly, None };
Description describeDramChain(Generation generation, const std::vector<Layer> &layers,
                              DramEdgeForm form, std::uint32_t scratchRegionBytes);
h13::Program encodeConv(Generation generation, ConvShape shape,
                        const std::uint8_t *weights, std::size_t weightBytes,
                        const std::uint8_t *bias = nullptr, std::size_t biasBytes = 0);
void fuseRelu(h13::Program &program);
}
