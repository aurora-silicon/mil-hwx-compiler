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
    // Kernel bytes per lane when they depend on the weights (sparse
    // stride-2 sections); all zero derives them from the shape.
    std::array<std::uint32_t,16> kernelChunks{};
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
// Apple's zero-skipping stride-2 kernel section (3x3 depthwise C64..C512 and
// the 3->32 stem), with the per-lane byte counts the descriptor needs.
std::string sparseStride2Refusal(ConvShape shape);
std::vector<std::uint8_t> packSparseStride2(ConvShape shape, const std::uint8_t *weights,
                                            const std::uint8_t *bias,
                                            std::array<std::uint32_t,16> &chunkBytes);
// A chain of measured standalone tasks whose intermediates go through the
// scratch BAR (slot 3) in DRAM, ping-ponging between two regions. Measured
// uses the r2 DRAM-chain producer/consumer fields; DependencyOnly sets only
// the DependencyMode bit and producer flag; None links with no dependency.
// Device verification pending.
enum class DramEdgeForm { Measured, DependencyOnly, None };
Description describeDramChain(Generation generation, const std::vector<Layer> &layers,
                              DramEdgeForm form, std::uint32_t scratchRegionBytes);
// Three explicitly staged H16G MobileNetV1 policies.  The placement policy
// keeps 26 of the graph's 27 internal edges in the 2 MiB L2 ring and spills
// edge 1 through one channel-3 region.  Schedule additionally selects the
// measured Common words; Header adds the measured cost/dispatch/mask policy
// and compact register write set.  These policies deliberately refuse every
// graph except the named 28-layer MobileNetV1 224 shape.
enum class MobileNetL2Policy { Placement, Schedule, Header };
// One-field-group swaps used only to bisect the first resident MobileNet
// edge against the proven linear and resident encodings.
enum class MobileNetL2FieldSwap {
    None,
    LegacyTerminalDirect,
    LinearKernelDMA,
    SentinelFirstEdgeGeometry,
};
struct MobileNetL2Variant {
    MobileNetL2Policy policy = MobileNetL2Policy::Placement;
    // Number of resident edges enabled in Apple placement order
    // (edge 0, then edges 2..26).  The remaining edges use the proven DRAM
    // ping-pong form.  This is a hardware-bisection facility, not a general
    // placement heuristic.
    unsigned residentPrefix = 26;
    // The full parity rungs and safe resident prefixes use decoded spill
    // bridges.  False is retained only for the old-form negative control.
    bool measuredSpill = true;
    std::uint32_t scratchRegionBytes = 0x1c0000;
    MobileNetL2FieldSwap fieldSwap = MobileNetL2FieldSwap::None;
};
Description describeMobileNetL2(Generation generation,
                                const std::vector<Layer> &layers,
                                MobileNetL2Policy policy);
Description describeMobileNetL2(Generation generation,
                                const std::vector<Layer> &layers,
                                MobileNetL2Variant variant);
h13::Program encodeConv(Generation generation, ConvShape shape,
                        const std::uint8_t *weights, std::size_t weightBytes,
                        const std::uint8_t *bias = nullptr, std::size_t biasBytes = 0,
                        Layer *layerOut = nullptr);
// One program of the encoded standalone layers linked through DRAM scratch
// (slot 3, two ping-pong regions) with the DependencyMode edge form, which
// ran exactly on the J700 (MobileNetV1 224, 28 tasks).
h13::Program linkDramChain(Generation generation, const std::vector<Layer> &layers,
                           const std::vector<h13::Program> &programs,
                           std::vector<h13::LinkedLayer> *layout);
h13::Program linkMobileNetL2(Generation generation,
                             const std::vector<Layer> &layers,
                             const std::vector<h13::Program> &programs,
                             MobileNetL2Policy policy,
                             std::vector<h13::LinkedLayer> *layout);
h13::Program linkMobileNetL2(Generation generation,
                             const std::vector<Layer> &layers,
                             const std::vector<h13::Program> &programs,
                             MobileNetL2Variant variant,
                             std::vector<h13::LinkedLayer> *layout);
void fuseRelu(h13::Program &program);
}
