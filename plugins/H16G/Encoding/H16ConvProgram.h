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
    std::uint32_t kernelSlotFlags = 0x24000000;
    bool transformerStagingRecords = false;
    bool transformerContinuationRecords = false;
};
struct Description {
    std::vector<Task> tasks;
    // Includes the zero prefix and inter-task alignment, as in HWX __text.
    std::vector<std::uint8_t> stream;
    std::vector<std::size_t> kernelRelocations;
    std::size_t kernelBytes = 0;
};

/// Formula inputs for the constant-weight transformer projection family.
/// The logical MIL surface is [1,S,C]; H16G records it as [1,1,S,C].
/// `outputScale` is an fp16 bit pattern (0x3c00 is one).  GELU uses the
/// public H16G 128-byte table supplied by the caller so this C++ packet
/// builder remains independent of the Objective-C table registry.
struct TransformerLinearOptions {
    bool bias = false;
    bool gelu = false;
    std::uint16_t outputScale = 0x3c00;
    const std::uint8_t *geluTable = nullptr;
    std::size_t geluTableBytes = 0;
};

/// Object-writer hints decoded alongside the formula-built program.  The
/// task descriptors and kernel section do not depend on these scheduler cost
/// hints; they are carried separately because the H16G Mach-O record does.
struct TransformerLinearProgram {
    h13::Program program;
    std::uint32_t programRecordCount = 0;
    std::uint32_t programFormatCode = 0;
    bool scratchBacked = false;
    std::size_t scratchBytes = 0;
};

/// Native transformer data movement over a contiguous fp16 tensor.  Split
/// maps [1,S,H*D] to [1,H,S,D], SplitTranspose maps it to [1,H,D,S], and
/// Merge is the inverse [1,H,S,D] to [1,S,H*D].  The measured rows have
/// D=64; other requested shapes are emitted only as explicitly unrevealed
/// predictions.
enum class AttentionLayout { Split, SplitTranspose, Merge };
struct TransformerAttentionProgram {
    h13::Program program;
    std::uint32_t programRecordCount = 0;
    std::uint32_t programFormatCode = 0;
    // Bytes the operation descriptor exposes to the task.  The backing
    // allocation may be larger when the engine uses an implicit ping-pong
    // half; h13::Program::scratchAllocationBytes carries that extent.
    std::size_t scratchBytes = 0;
    bool revealed = false;
};

/// Native last-axis layer normalization over [1,S,C].  The supplied H16G
/// corpus proves the epsilon=fp16(1e-5), identity-gamma, zero-beta form used
/// by the BERT and Whisper encoder programs.  Non-identity affine operands
/// remain outside this API: their constants are present in the full-program
/// constant stream, but no captured ANE task consumes non-identity values.
struct TransformerLayerNormProgram {
    h13::Program program;
    std::uint32_t programRecordCount = 0;
    std::uint32_t programFormatCode = 0;
    std::size_t scratchBytes = 0;
    bool revealedInputPrefix = false;
    bool revealedOutputBoundary = false;
};

std::string transformerLayerNormRefusal(std::uint32_t tokens,
                                        std::uint32_t channels);
TransformerLayerNormProgram encodeTransformerLayerNorm(
    std::uint32_t tokens, std::uint32_t channels);

std::string attentionLayoutRefusal(std::uint32_t tokens,
                                   std::uint32_t heads,
                                   std::uint32_t headDimension,
                                   AttentionLayout operation);
TransformerAttentionProgram encodeAttentionLayout(
    std::uint32_t tokens, std::uint32_t heads,
    std::uint32_t headDimension, AttentionLayout operation);

/// Runtime/runtime attention matrix products. QK consumes [1,H,S,D] and
/// [1,H,D,S]; QKTransposeY consumes two [1,H,S,D] tensors; PV consumes
/// [1,H,S,S] and [1,H,S,D]. `outputScale` is folded into QK's final result
/// word; other operations retain their qualified numerical scale.
enum class AttentionMatmul { QK, QKTransposeY, PV };
std::string attentionMatmulRefusal(std::uint32_t tokens,
                                   std::uint32_t heads,
                                   std::uint32_t headDimension,
                                   AttentionMatmul operation);
TransformerAttentionProgram encodeAttentionMatmul(
    std::uint32_t tokens, std::uint32_t heads,
    std::uint32_t headDimension, AttentionMatmul operation,
    std::uint16_t outputScale = 0x3c00);

/// Native last-axis softmax over [1,H,S,S]. The measured S=1500,H=8 program
/// is the physical form; callers with H<=8 zero-pad its trailing planes.
std::string attentionSoftmaxRefusal(std::uint32_t tokens,
                                    std::uint32_t heads);
TransformerAttentionProgram encodeAttentionSoftmax(std::uint32_t tokens,
                                                   std::uint32_t heads);

/// Fused scale-QK-softmax-PV over Q [1,H,S,D], K [1,H,D,S], and
/// V [1,H,S,D].
TransformerAttentionProgram encodeFusedAttention(
    std::uint32_t tokens, std::uint32_t heads,
    std::uint32_t headDimension);

/// Empty when the transformer-linear formula has measured coverage.
std::string transformerLinearRefusal(std::uint32_t tokens,
                                     std::uint32_t inputChannels,
                                     std::uint32_t outputChannels,
                                     TransformerLinearOptions options);

/// Encode a constant fp16 linear projection over S tokens.  `weights` is the
/// MIL row-major [Cout,Cin] tensor and `bias` is [Cout] when enabled.
TransformerLinearProgram encodeTransformerLinear(
    std::uint32_t tokens, std::uint32_t inputChannels,
    std::uint32_t outputChannels, const std::uint8_t *weights,
    std::size_t weightBytes, const std::uint8_t *bias,
    std::size_t biasBytes, TransformerLinearOptions options = {});

/// Encode Whisper's biased 1x3, stride-1 stem projection. The logical MIL
/// tensors are [1,80,T] -> [1,C,T]; the object records [1,1,80,T] and
/// [1,1,C,T]. The optional GELU uses the same table contract as linear.
TransformerLinearProgram encodeWhisperStem1(
    std::uint32_t tokens, std::uint32_t outputChannels,
    const std::uint8_t *weights, std::size_t weightBytes,
    const std::uint8_t *bias, std::size_t biasBytes,
    TransformerLinearOptions options = {});

/// Encode Whisper's biased 1x3, stride-2 stem projection. The measured
/// programs are [1,C,3000] -> [1,C,1500] for C in {384,512}. The kernel
/// writer is data dependent: both fp16 signed zeros clear their sparse mask
/// bit and consume no coefficient bytes.
TransformerLinearProgram encodeWhisperStem2(
    std::uint32_t inputChannels, const std::uint8_t *weights,
    std::size_t weightBytes, const std::uint8_t *bias,
    std::size_t biasBytes, TransformerLinearOptions options = {});

/// Materialize Whisper's encoder-axis transpose
/// [1,C,1500] -> [1,1500,C] for C in {384,512}.
TransformerLinearProgram encodeWhisperTranspose(std::uint32_t channels);
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
// Full-model ResNet packing facts.  Unlike the isolated MobileNet sparse
// form above, these pack dense groups-1 convolutions into sixteen KernelDMA
// regions.  The 7x7 form is decoded only for the 3->64/s2 stem; the regular
// form is decoded for biased 1x1/s2 and 3x3/s2 layers whose output count is a
// multiple of sixteen.  These APIs encode kernel regions only: they do not
// claim a standalone task schedule or a generic residual-DAG policy.
std::vector<std::uint8_t> packResNetStride2(ConvShape shape,
                                            const std::uint8_t *weights,
                                            const std::uint8_t *bias,
                                            std::array<std::uint32_t,16> &chunkBytes);
std::vector<std::uint8_t> packResNetDense(ConvShape shape,
                                          const std::uint8_t *weights,
                                          const std::uint8_t *bias,
                                          std::array<std::uint32_t,16> &chunkBytes);
std::vector<std::uint8_t> packResNetLinearHead(
    std::uint32_t inputChannels, std::uint32_t outputChannels,
    const std::uint8_t *weights, const std::uint8_t *bias,
    std::array<std::uint32_t,16> &chunkBytes);
std::vector<std::uint8_t> packResNetStem7x7(ConvShape shape,
                                            const std::uint8_t *weights,
                                            const std::uint8_t *bias,
                                            std::array<std::uint32_t,16> &chunkBytes);
struct ResNet50ConvWeights {
    ConvShape shape;
    const std::uint8_t *weights = nullptr;
    std::size_t weightBytes = 0;
    const std::uint8_t *bias = nullptr;
    std::size_t biasBytes = 0;
};
struct ResNet50Program {
    h13::Program program;
    std::uint32_t programRecordCount = 0;
    std::uint32_t programFormatCode = 0;
    std::size_t scratchBytes = 0;
};
// Exact public random-weight ResNet-50 topology: 53 biased convolutions,
// max-pool, sixteen residual add+ReLUs, global mean, and a 2048->1000 head.
// Convolution entries are supplied in the measured task-schedule order.
ResNet50Program encodeResNet50(
    const std::vector<ResNet50ConvWeights> &convolutions,
    const std::uint8_t *headWeights, std::size_t headWeightBytes,
    const std::uint8_t *headBias, std::size_t headBiasBytes);
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
