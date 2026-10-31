#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ane::h13 {
constexpr std::size_t anecHeaderBytes = 0x1000;

constexpr std::size_t taskBytes = 0x274;
constexpr std::size_t tileBytes = 0x4000;
constexpr std::size_t constantOffset = 0x280;

enum class BinaryOperation { Add, Multiply, Maximum, Minimum, Subtract, RealDivide };
enum class UnaryOperation {
    Absolute,
    Exponential,
    Gelu,
    LeakyRelu,
    Relu,
    ReciprocalSquareRoot,
    Sigmoid,
    Silu,
    SquareRoot,
    Tanh,
};

enum class NormOperation { Softmax, LayerNorm, ReduceSum, ReduceMax, ReduceMean };

struct ElementwiseShape {
    std::uint32_t channels;
    std::uint32_t height;
    std::uint32_t width;
};

/// One decoded Apple elementwise surface with its batch. Apple records the
/// batch in the tensor descriptor's shape while its declared size covers a
/// single batch element, so the batch never enters the row or plane stride.
struct BatchedShape {
    std::uint32_t batch;
    std::uint32_t channels;
    std::uint32_t height;
    std::uint32_t width;
};

/// How a binary operation's second operand reaches the program: another
/// runtime surface, an inline fp16 scalar, or a per-channel constant Apple
/// folds into the bias and scale blocks of the constant section.
enum class BroadcastOperand : std::uint8_t { Runtime, Scalar, Constant };

/// One decoded Apple broadcast geometry. `y` is zero for a scalar operand;
/// otherwise it is the second operand's NCHW surface, which may differ from
/// `x` in any axis Apple broadcasts.
struct BroadcastShape {
    BatchedShape x;
    BatchedShape y;
};

/// One decoded Apple matmul geometry: `rows` logical x rows, a `reduction`
/// long inner product, `columns` output columns, the two MIL transpose flags,
/// and whether the second operand is a runtime surface instead of a constant.
struct MatmulShape {
    std::uint32_t rows;
    std::uint32_t reduction;
    std::uint32_t columns;
    bool transposeX = false;
    bool transposeY = true;
    bool runtimeWeight = false;
};

/// One decoded Apple normalization or reduction geometry: the input and output
/// CHW surfaces, which NCHW axes the operation reduces (bit `i` for axis `i`),
/// and whether the MIL result keeps the reduced axes.
struct NormShape {
    ElementwiseShape input;
    ElementwiseShape output;
    std::uint32_t axisMask;
    bool keepDims;
};

/// One decoded Apple convolution geometry: the MIL kernel extent, stride and
/// group count, whether a bias is folded into the constant section, and the
/// input and output CHW surfaces. The padding follows from those, so a
/// `pad_type` of `same` and `valid` share a shape whenever they agree on the
/// output surface.
struct ConvShape {
    std::uint32_t kernel;
    std::uint32_t kernelWidth;
    std::uint32_t stride;
    std::uint32_t groups;
    bool bias;
    ElementwiseShape input;
    ElementwiseShape output;
};

struct TensorLayout {
    std::uint32_t index;
    std::array<std::uint64_t, 6> nchw;
    std::uint64_t allocationBytes;
    /// Element size in bytes: 2 for the fp16 surfaces every decoded form
    /// used before the boolean ops, 1 for the bool compare/cond surfaces
    /// the boolean captures bind. Defaults keep old initializers valid.
    std::uint32_t elementSize = 2;
};

struct Program {
    std::vector<std::uint8_t> task;
    std::vector<std::uint8_t> constants;
    std::vector<TensorLayout> inputs;
    TensorLayout output;
    std::size_t firstTaskBytes = taskBytes;
    std::uint32_t taskCount = 1;
    std::size_t constantOffsetBytes = constantOffset;
    /// Task-descriptor byte offsets holding kernel-table addends the HWX
    /// writer must relocate; empty when the task carries absolute addresses.
    std::vector<std::size_t> kernelRelocations;
    /// The zero-filled allocation Apple places below every surface; the HWX
    /// writer emits it as __DATA/__bss and every surface address shifts by it.
    std::uint64_t scratchAllocationBytes = 0;
    /// Where the output surface sits in Apple's surface order: `inputs.size()`
    /// for last, 0 for a matmul, which lays the output out first, 1 for a
    /// broadcast, which puts it between the two operands.
    std::size_t outputBindingIndex = static_cast<std::size_t>(-1);
    /// Descriptor channels in output, input0, input1 order, before ANEC rebinding.
    std::array<std::uint32_t, 4> taskSurfaceChannels{4, 5, 6, 7};
    /// Built by `encodeConvGeneric` from the per-register formulas rather
    /// than replayed from a decoded template.
    bool formulaBuilt = false;
};

Program encodeBinary(BinaryOperation operation);
bool supportsElementwise(BinaryOperation operation, ElementwiseShape shape,
                         bool scalarConstant = false);
bool supportsElementwise(UnaryOperation operation, ElementwiseShape shape);
Program encodeElementwise(BinaryOperation operation, ElementwiseShape shape,
                          bool scalarConstant = false,
                          std::uint16_t scalarBits = 0x3800);
bool supportsElementwiseConstant(BinaryOperation operation,
                                 ElementwiseShape shape);
/// Encodes Apple's constant-blob twin: one runtime surface whose second
/// source reads the operation's constant straight from the constant
/// section. `constant` is the whole fp16 blob (elements * 2 bytes) exactly
/// as the MIL resolves it.
Program encodeElementwiseConstant(BinaryOperation operation,
                                  ElementwiseShape shape,
                                  const std::uint8_t *constant,
                                  std::size_t constantBytes);
Program encodeElementwise(UnaryOperation operation, ElementwiseShape shape);

/// True when the decoded Apple corpus covers this transpose as one 1-task
/// program: the elementwise triples are the Apple-normalized NCHW surfaces of
/// the MIL input and result (rank 3 [1, A, B] -> (1, A, B), rank 4
/// [1, C, H, W] -> (C, H, W)).
bool supportsTransposeParity(ElementwiseShape input, ElementwiseShape output);
Program encodeTransposeParity(ElementwiseShape input, ElementwiseShape output);

/// True when the decoded Apple corpus covers this slice_by_index as one
/// 1-task program: the elementwise triples are the Apple-normalized NCHW
/// surfaces of the MIL input and result (rank 4 [1, C, H, W] -> (C, H, W)),
/// with only the last axis narrowed from element 0.
bool supportsSliceParity(ElementwiseShape input, ElementwiseShape output);
Program encodeSliceParity(ElementwiseShape input, ElementwiseShape output);

// Weights are row-major little-endian fp16; transposeY selects [512,K] instead of [K,512].
Program encodeMatvec(std::uint32_t reduction,
                     const std::uint8_t *weights,
                     std::size_t weightBytes, bool transposeY);
/// True when the decoded Apple corpus covers this geometry as one program.
bool supportsMatmulParity(MatmulShape shape);
/// Encodes Apple's own matmul task stream for the geometry. `weights` is the
/// [columns, reduction] row-major fp16 constant, exactly the bytes the MIL
/// blob resolves to, and must be null when the second operand is runtime.
Program encodeMatmulParity(MatmulShape shape, const std::uint8_t *weights,
                           std::size_t weightBytes);
/// One decoded Apple batched-matmul geometry: B independent per-batch GEMMs
/// of [rows, reduction] by [reduction, columns] over batch-major contiguous
/// operand planes, lowered as one task stream of an optional prefix task
/// plus 26 tasks per batch (the fold-flag forms carry the prefix).
struct BatchedMatmulShape {
    std::uint32_t batch = 1;
    std::uint32_t rows;
    std::uint32_t reduction;
    std::uint32_t columns;
    bool transposeX = false;
    bool transposeY = false;
    bool runtimeWeight = false;
};

/// True when the decoded batched corpus covers this geometry as one program.
bool supportsBatchedMatmul(BatchedMatmulShape shape);

/// Packs a constant weight for a covered packed geometry, byte-exactly as
/// the decoded sections carry: the template's kernel header, then the
/// weight matrix — transposed for transpose_y forms — cut into packCols
/// chunks placed at the padded row stride with a -64 phase, final 64
/// source halves unwritten. `weights` is the dense row-major
/// [B, packRows, packCols] fp16 blob exactly as the MIL resolves it.
std::vector<std::uint8_t> packBatchedWeights(BatchedMatmulShape shape,
                                             const std::uint8_t *weights,
                                             std::size_t weightBytes);

/// Encodes Apple's own batched task stream for the geometry. `packed` is
/// the packBatchedWeights output for a constant second operand and null
/// for a runtime one.

/// One decoded boolean-op geometry: the compare/floor/select/floor_div
/// families with their fixed task streams. `channels/height/width` is the
/// CHW surface; `constInput` selects the captured constant-operand twin
/// (blob x for floor, the scalar-2.0 divisor for floor_div). The element
/// dtype follows the family: less emits a bool result and select reads a
/// bool cond — 1-byte surfaces — while every operand stays fp16.
enum class H13BooleanKind : std::uint8_t { Less, Floor, Select, FloorDiv };

struct H13BooleanShape {
    H13BooleanKind kind = H13BooleanKind::Floor;
    bool constInput = false;
    std::uint32_t channels = 1;
    std::uint32_t height = 1;
    std::uint32_t width = 1;
};

/// True when the decoded boolean corpus covers this geometry.
bool supportsBooleanOp(H13BooleanShape shape);

/// Encodes the captured task stream verbatim with its constant section.
/// scalarInput carries the single fp16 lane of a constant-operand twin
/// (floor over a blob x); it replaces the section's head word and must be
/// null for runtime-operand forms.
ane::h13::Program encodeBooleanOp(H13BooleanShape shape,
                                  const std::uint8_t *scalarInput = nullptr,
                                  std::size_t scalarBytes = 0);

Program encodeBatchedMatmul(BatchedMatmulShape shape,
                            const std::uint8_t *packed,
                            std::size_t packedBytes);
/// Apple's constant-section permutation for a [columns, reduction] fp16
/// weight, whose row-group size depends on `rows` and `reduction`.
std::vector<std::uint8_t> packMatvecWeights(MatmulShape shape,
                                            const std::uint8_t *weights,
                                            std::size_t weightBytes);

/// How a rank-3 linear's bias rides: absent or all-equal halves fold into a
/// scalar register (the captured three-task form); distinct halves ride as
/// per-column bias groups appended to the section.
enum class LinearBiasMode : std::uint8_t { None, Uniform, Block };

/// True when the decoded encoder corpus covers this rank-3 linear geometry
/// (x [1, rows, reduction], weight [columns, reduction], bias [columns]).
bool supportsLinearParity(std::uint32_t rows, std::uint32_t reduction,
                          std::uint32_t columns, LinearBiasMode biasMode);

/// Encodes Apple's own linear task stream. `weights` is the row-major
/// [columns, reduction] fp16 constant; `bias` is the [columns] fp16 constant
/// and must be null for LinearBiasMode::None. Block modes append the
/// captured per-column bias tiles; the others take no section beyond the
/// packed weights. For LinearBiasMode::Uniform the decoded task stream
/// carries the captured uniform bias as a scalar-register immediate
/// (originally 0x3401 from the oracle captures); pass `uniformBiasHalves`
/// to stamp the real value instead — 0x0000 if the model has no bias.
Program encodeLinearParity(std::uint32_t rows, std::uint32_t reduction,
                           std::uint32_t columns, LinearBiasMode biasMode,
                           const std::uint8_t *weights,
                           std::size_t weightBytes, const std::uint8_t *bias,
                           std::size_t biasBytes,
                           std::uint16_t uniformBiasHalves = 0x3401);

/// True when the decoded corpus covers the d1024 s375 FFN chain
/// (matmul → bias add → silu → matmul → bias add) as one program.
bool supportsFFNChain(std::uint32_t rows, std::uint32_t inner1,
                      std::uint32_t middle, std::uint32_t inner2,
                      std::uint32_t columns);

/// Encodes the captured 28-task chain stream. The four constants are the
/// row-major fp16 blobs the MIL resolves to: w1 [middle, inner1],
/// b1 [middle], w2 [columns, middle], b2 [columns].
Program encodeFFNChain(std::uint32_t rows, std::uint32_t inner1,
                       std::uint32_t middle, std::uint32_t inner2,
                       std::uint32_t columns, const std::uint8_t *weights1,
                       const std::uint8_t *bias1, const std::uint8_t *weights2,
                       const std::uint8_t *bias2);
/// True when the decoded Apple corpus covers this broadcast as one program.
bool supportsBroadcast(BinaryOperation operation, BroadcastOperand operand,
                       BroadcastShape shape);
/// Encodes Apple's broadcast task stream. `constant` holds one fp16 value per
/// channel for `BroadcastOperand::Constant` and is null otherwise;
/// `scalarBits` carries the inline fp16 operand for `BroadcastOperand::Scalar`.
Program encodeBroadcast(BinaryOperation operation, BroadcastOperand operand,
                        BroadcastShape shape,
                        const std::uint8_t *constant = nullptr,
                        std::size_t constantBytes = 0,
                        std::uint16_t scalarBits = 0x3800);
/// True when the decoded Apple corpus covers this softmax, layer_norm, or
/// reduction geometry as one multi-task program.
bool supportsNormParity(NormOperation operation, NormShape shape);
/// Encodes Apple's own task stream for the geometry, with the LUT constant
/// section the decoded oracle carries.
Program encodeNormParity(NormOperation operation, NormShape shape);
/// True when the decoded Apple corpus covers this convolution as one program.
bool supportsConvParity(ConvShape shape);
/// Apple's constant-section layout for a convolution weight. `weights` is the
/// MIL `[Cout, Cin / groups, kh, kw]` row-major fp16 constant, exactly the
/// bytes the blob resolves to, and `bias` holds one fp16 value per output
/// channel or is null.
std::vector<std::uint8_t> packConvWeights(ConvShape shape,
                                          const std::uint8_t *weights,
                                          std::size_t weightBytes,
                                          const std::uint8_t *bias = nullptr,
                                          std::size_t biasBytes = 0);
/// Why a convolution has no generic (formula-built) encoding, or empty when
/// `encodeConvGeneric` accepts it. The generic encoder covers dense (groups
/// 1) and depthwise convolutions with a square 1x1 or 3x3 kernel, unit
/// dilation, `same` or `valid` padding, stride 1 (and stride 2 for 1x1),
/// channel counts that are multiples of 16, and any spatial extent whose
/// rows fit the L2 tiling budget.
std::string convGenericRefusal(ConvShape shape);
/// Builds a convolution task stream from the decoded per-register formulas
/// (see docs/ane/fused-conv.md) instead of a captured template: the closest
/// decoded template of the same family supplies the invariant words and every
/// shape-dependent register -- surfaces, L2 strides and result base, tile
/// height, channel counts, kernel-DMA chunk table -- is recomputed. Weights
/// over the per-task kernel budget split into independent output-channel
/// tasks in one program.
Program encodeConvGeneric(ConvShape shape, const std::uint8_t *weights,
                          std::size_t weightBytes,
                          const std::uint8_t *bias = nullptr,
                          std::size_t biasBytes = 0);
/// The constant-section size the decoded template for this geometry records.
std::size_t convParitySectionBytes(ConvShape shape);
/// Encodes Apple's own convolution task stream for the geometry.
Program encodeConvParity(ConvShape shape, const std::uint8_t *weights,
                         std::size_t weightBytes,
                         const std::uint8_t *bias = nullptr,
                         std::size_t biasBytes = 0);
/// A decoded matmul or elementwise epilogue Apple folds into the producer's
/// own task instead of emitting the consumer as a separate program.
enum class PostOperation : std::uint8_t { Relu };

/// Fuses a decoded post-operation into an Apple-parity matmul program
/// (two tasks, constant weight). Verified byte-exact against the
/// `chain_pair_mm_relu` oracles: the compute task's NE word `0x0c804`
/// carries the clamp bit `0x00010000` and no other word moves.
/// Throws unless the program is exactly the decoded no-post-op form.
void fuseMatmulPostOperation(Program &program, PostOperation operation);

/// Fuses a relu into a convolution program's NE epilogue. The NE MAC
/// configuration word `0x0c804` carries the same clamp bit `0x00010000`
/// the decoded `chain_pair_mm_relu` matmul epilogue sets; every conv task
/// writes that register, so a multi-task convolution gets the bit in each
/// task. Throws unless every task is a decoded no-post-op form
/// (`0x00101c00`, or `0x00101c10` with a per-channel bias).
void fuseConvPostOperation(Program &program, PostOperation operation);

/// Fuses a relu into a one-task Apple elementwise program. Verified byte-exact
/// against `chain_add_relu_c512` on both targets: the PE word `0x08800`
/// (H14 `0x900`) carries the clamp bit `0x20` and no other word moves.
/// Throws unless the PE word is the decoded no-post-op form `0x00080000`.
void fuseElementwisePostOperation(Program &program, PostOperation operation);

/// Relinks already-routed programs into one task stream. Chains are accepted
/// only through the fusion path above, so this composes a single program
/// unchanged and refuses a multi-program list: relinking standalone task
/// streams keeps every task's own surface routing, and the decoded corpus
/// provides no task-DMA addressing for declared intermediate surfaces (reads
/// are wired to the input channels, writes to the output channel or L2), so
/// any multi-program relink would emit tasks that never exchange data.
Program composePrograms(const std::vector<Program> &programs);
std::vector<std::uint8_t> encodeANEC(const Program &program);

/// One decoded tile geometry: the input CHW surface replicated repC/repH/repW
/// times into the output surface. Apple's form is materialized — input and
/// output are distinct bindings, and the const-x twin carries the
/// unexpanded payload in the constant section.
struct H13TileShape {
    std::uint32_t inChannels = 1;
    std::uint32_t inHeight = 1;
    std::uint32_t inWidth = 1;
    std::uint32_t repChannels = 1;
    std::uint32_t repHeight = 1;
    std::uint32_t repWidth = 1;
    bool runtimeInput = true;
};

/// True when the decoded tile corpus covers this geometry.
bool supportsTileOp(H13TileShape shape);

/// Encodes the captured task stream with its constant section.
ane::h13::Program encodeTileOp(H13TileShape shape);

}

