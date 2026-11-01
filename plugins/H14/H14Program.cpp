#include "H14Program.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ane::h14 {
namespace {

enum class ElementwiseKind : std::uint8_t { BinaryRuntime, BinaryScalar, Unary };

/// One nonzero fp16 halfword run of a decoded constant section.
struct ConstantRun {
    std::uint32_t index;
    std::uint16_t bits;
    std::uint32_t count;
};

struct OracleTaskTemplate {
    ElementwiseKind kind;
    std::uint8_t operation;
    ElementwiseShape shape;
    /// The second runtime operand's surface, which the decoded broadcast
    /// forms let differ from the result surface; equal to `shape` otherwise.
    ElementwiseShape operand;
    const std::uint32_t *text;
    std::size_t textWords;
    std::uint32_t taskCount;
    const ConstantRun *constants;
    std::size_t constantRuns;
    std::size_t constantBytes;
    std::uint32_t programRecordCount;
    std::uint32_t unresolvedDescriptorWord;
    /// H17 and H18 keep the 128-byte kernel table (LUT and scalar operands)
    /// that H14 keeps in its constant section in a __KERN_0 segment instead.
    const ConstantRun *kernel = nullptr;
    std::size_t kernelRuns = 0;
    std::size_t kernelBytes = 0;
};

/// One decoded Apple program for the scaled exp chain: exp, mul, add, mul, mul as one MIL program, three tasks. The constant section is the
/// exponential table; the scalar constants of the MIL sit in the task words.
struct OracleExpChainTemplate {
    std::uint32_t channels;
    const std::uint32_t *text;
    std::size_t textWords;
    std::uint32_t taskCount;
    std::size_t constantBytes;
    std::uint32_t programRecordCount;
    std::uint32_t unresolvedDescriptorWord;
    std::uint32_t scratchDescriptorWord;
};

struct OracleMatvecTemplate {
    std::uint32_t rows;
    std::uint32_t reduction;
    std::uint32_t columns;
    const std::uint32_t *text;
    std::size_t textWords;
    std::uint32_t taskCount;
    std::uint32_t programRecordCount;
    std::uint32_t unresolvedDescriptorWord;
};

/// Which LUT layout a decoded normalization program's constant section holds.
/// The generator derives this from the recorded section hash, so the encoder
/// rebuilds the same bytes instead of guessing them from the operation.
enum class NormConstants : std::uint8_t {
    Zero,
    Exponential,
    ExponentialReciprocal,
};

struct OracleNormTemplate {
    NormOperation operation;
    ElementwiseShape input;
    ElementwiseShape output;
    std::uint32_t axisMask;
    bool keepDims;
    NormConstants constants;
    const std::uint32_t *text;
    std::size_t textWords;
    std::uint32_t taskCount;
    std::size_t constantBytes;
    std::uint32_t programRecordCount;
    std::uint32_t unresolvedDescriptorWord;
    std::uint32_t scratchDescriptorWord;
};


/// One decoded Apple convolution program, keyed by everything the compiler
/// knows before it picks one: the kernel, stride, group count, whether a bias
/// is folded in, and the two CHW surfaces.
struct OracleConvTemplate {
    std::uint32_t kernel;
    std::uint32_t kernelWidth;
    std::uint32_t stride;
    std::uint32_t groups;
    bool bias;
    ElementwiseShape input;
    ElementwiseShape output;
    const std::uint32_t *text;
    std::size_t textWords;
    std::uint32_t taskCount;
    std::size_t constantBytes;
    std::uint32_t programRecordCount;
    std::uint32_t unresolvedDescriptorWord;
    std::uint32_t scratchDescriptorWord;
    bool nativeLayer;
};

#include "H14ElementwiseTemplates.inc"
#include "H17ElementwiseTemplates.inc"
#include "H18ElementwiseTemplates.inc"
#include "H14MatvecTemplates.inc"
#include "H14ExpChainTemplates.inc"
#include "H14NormTemplates.inc"
#include "H14ConvTemplates.inc"
// H14's exponential and reciprocal sections are the H13 tables byte-for-byte:
// research/mint_h14_norm_probes.py resolves every decoded H14 section against
// these words by SHA-256 before it emits a NormConstants kind.
#include "H13ElementwiseConstants.inc"

bool sameShape(ElementwiseShape left, ElementwiseShape right) {
    return left.channels == right.channels && left.height == right.height &&
           left.width == right.width;
}

const char *generationName(Generation generation) {
    switch (generation) {
        case Generation::H14: return "H14";
        case Generation::H17: return "H17";
        case Generation::H18: return "H18";
    }
    throw std::logic_error("unknown parity generation");
}

template <std::size_t N>
const OracleTaskTemplate *findElementwise(const OracleTaskTemplate (&table)[N],
                                          ElementwiseKind kind,
                                          std::uint8_t operation,
                                          ElementwiseShape shape,
                                          ElementwiseShape operand) {
    for (const auto &candidate : table)
        if (candidate.kind == kind && candidate.operation == operation &&
            sameShape(candidate.shape, shape) &&
            sameShape(candidate.operand, operand)) return &candidate;
    return nullptr;
}

const OracleTaskTemplate *elementwiseTemplate(Generation generation,
                                              ElementwiseKind kind,
                                              std::uint8_t operation,
                                              ElementwiseShape shape,
                                              ElementwiseShape operand) {
    switch (generation) {
        case Generation::H14:
            return findElementwise(kH14ElementwiseTasks, kind, operation, shape, operand);
        case Generation::H17:
            return findElementwise(kH17ElementwiseTasks, kind, operation, shape, operand);
        case Generation::H18:
            return findElementwise(kH18ElementwiseTasks, kind, operation, shape, operand);
    }
    throw std::logic_error("unknown parity generation");
}

const OracleExpChainTemplate *expChainTemplate(std::uint32_t channels) {
    for (const auto &candidate : kExpChainTasks)
        if (candidate.channels == channels) return &candidate;
    return nullptr;
}

const OracleMatvecTemplate *matvecTemplate(MatvecShape shape) {
    for (const auto &candidate : kMatvecTasks)
        if (candidate.rows == shape.rows &&
            candidate.reduction == shape.reduction &&
            candidate.columns == shape.columns) return &candidate;
    return nullptr;
}

const OracleNormTemplate *normTemplate(NormOperation operation,
                                       NormShape shape) {
    for (const auto &candidate : kH14NormTasks)
        if (candidate.operation == operation &&
            candidate.axisMask == shape.axisMask &&
            candidate.keepDims == shape.keepDims &&
            sameShape(candidate.input, shape.input) &&
            sameShape(candidate.output, shape.output)) return &candidate;
    return nullptr;
}

template <std::size_t N>
void putWords(std::vector<std::uint8_t> &bytes, std::size_t offset,
              const std::uint32_t (&words)[N]) {
    if (offset + N * sizeof(std::uint32_t) > bytes.size())
        throw std::logic_error("H14 constant table exceeds its decoded section");
    for (std::size_t index = 0; index != N; ++index)
        for (std::size_t byte = 0; byte != sizeof(std::uint32_t); ++byte)
            bytes[offset + index * sizeof(std::uint32_t) + byte] =
                static_cast<std::uint8_t>(words[index] >> (byte * 8));
}

/// Apple's H14 softmax sections hold the exponential table at offset 0 and,
/// when the reduced axis is not the last one, the reciprocal table in the
/// final 128 bytes; layer_norm and every reduction leave the section zero.
std::vector<std::uint8_t> normConstants(NormConstants kind, std::size_t size) {
    std::vector<std::uint8_t> bytes(size, 0);
    if (kind == NormConstants::Zero) return bytes;
    putWords(bytes, 0, kExpKERNWords);
    if (kind == NormConstants::ExponentialReciprocal)
        putWords(bytes, size - sizeof(kRecipKERNWords), kRecipKERNWords);
    return bytes;
}

std::vector<std::uint8_t> streamBytes(const std::uint32_t *text,
                                      std::size_t textWords) {
    std::vector<std::uint8_t> bytes(textWords * sizeof(std::uint32_t));
    for (std::size_t index = 0; index != textWords; ++index) {
        const auto value = text[index];
        for (std::size_t byte = 0; byte != sizeof(value); ++byte)
            bytes[index * sizeof(value) + byte] =
                static_cast<std::uint8_t>(value >> (byte * 8));
    }
    return bytes;
}

std::vector<std::uint8_t> runBytes(const ConstantRun *runs, std::size_t count,
                                   std::size_t size) {
    std::vector<std::uint8_t> bytes(size, 0);
    for (std::size_t run = 0; run != count; ++run) {
        const auto &entry = runs[run];
        for (std::uint32_t offset = 0; offset != entry.count; ++offset) {
            const std::size_t index = (entry.index + offset) * 2;
            if (index + 1 >= bytes.size())
                throw std::logic_error("H14 constant run leaves its section");
            bytes[index] = static_cast<std::uint8_t>(entry.bits);
            bytes[index + 1] = static_cast<std::uint8_t>(entry.bits >> 8);
        }
    }
    return bytes;
}

std::uint64_t alignTile(std::uint64_t value) {
    return (value + tileBytes - 1) & ~(static_cast<std::uint64_t>(tileBytes) - 1);
}

TensorLayout elementwiseTensor(std::uint32_t index, ElementwiseShape shape) {
    const std::uint64_t row = std::max<std::uint64_t>(64, shape.width * 2);
    const std::uint64_t plane = row * shape.height;
    const std::uint64_t bytes = plane * shape.channels;
    return {index, {1, shape.channels, shape.height, shape.width, plane, row},
            alignTile(bytes)};
}

/// Apple's matvec surface: one dense [1, 1, rows, width] fp16 plane whose row
/// stride is the logical row, without the 64-byte elementwise row padding.
TensorLayout matvecTensor(std::uint32_t index, std::uint32_t rows,
                          std::uint32_t width) {
    const std::uint64_t row = static_cast<std::uint64_t>(width) * 2;
    const std::uint64_t plane = row * rows;
    return {index, {1, 1, rows, width, plane, row}, alignTile(plane)};
}

/// The task stream and descriptor metadata every oracle template carries; the
/// caller supplies the constants and the surfaces.
Program streamProgram(const std::uint32_t *text, std::size_t textWords,
                      std::uint32_t taskCount,
                      std::uint32_t programRecordCount,
                      std::uint32_t unresolvedDescriptorWord) {
    Program program;
    program.taskStream = streamBytes(text, textWords);
    const auto sizes = taskSizes(program.taskStream);
    if (sizes.size() != taskCount)
        throw std::logic_error("H14 task stream task count differs from the oracle");
    program.firstTaskBytes = sizes.front();
    program.taskCount = taskCount;
    program.constantOffsetBytes =
        (program.taskStream.size() + 0x3f) & ~std::size_t(0x3f);
    program.programRecordCount = programRecordCount;
    program.unresolvedDescriptorWord = unresolvedDescriptorWord;
    return program;
}

Program oracleProgram(const OracleTaskTemplate &source, std::size_t inputCount) {
    if (inputCount != 1 && inputCount != 2)
        throw std::logic_error("H14 elementwise programs take one or two inputs");
    Program program = streamProgram(source.text, source.textWords,
                                    source.taskCount, source.programRecordCount,
                                    source.unresolvedDescriptorWord);
    program.constants = runBytes(source.constants, source.constantRuns,
                                 source.constantBytes);
    program.kernelTable = runBytes(source.kernel, source.kernelRuns,
                                   source.kernelBytes);
    program.inputs.reserve(inputCount);
    program.inputs.push_back(elementwiseTensor(5, source.shape));
    if (inputCount == 2)
        program.inputs.push_back(elementwiseTensor(6, source.operand));
    program.output = elementwiseTensor(4, source.shape);
    return program;
}

std::uint32_t loadLE32(const std::vector<std::uint8_t> &bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index != 4; ++index)
        value |= static_cast<std::uint32_t>(bytes[offset + index]) << (index * 8);
    return value;
}

void storeLE32(std::vector<std::uint8_t> &bytes, std::size_t offset,
               std::uint32_t value) {
    for (std::size_t index = 0; index != 4; ++index)
        bytes[offset + index] = static_cast<std::uint8_t>(value >> (index * 8));
}

/// Byte offset of one 32-bit register word inside one H14 task image: eight
/// header words, one extra word when header[7] bit 1 is set, then register
/// records. A dense record (bit 31 clear) holds a base word index in bits
/// 0:14 and a word count minus one in bits 15:20. A scatter record (bit 31
/// set) holds the same base and a 16-bit following-word mask in bits 15:30;
/// its payload is the base word followed by every masked word, ascending.
std::size_t h14RegisterOffset(const std::vector<std::uint8_t> &stream,
                              std::size_t offset, std::size_t bytes,
                              std::uint32_t address) {
    if (offset + 32 > stream.size() || bytes < 32 || offset + bytes > stream.size())
        throw std::invalid_argument("h14.chain-unrepresentable-edge: task image is truncated");
    if (address % sizeof(std::uint32_t))
        throw std::invalid_argument("h14.chain-unrepresentable-edge: register address is not word-aligned");
    const std::size_t extra =
        (loadLE32(stream, offset + 28) & 3) == 3 ? sizeof(std::uint32_t) : 0;
    std::size_t cursor = offset + 32 + extra;
    const std::size_t end = offset + bytes;
    while (cursor + sizeof(std::uint32_t) <= end) {
        const auto header = loadLE32(stream, cursor);
        const auto base = header & 0x7fff;
        if (header & 0x80000000u) {
            const auto mask = static_cast<std::uint16_t>((header >> 15) & 0xffff);
            if (address == base * sizeof(std::uint32_t))
                return cursor + sizeof(std::uint32_t);
            if (address > base * sizeof(std::uint32_t)) {
                const auto bit = address / sizeof(std::uint32_t) - base - 1;
                if (bit < 16 && (mask >> bit) & 1) {
                    std::size_t preceding = 1;
                    for (std::size_t lower = 0; lower != bit; ++lower)
                        preceding += (mask >> lower) & 1;
                    return cursor + sizeof(std::uint32_t) +
                           preceding * sizeof(std::uint32_t);
                }
            }
            std::size_t payload = 1;
            for (std::size_t bit = 0; bit != 16; ++bit) payload += (mask >> bit) & 1;
            cursor += sizeof(std::uint32_t) + payload * sizeof(std::uint32_t);
        } else {
            const auto count = ((header >> 15) & 0x3f) + 1;
            if (address >= base * sizeof(std::uint32_t) &&
                address < (base + count) * sizeof(std::uint32_t))
                return cursor + sizeof(std::uint32_t) + address -
                       base * sizeof(std::uint32_t);
            cursor += sizeof(std::uint32_t) + count * sizeof(std::uint32_t);
        }
    }
    throw std::invalid_argument(
        "h14.chain-unrepresentable-edge: task image does not carry the decoded "
        "register record");
}

/// Byte span of one task inside the stream: H14 pads every task to 16 bytes
/// and starts the stream with a zero-size prefix frame.
std::pair<std::size_t, std::size_t> h14TaskSpan(
    const std::vector<std::uint8_t> &stream, std::size_t index) {
    std::size_t offset = 0, seen = 0;
    while (offset != stream.size()) {
        const auto words = (loadLE32(stream, offset) >> 16) & 0x7ff;
        if (!words) {
            offset = std::min(offset + taskAlignment, stream.size());
            continue;
        }
        const std::size_t bytes = words * sizeof(std::uint32_t);
        if (seen == index) return {offset, bytes};
        ++seen;
        offset = std::min((offset + bytes + taskAlignment - 1) &
                              ~(taskAlignment - 1), stream.size());
    }
    throw std::invalid_argument("h14.chain-unrepresentable-edge: task index is out of range");
}

void appendLE(std::vector<std::uint8_t> &bytes, std::uint64_t value,
              std::size_t width) {
    for (std::size_t index = 0; index != width; ++index)
        bytes.push_back(static_cast<std::uint8_t>(value >> (index * 8)));
}

std::uint64_t checkedAdd(std::uint64_t left, std::uint64_t right,
                         const char *what) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::invalid_argument(what);
    return left + right;
}

std::uint64_t checkedMultiply(std::uint64_t left, std::uint64_t right,
                              const char *what) {
    if (left && right > std::numeric_limits<std::uint64_t>::max() / left)
        throw std::invalid_argument(what);
    return left * right;
}

std::uint32_t tileCount(std::uint64_t bytes, const char *what) {
    if (!bytes) throw std::invalid_argument(what);
    const auto tiles = checkedAdd(bytes, tileBytes - 1, what) / tileBytes;
    if (tiles > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(what);
    return static_cast<std::uint32_t>(tiles);
}

void validateTensor(const TensorLayout &tensor, std::uint32_t expectedIndex,
                    const char *kind) {
    if (tensor.index != expectedIndex)
        throw std::invalid_argument("unsupported ANEC channel index");
    const auto &nchw = tensor.nchw;
    if (!nchw[0] || !nchw[1] || !nchw[2] || !nchw[3] || !nchw[4] || !nchw[5])
        throw std::invalid_argument("tensor layout has a zero dimension or stride");
    if ((nchw[5] % 2) || (nchw[4] % nchw[5]) || nchw[5] / 2 < nchw[3])
        throw std::invalid_argument("unsupported tensor tiling layout");
    const auto minimumPlane = checkedMultiply(nchw[2], nchw[5],
                                              "tensor row span overflows");
    if (nchw[4] < minimumPlane)
        throw std::invalid_argument("unsupported tensor plane stride");
    const auto physicalSpan = checkedMultiply(
        checkedMultiply(nchw[0], nchw[1], "tensor channel span overflows"),
        nchw[4], "tensor physical span overflows");
    if (tensor.allocationBytes < physicalSpan)
        throw std::invalid_argument(kind);
}

constexpr std::size_t headerBytes = 0x1000;
constexpr std::size_t headerFieldsBytes = 0x6a8;
constexpr std::size_t channelCount = 32;
constexpr std::size_t tensorFields = 6;

} // namespace

std::vector<std::size_t> taskSizes(const std::vector<std::uint8_t> &stream) {
    if (stream.empty() || stream.size() % sizeof(std::uint32_t))
        throw std::invalid_argument("H14 task stream must be nonempty and word-aligned");
    std::vector<std::size_t> sizes;
    std::size_t offset = 0;
    while (offset != stream.size()) {
        if (stream.size() - offset < sizeof(std::uint32_t))
            throw std::invalid_argument("H14 task stream ends inside a task header");
        const auto words = (loadLE32(stream, offset) >> 16) & 0x7ff;
        if (!words) {
            // A zero-size frame consumes one 16-byte slot, as Apple's prefix does.
            const auto next = std::min(offset + taskAlignment, stream.size());
            for (std::size_t index = offset; index != next; ++index)
                if (stream[index])
                    throw std::invalid_argument("H14 zero-size task frame is nonzero");
            offset = next;
            continue;
        }
        if (words < taskHeaderWords)
            throw std::invalid_argument("H14 task declares fewer words than its header");
        const std::size_t bytes = words * sizeof(std::uint32_t);
        if (bytes > stream.size() - offset)
            throw std::invalid_argument("H14 task extends beyond its task stream");
        sizes.push_back(bytes);
        const auto next = std::min((offset + bytes + taskAlignment - 1) &
                                       ~(taskAlignment - 1), stream.size());
        for (std::size_t index = offset + bytes; index != next; ++index)
            if (stream[index])
                throw std::invalid_argument("H14 task alignment padding is nonzero");
        offset = next;
    }
    if (sizes.empty())
        throw std::invalid_argument("H14 task stream carries no tasks");
    return sizes;
}

bool supportsElementwise(Generation generation, BinaryOperation operation,
                         ElementwiseShape shape, ElementwiseShape operand,
                         bool scalarConstant) {
    const auto kind = scalarConstant ? ElementwiseKind::BinaryScalar
                                     : ElementwiseKind::BinaryRuntime;
    return elementwiseTemplate(generation, kind,
                               static_cast<std::uint8_t>(operation), shape,
                               operand);
}

bool supportsElementwise(Generation generation, UnaryOperation operation,
                         ElementwiseShape shape) {
    return elementwiseTemplate(generation, ElementwiseKind::Unary,
                               static_cast<std::uint8_t>(operation), shape, shape);
}

Program encodeElementwise(Generation generation, BinaryOperation operation,
                          ElementwiseShape shape, ElementwiseShape operand,
                          bool scalarConstant, std::uint16_t scalarBits) {
    const auto kind = scalarConstant ? ElementwiseKind::BinaryScalar
                                     : ElementwiseKind::BinaryRuntime;
    const auto *source = elementwiseTemplate(
        generation, kind, static_cast<std::uint8_t>(operation), shape, operand);
    if (!source)
        throw std::invalid_argument(std::string(generationName(generation)) +
            " binary operation is outside the decoded parity envelope");
    if (scalarConstant && scalarBits != 0x3800)
        throw std::invalid_argument(std::string(generationName(generation)) +
            " scalar operation requires the decoded fp16 0.5 operand");
    return oracleProgram(*source, scalarConstant ? 1 : 2);
}

Program encodeElementwise(Generation generation, UnaryOperation operation,
                          ElementwiseShape shape) {
    const auto *source = elementwiseTemplate(
        generation, ElementwiseKind::Unary, static_cast<std::uint8_t>(operation),
        shape, shape);
    if (!source)
        throw std::invalid_argument(std::string(generationName(generation)) +
            " unary operation is outside the decoded parity envelope");
    return oracleProgram(*source, 1);
}

bool supportsMatvecParity(MatvecShape shape) {
    return matvecTemplate(shape);
}

std::vector<std::uint8_t> packMatvecWeights(MatvecShape shape,
                                            const std::uint8_t *weights,
                                            std::size_t weightBytes) {
    if (!weights)
        throw std::invalid_argument("H14 matvec weights must not be null");
    if (!shape.reduction || shape.columns < 16 || shape.columns % 16 ||
        (shape.columns > 256 && shape.columns % 256))
        throw std::invalid_argument(
            "H14 matvec packing needs a positive reduction and 16 columns per "
            "row group, 256 per plane group above 256 columns");
    if (weightBytes !=
        static_cast<std::size_t>(shape.reduction) * shape.columns * 2)
        throw std::invalid_argument(
            "H14 matvec requires columns * reduction fp16 weights");
    // Apple interleaves `group` weight rows at halfword granularity, packs
    // each group of rows as one contiguous reduction-major plane, and orders
    // the planes by the low four bits of the group index. Identical to the
    // H13 permutation: research/mint_h14_matvec_probes.py rebuilds all 125
    // known-weight H14 probe sections byte-for-byte from this rule.
    const std::uint32_t group = std::min<std::uint32_t>(16, shape.columns / 16);
    const std::uint32_t groups = shape.columns / group;
    std::vector<std::uint8_t> packed(weightBytes);
    for (std::uint32_t column = 0; column != shape.columns; ++column) {
        const std::uint32_t plane = column / group;
        const std::uint32_t destinationPlane =
            (plane % 16) * (groups / 16) + plane / 16;
        std::size_t destination =
            (static_cast<std::size_t>(destinationPlane) * shape.reduction *
                 group + column % group) * 2;
        std::size_t source = static_cast<std::size_t>(column) * shape.reduction * 2;
        for (std::uint32_t index = 0; index != shape.reduction; ++index) {
            packed[destination] = weights[source];
            packed[destination + 1] = weights[source + 1];
            destination += static_cast<std::size_t>(group) * 2;
            source += 2;
        }
    }
    return packed;
}

Program encodeMatvecParity(MatvecShape shape, const std::uint8_t *weights,
                           std::size_t weightBytes) {
    const auto *source = matvecTemplate(shape);
    if (!source)
        throw std::invalid_argument(
            "H14 matmul geometry is outside the decoded parity envelope");
    Program program = streamProgram(source->text, source->textWords,
                                    source->taskCount,
                                    source->programRecordCount,
                                    source->unresolvedDescriptorWord);
    program.constants = packMatvecWeights(shape, weights, weightBytes);
    program.inputs = {matvecTensor(5, shape.rows, shape.reduction)};
    program.output = matvecTensor(4, shape.rows, shape.columns);
    return program;
}

bool supportsExpScaledChainParity(std::uint32_t channels) {
    return expChainTemplate(channels);
}

Program encodeExpScaledChainParity(std::uint32_t channels) {
    const auto *source = expChainTemplate(channels);
    if (!source)
        throw std::invalid_argument(
            "H14 scaled exp chain is outside the decoded parity envelope");
    Program program = streamProgram(source->text, source->textWords,
                                    source->taskCount,
                                    source->programRecordCount,
                                    source->unresolvedDescriptorWord);
    program.constants = normConstants(NormConstants::Exponential, source->constantBytes);
    program.scratchDescriptorWord = source->scratchDescriptorWord;
    const ElementwiseShape surface{source->channels, 1, 1};
    program.inputs = {elementwiseTensor(5, surface)};
    program.output = elementwiseTensor(4, surface);
    return program;
}

bool supportsNormParity(NormOperation operation, NormShape shape) {
    return normTemplate(operation, shape);
}

Program encodeNormParity(NormOperation operation, NormShape shape) {
    const auto *source = normTemplate(operation, shape);
    if (!source)
        throw std::invalid_argument(
            "H14 normalization geometry is outside the decoded parity envelope");
    Program program = streamProgram(source->text, source->textWords,
                                    source->taskCount,
                                    source->programRecordCount,
                                    source->unresolvedDescriptorWord);
    program.constants = normConstants(source->constants, source->constantBytes);
    program.scratchDescriptorWord = source->scratchDescriptorWord;
    program.inputs = {elementwiseTensor(5, source->input)};
    program.output = elementwiseTensor(4, source->output);
    return program;
}

std::vector<std::uint8_t> encodeANEC(const Program &program) {
    const auto sizes = taskSizes(program.taskStream);
    if (sizes.size() != program.taskCount || sizes.front() != program.firstTaskBytes)
        throw std::invalid_argument("H14 ANEC task metadata does not match its stream");
    if (program.constantOffsetBytes < program.taskStream.size() ||
        program.constantOffsetBytes % 0x40)
        throw std::invalid_argument("H14 ANEC constant offset is invalid");
    if (program.inputs.empty() || program.inputs.size() > 2)
        throw std::invalid_argument("H14 ANEC requires one or two input tensors");

    validateTensor(program.output, 4, "output allocation does not cover its physical span");
    for (std::size_t index = 0; index != program.inputs.size(); ++index)
        validateTensor(program.inputs[index], static_cast<std::uint32_t>(5 + index),
                       "input allocation does not cover its physical span");

    const auto constantsSize = static_cast<std::uint64_t>(program.constants.size());
    const auto contentSize = checkedAdd(program.constantOffsetBytes, constantsSize,
                                        "ANEC content size overflows");
    if (contentSize > std::numeric_limits<std::size_t>::max() - headerBytes)
        throw std::invalid_argument("ANEC output size overflows");

    std::array<std::uint32_t, channelCount> tiles{};
    tiles[0] = tileCount(contentSize, "ANEC content tile count overflows");
    if (program.scratchAllocationBytes)
        tiles[3] = tileCount(program.scratchAllocationBytes, "scratch tile count overflows");
    tiles[4] = tileCount(program.output.allocationBytes,
                         "output tile count overflows");
    for (std::size_t index = 0; index != program.inputs.size(); ++index)
        tiles[5 + index] = tileCount(program.inputs[index].allocationBytes,
                                     "input tile count overflows");

    std::array<std::uint64_t, channelCount * tensorFields> layouts{};
    std::copy(program.output.nchw.begin(), program.output.nchw.end(),
              layouts.begin() + 4 * tensorFields);
    for (std::size_t index = 0; index != program.inputs.size(); ++index)
        std::copy(program.inputs[index].nchw.begin(),
                  program.inputs[index].nchw.end(),
                  layouts.begin() + (5 + index) * tensorFields);

    std::vector<std::uint8_t> anec;
    anec.reserve(headerBytes + static_cast<std::size_t>(contentSize));
    appendLE(anec, contentSize, 8);
    appendLE(anec, program.firstTaskBytes, 4);
    appendLE(anec, program.taskCount, 4);
    appendLE(anec, program.taskStream.size(), 8);
    appendLE(anec, constantsSize, 8);
    appendLE(anec, program.inputs.size(), 4);
    appendLE(anec, 1, 4);
    for (const auto tile : tiles) appendLE(anec, tile, 4);
    for (const auto field : layouts) appendLE(anec, field, 8);
    if (anec.size() != headerFieldsBytes)
        throw std::logic_error("ANEC header layout changed");
    anec.resize(headerBytes, 0);
    anec.insert(anec.end(), program.taskStream.begin(), program.taskStream.end());
    anec.resize(headerBytes + program.constantOffsetBytes, 0);
    anec.insert(anec.end(), program.constants.begin(), program.constants.end());
    return anec;
}

namespace {

const OracleConvTemplate *convTemplate(ConvShape shape) {
    for (const auto &candidate : kConvTasks)
        if (candidate.kernel == shape.kernel &&
            candidate.kernelWidth == shape.kernelWidth &&
            candidate.stride == shape.stride &&
            candidate.groups == shape.groups && candidate.bias == shape.bias &&
            sameShape(candidate.input, shape.input) &&
            sameShape(candidate.output, shape.output)) return &candidate;
    return nullptr;
}

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

/// Apple's convolution surface. The row is the width padded to 64 bytes, not
/// merely floored at it: a `pad_type="valid"` result 62 columns wide takes a
/// 128-byte row, which is what the decoded oracles record.
TensorLayout convTensor(std::uint32_t index, ElementwiseShape shape) {
    const std::uint64_t row =
        alignUp(static_cast<std::uint64_t>(shape.width) * 2, 64);
    const std::uint64_t plane = row * shape.height;
    const std::uint64_t element = plane * shape.channels;
    return {index, {1, shape.channels, shape.height, shape.width, plane, row},
            alignTile(element)};
}

/// How Apple splits the output channels into halfword interleave widths.
///
/// Every plane group holds 16 planes, so the channels are consumed in chunks
/// of `16 * lanes`, where `lanes` is the largest power of two at or below
/// `outputs / 16` and at most the cap: 32 for an input surface of 8 or fewer
/// pixels per side and 16 above it. 768 outputs at cap 32 is the case that
/// proves the split -- 512 channels interleave 32 wide, the remaining 256
/// interleave 16 wide -- and known-weight probes pin the cap's boundary
/// between 8 and 9 pixels.
std::vector<std::uint32_t> laneChunks(std::uint32_t outputs, std::uint32_t cap) {
    std::vector<std::uint32_t> chunks;
    std::uint32_t remaining = outputs;
    while (remaining) {
        std::uint32_t lanes = 1;
        if (remaining >= 16) {
            lanes = cap;
            while (lanes > remaining / 16) lanes /= 2;
        }
        chunks.push_back(lanes);
        remaining -= std::min(remaining, 16 * lanes);
    }
    return chunks;
}

std::uint32_t laneCap(ElementwiseShape input) {
    return std::max(input.height, input.width) <= 8 ? 32 : 16;
}

void putHalfword(std::vector<std::uint8_t> &bytes, std::size_t offset,
                 const std::uint8_t *source) {
    bytes[offset] = source[0];
    bytes[offset + 1] = source[1];
}

/// Native small-output packing (c3/c8/c24): distribute adjacent outputs
/// evenly over up to 16 cores, padding each core's interleaved plane.
std::vector<std::uint8_t> packConvSmallDense(std::uint32_t reduction,
    std::uint32_t outputs, const std::uint8_t *weights, const std::uint8_t *bias) {
    const auto cores = std::min(16u, outputs);
    const std::uint64_t rows = reduction + (bias ? 1 : 0);
    std::vector<std::uint8_t> packed;
    unsigned first = 0;
    for (unsigned core = 0; core != cores; ++core) {
        const unsigned lanes = outputs / cores + (core < outputs % cores);
        const auto base = packed.size();
        packed.resize(base + alignUp(rows * lanes * 2, 64), 0);
        for (unsigned lane = 0; lane != lanes; ++lane) {
            auto cursor = base + lane * 2;
            if (bias) {
                putHalfword(packed, cursor, bias + (first + lane) * 2);
                cursor += lanes * 2;
            }
            for (unsigned row = 0; row != reduction; ++row) {
                putHalfword(packed, cursor, weights + (static_cast<std::size_t>(first + lane) * reduction + row) * 2);
                cursor += lanes * 2;
            }
        }
        first += lanes;
    }
    return packed;
}

/// Apple's dense convolution section. A plane interleaves `lanes` output
/// channels at halfword granularity over `reduction` rows, preceded by one
/// bias row when the convolution has a bias; 16 plane groups follow, each
/// holding one plane per chunk. A grouped convolution pads every plane to 64
/// bytes and a groups-1 convolution pads only the plane group.
std::vector<std::uint8_t> packConvDense(std::uint32_t reduction,
                                        const std::vector<std::uint32_t> &chunks,
                                        const std::uint8_t *weights,
                                        const std::uint8_t *bias,
                                        bool planePadding) {
    const std::uint64_t rows = reduction + (bias ? 1 : 0);
    std::vector<std::uint64_t> planeBytes;
    planeBytes.reserve(chunks.size());
    for (const auto lanes : chunks) {
        const std::uint64_t span = rows * lanes * 2;
        planeBytes.push_back(planePadding ? alignUp(span, 64) : span);
    }
    std::uint64_t total = 0;
    for (const auto span : planeBytes) total += span;
    const std::uint64_t groupBytes = alignUp(total, 64);
    std::vector<std::uint8_t> packed(groupBytes * 16, 0);
    std::uint32_t start = 0;
    for (std::size_t chunk = 0; chunk != chunks.size(); ++chunk) {
        const std::uint32_t lanes = chunks[chunk];
        std::uint64_t offset = 0;
        for (std::size_t earlier = 0; earlier != chunk; ++earlier)
            offset += planeBytes[earlier];
        for (std::uint32_t group = 0; group != 16; ++group) {
            for (std::uint32_t lane = 0; lane != lanes; ++lane) {
                const std::uint32_t column = start + group * lanes + lane;
                std::size_t cursor = static_cast<std::size_t>(
                    group * groupBytes + offset + lane * 2);
                if (bias) {
                    putHalfword(packed, cursor, bias + column * 2);
                    cursor += lanes * 2;
                }
                const std::uint8_t *source =
                    weights + static_cast<std::size_t>(column) * reduction * 2;
                for (std::uint32_t row = 0; row != reduction; ++row) {
                    putHalfword(packed, cursor, source);
                    cursor += lanes * 2;
                    source += 2;
                }
            }
        }
        start += 16 * lanes;
    }
    return packed;
}

/// Apple's depthwise section: 16 lanes, each holding `outputs / 16` channels
/// back to back with no padding between them and the lane padded to 64 bytes.
/// A bias precedes each channel's taps.
std::vector<std::uint8_t> packConvDepthwise(std::uint32_t taps,
                                            std::uint32_t outputs,
                                            const std::uint8_t *weights,
                                            const std::uint8_t *bias) {
    const std::uint32_t lanes = std::min<std::uint32_t>(16, outputs);
    const std::uint64_t rows = taps + (bias ? 1 : 0);
    // A non-multiple of 16 has unequal core chunks: c24 gives two channels
    // to cores 0..7 and one to 8..15. Pad each core's actual span separately.
    std::vector<std::uint64_t> laneBases(lanes);
    std::uint64_t total = 0;
    for (std::uint32_t lane = 0; lane != lanes; ++lane) {
        laneBases[lane] = total;
        const auto slots = (outputs - lane + lanes - 1) / lanes;
        total += alignUp(slots * rows * 2, 64);
    }
    std::vector<std::uint8_t> packed(total, 0);
    for (std::uint32_t column = 0; column != outputs; ++column) {
        std::size_t cursor = static_cast<std::size_t>(
            laneBases[column % lanes] + (column / lanes) * rows * 2);
        if (bias) {
            putHalfword(packed, cursor, bias + column * 2);
            cursor += 2;
        }
        const std::uint8_t *source =
            weights + static_cast<std::size_t>(column) * taps * 2;
        for (std::uint32_t tap = 0; tap != taps; ++tap)
            putHalfword(packed, cursor + tap * 2, source + tap * 2);
    }
    return packed;
}

/// Apple's stride-2 section, which skips zero weights.
///
/// A plane holds the plane's `lanes` bias halfwords when the convolution has
/// a bias, a 16-bit count of the body bytes, one zero lead byte, then one row
/// per reduction step. A row carries, per group of eight lanes, a mask byte
/// whose bit `l` marks a lane whose weight is not zero, a zero byte, and the
/// marked lanes' halfwords, and closes with `lanes / 2 - 2` zero bytes -- the
/// H13 body with the lead byte hoisted and the mask byte padded, which counts
/// the same. Both signed zeros count as zero, so the section's size depends on
/// the weight values.
std::vector<std::uint8_t> packConvStrided(std::uint32_t reduction,
                                          const std::vector<std::uint32_t> &chunks,
                                          const std::uint8_t *weights,
                                          const std::uint8_t *bias) {
    std::vector<std::uint8_t> packed;
    std::uint32_t start = 0;
    for (const auto lanes : chunks) {
        const std::uint32_t padding = lanes >= 4 ? lanes / 2 - 2 : 0;
        for (std::uint32_t group = 0; group != 16; ++group) {
            const std::uint32_t first = start + group * lanes;
            std::vector<std::uint8_t> body;
            std::size_t count = 0;
            body.push_back(0);
            for (std::uint32_t row = 0; row != reduction; ++row) {
                ++count;
                for (std::uint32_t base = 0; base < lanes; base += 8) {
                    const std::uint32_t end = std::min(base + 8, lanes);
                    std::uint8_t mask = 0;
                    std::vector<std::uint8_t> values;
                    for (std::uint32_t lane = base; lane != end; ++lane) {
                        const std::uint8_t *value = weights +
                            (static_cast<std::size_t>(first + lane) * reduction +
                             row) * 2;
                        if (value[0] || (value[1] & 0x7f)) {
                            mask |= static_cast<std::uint8_t>(1u << (lane - base));
                            values.push_back(value[0]);
                            values.push_back(value[1]);
                        }
                    }
                    body.push_back(mask);
                    body.push_back(0);
                    count += 1 + values.size();
                    body.insert(body.end(), values.begin(), values.end());
                }
                body.insert(body.end(), padding, 0);
                count += padding;
            }
            if (count > 0xffff)
                throw std::invalid_argument(
                    "H14 strided convolution plane exceeds a 16-bit body count");
            std::vector<std::uint8_t> plane;
            if (bias)
                plane.insert(plane.end(), bias + first * 2,
                             bias + (first + lanes) * 2);
            plane.push_back(static_cast<std::uint8_t>(count));
            plane.push_back(static_cast<std::uint8_t>(count >> 8));
            plane.insert(plane.end(), body.begin(), body.end());
            plane.resize(alignUp(plane.size(), 64), 0);
            packed.insert(packed.end(), plane.begin(), plane.end());
        }
        start += 16 * lanes;
    }
    return packed;
}

} // namespace

bool supportsConvParity(ConvShape shape) { return convTemplate(shape); }

bool isNativeConv(ConvShape shape) {
    const auto *source = convTemplate(shape);
    return source && source->nativeLayer;
}

std::vector<std::uint8_t> packConvWeights(ConvShape shape,
                                          const std::uint8_t *weights,
                                          std::size_t weightBytes,
                                          const std::uint8_t *bias,
                                          std::size_t biasBytes) {
    const std::uint32_t taps = shape.kernel * shape.kernelWidth;
    if (!shape.groups || shape.input.channels % shape.groups ||
        shape.output.channels % shape.groups)
        throw std::invalid_argument(
            "H14 convolution groups must divide both channel counts");
    const std::uint32_t reduction = shape.input.channels / shape.groups * taps;
    if (!weights || weightBytes !=
            static_cast<std::size_t>(shape.output.channels) * reduction * 2)
        throw std::invalid_argument(
            "H14 convolution requires Cout * Cin / groups * kh * kw fp16 weights");
    if (shape.bias != (bias != nullptr) ||
        (bias && biasBytes != static_cast<std::size_t>(shape.output.channels) * 2))
        throw std::invalid_argument(
            "H14 convolution bias must hold one fp16 value per output channel");
    if (shape.groups == shape.input.channels &&
        shape.groups == shape.output.channels)
        return packConvDepthwise(taps, shape.output.channels, weights, bias);
    if (shape.kernel == 3 && shape.kernelWidth == 3 && shape.stride == 1 &&
        shape.groups == 1 && ((shape.input.channels == 128 && shape.output.channels == 512) ||
                             (shape.input.channels == 512 && shape.output.channels == 128)) &&
        shape.input.height == 35 && shape.input.width == 35 &&
        shape.output.height == 35 && shape.output.width == 35) {
        // The native 35x35 schedule writes the upper output slice first.
        // Indexed-weight hashes prove the two contiguous coefficient groups.
        std::vector<std::uint8_t> packed;
        const unsigned split = shape.output.channels == 512 ? 256 : 96;
        for (const unsigned first : {split, 0u}) {
            const unsigned count = first ? shape.output.channels - split : split;
            const auto part = packConvSmallDense(reduction, count,
                weights + static_cast<std::size_t>(first) * reduction * 2,
                bias ? bias + first * 2 : nullptr);
            packed.insert(packed.end(), part.begin(), part.end());
        }
        return packed;
    }
    if (shape.output.channels % 16) {
        if (shape.groups != 1 || shape.stride != 1)
            throw std::invalid_argument("small dense outputs are qualified only for stride 1, groups 1");
        return packConvSmallDense(reduction, shape.output.channels, weights, bias);
    }
    const auto chunks = laneChunks(shape.output.channels, laneCap(shape.input));
    if (shape.stride > 1) {
        if (shape.groups != 1 || taps != 1 ||
            *std::max_element(chunks.begin(), chunks.end()) > 8)
            throw std::invalid_argument(
                "H14 has no derived strided packing for grouped, multi-tap, or "
                "16-lane convolutions");
        return packConvStrided(reduction, chunks, weights, bias);
    }
    if (shape.groups > 1) {
        const std::uint32_t lanes =
            std::max<std::uint32_t>(1, shape.output.channels / 64);
        return packConvDense(reduction,
                             std::vector<std::uint32_t>(
                                 shape.output.channels / lanes / 16, lanes),
                             weights, bias, true);
    }
    return packConvDense(reduction, chunks, weights, bias, false);
}

Program encodeConvParity(ConvShape shape, const std::uint8_t *weights,
                         std::size_t weightBytes, const std::uint8_t *bias,
                         std::size_t biasBytes) {
    const auto *source = convTemplate(shape);
    if (!source)
        throw std::invalid_argument(
            "H14 convolution geometry is outside the decoded parity envelope");
    Program program = streamProgram(source->text, source->textWords,
                                    source->taskCount,
                                    source->programRecordCount,
                                    source->unresolvedDescriptorWord);
    program.constants =
        packConvWeights(shape, weights, weightBytes, bias, biasBytes);
    if (program.constants.size() != source->constantBytes)
        throw std::logic_error(
            "H14 packed convolution section differs from the decoded size");
    program.inputs = {convTensor(5, shape.input)};
    program.output = convTensor(4, shape.output);
    program.scratchDescriptorWord = source->scratchDescriptorWord;
    return program;
}

#include "H14ConvGeneric.inc"

Program composePrograms(const std::vector<Program> &programs) {
    if (programs.empty())
        throw std::invalid_argument("H14 chain requires at least one program");
    if (programs.size() == 1) return programs.front();
    throw std::invalid_argument(
        "h14.chain-unrepresentable-edge: relinking standalone programs keeps "
        "every task's own surface routing; the decoded corpus wires task DMA to "
        "the fixed boundary channels (reads through Src1/Src2, writes through "
        "Dst to the output channel or L2) and provides no addressing for "
        "declared intermediate surfaces, so no multi-program relink can be "
        "emitted correctly");
}

void fuseElementwisePostOperation(Program &program, PostOperation operation) {
    if (operation != PostOperation::Relu)
        throw std::invalid_argument(
            "h14.chain-unrepresentable-edge: only the relu post-operation has a "
            "decoded elementwise epilogue");
    if (program.taskCount != 1)
        throw std::invalid_argument(
            "h14.chain-unrepresentable-edge: elementwise post-operation fusion "
            "is decoded only for one-task programs");
    const auto span = h14TaskSpan(program.taskStream, 0);
    const auto word = h14RegisterOffset(program.taskStream, span.first,
                                        span.second, 0x900);
    const auto value = loadLE32(program.taskStream, word);
    if (value != 0x00080000)
        throw std::invalid_argument(
            "h14.chain-unrepresentable-edge: elementwise PE word is not the "
            "decoded no-post-operation form 0x00080000");
    storeLE32(program.taskStream, word, value | 0x20u);
}

void fuseMatmulPostOperation(Program &program, PostOperation operation) {
    if (operation != PostOperation::Relu)
        throw std::invalid_argument(
            "h14.chain-unrepresentable-edge: only the relu post-operation has a "
            "decoded matmul epilogue; lookup-table and bias epilogues extend the "
            "per-lane kernel-DMA layout with section bytes the decoded records "
            "do not retain");
    if (program.taskCount != 2)
        throw std::invalid_argument(
            "h14.chain-unrepresentable-edge: matmul post-operation fusion is "
            "decoded only for the two-task parity form");
    const auto span = h14TaskSpan(program.taskStream, 1);
    const auto word = h14RegisterOffset(program.taskStream, span.first,
                                        span.second, 0xd04);
    const auto value = loadLE32(program.taskStream, word);
    if (value != 0x00101c00)
        throw std::invalid_argument(
            "h14.chain-unrepresentable-edge: matmul compute task is not the "
            "decoded no-post-operation form 0x00101c00");
    storeLE32(program.taskStream, word, value | 0x00010000u);
}


} // namespace ane::h14
