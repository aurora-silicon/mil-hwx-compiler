#include "../plugins/H13/H13Program.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kBlockBytes = 0x8000;
constexpr std::size_t kRowsPerBlock = 32;
constexpr std::size_t kOutputRows = 512;

void putLE32(std::vector<std::uint8_t> &bytes, std::size_t offset,
             std::uint32_t value) {
    for (std::size_t i = 0; i != 4; ++i)
        bytes.at(offset + i) = static_cast<std::uint8_t>(value >> (8 * i));
}

void putBE32(std::vector<std::uint8_t> &bytes, std::size_t offset,
             std::uint32_t value) {
    for (std::size_t i = 0; i != 4; ++i)
        bytes.at(offset + i) = static_cast<std::uint8_t>(value >> (8 * (3 - i)));
}

std::uint16_t le16(const std::vector<std::uint8_t> &bytes, std::size_t offset) {
    return static_cast<std::uint16_t>(bytes.at(offset)) |
           static_cast<std::uint16_t>(bytes.at(offset + 1)) << 8;
}

constexpr std::uint32_t streamHeader(std::uint32_t address,
                                     std::uint32_t words) {
    return ((words - 1) << 26) | address;
}

std::vector<std::uint8_t> expectedBinary(std::uint32_t operation) {
    std::vector<std::uint8_t> bytes(ane::h13::taskBytes, 0);
    putLE32(bytes, 0x00, (0x40u << 16) | (1u << 25));
    putLE32(bytes, 0x08, 1058);
    putLE32(bytes, 0x10, 0x00fff86a);
    putLE32(bytes, 0x18, (38u << 10) | (3u << 28));
    putLE32(bytes, 0x20,
            6u | (1u << 5) | (5u << 6) | (1u << 11) | (4u << 12) |
                (1u << 17));
    putLE32(bytes, 0x28, streamHeader(0x1f800, 62));

    putLE32(bytes, 0x124, streamHeader(0x00000, 16));
    putLE32(bytes, 0x128, (1u << 16) | 1u);
    putLE32(bytes, 0x12c, 1);
    putLE32(bytes, 0x130, 2u | (2u << 2) | (2u << 4));
    putLE32(bytes, 0x134, 64);
    putLE32(bytes, 0x138, 64);
    putLE32(bytes, 0x13c, (1u << 16) | 1u);
    putLE32(bytes, 0x140, 1);
    putLE32(bytes, 0x144,
            1u | (1u << 5) | (1u << 13) | (1u << 15) | (1u << 28) |
                (1u << 30));
    putLE32(bytes, 0x148, 0x2041);
    putLE32(bytes, 0x14c, 1u | (1u << 16));
    putLE32(bytes, 0x150, 1);
    putLE32(bytes, 0x154, 4);
    putLE32(bytes, 0x15c, 3u | (6u << 3));

    putLE32(bytes, 0x168, streamHeader(0x13800, 28));
    putLE32(bytes, 0x16c, 1u | (8u << 4) | (8u << 8) | (3u << 12) |
                                  (3u << 16));
    putLE32(bytes, 0x170, 0x33880);
    putLE32(bytes, 0x178, 0x40);
    putLE32(bytes, 0x17c, 0x40);
    putLE32(bytes, 0x180, 0x1000);
    putLE32(bytes, 0x18c, 0x40);
    putLE32(bytes, 0x190, 0x40);
    putLE32(bytes, 0x194, 0x1000);
    putLE32(bytes, 0x1a4, 1u | (3u << 4) | (2u << 12) | (1u << 24));
    putLE32(bytes, 0x1a8, 0x2030);

    putLE32(bytes, 0x1dc, streamHeader(0x04800, 18));
    putLE32(bytes, 0x1e4,
            2u | (1u << 4) | (1u << 5) | (1u << 6) | (1u << 8) |
                (1u << 20) | (1u << 22) | (1u << 24));
    putLE32(bytes, 0x1ec, 0x10);
    putLE32(bytes, 0x1f0, 0x420);
    putLE32(bytes, 0x1f4, 0x400);
    putLE32(bytes, 0x1f8, 0x400);
    putLE32(bytes, 0x1fc, 0x440);
    putLE32(bytes, 0x200, 0x10);
    putLE32(bytes, 0x204, 0x420);
    putLE32(bytes, 0x208, 0x400);
    putLE32(bytes, 0x20c, 0x400);
    putLE32(bytes, 0x210,
            2u | (2u << 2) | (1u << 4) | (1u << 5) | (1u << 6) |
                (1u << 8) | (1u << 20) | (1u << 22));
    putLE32(bytes, 0x214, 0x860);

    putLE32(bytes, 0x228, streamHeader(0x08800, 4));
    putLE32(bytes, 0x22c, (2u << 18) | (operation << 2));
    putLE32(bytes, 0x230, 0x3c00u << 16);
    putLE32(bytes, 0x234, 0x3c00u << 16);
    putLE32(bytes, 0x238, 0x3f800000);
    putLE32(bytes, 0x23c, streamHeader(0x0c800, 5));
    putLE32(bytes, 0x244, operation == 1 ? 0x30 : 0);

    putLE32(bytes, 0x254, streamHeader(0x17800, 7));
    putLE32(bytes, 0x258, 1u | (12u << 4) | (1u << 26));
    putLE32(bytes, 0x260, 0x40);
    putLE32(bytes, 0x264, 0x40);
    putLE32(bytes, 0x268, 0x1000);
    putLE32(bytes, 0x270, 1u | (3u << 4) | (2u << 12) | (1u << 24));
    return bytes;
}

std::vector<std::uint8_t> expectedMatvecTask() {
    std::vector<std::uint8_t> bytes(ane::h13::taskBytes, 0);
    putLE32(bytes, 0x00, (0x40u << 16) | (1u << 25));
    putLE32(bytes, 0x08, 1058);
    putLE32(bytes, 0x10, 0x00fff86a);
    putLE32(bytes, 0x18, (38u << 10) | (3u << 28));
    putLE32(bytes, 0x20,
            5u | (1u << 5) | (36u << 12) | (1u << 24) | (1u << 26));
    putLE32(bytes, 0x24, 0x21);
    putLE32(bytes, 0x28, streamHeader(0x1f800, 62));

    std::size_t dma = 0x2c;
    putBE32(bytes, dma, 0x40000000);
    dma += 8;
    for (std::size_t i = 0; i != 16; ++i, dma += 4)
        putBE32(bytes, dma, 0x81000000);
    for (std::uint32_t i = 0; i != 16; ++i, dma += 4)
        putLE32(bytes, dma, i * 0x8000);
    for (std::size_t i = 0; i != 16; ++i, dma += 4)
        putLE32(bytes, dma, 0x8000);
    for (std::size_t i = 0; i != 4; ++i, dma += 4)
        putBE32(bytes, dma, 0x80000000);
    assert(dma + 8 * 4 == 0x124);

    putLE32(bytes, 0x124, streamHeader(0x00000, 16));
    putLE32(bytes, 0x128, (1u << 16) | 1u);
    putLE32(bytes, 0x12c, 1);
    putLE32(bytes, 0x130, 2u | (2u << 4));
    putLE32(bytes, 0x134, 512);
    putLE32(bytes, 0x138, 512);
    putLE32(bytes, 0x13c, (1u << 16) | 1u);
    putLE32(bytes, 0x140, 1);
    putLE32(bytes, 0x144, 0x5000b421);
    putLE32(bytes, 0x148, 0x2041);
    putLE32(bytes, 0x14c, 0x00010001);
    putLE32(bytes, 0x150, 1);
    putLE32(bytes, 0x15c, 0x00244405);
    putLE32(bytes, 0x160, 1u << 20);

    putLE32(bytes, 0x168, streamHeader(0x13800, 28));
    putLE32(bytes, 0x16c, 1u | (8u << 4) | (8u << 8) | (3u << 12) |
                                  (3u << 16));
    putLE32(bytes, 0x170, 0x8880);
    putLE32(bytes, 0x178, 0x40);
    putLE32(bytes, 0x17c, 0x40);
    putLE32(bytes, 0x180, 0x8000);
    putLE32(bytes, 0x1a4, 1u | (3u << 4) | (2u << 12) | (1u << 24));
    putLE32(bytes, 0x1ac, 0x100);

    putLE32(bytes, 0x1dc, streamHeader(0x04800, 18));
    putLE32(bytes, 0x1e4, 0x00500172);
    putLE32(bytes, 0x1ec, 0x10);
    putLE32(bytes, 0x1f0, 0x2030);
    putLE32(bytes, 0x1f4, 0x2000);
    putLE32(bytes, 0x1f8, 0x2000);
    putLE32(bytes, 0x210, 0x00500172);
    putLE32(bytes, 0x214, 0x2030);
    putLE32(bytes, 0x218, 0x10);
    putLE32(bytes, 0x21c, 0x2020);
    putLE32(bytes, 0x220, 0x2000);
    putLE32(bytes, 0x224, 0x2000);

    putLE32(bytes, 0x228, streamHeader(0x08800, 4));
    putLE32(bytes, 0x23c, streamHeader(0x0c800, 5));
    putLE32(bytes, 0x240, 0x82);
    putLE32(bytes, 0x244, 0x00101c00);
    putLE32(bytes, 0x250, 0x3c00);

    putLE32(bytes, 0x254, streamHeader(0x17800, 7));
    putLE32(bytes, 0x258, 1u | (12u << 4));
    putLE32(bytes, 0x260, 0x40);
    putLE32(bytes, 0x264, 0x40);
    putLE32(bytes, 0x268, 0x8000);
    putLE32(bytes, 0x270,
            1u | (3u << 4) | (2u << 12) | (3u << 20) | (1u << 24));
    return bytes;
}

template <typename Function>
void rejects(Function &&function) {
    try {
        function();
        assert(false);
    } catch (const std::invalid_argument &) {
    }
}

void checkLayout(const ane::h13::TensorLayout &layout, std::uint32_t index,
                 std::uint64_t channels, std::uint64_t allocation) {
    assert(layout.index == index);
    assert((layout.nchw == std::array<std::uint64_t, 6>{1, channels, 1, 1, 64, 64}));
    assert(layout.allocationBytes == allocation);
}

std::uint16_t weight(std::size_t row, std::size_t column) {
    return static_cast<std::uint16_t>((row * 521 + column * 17 + 1) & 0xffff);
}

void checkBinary() {
    using ane::h13::BinaryOperation;
    const std::array operations{BinaryOperation::Add, BinaryOperation::Multiply,
                                BinaryOperation::Maximum, BinaryOperation::Minimum};
    for (std::uint32_t i = 0; i != operations.size(); ++i) {
        const auto program = ane::h13::encodeBinary(operations[i]);
        assert(program.task == expectedBinary(i));
        assert(program.constants.empty());
        assert(program.inputs.size() == 2);
        checkLayout(program.inputs[0], 5, 64, 0x4000);
        checkLayout(program.inputs[1], 6, 64, 0x4000);
        checkLayout(program.output, 4, 64, 0x4000);
    }
    rejects([] { ane::h13::encodeBinary(static_cast<BinaryOperation>(99)); });
}

void checkMatvec(std::uint32_t reduction) {
    std::vector<std::uint8_t> weights(kOutputRows * reduction * 2);
    for (std::size_t row = 0; row != kOutputRows; ++row)
        for (std::size_t column = 0; column != reduction; ++column) {
            const auto value = weight(row, column);
            const auto offset = (row * reduction + column) * 2;
            weights[offset] = static_cast<std::uint8_t>(value);
            weights[offset + 1] = static_cast<std::uint8_t>(value >> 8);
        }

    const auto program = ane::h13::encodeMatvec(reduction, weights.data(), weights.size(), true);
    assert(program.task == expectedMatvecTask());
    assert(program.inputs.size() == 1);
    checkLayout(program.inputs[0], 5, 512, 512 * 64);
    checkLayout(program.output, 4, 512, 0x8000);

    assert(program.constants.size() == 16 * kBlockBytes);
    const auto packedAt = [&](std::size_t group, std::size_t column,
                              std::size_t rowInGroup) {
        return le16(program.constants,
                    group * kBlockBytes + (column * kRowsPerBlock + rowInGroup) * 2);
    };
    assert(packedAt(0, 0, 0) == weight(0, 0));
    assert(packedAt(0, 1, 0) == weight(0, 1));
    assert(packedAt(0, 0, 31) == weight(31, 0));
    assert(packedAt(7, reduction / 2, 13) == weight(7 * 32 + 13, reduction / 2));
    assert(packedAt(15, reduction - 1, 31) == weight(511, reduction - 1));

    if (reduction == 256) {
        for (std::size_t group = 0; group != 16; ++group) {
            const auto padding = program.constants.begin() + group * kBlockBytes + 0x4000;
            assert(std::all_of(padding, padding + 0x4000,
                               [](std::uint8_t byte) { return byte == 0; }));
        }
    }
}

void checkMatvecValidation() {
    std::vector<std::uint8_t> weights(512 * 256 * 2);
    rejects([&] { ane::h13::encodeMatvec(128, weights.data(), weights.size(), true); });
    rejects([&] { ane::h13::encodeMatvec(256, weights.data(), weights.size() - 1, true); });
    rejects([&] { ane::h13::encodeMatvec(256, nullptr, weights.size(), true); });
}

void checkEncoderLinearEnvelopeHole() {
    using ane::h13::BinaryOperation;
    using ane::h13::supportsElementwise;
    using ane::h13::supportsMatmulParity;
    // Per-head encoder linear [1,375,1024] x [128,1024], ty=1, constant W.
    assert(!supportsMatmulParity({375, 1024, 128, false, true, false}));
    assert(!supportsMatmulParity({1, 1024, 128, false, true, false}));
    assert(!supportsMatmulParity({1, 512, 128, false, true, false}));
    assert(supportsMatmulParity({1, 256, 128, false, true, false}));
    assert(supportsMatmulParity({1, 1024, 256, false, true, false}));
    assert(supportsMatmulParity({1, 1024, 512, false, true, false}));
    assert(supportsMatmulParity({1, 1024, 1024, false, true, false}));
    assert(supportsElementwise(BinaryOperation::Add, {128, 1, 1}, false));
    assert(!supportsElementwise(BinaryOperation::Add, {128, 1, 1}, true));
}

void checkAttentionBatchedEnvelope() {
    using ane::h13::supportsBatchedMatmul;
    using ane::h13::supportsMatmulParity;
    // Encoder attention [1,8,375,128] x [1,8,128,375], tx=0, ty=0, runtime y.
    assert(supportsBatchedMatmul({8, 375, 128, 375, false, false, true}));
    // The 2026-09-16 head-projection capture decodes [8,375,1024] x
    // [8,128,1024] at ty=1 with identity packing; B=1 stays outside.
    assert(supportsBatchedMatmul({8, 375, 1024, 128, false, true, false}));
    assert(!supportsBatchedMatmul({1, 375, 1024, 128, false, true, false}));
    assert(!supportsBatchedMatmul({2, 375, 1024, 128, false, true, false}));
    assert(!supportsMatmulParity({375, 128, 375, false, false, true}));
}

void checkEncoderUnaryEnvelope() {
    using ane::h13::UnaryOperation;
    using ane::h13::supportsElementwise;
    assert(supportsElementwise(UnaryOperation::Silu, {64, 1, 1}));
    assert(supportsElementwise(UnaryOperation::Silu, {512, 1, 1}));
    assert(supportsElementwise(UnaryOperation::Sigmoid, {64, 1, 1}));
    assert(supportsElementwise(UnaryOperation::Sigmoid, {512, 1, 1}));
    assert(!supportsElementwise(UnaryOperation::Silu, {128, 1, 1}));
    assert(!supportsElementwise(UnaryOperation::Silu, {1024, 1, 1}));
    assert(!supportsElementwise(UnaryOperation::Silu, {384000, 1, 1}));
    assert(!supportsElementwise(UnaryOperation::Silu, {1024, 375, 1}));
    // The encoder-geometry oracles (2026-09-16) decode the rank-3 spell
    // [1, A, B] as the [1, 1, A, B] surface.
    assert(supportsElementwise(UnaryOperation::Silu, {1, 1024, 375}));
    assert(supportsElementwise(UnaryOperation::Silu, {1, 375, 4096}));
    assert(supportsElementwise(UnaryOperation::Sigmoid, {1, 1024, 375}));
    assert(!supportsElementwise(UnaryOperation::Sigmoid, {384000, 1, 1}));
    assert(!supportsElementwise(UnaryOperation::Sigmoid, {1024, 375, 1}));
}

void checkEncoderNormEnvelope() {
    using ane::h13::NormOperation;
    using ane::h13::supportsNormParity;
    assert(supportsNormParity(NormOperation::LayerNorm,
                              {{1024, 1, 1}, {1024, 1, 1}, 0x02, true}));
    assert(supportsNormParity(NormOperation::LayerNorm,
                              {{512, 1, 1}, {512, 1, 1}, 0x02, true}));
    assert(supportsNormParity(NormOperation::Softmax,
                              {{512, 1, 1}, {512, 1, 1}, 0x02, true}));
    assert(supportsNormParity(NormOperation::Softmax,
                              {{8, 128, 128}, {8, 128, 128}, 0x08, true}));
    // The encoder-geometry oracles (2026-09-16) decode the encoder norm
    // shapes: last-axis layer_norm at [1,1,375,1024] and last-axis softmax
    // over one and eight heads at [375,375].
    assert(supportsNormParity(NormOperation::LayerNorm,
                              {{1, 375, 1024}, {1, 375, 1024}, 0x08, true}));
    assert(!supportsNormParity(NormOperation::LayerNorm,
                               {{375, 1024, 1}, {375, 1024, 1}, 0x04, true}));
    assert(supportsNormParity(NormOperation::Softmax,
                              {{8, 375, 375}, {8, 375, 375}, 0x08, true}));
    assert(supportsNormParity(NormOperation::Softmax,
                              {{1, 375, 375}, {1, 375, 375}, 0x08, true}));
}

void checkEncoderConvEnvelope() {
    using ane::h13::supportsConvParity;
    assert(supportsConvParity({1, 1, 1, 1, true, {256, 32, 32}, {256, 32, 32}}));
    assert(supportsConvParity({1, 1, 1, 1, true, {256, 16, 16}, {256, 16, 16}}));
    assert(supportsConvParity({1, 1, 2, 1, false, {64, 16, 16}, {64, 8, 8}}));
    // The decoded encoder rows: the W-major rank-3 out-projection respell
    // (one linked two-task program), the H-major in-projection capture, the
    // W-major rank-3 in-projection respell (the decoded colgroup-order
    // four-task program), the rect depthwise 9x1 key, the W-major
    // bias-bearing depthwise respell (its own two-task slotted section),
    // and the small bias1 pointwise pairs whose uniform captures were
    // value-collision degenerate.
    assert(supportsConvParity({1, 1, 1, 1, false, {1024, 1, 375}, {1024, 1, 375}}));
    assert(supportsConvParity({1, 1, 1, 1, false, {1024, 375, 1}, {2048, 375, 1}}));
    assert(supportsConvParity({1, 1, 1, 1, false, {1024, 1, 375}, {2048, 1, 375}}));
    assert(supportsConvParity({9, 1, 1, 1024, false, {1024, 1, 375}, {1024, 1, 375}}));
    assert(supportsConvParity({1, 9, 1, 1024, true, {1024, 1, 375}, {1024, 1, 375}}));
    assert(supportsConvParity({9, 1, 1, 1024, true, {1024, 375, 1}, {1024, 375, 1}}));
    assert(supportsConvParity({1, 1, 1, 1, true, {256, 750, 32}, {256, 750, 32}}));
    assert(supportsConvParity({1, 1, 1, 1, true, {256, 375, 16}, {256, 375, 16}}));
    assert(!supportsConvParity({3, 3, 2, 1, true, {1, 3000, 128}, {256, 1500, 64}}));
    assert(!supportsConvParity({3, 3, 2, 256, true, {256, 1500, 64}, {256, 750, 32}}));
    assert(!supportsConvParity({3, 3, 2, 256, true, {256, 750, 32}, {256, 375, 16}}));
    // The W-padded rel-pos padconv lowers through its own two-task capture.
    assert(supportsConvParity({1, 1, 1, 8, false, {8, 375, 749}, {8, 375, 750}}));
    assert(supportsConvParity({1, 1, 1, 8, false, {8, 8, 7}, {8, 8, 8}}));
    // An asymmetric padconv surface pair with no capture stays refused.
    assert(!supportsConvParity({1, 1, 1, 8, false, {8, 100, 49}, {8, 100, 50}}));
    // The W-major k1x1 row does not cover the transposed surfaces, and the
    // rect key does not survive a kernel-extent flip or a bias.
    assert(!supportsConvParity({1, 1, 1, 1, false, {1024, 375, 1}, {1024, 375, 1}}));
    assert(!supportsConvParity({1, 9, 1, 1024, false, {1024, 1, 375}, {1024, 1, 375}}));
    assert(!supportsConvParity({9, 1, 1, 1024, true, {1024, 1, 375}, {1024, 1, 375}}));
    assert(!supportsConvParity({3, 3, 2, 1, false, {64, 16, 16}, {64, 8, 8}}));
}

void checkEncoderSelectEnvelope() {
    using ane::h13::H13BooleanKind;
    using ane::h13::supportsBooleanOp;
    assert(supportsBooleanOp({H13BooleanKind::Select, false, 8, 375, 375}));
    assert(supportsBooleanOp({H13BooleanKind::Select, false, 64, 1, 1}));
    assert(supportsBooleanOp({H13BooleanKind::Select, true, 8, 375, 375}));
    assert(!supportsBooleanOp({H13BooleanKind::Select, true, 64, 1, 1}));
    assert(!supportsBooleanOp({H13BooleanKind::Select, false, 1024, 375, 1}));
    assert(!supportsBooleanOp({H13BooleanKind::Select, false, 1, 1024, 375}));
    assert(!supportsBooleanOp({H13BooleanKind::Select, false, 8, 128, 128}));
}



} // namespace

int main() {
    checkBinary();
    checkMatvec(256);
    checkMatvec(512);
    checkMatvecValidation();
    checkEncoderLinearEnvelopeHole();
    checkAttentionBatchedEnvelope();
    checkEncoderUnaryEnvelope();
    checkEncoderNormEnvelope();
    checkEncoderConvEnvelope();
    checkEncoderSelectEnvelope();
    std::cout << "H13_ENCODING_OK\n";
}
