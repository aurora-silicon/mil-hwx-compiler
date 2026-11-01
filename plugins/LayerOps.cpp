#include "LayerOps.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace ane::layers {
namespace {
struct NativeLayer {
    bool h14;
    Operation operation;
    Shape shape;
    const std::uint32_t *words;
    std::size_t wordCount, firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffset, constantBytes;
    std::uint32_t recordCount, unresolved;
};
#include "NativeLayerTemplates.inc"

const NativeLayer *find(bool h14, Operation operation, Shape shape) {
    for (const auto &row : kNativeLayers)
        if (row.h14 == h14 && row.operation == operation &&
            row.shape.channels == shape.channels && row.shape.height == shape.height &&
            row.shape.width == shape.width && row.shape.otherChannels == shape.otherChannels)
            return &row;
    return nullptr;
}

void half(std::vector<std::uint8_t> &out, std::size_t offset, std::uint16_t value) {
    out.at(offset) = static_cast<std::uint8_t>(value);
    out.at(offset + 1) = static_cast<std::uint8_t>(value >> 8);
}

std::vector<std::uint8_t> resizeConstants(Operation op, std::uint32_t channels, bool h14) {
    const auto cores = std::min(16u, channels);
    std::vector<std::uint8_t> out;
    if (op == Operation::Nearest2x) {
        for (std::uint32_t core = 0; core != cores; ++core) {
            const auto base = out.size();
            out.resize(base + 128, 0);
            for (unsigned i = 1; i != 16; ++i) half(out, base + i * 2, 0x3c00);
            const auto slots = (channels - core + 15) / 16;
            if (slots > 32) throw std::invalid_argument("layer nearest mask capacity exceeded");
            for (unsigned slot = 0; slot != slots; ++slot) half(out, base + 64 + slot * 2, 0x1111);
        }
        return out;
    }
    // Four phase filters of separable half-pixel interpolation. Entries are
    // numerator/16, so their fp16 representations are exact.
    constexpr unsigned weights[9][4] = {
        {1,0,0,0}, {3,3,0,0}, {0,1,0,0}, {3,0,3,0}, {9,9,9,9},
        {0,3,0,3}, {0,0,1,0}, {0,0,3,3}, {0,0,0,1}};
    std::vector<std::uint8_t> channel(3, 0); // u16 count, then sparse prefix
    const unsigned step = h14 ? 4 : 2;
    for (unsigned start = 0; start < 9; start += step) {
        for (unsigned pair = 0; pair != step; pair += 2) {
            unsigned mask = 0;
            for (unsigned j = 0; j != 2; ++j)
                if (start + pair + j < 9)
                    for (unsigned phase = 0; phase != 4; ++phase)
                        if (weights[start + pair + j][phase]) mask |= 1u << (phase + 4 * j);
            channel.push_back(static_cast<std::uint8_t>(mask));
        }
        for (unsigned row = start; row < std::min(9u, start + step); ++row)
            for (const auto value : weights[row]) {
                if (!value) continue;
                const std::uint16_t bits = value == 1 ? 0x2c00 : value == 3 ? 0x3200 : 0x3880;
                channel.push_back(static_cast<std::uint8_t>(bits));
                channel.push_back(static_cast<std::uint8_t>(bits >> 8));
            }
    }
    half(channel, 0, static_cast<std::uint16_t>(channel.size() - 3));
    for (unsigned core = 0; core != cores; ++core) {
        for (unsigned slot = 0; slot != (channels - core + 15) / 16; ++slot)
            out.insert(out.end(), channel.begin(), channel.end());
        out.resize((out.size() + 63) / 64 * 64, 0);
    }
    return out;
}

std::vector<std::uint8_t> constants(const NativeLayer &source) {
    if (source.operation == Operation::Nearest2x || source.operation == Operation::BilinearHalf2x)
        return resizeConstants(source.operation, source.shape.channels, source.h14);
    if (source.operation == Operation::Sigmoid) {
        auto out = source.h14
            ? h14::encodeElementwise(h14::Generation::H14, h14::UnaryOperation::Sigmoid, {64,1,1}).constants
            : h13::encodeElementwise(h13::UnaryOperation::Sigmoid, {64,1,1}).constants;
        out.resize(source.constantBytes, 0);
        return out;
    }
    std::vector<std::uint8_t> out(source.constantBytes, 0);
    if (source.operation == Operation::HardSwish) {
        const auto channels = source.shape.channels;
        const auto block = (channels + 31) / 32 * 32;
        if (channels > 1)
            for (unsigned c = 0; c != channels; ++c) half(out, (block + c) * 2, 0x3155);
        if (source.h14) return out;
        const auto base = channels > 1 ? block * 4 : 0;
        // The relu6 LUT. The task's PE immediates carry +3 and /6.
        half(out, base + 2, 0x4600); half(out, base + 6, 0x4600);
        half(out, base + 74, 0x3c00); half(out, base + 78, 0x3c00);
        half(out, base + 82, 64); half(out, base + 84, 1);
    }
    return out;
}

template<class Layout>
Layout surface(unsigned index, unsigned channels, unsigned height, unsigned width) {
    const std::uint64_t row = (static_cast<std::uint64_t>(width) * 2 + 63) / 64 * 64;
    const auto plane = row * height;
    return {index, {1, channels, height, width, plane, row}, (plane * channels + 0x3fff) / 0x4000 * 0x4000};
}

template<class Layout>
void surfaces(const NativeLayer &source, std::vector<Layout> &inputs, Layout &output) {
    const auto &s = source.shape;
    inputs = {surface<Layout>(5, s.channels, s.height, s.width)};
    if (source.operation == Operation::Concat)
        inputs.push_back(surface<Layout>(6, s.otherChannels, s.height, s.width));
    const unsigned scale = source.operation == Operation::Nearest2x || source.operation == Operation::BilinearHalf2x ? 2 : 1;
    output = surface<Layout>(4, s.channels + s.otherChannels, s.height * scale, s.width * scale);
}

std::vector<std::uint8_t> stream(const NativeLayer &source) {
    std::vector<std::uint8_t> out(source.wordCount * 4);
    for (std::size_t i = 0; i != source.wordCount; ++i)
        for (unsigned byte = 0; byte != 4; ++byte)
            out[i * 4 + byte] = static_cast<std::uint8_t>(source.words[i] >> (8 * byte));
    return out;
}

void bindH14(std::vector<std::uint8_t> &bytes, bool linux) {
    auto get = [&](std::size_t pos) {
        std::uint32_t v = 0;
        for (unsigned j = 0; j != 4; ++j) v |= static_cast<std::uint32_t>(bytes.at(pos+j)) << (8*j);
        return v;
    };
    for (std::size_t offset = 16; offset < bytes.size();) {
        const auto size = ((get(offset) >> 16) & 0x7ff) * 4;
        if (size < 32 || offset + size > bytes.size()) throw std::logic_error("invalid native H14 task");
        auto cursor = offset + 32 + ((get(offset+28) & 3) == 3 ? 4 : 0);
        while (cursor < offset + size) {
            auto h = get(cursor);
            unsigned count = ((h >> 15) & 0x3f) + 1;
            if (h & 0x80000000u) {
                count = 1;
                for (unsigned j = 0; j != 16; ++j) count += (h >> (15+j)) & 1;
            } else if (h & 0x20000000u) {
                const auto address = (h & 0x7fff) * 4;
                const auto slot = (h >> 23) & 63;
                unsigned bound = slot;
                if ((address == 0x1110 || address == 0x1128) && slot == (linux ? 4u : 5u)) bound = linux ? 5 : 4;
                if (address == 0x1508 && slot == (linux ? 5u : 4u)) bound = linux ? 4 : 5;
                h = (h & ~(63u << 23)) | (bound << 23);
                for (unsigned j = 0; j != 4; ++j) bytes[cursor+j] = static_cast<std::uint8_t>(h >> (8*j));
            }
            cursor += 4 * (count + 1);
        }
        offset = std::min((offset + size + 15) / 16 * 16, bytes.size());
    }
}
} // namespace

bool supports(bool h14, Operation operation, Shape shape) { return find(h14, operation, shape); }

h13::Program encodeH13(Operation operation, Shape shape) {
    const auto *source = find(false, operation, shape);
    if (!source) throw std::invalid_argument("h13.layer-outside-envelope: no decoded native layer geometry");
    h13::Program out;
    out.task = stream(*source);
    out.constants = constants(*source);
    if (out.constants.size() != source->constantBytes) throw std::logic_error("H13 native constant size mismatch");
    surfaces(*source, out.inputs, out.output);
    out.firstTaskBytes = source->firstTaskBytes;
    out.taskCount = source->taskCount;
    out.constantOffsetBytes = source->constantOffset;
    out.taskSurfaceChannels = {5,4,6,7};
    out.nativeDramLayer = true;
    return out;
}

h14::Program encodeH14(Operation operation, Shape shape) {
    const auto *source = find(true, operation, shape);
    if (!source) throw std::invalid_argument("h14.layer-outside-envelope: no decoded native layer geometry");
    h14::Program out;
    out.taskStream = stream(*source);
    bindH14(out.taskStream, true);
    out.normalizedBarRoles = true;
    out.constants = constants(*source);
    if (out.constants.size() != source->constantBytes) throw std::logic_error("H14 native constant size mismatch");
    surfaces(*source, out.inputs, out.output);
    out.firstTaskBytes = source->firstTaskBytes;
    out.taskCount = source->taskCount;
    out.constantOffsetBytes = source->constantOffset;
    out.programRecordCount = source->recordCount;
    out.unresolvedDescriptorWord = source->unresolved;
    return out;
}

std::vector<std::uint8_t> h14HWXStream(const h14::Program &program) {
    auto bytes = program.taskStream;
    if (program.normalizedBarRoles) bindH14(bytes, false);
    return bytes;
}
} // namespace ane::layers
