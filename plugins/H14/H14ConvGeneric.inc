// Formula-built H14 convolution and DRAM linking. Included in H14Program.cpp
// to share the packed-record reader and the independently derived packers.
namespace {

std::uint32_t log2Floor(std::uint32_t n) {
    std::uint32_t v = 0;
    while (n > 1) { n >>= 1; ++v; }
    return v;
}

bool depthwise(const ConvShape &s) {
    return s.groups == s.input.channels && s.groups == s.output.channels;
}

int padding(const ConvShape &s) {
    for (int p : {static_cast<int>((s.kernel - 1) / 2), 0}) {
        auto extent = [&](std::uint32_t n) -> std::int64_t {
            return (static_cast<std::int64_t>(n) + 2 * p - s.kernel) / s.stride + 1;
        };
        if (extent(s.input.height) == s.output.height &&
            extent(s.input.width) == s.output.width) return p;
    }
    return -1;
}

const OracleConvTemplate *familyBase(const ConvShape &s) {
    const OracleConvTemplate *best = nullptr;
    for (const auto &c : kConvTasks) {
        ConvShape probe{c.kernel, c.kernelWidth, c.stride, c.groups, c.bias,
                        c.input, c.output};
        if (c.taskCount != 1 || c.kernel != s.kernel ||
            c.kernelWidth != s.kernelWidth || c.stride != s.stride ||
            c.bias != s.bias || (c.groups != 1 && !depthwise(probe)) ||
            depthwise(probe) != depthwise(s) ||
            (padding(probe) == 0) != (padding(s) == 0)) continue;
        if (!best || c.input.channels < best->input.channels ||
            (c.input.channels == best->input.channels &&
             c.input.width < best->input.width)) best = &c;
    }
    return best;
}

// Last write wins: the zero Prefetch record precedes the BAR relocation
// to the same register pair. Patching the first would miss the relocation.
std::size_t lastRegister(const Program &p, std::size_t off, std::size_t bytes,
                         std::uint32_t address) {
    std::size_t found = p.taskStream.size();
    for (std::size_t c = off + 32; c < off + bytes;) {
        const auto h = loadLE32(p.taskStream, c);
        const auto base = (h & 0x7fff) * 4;
        std::size_t n = 0;
        if (h & 0x80000000u) {
            if (address == base) found = c + 4;
            n = 1;
            for (unsigned b = 0; b < 16; ++b) if ((h >> (15 + b)) & 1) {
                if (address == base + (b + 1) * 4) found = c + 4 * (n + 1);
                ++n;
            }
        } else {
            n = ((h >> 15) & 0x3f) + 1;
            if (address >= base && address < base + n * 4)
                found = c + 4 + address - base;
        }
        c += (n + 1) * 4;
    }
    if (found == p.taskStream.size()) throw std::invalid_argument("H14 register absent");
    return found;
}

// A missing word is the reset/default value in the family. New words are
// emitted as one-word records, so a formula never relies on a base's shape.
std::uint32_t get(const Program &p, std::size_t off, std::size_t bytes,
                  std::uint32_t address) {
    try { return loadLE32(p.taskStream,
        lastRegister(p, off, bytes, address)); }
    catch (const std::invalid_argument &) { return 0; }
}

void put(Program &p, std::size_t off, std::size_t &bytes,
          std::uint32_t address, std::uint32_t value) {
    try {
        storeLE32(p.taskStream, lastRegister(p, off, bytes, address), value);
        return;
    } catch (const std::invalid_argument &) {
        if (!value) return;
    }
    if (off + bytes != p.taskStream.size())
        throw std::logic_error("H14 register insertion requires the final task");
    appendLE(p.taskStream, address / 4, 4);
    appendLE(p.taskStream, value, 4);
    bytes += 8;
    if (bytes / 4 > 0x7ff)
        throw std::invalid_argument("h14.conv-outside-envelope: task is too large");
    storeLE32(p.taskStream, off, (static_cast<std::uint32_t>(bytes / 4) << 16) |
              (loadLE32(p.taskStream, off) & 0xffff));
}

// BAR records are dense records with bit 29. Bits 28:23 name the BAR slot;
// unlike H13, the routing is on the RECORD, not in an extra header word.
void route(Program &p, std::size_t off, std::size_t bytes,
            std::uint32_t address, std::uint32_t slot) {
    std::size_t cursor = off + 32;
    while (cursor < off + bytes) {
        const auto h = loadLE32(p.taskStream, cursor);
        std::size_t count = 0;
        if (h & 0x80000000u) {
            count = 1;
            for (unsigned b = 0; b < 16; ++b) count += (h >> (15 + b)) & 1;
        } else {
            count = ((h >> 15) & 0x3f) + 1;
            if ((h & 0x20000000u) && (h & 0x7fff) * 4 == address) {
                storeLE32(p.taskStream, cursor,
                    (h & ~(0x3fu << 23)) | (slot << 23));
                return;
            }
        }
        cursor += (count + 1) * 4;
    }
    throw std::invalid_argument("h14.linked-outside-envelope: missing BAR record");
}

std::uint32_t skew(std::uint32_t stride, std::uint32_t width,
                    std::uint32_t channels) {
    if (stride == 2) return width >= 64 ? 0x30 : 0x20;
    if (width >= 64) return 0;
    if (width > 16) return 0x50;
    return channels >= 32 ? 0x30 : 0x20;
}

} // namespace

std::string convGenericRefusal(ConvShape s) {
    if (!s.input.channels || !s.output.channels || !s.input.height ||
        !s.input.width || !s.output.height || !s.output.width || !s.groups)
        return "convolution surfaces and groups must be nonzero";
    if (s.kernel != s.kernelWidth || (s.kernel != 1 && s.kernel != 3))
        return "generic kernels must be square 1x1 or 3x3";
    if (s.stride != 1 && s.stride != 2) return "generic strides must be 1 or 2";
    if (s.groups != 1 && !depthwise(s)) return "generic groups must be 1 or depthwise";
    if (s.stride == 2 && (s.kernel != 1 || s.groups != 1))
        return "generic stride 2 requires dense 1x1";
    if (s.input.channels % 16 || s.output.channels % 16)
        return "generic channel counts must be multiples of 16";
    if (s.input.channels > 4096 || s.output.channels > 4096 ||
        s.input.width > 4096 || s.input.height > 16384 ||
        s.output.width > 4096 || s.output.height > 16384)
        return "generic dimensions exceed the derived envelope";
    if (padding(s) < 0) return "generic padding must be symmetric same or valid";
    if (!familyBase(s)) return "no decoded family supplies the invariant registers";
    if (s.stride == 2 && s.output.channels > 128 && s.output.channels % 128)
        return "wide stride 2 requires a multiple of 128 outputs";
    return {};
}

Program encodeConvGeneric(ConvShape s, const std::uint8_t *weights,
                          std::size_t weightBytes, const std::uint8_t *bias,
                          std::size_t biasBytes) {
    const auto refusal = convGenericRefusal(s);
    if (!refusal.empty()) throw std::invalid_argument("h14.conv-outside-envelope: " + refusal);
    const bool dw = depthwise(s);
    const auto reduction = (dw ? 1u : s.input.channels) * s.kernel * s.kernelWidth;
    if (!weights || weightBytes != static_cast<std::size_t>(s.output.channels) * reduction * 2 ||
        s.bias != (bias != nullptr) || (bias && biasBytes != s.output.channels * 2ull))
        throw std::invalid_argument("h14.conv-outside-envelope: invalid weight or bias length");
    // Output-channel partitioning stays within the proven 1 MiB H13 kernel
    // budget, even though H14's corpus includes larger single-task sections.
    auto slice = s.output.channels;
    if (s.stride == 2) slice = std::min(slice, 128u);
    const auto perChannel = (reduction + (s.bias ? 1 : 0)) * 2ull;
    if (!dw && perChannel * slice > 0x100000) {
        const auto fit = static_cast<std::uint32_t>(0x100000 / perChannel);
        slice = fit >= 256 ? fit / 256 * 256 :
            (fit >= 16 ? 16u << log2Floor(fit / 16) : 0u);
    }
    if (slice && (s.output.channels + slice - 1) / slice > 254)
        throw std::invalid_argument("h14.conv-outside-envelope: DMA task-tag limit exceeded");
    if (!slice) throw std::invalid_argument("h14.conv-outside-envelope: kernel budget exceeded");
    const auto base = familyBase(s);
    Program result;
    result.formulaBuilt = true;
    result.depthwiseConv = dw;
    result.inputs = {convTensor(5, s.input)};
    result.output = convTensor(4, s.output);
    if (result.inputs[0].allocationBytes > UINT32_MAX || result.output.allocationBytes > UINT32_MAX)
        throw std::invalid_argument("h14.conv-outside-envelope: surface exceeds 32-bit DMA");
    result.taskStream.assign(16, 0);
    const auto row = static_cast<std::uint32_t>(alignUp(std::max(16u, s.input.width * 2), 16));
    const auto sourceRow = row * s.input.channels + skew(s.stride, s.input.width, s.input.channels);
    const auto outRow = static_cast<std::uint32_t>(alignUp(std::max(16u, s.output.width * 2), 16));
    std::uint32_t tile = s.output.height;
    while (tile && static_cast<std::uint64_t>(std::min(s.input.height,
            tile * s.stride + s.kernel - 1)) * sourceRow +
            static_cast<std::uint64_t>(tile) * slice * outRow > 0x1e0000) --tile;
    if (!tile) throw std::invalid_argument("h14.conv-outside-envelope: one row exceeds L2 budget");
    const auto rows = std::min(s.input.height, tile * s.stride + s.kernel - 1);
    auto cls = s.output.width <= 8 ? 4u :
        7 - std::min(7u, log2Floor(s.output.width - 1) + 1);
    if (s.kernel == 1 && !dw) cls = std::min(cls, log2Floor(s.output.channels / 16) + 1);
    if (s.input.height == 1 && s.input.width == 1 && s.kernel == 1) cls = 0;
    const bool repeat = cls == 4 || (cls >= 2 && !dw &&
        (s.kernel == 3 || (s.input.channels >= 512 && s.output.channels >= 512))) ||
        (s.output.width <= 8 && s.kernel == 1);
    const auto cfg = 0x200000u | (cls << 16) |
        (repeat ? (cls << 12) | (cls << 8) : 0u) |
        (s.output.width <= 8 ? 5u : 1u);
    const auto taskCount = (s.output.channels + slice - 1) / slice;
    std::uint32_t id = 0;
    for (std::uint32_t first = 0; first < s.output.channels; first += slice, ++id) {
        auto part = s;
        part.output.channels = std::min(slice, s.output.channels - first);
        const auto *w = weights + static_cast<std::size_t>(first) * reduction * 2;
        const auto *b = bias ? bias + first * 2 : nullptr;
        const auto chunks = laneChunks(part.output.channels, laneCap(s.input));
        if (!dw && std::any_of(chunks.begin(), chunks.end(),
                [&](std::uint32_t v) { return v != chunks.front(); }))
            throw std::invalid_argument("h14.conv-outside-envelope: mixed lane chunks");
        if (s.stride == 2 && (chunks.size() != 1 || chunks.front() > 8))
            throw std::invalid_argument("h14.conv-outside-envelope: sparse forms above eight lanes refused");
        auto packed = dw ? packConvDepthwise(s.kernel * s.kernel, part.output.channels, w, b)
            : s.stride == 2 ? packConvStrided(reduction, chunks, w, b)
            : packConvDense(reduction, chunks, w, b, false);
        std::array<std::uint32_t, 16> offsets{}, sizes{};
        std::size_t cursor = 0;
        for (unsigned i = 0; i < 16; ++i) {
            offsets[i] = static_cast<std::uint32_t>(cursor);
            auto size = packed.size() / 16;
            if (s.stride == 2) {
                const auto biasSpan = s.bias ? chunks.front() * 2 : 0;
                const auto at = cursor + biasSpan;
                if (at + 2 > packed.size()) throw std::logic_error("H14 sparse plane is truncated");
                const auto count = packed[at] | (packed[at + 1] << 8);
                // H14 count excludes the body's hoisted lead byte.
                size = alignUp(biasSpan + 2 + count + 1, 64);
            }
            sizes[i] = static_cast<std::uint32_t>(size);
            cursor += size;
        }
        if (cursor != packed.size()) throw std::logic_error("H14 kernel chunks do not cover section");
        Program task = streamProgram(base->text, base->textWords, 1, 0, 0);
        const auto span = h14TaskSpan(task.taskStream, 0);
        auto bytes = span.second;
        auto set = [&](std::uint32_t a, std::uint32_t v) { put(task, span.first, bytes, a, v); };
        set(0, (s.input.height << 16) | s.input.width);
        set(0xc, s.input.channels); set(0x10, part.output.channels);
        set(0x14, (s.output.height << 16) | s.output.width);
        set(0x1c, dw ? 0 : log2Floor(chunks.front()));
        if (s.kernel != 1 || s.stride != 1)
            set(0x20, 0x50000000u | s.kernel | (s.kernel << 6) |
                (s.stride << 13) | (s.stride << 15) |
                (static_cast<std::uint32_t>(padding(s)) << 17) |
                (static_cast<std::uint32_t>(padding(s)) << 22));
        set(0x2c, tile); set(0x38, cfg);
        set(0x50c, row); set(0x510, sourceRow);
        set(0x514, row * s.input.channels); set(0x518, row * s.input.channels);
        set(0x534, rows * sourceRow);
        set(0x1118, static_cast<std::uint32_t>(result.inputs[0].nchw[5]));
        set(0x111c, static_cast<std::uint32_t>(result.inputs[0].nchw[4]));
        set(0x1508, static_cast<std::uint32_t>(result.output.nchw[4] * first));
        set(0x1510, static_cast<std::uint32_t>(result.output.nchw[5]));
        set(0x1514, static_cast<std::uint32_t>(result.output.nchw[4]));
        set(0x1908, static_cast<std::uint32_t>(result.constants.size()));
        for (unsigned i = 0; i < 16; ++i) {
            set(0x1958 + 4 * i, offsets[i]); set(0x1998 + 4 * i, sizes[i]);
        }
        // The decoded multi-task H14 corpus advances each DMA's task tag.
        set(0x1100, (get(task, span.first, bytes, 0x1100) & ~0xff0000u) | (id << 16));
        set(0x1500, (get(task, span.first, bytes, 0x1500) & ~0xff0000u) | (id << 16));
        set(0x1900, (get(task, span.first, bytes, 0x1900) & ~0xff0000u) | ((id + 1) << 16));
        for (unsigned i = 0; i < 16; ++i)
            set(0x1918 + 4 * i, (get(task, span.first, bytes, 0x1918 + 4 * i) & ~0xff0000u) | (id << 16));
        storeLE32(task.taskStream, span.first,
            (static_cast<std::uint32_t>(bytes / 4) << 16) | id);
        // Header word 2: 0x08 opens the program on its first task and 0x22
        // closes it on its last, as in every decoded multi-task H14 program
        // (h14-mobile chain and s2link oracles). A single task carries both
        // (0x2a). Setting both on every task of a multi-task program made the
        // firmware's next command never complete on the M2.
        storeLE32(task.taskStream, span.first + 8,
            (id == 0 ? 0x08u : 0u) | (id + 1 == taskCount ? 0x22u : 0u));
        storeLE32(task.taskStream, span.first + 24, 0);
        storeLE32(task.taskStream, span.first + 28, 0x50001);
        result.taskStream.resize(alignUp(result.taskStream.size(), 16), 0);
        result.taskStream.insert(result.taskStream.end(), task.taskStream.begin() +
            static_cast<std::ptrdiff_t>(span.first), task.taskStream.end());
        result.constants.insert(result.constants.end(), packed.begin(), packed.end());
        if (!id) result.firstTaskBytes = bytes;
    }
    result.taskCount = id;
    result.constantOffsetBytes = alignUp(result.taskStream.size(), 64);
    return result;
}

void fuseConvRelu(Program &p) {
    if (!p.formulaBuilt) throw std::invalid_argument("H14 relu fusion requires formula-built conv");
    if (p.mobileRingRuleBuilt) {
        if (p.mobileRingActivation)
            throw std::invalid_argument("H14 native ring convolution already has a post-operation");
        p.mobileRingActivation = 1;
        return;
    }
    for (std::uint32_t i = 0; i < p.taskCount; ++i) {
        const auto span = h14TaskSpan(p.taskStream, i);
        const auto at = h14RegisterOffset(p.taskStream, span.first, span.second, 0xd04);
        storeLE32(p.taskStream, at, loadLE32(p.taskStream, at) | 0x10000u);
    }
}

namespace {

std::array<std::uint8_t, 128> clipBoundsDescriptor(std::uint16_t lower,
                                                   std::uint16_t upper) {
    std::array<std::uint8_t, 128> descriptor{};
    const auto slot = [&](std::size_t index, std::uint16_t value) {
        descriptor[index * 2] = static_cast<std::uint8_t>(value);
        descriptor[index * 2 + 1] = static_cast<std::uint8_t>(value >> 8);
    };
    slot(0, lower); slot(1, upper); slot(2, lower); slot(3, upper);
    slot(37, 0x3c00); slot(39, 0x3c00);
    slot(41, 0x0040); slot(42, 0x0001);
    return descriptor;
}

std::vector<std::uint8_t> h14ClipDepthwiseTask(
        const std::vector<std::uint8_t> &task,
        const std::array<std::uint32_t, 31> &values) {
    if (task.size() < 32 || task.size() % 4)
        throw std::invalid_argument(
            "h14.conv-clip-outside-envelope: malformed depthwise task");
    std::vector<std::uint8_t> out(task.begin(), task.begin() + 32);
    bool inserted = false;
    for (std::size_t cursor = 32; cursor < task.size();) {
        const auto header = loadLE32(task, cursor);
        const auto base = (header & 0x7fffu) * 4;
        std::vector<std::uint32_t> addresses{base};
        if (header & 0x80000000u) {
            for (unsigned bit = 0; bit != 16; ++bit)
                if ((header >> (15 + bit)) & 1)
                    addresses.push_back(base + (bit + 1) * 4);
        } else {
            for (std::uint32_t count = (header >> 15) & 0x3fu;
                 count; --count)
                addresses.push_back(addresses.back() + 4);
        }
        const auto end = cursor + (addresses.size() + 1) * 4;
        if (end > task.size())
            throw std::invalid_argument(
                "h14.conv-clip-outside-envelope: truncated packed record");
        const bool touches = std::any_of(addresses.begin(), addresses.end(),
            [](std::uint32_t address) {
                return address >= 0x195c && address <= 0x19d4;
            });
        if (touches) {
            if (std::any_of(addresses.begin(), addresses.end(),
                    [](std::uint32_t address) {
                        return address < 0x195c || address > 0x19d4;
                    }))
                throw std::invalid_argument(
                    "h14.conv-clip-outside-envelope: kernel-DMA record "
                    "straddles the clip table");
            if (!inserted) {
                appendLE(out, 0x000f0657u, 4);
                for (const auto value : values) appendLE(out, value, 4);
                inserted = true;
            }
        } else {
            out.insert(out.end(),
                       task.begin() + static_cast<std::ptrdiff_t>(cursor),
                       task.begin() + static_cast<std::ptrdiff_t>(end));
        }
        cursor = end;
    }
    if (!inserted)
        throw std::invalid_argument(
            "h14.conv-clip-outside-envelope: depthwise task has no "
            "kernel-DMA lane record");
    storeLE32(out, 0, (static_cast<std::uint32_t>(out.size() / 4) << 16) |
                      (loadLE32(out, 0) & 0xffffu));
    storeLE32(out, 4, (loadLE32(out, 4) & 0xffff0000u) | 2u);
    return out;
}

} // namespace

void fuseConvClip(Program &p, std::uint16_t lower, std::uint16_t upper) {
    if (!p.formulaBuilt)
        throw std::invalid_argument(
            "h14.conv-clip-outside-envelope: clip fusion requires a "
            "formula-built convolution");
    const auto descriptor = clipBoundsDescriptor(lower, upper);
    if (p.mobileRingRuleBuilt) {
        if (p.mobileRingActivation)
            throw std::invalid_argument(
                "h14.conv-clip-outside-envelope: native ring convolution "
                "already has a post-operation");
        std::vector<std::uint8_t> constants;
        for (std::size_t lane = 0; lane != 16; ++lane) {
            const std::size_t offset = p.mobileRingKernelOffsets[lane];
            const std::size_t size = p.mobileRingKernelSizes[lane];
            if (offset + size > p.constants.size())
                throw std::logic_error(
                    "H14 native ring coefficient lane exceeds its section");
            p.mobileRingKernelOffsets[lane] =
                static_cast<std::uint32_t>(constants.size());
            constants.insert(constants.end(), descriptor.begin(),
                             descriptor.end());
            constants.insert(constants.end(),
                p.constants.begin() + static_cast<std::ptrdiff_t>(offset),
                p.constants.begin() + static_cast<std::ptrdiff_t>(offset + size));
            p.mobileRingKernelSizes[lane] =
                static_cast<std::uint32_t>(size + descriptor.size());
        }
        p.constants = std::move(constants);
        p.mobileRingActivation = 2;
        p.mobileRingClipLower = lower;
        p.mobileRingClipUpper = upper;
        return;
    }
    std::vector<std::vector<std::uint8_t>> tasks;
    tasks.reserve(p.taskCount);
    for (std::uint32_t index = 0; index != p.taskCount; ++index) {
        const auto span = h14TaskSpan(p.taskStream, index);
        tasks.emplace_back(
            p.taskStream.begin() + static_cast<std::ptrdiff_t>(span.first),
            p.taskStream.begin() +
                static_cast<std::ptrdiff_t>(span.first + span.second));
    }

    std::vector<std::uint8_t> constants;
    constants.reserve(p.constants.size() +
                      static_cast<std::size_t>(p.taskCount) * 16 *
                          descriptor.size());
    for (auto &bytes : tasks) {
        Program task;
        task.taskStream.assign(16, 0);
        task.taskStream.insert(task.taskStream.end(), bytes.begin(), bytes.end());
        const std::size_t off = 16;
        const std::size_t size = bytes.size();
        const auto selectorAt = h14RegisterOffset(task.taskStream, off, size,
                                                  0x0d04);
        const auto selector = loadLE32(task.taskStream, selectorAt);
        if (selector != 0x00101c00 && selector != 0x00101c10)
            throw std::invalid_argument(
                "h14.conv-clip-outside-envelope: convolution NE word is not "
                "a decoded no-post-operation form");
        storeLE32(task.taskStream, selectorAt,
                  (selector & ~0x00010000u) | 0x00020000u);

        const auto oldBase = get(task, off, size, 0x1908);
        if (constants.size() > UINT32_MAX)
            throw std::invalid_argument(
                "h14.conv-clip-outside-envelope: kernel base overflows");
        const auto newBase = static_cast<std::uint32_t>(constants.size());
        storeLE32(task.taskStream,
                  lastRegister(task, off, size, 0x1908), newBase);
        std::array<std::uint32_t, 16> newOffsets{}, newSizes{};
        for (std::uint32_t lane = 0; lane != 16; ++lane) {
            const std::size_t oldOffset = get(task, off, size,
                                              0x1958 + 4 * lane);
            const std::size_t oldSize = get(task, off, size,
                                            0x1998 + 4 * lane);
            const std::size_t oldGlobal = oldBase + oldOffset;
            if (oldGlobal + oldSize > p.constants.size())
                throw std::invalid_argument(
                    "h14.conv-clip-outside-envelope: kernel lane exceeds the "
                    "constant section");
            if (oldOffset > UINT32_MAX - 128 * lane ||
                oldSize > UINT32_MAX - 128)
                throw std::invalid_argument(
                    "h14.conv-clip-outside-envelope: kernel lane offset or "
                    "size overflows");
            newOffsets[lane] = static_cast<std::uint32_t>(oldOffset +
                                                           128 * lane);
            newSizes[lane] = static_cast<std::uint32_t>(oldSize + 128);
            constants.insert(constants.end(), descriptor.begin(),
                             descriptor.end());
            constants.insert(constants.end(),
                p.constants.begin() + static_cast<std::ptrdiff_t>(oldGlobal),
                p.constants.begin() +
                    static_cast<std::ptrdiff_t>(oldGlobal + oldSize));
        }
        if (p.depthwiseConv) {
            std::array<std::uint32_t, 31> values{};
            std::copy(newOffsets.begin() + 1, newOffsets.end(),
                      values.begin());
            std::copy(newSizes.begin(), newSizes.end(), values.begin() + 15);
            bytes.assign(task.taskStream.begin() + 16, task.taskStream.end());
            bytes = h14ClipDepthwiseTask(bytes, values);
        } else {
            for (std::uint32_t lane = 1; lane != 16; ++lane)
                storeLE32(task.taskStream,
                    lastRegister(task, off, size, 0x1958 + 4 * lane),
                    newOffsets[lane]);
            for (std::uint32_t lane = 0; lane != 16; ++lane)
                storeLE32(task.taskStream,
                    lastRegister(task, off, size, 0x1998 + 4 * lane),
                    newSizes[lane]);
            bytes.assign(task.taskStream.begin() + 16, task.taskStream.end());
        }
    }

    p.taskStream.assign(16, 0);
    for (const auto &task : tasks) {
        p.taskStream.resize(alignUp(p.taskStream.size(), 16), 0);
        p.taskStream.insert(p.taskStream.end(), task.begin(), task.end());
    }
    p.constants = std::move(constants);
    p.firstTaskBytes = tasks.empty() ? 0 : tasks.front().size();
    p.constantOffsetBytes = alignUp(p.taskStream.size(), 64);
}

// Later output slices of a multi-task layer reuse the input the layer's
// first task staged in L2, as Apple's linked H14 programs do (decoded
// chain3_c1024_s28 tasks 1-3 and 6-8, and the stride-2 consumer of
// h14_s2link_pw1024_pws2_2048, tasks 5-9). Such a task carries no source TileDMA
// records and no SourceCfg/depth-stride words (0x504, 0x514, 0x518). Its
// NE repeat fields (0x38 bits 15:8) take the NE class and ResultCfg (0x530)
// drops the bits a DMA-loaded source sets (0x500030). The task must be the
// last one in the stream.
void reuseLayerInput(Program &p, std::size_t off, std::size_t &bytes) {
    if (off + bytes != p.taskStream.size())
        throw std::logic_error("H14 input reuse requires the final task");
    const auto ne = get(p, off, bytes, 0x38);
    const auto cls = (ne >> 16) & 0xfu;
    const auto resultCfg = get(p, off, bytes, 0x530) & ~0x500030u;
    const std::size_t extra = (loadLE32(p.taskStream, off + 28) & 3) == 3 ? 4 : 0;
    std::vector<std::uint8_t> out(p.taskStream.begin() + static_cast<std::ptrdiff_t>(off),
        p.taskStream.begin() + static_cast<std::ptrdiff_t>(off + 32 + extra));
    for (std::size_t c = off + 32 + extra; c < off + bytes;) {
        const auto h = loadLE32(p.taskStream, c);
        const auto base = (h & 0x7fff) * 4;
        std::vector<std::uint32_t> addrs{base};
        if (h & 0x80000000u) {
            for (unsigned b = 0; b < 16; ++b)
                if ((h >> (15 + b)) & 1) addrs.push_back(base + (b + 1) * 4);
        } else {
            for (std::uint32_t n = ((h >> 15) & 0x3f); n; --n)
                addrs.push_back(addrs.back() + 4);
        }
        const auto end = c + (addrs.size() + 1) * 4;
        const bool source = base >= 0x1100 && base < 0x1500;
        if (source != (addrs.back() >= 0x1100 && addrs.back() < 0x1500))
            throw std::logic_error("H14 record straddles the source TileDMA block");
        if (source) { c = end; continue; }
        const bool sourceCfg = std::any_of(addrs.begin(), addrs.end(), [](std::uint32_t a) {
            return a == 0x504 || a == 0x514 || a == 0x518; });
        if (!sourceCfg) {
            out.insert(out.end(), p.taskStream.begin() + static_cast<std::ptrdiff_t>(c),
                       p.taskStream.begin() + static_cast<std::ptrdiff_t>(end));
            c = end;
            continue;
        }
        if (!(h & 0x80000000u))
            throw std::logic_error("H14 input reuse expects the L2 words in a scatter record");
        std::vector<std::pair<std::uint32_t, std::uint32_t>> kept;
        for (std::size_t k = 0; k < addrs.size(); ++k) {
            const auto a = addrs[k];
            if (a == 0x504 || a == 0x514 || a == 0x518) continue;
            kept.emplace_back(a, a == 0x530 ? resultCfg : loadLE32(p.taskStream, c + 4 + 4 * k));
        }
        if (!kept.empty()) {
            const auto first = kept.front().first / 4;
            std::uint32_t mask = 0;
            for (std::size_t k = 1; k < kept.size(); ++k) {
                const auto bit = kept[k].first / 4 - first - 1;
                if (bit >= 16) throw std::logic_error("H14 L2 scatter record does not fit");
                mask |= 1u << bit;
            }
            appendLE(out, 0x80000000u | (mask << 15) | first, 4);
            for (const auto &kv : kept) appendLE(out, kv.second, 4);
        }
        c = end;
    }
    storeLE32(out, 0, (static_cast<std::uint32_t>(out.size() / 4) << 16) |
              (loadLE32(out, 0) & 0xffff));
    p.taskStream.resize(off);
    p.taskStream.insert(p.taskStream.end(), out.begin(), out.end());
    bytes = out.size();
    put(p, off, bytes, 0x38, (ne & ~0xff00u) | (cls << 12) | (cls << 8));
}

Program linkConvChain(const std::vector<Program> &layers,
                      std::vector<LinkedLayer> *layout) {
    if (layers.size() < 2) throw std::invalid_argument("h14.linked-outside-envelope: need at least two conv layers");
    Program result;
    result.taskCount = 0;
    result.formulaBuilt = true;
    result.taskStream.assign(16, 0);
    std::vector<std::uint64_t> regions;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto &p = layers[i];
        if (!p.formulaBuilt || p.inputs.size() != 1 || p.inputs[0].index != 5 ||
            p.output.index != 4 || p.scratchAllocationBytes ||
            !p.taskCount || (i && layers[i - 1].output.nchw != p.inputs[0].nchw))
            throw std::invalid_argument("h14.linked-outside-envelope: incompatible formula-built layers");
        if (taskSizes(p.taskStream).size() != p.taskCount)
            throw std::invalid_argument("h14.linked-outside-envelope: task count differs from stream");
        validateTensor(p.inputs[0], 5, "H14 linked input allocation is too small");
        validateTensor(p.output, 4, "H14 linked output allocation is too small");
        if (p.constants.empty() || p.constantOffsetBytes != alignUp(p.taskStream.size(), 64))
            throw std::invalid_argument("h14.linked-outside-envelope: invalid layer constant section");
        for (std::uint32_t t = 0; t < p.taskCount; ++t) {
            const auto span = h14TaskSpan(p.taskStream, t);
            auto r = [&](std::uint32_t a) { return get(p, span.first, span.second, a); };
            const auto readEnd = static_cast<std::uint64_t>(r(0x1110)) +
                static_cast<std::uint64_t>(r(0x111c)) * r(0xc);
            const auto writeEnd = static_cast<std::uint64_t>(r(0x1508)) +
                static_cast<std::uint64_t>(r(0x1514)) * r(0x10);
            if (readEnd > p.inputs[0].allocationBytes || writeEnd > p.output.allocationBytes ||
                r(0x1118) != p.inputs[0].nchw[5] || r(0x111c) != p.inputs[0].nchw[4] ||
                r(0x1510) != p.output.nchw[5] || r(0x1514) != p.output.nchw[4])
                throw std::invalid_argument("h14.linked-outside-envelope: DMA span exceeds its surface");
            for (unsigned j = 0; j < 16; ++j)
                if (static_cast<std::uint64_t>(r(0x1908)) + r(0x1958 + 4 * j) +
                    r(0x1998 + 4 * j) > p.constants.size())
                    throw std::invalid_argument("h14.linked-outside-envelope: coefficient span exceeds constants");
        }
        if (i + 1 < layers.size()) {
            regions.push_back(result.scratchAllocationBytes);
            result.scratchAllocationBytes += alignTile(p.output.allocationBytes);
        }
    }
    if (result.scratchAllocationBytes > UINT32_MAX)
        throw std::invalid_argument("h14.linked-outside-envelope: scratch arena exceeds 32-bit DMA");
    result.inputs = layers.front().inputs;
    result.output = layers.back().output;
    if (layout) layout->clear();
    for (std::size_t l = 0; l < layers.size(); ++l) {
        const auto &p = layers[l];
        if (layout) layout->push_back({result.taskCount, p.taskCount,
            l ? regions[l - 1] : 0, l + 1 < layers.size() ? regions[l] : 0});
        for (std::uint32_t i = 0; i < p.taskCount; ++i) {
            const auto src = h14TaskSpan(p.taskStream, i);
            const auto off = static_cast<std::size_t>(alignUp(result.taskStream.size(), 16));
            result.taskStream.resize(off, 0);
            result.taskStream.insert(result.taskStream.end(),
                p.taskStream.begin() + static_cast<std::ptrdiff_t>(src.first),
                p.taskStream.begin() + static_cast<std::ptrdiff_t>(src.first + src.second));
            auto bytes = src.second;
            auto set = [&](std::uint32_t a, std::uint32_t v) { put(result, off, bytes, a, v); };
            const bool reuse = p.taskCount > 1 && i > 0;
            if (reuse) reuseLayerInput(result, off, bytes);
            set(0x1908, get(p, src.first, src.second, 0x1908) +
                static_cast<std::uint32_t>(result.constants.size()));
            if (l && !reuse) {
                route(result, off, bytes, 0x1110, 3);
                set(0x1110, static_cast<std::uint32_t>(regions[l - 1]));
                // Native H14 scratch hints + dependency, not H13's 0x7bb81.
                set(0x1100, 0x00000e41u | (i == 0 ? 0x10000000u : 0u));
                set(0x1108, 0x4e);
            }
            if (l + 1 < layers.size()) {
                route(result, off, bytes, 0x1508, 3);
                set(0x1508, get(p, src.first, src.second, 0x1508) +
                    static_cast<std::uint32_t>(regions[l]));
                set(0x1500, (get(p, src.first, src.second, 0x1500) & ~0xf0u) | 0xf0u);
            }
            if (result.taskCount >= 254)
                throw std::invalid_argument("h14.linked-outside-envelope: DMA task-tag limit exceeded");
            const auto tag = result.taskCount << 16;
            if (!reuse)
                set(0x1100, (get(result, off, bytes, 0x1100) & ~0xff0000u) | tag);
            set(0x1500, (get(result, off, bytes, 0x1500) & ~0xff0000u) | tag);
            set(0x1900, (get(result, off, bytes, 0x1900) & ~0xff0000u) | ((result.taskCount + 1) << 16));
            for (unsigned j = 0; j < 16; ++j)
                set(0x1918 + 4 * j, (get(result, off, bytes, 0x1918 + 4 * j) & ~0xff0000u) | tag);
            const bool last = l + 1 == layers.size() && i + 1 == p.taskCount;
            storeLE32(result.taskStream, off,
                (static_cast<std::uint32_t>(bytes / 4) << 16) | result.taskCount);
            storeLE32(result.taskStream, off + 8, (result.taskCount == 0 ? 8u : 0u) | (last ? 0x22u : 0u));
            // LiveOuts: every task of a multi-task layer keeps L2 live, as
            // in Apple's linked programs, so later slices can reuse the
            // staged input; single tasks and the program's last task close
            // it. TSR remains set. H14 has no H13 header[6] TSE flag.
            storeLE32(result.taskStream, off + 24,
                      p.taskCount > 1 && !last ? 0x1fffffffu : 0u);
            storeLE32(result.taskStream, off + 28, 0x50001);
            if (!result.taskCount) result.firstTaskBytes = bytes;
            ++result.taskCount;
        }
        result.constants.insert(result.constants.end(), p.constants.begin(), p.constants.end());
    }
    if (result.taskCount > 0xffff || result.constants.size() > UINT32_MAX)
        throw std::invalid_argument("h14.linked-outside-envelope: task or constant limit exceeded");
    result.constantOffsetBytes = alignUp(result.taskStream.size(), 64);
    return result;
}

namespace {

constexpr std::uint64_t kMobileL2Bytes = 2ull * 1024 * 1024;
constexpr std::uint64_t kMobileL2BankBytes = 64ull * 1024;

std::uint64_t align16(std::uint64_t value) {
    return (value + 15) & ~std::uint64_t(15);
}

bool mobileOverlap(std::uint64_t left, std::uint64_t leftBytes,
                   std::uint64_t right, std::uint64_t rightBytes) {
    if (leftBytes + rightBytes > kMobileL2Bytes) return true;
    for (const auto l : {left % kMobileL2Bytes,
                         left % kMobileL2Bytes + kMobileL2Bytes})
        for (const auto r : {right % kMobileL2Bytes,
                             right % kMobileL2Bytes + kMobileL2Bytes})
            if (l < r + rightBytes && r < l + leftBytes) return true;
    return false;
}

std::uint32_t mobileSkew(const ConvShape &shape) {
    const auto width = shape.input.width;
    if (width >= 64) return shape.stride == 2 ? 0x20u : 0x30u;
    if (width >= 32)
        return shape.input.channels <= 64 ? 0x80u : 0x30u;
    if (width >= 16) return shape.stride == 2 ? 0x40u : 0x50u;
    if (width >= 8) return shape.stride == 2 ? 0x20u : 0x30u;
    return 0x30u;
}

struct MobileGeometry {
    ConvShape shape{};
    bool depthwise = false;
    std::uint32_t sourceChannel = 0;
    std::uint32_t sourceRow = 0;
    std::uint64_t sourceSpan = 0;
};

std::uint32_t mobileCommonClass(const MobileGeometry &g) {
    const auto width = g.shape.output.width;
    std::uint32_t ceilLog = 0;
    for (auto value = width - 1; value; value >>= 1) ++ceilLog;
    const auto stage = ceilLog >= 7 ? 0u : 7u - ceilLog;
    if (!stage)
        return 0x00230001u |
            ((!g.depthwise && g.shape.kernel == 1) ? 0x00001100u : 0u);
    if (stage == 1)
        return g.shape.stride == 2 ? 0x00211101u : 0x00243301u;
    const auto repeated = std::min(stage, 4u);
    return 0x00200001u | (repeated << 16) | (repeated << 12) |
        (repeated << 8) | (repeated == 4 ? 4u : 0u);
}

std::uint32_t mobileCost(const MobileGeometry &g, bool terminal) {
    const auto width = g.shape.input.width;
    if (g.shape.input.channels == 3 && g.shape.output.channels == 32)
        return 8;
    if (g.depthwise) {
        if (g.shape.stride == 2) return std::max(1u, width / 8);
        if (width >= 64) return 0x17;
        if (width >= 32) return 9;
        if (width >= 16) return 3;
        if (width >= 8) return 1;
        return 2;
    }
    const bool expands = g.shape.output.channels > g.shape.input.channels;
    if (expands) {
        if (width >= 64) return 0x17;
        if (width >= 32) return 3;
        if (width >= 16) return 3;
        if (width >= 8) return 7;
        return 0x1e;
    }
    if (width >= 32) return 6;
    if (width >= 16) return 7;
    if (width >= 8) return 0xf;
    return terminal ? 0x4a : 0x3d;
}

void appendHalf(std::vector<std::uint8_t> &out, const std::uint8_t *value) {
    out.push_back(value[0]);
    out.push_back(value[1]);
}

bool nonzeroHalf(const std::uint8_t *value) {
    return value[0] || (value[1] & 0x7f);
}

using MobilePick = std::pair<unsigned, unsigned>;

std::vector<MobilePick> mobilePicks(bool wide, unsigned row,
                                    unsigned mask) {
    if (!wide) {
        if (row == 0 && mask == 0)
            return {{0, 0}, {2, 2}, {4, 6}, {6, 8}};
        if (row == 0) return {{0, 1}, {4, 7}};
        if (mask == 0) return {{0, 3}, {2, 5}};
        return {{0, 4}};
    }
    if (row == 0 && mask == 0) return {{0, 0}, {4, 2}};
    if (row == 0) return {{0, 6}, {4, 8}};
    if (row == 1 && mask == 0) return {{0, 1}};
    if (row == 1) return {{0, 7}};
    if (row == 2 && mask == 0) return {{0, 3}, {4, 5}};
    if (row == 3 && mask == 0) return {{0, 4}};
    return {};
}

template <class Weight>
void appendMobileSparsePlane(std::vector<std::uint8_t> &out, bool wide,
                             unsigned repeats, unsigned lanes,
                             Weight weight) {
    const auto start = out.size();
    out.insert(out.end(), 3, 0); // u16 body length and structural zero
    const unsigned rows = wide ? 4 : 2;
    for (unsigned repeat = 0; repeat != repeats; ++repeat) {
        for (unsigned row = 0; row != rows; ++row) {
            std::array<std::uint8_t, 2> masks{};
            std::array<std::vector<std::uint8_t>, 2> values;
            for (unsigned which = 0; which != 2; ++which) {
                for (const auto &[bit, tap] : mobilePicks(wide, row, which)) {
                    for (unsigned lane = 0; lane != lanes; ++lane) {
                        const auto *value = weight(repeat, lane, tap);
                        if (!nonzeroHalf(value)) continue;
                        masks[which] |= static_cast<std::uint8_t>(1u << (bit + lane));
                        appendHalf(values[which], value);
                    }
                }
            }
            // H14's phase grammar places both masks before either payload.
            out.push_back(masks[0]);
            out.push_back(masks[1]);
            out.insert(out.end(), values[0].begin(), values[0].end());
            out.insert(out.end(), values[1].begin(), values[1].end());
        }
    }
    const auto body = out.size() - start - 3;
    if (body > 0xffff)
        throw std::invalid_argument(
            "H14 native stride-2 sparse plane exceeds its 16-bit length");
    out[start] = static_cast<std::uint8_t>(body);
    out[start + 1] = static_cast<std::uint8_t>(body >> 8);
}

std::vector<std::uint8_t> packMobileSparse(
        const ConvShape &shape, const std::uint8_t *weights,
        const std::uint8_t *bias, std::array<std::uint32_t, 16> &offsets,
        std::array<std::uint32_t, 16> &sizes) {
    const bool dw = depthwise(shape);
    const bool stem = shape.groups == 1 && shape.input.channels == 3 &&
        shape.output.channels == 32;
    if (shape.kernel != 3 || shape.kernelWidth != 3 || shape.stride != 2 ||
        (!dw && !stem))
        throw std::invalid_argument(
            "H14 native sparse packing covers a MobileNet stem or depthwise s2");
    std::vector<std::uint8_t> packed;
    for (unsigned chunk = 0; chunk != 16; ++chunk) {
        const auto start = packed.size();
        offsets[chunk] = static_cast<std::uint32_t>(start);
        if (dw) {
            const auto perChunk = shape.input.channels / 16;
            for (unsigned channel = chunk; channel < shape.input.channels;
                 channel += 16) {
                if (bias) appendHalf(packed, bias + 2 * channel);
                appendMobileSparsePlane(packed, perChunk >= 32, 1, 1,
                    [&](unsigned, unsigned, unsigned tap) {
                        return weights + 2 *
                            (static_cast<std::size_t>(channel) * 9 + tap);
                    });
            }
        } else {
            const auto output = 2 * chunk;
            if (bias) {
                appendHalf(packed, bias + 2 * output);
                appendHalf(packed, bias + 2 * (output + 1));
            }
            appendMobileSparsePlane(packed, false, shape.input.channels, 2,
                [&](unsigned input, unsigned lane, unsigned tap) {
                    return weights + 2 *
                        ((static_cast<std::size_t>(output + lane) *
                          shape.input.channels + input) * 9 + tap);
                });
        }
        packed.resize(static_cast<std::size_t>(alignUp(packed.size(), 64)), 0);
        sizes[chunk] = static_cast<std::uint32_t>(packed.size() - start);
    }
    return packed;
}

void ringScatter(std::vector<std::uint8_t> &task,
                 const std::vector<std::pair<std::uint32_t,
                                             std::uint32_t>> &words) {
    if (words.empty()) return;
    const auto base = words.front().first;
    if (base % 4) throw std::logic_error("H14 ring register is not aligned");
    std::uint32_t mask = 0;
    for (std::size_t i = 1; i != words.size(); ++i) {
        const auto bit = words[i].first / 4 - base / 4 - 1;
        if (bit >= 16) throw std::logic_error("H14 ring scatter span is too wide");
        mask |= 1u << bit;
    }
    appendLE(task, 0x80000000u | (mask << 15) | (base / 4), 4);
    for (const auto &word : words) appendLE(task, word.second, 4);
}

void ringDense(std::vector<std::uint8_t> &task, std::uint32_t address,
               const std::vector<std::uint32_t> &words,
               int barSlot = -1) {
    if (words.empty() || words.size() > 64 || address % 4)
        throw std::logic_error("invalid H14 ring dense record");
    auto header = static_cast<std::uint32_t>(address / 4) |
        (static_cast<std::uint32_t>(words.size() - 1) << 15);
    if (barSlot >= 0)
        header |= 1u << 29 | static_cast<std::uint32_t>(barSlot) << 23;
    appendLE(task, header, 4);
    for (const auto word : words) appendLE(task, word, 4);
}

std::uint32_t liveBanks(std::uint64_t base, std::uint64_t bytes) {
    std::uint32_t mask = 0;
    for (std::uint32_t bank = 0; bank != 32; ++bank)
        if (mobileOverlap(base, bytes,
                          static_cast<std::uint64_t>(bank) * kMobileL2BankBytes,
                          kMobileL2BankBytes))
            mask |= 1u << bank;
    return mask;
}

std::uint32_t denseLanes(const ConvShape &shape) {
    auto cap = std::max(shape.input.height, shape.input.width) <= 8 ? 32u : 16u;
    auto lanes = cap;
    while (lanes > shape.output.channels / 16) lanes /= 2;
    return std::max(1u, lanes);
}

std::vector<std::uint8_t> mobileTask(
        const Program &layer, const MobileGeometry &g, std::uint32_t index,
        bool first, bool last, bool readsScratch, bool writesScratch,
        bool inputResident, bool outputResident, std::uint32_t sourceCfg,
        std::uint64_t sourceBase, std::uint64_t sourceSpan,
        std::uint32_t resultCfg, std::uint64_t resultBase,
        std::uint64_t resultSpan, std::uint32_t resultChannel,
        std::uint32_t resultRow, std::uint64_t spillBytes,
        std::uint64_t coefficientBase) {
    std::vector<std::uint8_t> task(32, 0);
    const bool sparse = g.shape.kernel == 3 && g.shape.stride == 2;
    const auto mac = 0x00101c10u |
        (layer.mobileRingActivation == 1 ? 0x00010000u : 0u) |
        (layer.mobileRingActivation == 2 ? 0x00020000u : 0u);

    storeLE32(task, 4, mobileCost(g, last));
    storeLE32(task, 8, (first ? 8u : 0u) | (last ? 0x22u : 0u));
    storeLE32(task, 16, (first || last) ? 0x00fff868u : 0x00fff800u);
    std::uint32_t banks = 0;
    if (inputResident && !last) banks |= liveBanks(sourceBase, sourceSpan);
    if (outputResident) banks |= liveBanks(resultBase, resultSpan);
    storeLE32(task, 24, banks);
    storeLE32(task, 28, 0x00050001u);

    // Coefficient DMA is emitted first so the BAR relocation is a later,
    // last-write-wins record.  Its tag and addend come solely from stream
    // position and the concatenated coefficient allocator.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> kernel = {
        {0x1900, ((index + 1) << 16) | 0x240u |
                  (sparse ? 0x20u : 0u) | (last ? 0x80u : 0u)},
    };
    for (unsigned lane = 0; lane != 11; ++lane)
        kernel.emplace_back(0x1918 + 4 * lane, (index << 16) | 0x21u);
    ringScatter(task, kernel);
    std::vector<std::uint32_t> tailTags(5, (index << 16) | 0x21u);
    ringDense(task, 0x1944, tailTags);
    std::vector<std::uint32_t> laneWords;
    for (unsigned lane = 1; lane != 16; ++lane)
        laneWords.push_back(layer.mobileRingKernelOffsets[lane]);
    if (!writesScratch)
        for (const auto size : layer.mobileRingKernelSizes)
            laneWords.push_back(size);
    ringDense(task, 0x195c, laneWords);

    std::vector<std::pair<std::uint32_t, std::uint32_t>> common = {
        {0x0000, (g.shape.input.height << 16) | g.shape.input.width},
        {0x000c, g.shape.input.channels},
        {0x0010, g.shape.output.channels},
        {0x0014, (g.shape.output.height << 16) | g.shape.output.width},
    };
    const auto lanes = g.depthwise ? 1u : denseLanes(g.shape);
    const auto ocg = g.depthwise ? 0u : (lanes == 8 ? 0u : log2Floor(lanes));
    if (g.depthwise || ocg || sparse) common.emplace_back(0x001c, ocg);
    if (g.shape.kernel != 1) {
        const auto pad = sparse ? 0u : static_cast<std::uint32_t>(padding(g.shape));
        common.emplace_back(0x0020, 0x50000000u | g.shape.kernelWidth |
            (g.shape.kernel << 6) | (g.shape.stride << 13) |
            (g.shape.stride << 15) | (pad << 17) | (pad << 22));
    }
    if (g.depthwise) common.emplace_back(0x0028, 0x00014001u);
    common.emplace_back(0x002c, g.shape.output.height);
    common.emplace_back(0x0038, mobileCommonClass(g));
    common.emplace_back(0x003c, 1u << 20);
    ringScatter(task, common);

    std::vector<std::pair<std::uint32_t, std::uint32_t>> l2 = {
        {0x0504, sourceCfg},
    };
    if (sourceBase) l2.emplace_back(0x0508, static_cast<std::uint32_t>(sourceBase));
    l2.emplace_back(0x050c, g.sourceChannel);
    l2.emplace_back(0x0510, g.sourceRow);
    if (!inputResident) {
        const auto depth = g.shape.input.channels * g.sourceChannel;
        l2.emplace_back(0x0514, depth);
        l2.emplace_back(0x0518, depth);
    }
    l2.emplace_back(0x0530, resultCfg);
    if (resultBase) l2.emplace_back(0x0534, static_cast<std::uint32_t>(resultBase));
    if (outputResident) {
        l2.emplace_back(0x0538, resultChannel);
        l2.emplace_back(0x053c, resultRow);
    }
    ringScatter(task, l2);

    std::vector<std::pair<std::uint32_t, std::uint32_t>> ne;
    if (sparse) ne.emplace_back(0x0d00, 0x182u);
    ne.emplace_back(0x0d04, mac);
    ne.emplace_back(0x0d10, 0x3c00u);
    ringScatter(task, ne);

    if (first || readsScratch) {
        const auto row = first ? layer.inputs[0].nchw[5]
                               : 16ull * g.shape.input.width;
        const auto plane = first ? layer.inputs[0].nchw[4]
                                 : row * g.shape.input.height;
        ringScatter(task, {
            {0x1100, first ? 0x00000e21u
                           : 0x10000e41u | (index << 16)},
            {0x1108, first ? 0xceu : 0x4eu},
            {0x1118, static_cast<std::uint32_t>(row)},
            {0x111c, static_cast<std::uint32_t>(plane)},
        });
        ringDense(task, 0x1150, {first ? 0x01002031u : 0x08002031u});
    }

    if (writesScratch || last) {
        const auto row = writesScratch ? 16ull * g.shape.output.width
                                       : layer.output.nchw[5];
        const auto plane = writesScratch ? row * g.shape.output.height
                                         : layer.output.nchw[4];
        ringScatter(task, {
            {0x1500, writesScratch ? 0x01000ef1u | (index << 16)
                                   : 0x01000e31u | (index << 16)},
            {0x1510, static_cast<std::uint32_t>(row)},
            {0x1514, static_cast<std::uint32_t>(plane)},
            {0x1520, writesScratch ? 0x08302031u : 0x01302031u},
        });
        // A symbolic dead destination extension pads the compact role to the
        // minimum task size accepted by firmware; it is never BAR-routed.
        ringDense(task, 0x1524, {0xdead});
        ringDense(task, 0x1524, {0xdead});
    }
    ringDense(task, 0x1908,
              {static_cast<std::uint32_t>(coefficientBase), 0}, 1);
    if (first || readsScratch)
        ringDense(task, 0x1110, {0, 0}, first ? 5 : 3);
    if (writesScratch || last)
        ringDense(task, 0x1508, {0, 0}, writesScratch ? 3 : 4);

    // Firmware's compact-task envelope starts at 312 bytes.  Padding is a
    // harmless write to the documented dead destination extension, derived
    // from the encoded length rather than from a captured per-task size list.
    while (task.size() < 312) ringDense(task, 0x1524, {0xdead});

    storeLE32(task, 0,
              (static_cast<std::uint32_t>(task.size() / 4) << 16) | index);
    (void)spillBytes;
    return task;
}

} // namespace

std::string convLinkedRingRefusal(ConvShape shape) {
    const bool dw = depthwise(shape);
    const bool stem = shape.groups == 1 && shape.input.channels == 3 &&
        shape.output.channels == 32 && shape.kernel == 3 && shape.stride == 2;
    if (!shape.bias) return "native MobileNet ring convolutions require bias";
    if (!stem && !dw && !(shape.groups == 1 && shape.kernel == 1 &&
                          shape.kernelWidth == 1 && shape.stride == 1))
        return "native ring covers the stem, depthwise k3, and dense k1 forms";
    if (dw && (shape.kernel != 3 || shape.kernelWidth != 3 ||
               (shape.stride != 1 && shape.stride != 2)))
        return "native depthwise ring layers require k3 stride 1 or 2";
    if (!stem && (shape.input.channels % 16 || shape.output.channels % 16))
        return "native ring channels must be multiples of 16";
    if (padding(shape) < 0)
        return "native ring geometry must imply symmetric SAME or VALID padding";
    return {};
}

Program encodeConvLinkedRing(ConvShape shape, const std::uint8_t *weights,
                             std::size_t weightBytes,
                             const std::uint8_t *bias,
                             std::size_t biasBytes) {
    const auto refusal = convLinkedRingRefusal(shape);
    if (!refusal.empty())
        throw std::invalid_argument("h14.linked-ring-outside-envelope: " + refusal);
    const bool dw = depthwise(shape);
    const auto reduction = (dw ? 1u : shape.input.channels) * shape.kernel *
        shape.kernelWidth;
    if (!weights || weightBytes !=
            static_cast<std::size_t>(shape.output.channels) * reduction * 2 ||
        !bias || biasBytes != static_cast<std::size_t>(shape.output.channels) * 2)
        throw std::invalid_argument(
            "h14.linked-ring-outside-envelope: invalid coefficient payload");

    Program program;
    program.formulaBuilt = true;
    program.depthwiseConv = dw;
    program.mobileRingRuleBuilt = true;
    program.mobileRingShape = shape;
    program.inputs = {convTensor(5, shape.input)};
    program.output = convTensor(4, shape.output);
    program.taskStream.assign(16, 0); // no captured prefix; linker replaces it
    program.taskCount = 1;
    if (shape.kernel == 3 && shape.stride == 2) {
        program.constants = packMobileSparse(shape, weights, bias,
            program.mobileRingKernelOffsets, program.mobileRingKernelSizes);
    } else {
        program.constants = dw
            ? packConvDepthwise(shape.kernel * shape.kernelWidth,
                                shape.output.channels, weights, bias)
            : packConvDense(reduction,
                            laneChunks(shape.output.channels,
                                       laneCap(shape.input)),
                            weights, bias, false);
        if (program.constants.size() % 16)
            throw std::logic_error("H14 native coefficient section has unequal lanes");
        const auto laneBytes = static_cast<std::uint32_t>(
            program.constants.size() / 16);
        for (std::uint32_t lane = 0; lane != 16; ++lane) {
            program.mobileRingKernelOffsets[lane] = lane * laneBytes;
            program.mobileRingKernelSizes[lane] = laneBytes;
        }
    }
    program.constantOffsetBytes = 64;
    return program;
}

Program linkMobileNetRing(const std::vector<Program> &layers,
                          std::vector<LinkedLayer> *layout) {
    if (layers.empty() || layers.size() > 28)
        throw std::invalid_argument(
            "h14.linked-ring-outside-envelope: expected a prefix of 1..28 layers");
    std::vector<MobileGeometry> geometry;
    geometry.reserve(layers.size());
    for (std::size_t index = 0; index != layers.size(); ++index) {
        const auto &layer = layers[index];
        if (!layer.mobileRingRuleBuilt || !layer.formulaBuilt ||
            layer.taskCount != 1 || layer.inputs.size() != 1)
            throw std::invalid_argument(
                "h14.linked-ring-outside-envelope: every layer must be a "
                "rule-built native convolution");
        if (index && layers[index - 1].mobileRingShape.output.channels !=
                         layer.mobileRingShape.input.channels)
            throw std::invalid_argument(
                "h14.linked-ring-outside-envelope: channel chain is discontinuous");
        if (index && (layers[index - 1].mobileRingShape.output.height !=
                          layer.mobileRingShape.input.height ||
                      layers[index - 1].mobileRingShape.output.width !=
                          layer.mobileRingShape.input.width))
            throw std::invalid_argument(
                "h14.linked-ring-outside-envelope: spatial chain is discontinuous");
        MobileGeometry g;
        g.shape = layer.mobileRingShape;
        g.depthwise = layer.depthwiseConv;
        g.sourceChannel = static_cast<std::uint32_t>(align16(
            2ull * g.shape.input.width));
        g.sourceRow = g.shape.input.channels * g.sourceChannel +
            (index ? mobileSkew(g.shape) : 0u);
        g.sourceSpan = static_cast<std::uint64_t>(g.shape.input.height) *
            g.sourceRow;
        geometry.push_back(g);
    }
    const auto n = layers.size();
    if (n >= 4) {
        // The bridge width is allocator-selected.  224 narrows to 64 pixels;
        // a 192 input fits at full width, proving this is not a layer table.
        auto &bridge = geometry[2];
        const auto result = geometry[3].sourceSpan;
        auto channel = bridge.sourceChannel;
        while (channel > 16 && result +
                static_cast<std::uint64_t>(bridge.shape.input.height) *
                    (bridge.shape.input.channels * channel +
                     (channel == bridge.sourceChannel
                          ? mobileSkew(bridge.shape) : 0x50u)) > kMobileL2Bytes)
            channel -= 16;
        bridge.sourceChannel = channel;
        bridge.sourceRow = bridge.shape.input.channels * channel +
            (channel == static_cast<std::uint32_t>(align16(
                 2ull * bridge.shape.input.width))
                ? mobileSkew(bridge.shape) : 0x50u);
        bridge.sourceSpan = static_cast<std::uint64_t>(bridge.shape.input.height) *
            bridge.sourceRow;
        if (result + bridge.sourceSpan > kMobileL2Bytes)
            throw std::invalid_argument(
                "h14.linked-ring-outside-envelope: spill bridge exceeds L2");
    }

    std::vector<std::uint32_t> sourceCfg(n, 0x00500172u);
    std::vector<std::uint32_t> resultCfg(n, 0x14au);
    std::vector<std::uint64_t> sourceBase(n), resultBase(n), resultSpan(n);
    std::vector<bool> readsScratch(n), writesScratch(n), outputResident(n);
    const auto standaloneResultSpan = [&](std::size_t i) {
        const auto channel = align16(2ull * geometry[i].shape.output.width);
        const auto row = geometry[i].shape.output.channels * channel + 0x30u;
        return static_cast<std::uint64_t>(geometry[i].shape.output.height) * row;
    };
    if (n == 1) sourceBase[0] = align16(standaloneResultSpan(0));
    if (n >= 2) {
        outputResident[0] = true;
        resultCfg[0] = 0x00500170u;
        resultSpan[0] = geometry[1].sourceSpan;
        sourceBase[0] = resultSpan[0];
        sourceCfg[1] = 0x120u;
    }
    if (n == 2) {
        resultBase[1] = align16(geometry[1].sourceSpan) % kMobileL2Bytes;
    } else if (n == 3) {
        outputResident[1] = true;
        resultCfg[1] = 0x140u;
        resultSpan[1] = geometry[2].sourceSpan;
        resultBase[1] = align16(geometry[1].sourceSpan) % kMobileL2Bytes;
        sourceBase[2] = resultBase[1];
        sourceCfg[2] = 0x120u;
        resultBase[2] = align16(sourceBase[2] + geometry[2].sourceSpan) %
            kMobileL2Bytes;
    } else if (n >= 4) {
        writesScratch[1] = true;
        readsScratch[2] = true;
        resultCfg[1] = 0x84au;
        resultBase[1] = geometry[1].sourceSpan;
        sourceCfg[2] = 0x842u;
        outputResident[2] = true;
        resultCfg[2] = 0x140u;
        resultSpan[2] = geometry[3].sourceSpan;
        sourceBase[2] = resultSpan[2];
        sourceCfg[3] = 0x120u;
        for (std::size_t i = 3; i + 1 < n; ++i) {
            const auto &consumer = geometry[i + 1];
            const auto quarter = geometry.front().shape.input.width / 4;
            const auto eighth = geometry.front().shape.input.width / 8;
            const bool wide = consumer.shape.input.width > quarter ||
                (consumer.shape.input.width == quarter &&
                 consumer.shape.stride == 1) ||
                (consumer.shape.input.width >= eighth &&
                 consumer.shape.output.channels > consumer.shape.input.channels) ||
                (consumer.depthwise && geometry[i].shape.output.channels >
                                           geometry[i].shape.input.channels &&
                 consumer.shape.input.width >= eighth);
            sourceCfg[i + 1] = wide ? 0x120u : 0x104u;
            outputResident[i] = true;
            resultCfg[i] = 0x140u;
            resultSpan[i] = consumer.sourceSpan;
            auto candidate = align16(sourceBase[i] + geometry[i].sourceSpan) %
                kMobileL2Bytes;
            if (!geometry[i].depthwise &&
                geometry[i].shape.output.channels >
                    geometry[i].shape.input.channels) {
                candidate = 0;
            } else if (geometry[i].depthwise && i + 2 < n &&
                       geometry[i + 1].shape.output.channels >
                           geometry[i + 1].shape.input.channels) {
                const auto future = geometry[i + 2].sourceSpan;
                if (mobileOverlap(candidate, resultSpan[i], 0, future))
                    candidate = align16(future);
            }
            resultBase[i] = candidate;
            sourceBase[i + 1] = candidate;
        }
        const auto last = n - 1;
        outputResident[last] = false;
        resultCfg[last] = 0x14au;
        resultBase[last] = align16(sourceBase[last] +
                                   geometry[last].sourceSpan) % kMobileL2Bytes;
    }

    std::uint64_t spillBytes = 0;
    Program result;
    result.formulaBuilt = true;
    result.mobileRingRuleBuilt = true;
    result.inputs = layers.front().inputs;
    result.output = layers.back().output;
    if (n >= 4) {
        const auto row = 16ull * geometry[1].shape.output.width;
        const auto plane = row * geometry[1].shape.output.height;
        spillBytes = plane * (geometry[1].shape.output.channels / 8);
        result.scratchAllocationBytes =
            2 * layers[1].output.allocationBytes + spillBytes;
    }
    result.taskStream.assign(16, 0);
    if (layout) layout->clear();
    for (std::size_t i = 0; i != n; ++i) {
        const bool inputResident = i && !readsScratch[i];
        const auto task = mobileTask(layers[i], geometry[i],
            static_cast<std::uint32_t>(i), i == 0, i + 1 == n,
            readsScratch[i], writesScratch[i], inputResident,
            outputResident[i], sourceCfg[i], sourceBase[i],
            geometry[i].sourceSpan, resultCfg[i], resultBase[i],
            resultSpan[i], outputResident[i] ? geometry[i + 1].sourceChannel : 0,
            outputResident[i] ? geometry[i + 1].sourceRow : 0,
            spillBytes, result.constants.size());
        result.taskStream.resize(static_cast<std::size_t>(alignUp(
            result.taskStream.size(), taskAlignment)), 0);
        if (!i) result.firstTaskBytes = task.size();
        result.taskStream.insert(result.taskStream.end(), task.begin(), task.end());
        result.constants.insert(result.constants.end(), layers[i].constants.begin(),
                                layers[i].constants.end());
        if (layout) {
            LinkedLayer item{static_cast<std::uint32_t>(i), 1, 0, 0};
            item.readsScratch = readsScratch[i];
            item.writesScratch = writesScratch[i];
            item.inputEdge = inputResident ? (sourceCfg[i] == 0x104u ? 3 : 4) : 0;
            item.outputEdge = outputResident[i] && i + 1 < n
                ? (sourceCfg[i + 1] == 0x104u ? 3 : 4) : 0;
            item.l2Source = inputResident ? sourceBase[i] : 0;
            item.l2Result = outputResident[i] ? resultBase[i] : 0;
            item.l2ResultBytes = outputResident[i] ? resultSpan[i] : 0;
            layout->push_back(item);
        }
    }
    result.taskCount = static_cast<std::uint32_t>(n);
    result.constantOffsetBytes = static_cast<std::size_t>(alignUp(
        result.taskStream.size(), 64));
    return result;
}
