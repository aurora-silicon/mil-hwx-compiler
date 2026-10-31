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
        storeLE32(task.taskStream, span.first + 8, 0x2a);
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
    for (std::uint32_t i = 0; i < p.taskCount; ++i) {
        const auto span = h14TaskSpan(p.taskStream, i);
        const auto at = h14RegisterOffset(p.taskStream, span.first, span.second, 0xd04);
        storeLE32(p.taskStream, at, loadLE32(p.taskStream, at) | 0x10000u);
    }
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
