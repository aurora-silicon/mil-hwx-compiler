// Formula-built H14 convolution and DRAM linking. Included in H14Program.cpp
// to share the packed-record reader and the independently derived packers.
namespace {

using namespace fields;

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
        const auto base = recordAddress.get(h) * 4;
        std::size_t n = 0;
        if (h & scatterPresent.mask()) {
            if (address == base) found = c + 4;
            n = 1;
            for (unsigned b = 0; b < 16; ++b) if ((h >> (15 + b)) & 1) {
                if (address == base + (b + 1) * 4) found = c + 4 * (n + 1);
                ++n;
            }
        } else {
            n = denseCountMinusOne.get(h) + 1;
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
    if (bytes / 4 >= (std::uint32_t(1) << taskWordCount.width))
        throw std::invalid_argument("h14.conv-outside-envelope: task is too large");
    storeLE32(p.taskStream, off,
              taskHeader0(static_cast<std::uint32_t>(bytes / 4),
                          taskIdentifier.get(loadLE32(p.taskStream, off))));
}

// BAR records are dense records with bit 29. Bits 28:23 name the BAR slot;
// unlike H13, the routing is on the RECORD, not in an extra header word.
void route(Program &p, std::size_t off, std::size_t bytes,
            std::uint32_t address, std::uint32_t slot) {
    std::size_t cursor = off + 32;
    while (cursor < off + bytes) {
        const auto h = loadLE32(p.taskStream, cursor);
        std::size_t count = 0;
        if (h & scatterPresent.mask()) {
            count = 1;
            for (unsigned b = 0; b < 16; ++b) count += (h >> (15 + b)) & 1;
        } else {
            count = denseCountMinusOne.get(h) + 1;
            if ((h & barPresent.mask()) &&
                recordAddress.get(h) * 4 == address) {
                storeLE32(p.taskStream, cursor,
                    (h & ~barSlot.mask()) | barSlot.value(slot));
                return;
            }
        }
        cursor += (count + 1) * 4;
    }
    throw std::invalid_argument("h14.linked-outside-envelope: missing BAR record");
}

std::uint32_t skew(std::uint32_t stride, std::uint32_t width,
                    std::uint32_t channels) {
    constexpr std::uint32_t oneLane = 16 * sizeof(std::uint16_t);
    constexpr std::uint32_t oneAndHalfLanes = 24 * sizeof(std::uint16_t);
    constexpr std::uint32_t twoAndHalfLanes = 40 * sizeof(std::uint16_t);
    if (stride == 2) return width >= 64 ? oneAndHalfLanes : oneLane;
    if (width >= 64) return 0;
    if (width > 16) return twoAndHalfLanes;
    return channels >= 32 ? oneAndHalfLanes : oneLane;
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
    constexpr std::uint64_t kernelBudget = 1024ull * 1024;
    if (!dw && perChannel * slice > kernelBudget) {
        const auto fit = static_cast<std::uint32_t>(kernelBudget / perChannel);
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
            static_cast<std::uint64_t>(tile) * slice * outRow >
                30ull * 64 * 1024) --tile;
    if (!tile) throw std::invalid_argument("h14.conv-outside-envelope: one row exceeds L2 budget");
    const auto rows = std::min(s.input.height, tile * s.stride + s.kernel - 1);
    auto cls = s.output.width <= 8 ? 4u :
        7 - std::min(7u, log2Floor(s.output.width - 1) + 1);
    if (s.kernel == 1 && !dw) cls = std::min(cls, log2Floor(s.output.channels / 16) + 1);
    if (s.input.height == 1 && s.input.width == 1 && s.kernel == 1) cls = 0;
    const bool repeat = cls == 4 || (cls >= 2 && !dw &&
        (s.kernel == 3 || (s.input.channels >= 512 && s.output.channels >= 512))) ||
        (s.output.width <= 8 && s.kernel == 1);
    const auto cfg = scheduleConfig(cls, repeat ? cls : 0,
        repeat ? cls : 0, s.output.width <= 8);
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
        set(registers::inputSurface, surface(s.input.height, s.input.width));
        set(registers::inputChannels, s.input.channels);
        set(registers::outputChannels, part.output.channels);
        set(registers::outputSurface, surface(s.output.height, s.output.width));
        set(registers::outputChannelGroup,
            outputChannelGroup.value(dw ? 0 : log2Floor(chunks.front())));
        if (s.kernel != 1 || s.stride != 1)
            set(registers::convolution, convolutionConfig(s.kernel, s.kernel,
                s.stride, static_cast<std::uint32_t>(padding(s))));
        set(registers::tileHeight, tile); set(registers::schedule, cfg);
        set(registers::sourceChannelStride, row);
        set(registers::sourceRowStride, sourceRow);
        set(registers::sourceDepthStride, row * s.input.channels);
        set(registers::sourcePlaneStride, row * s.input.channels);
        set(registers::resultBase, rows * sourceRow);
        set(registers::sourceDmaRowStride,
            static_cast<std::uint32_t>(result.inputs[0].nchw[5]));
        set(registers::sourceDmaPlaneStride,
            static_cast<std::uint32_t>(result.inputs[0].nchw[4]));
        set(registers::resultDmaBase,
            static_cast<std::uint32_t>(result.output.nchw[4] * first));
        set(registers::resultDmaRowStride,
            static_cast<std::uint32_t>(result.output.nchw[5]));
        set(registers::resultDmaPlaneStride,
            static_cast<std::uint32_t>(result.output.nchw[4]));
        set(registers::kernelDmaBase,
            static_cast<std::uint32_t>(result.constants.size()));
        for (unsigned i = 0; i < 16; ++i) {
            set(registers::kernelLaneOffset + 4 * i, offsets[i]);
            set(registers::kernelLaneSize + 4 * i, sizes[i]);
        }
        // The decoded multi-task H14 corpus advances each DMA's task tag.
        set(registers::sourceDmaConfig,
            (get(task, span.first, bytes, registers::sourceDmaConfig) &
             ~dmaTaskTag.mask()) | dmaTaskTag.value(id));
        set(registers::resultDmaConfig,
            (get(task, span.first, bytes, registers::resultDmaConfig) &
             ~dmaTaskTag.mask()) | dmaTaskTag.value(id));
        set(registers::kernelDmaConfig,
            (get(task, span.first, bytes, registers::kernelDmaConfig) &
             ~kernelNextTaskTag.mask()) | kernelNextTaskTag.value(id + 1));
        for (unsigned i = 0; i < 16; ++i)
            set(registers::kernelLaneConfig + 4 * i,
                (get(task, span.first, bytes,
                     registers::kernelLaneConfig + 4 * i) & ~dmaTaskTag.mask()) |
                 dmaTaskTag.value(id));
        storeLE32(task.taskStream, span.first,
            taskHeader0(static_cast<std::uint32_t>(bytes / 4), id));
        // Header word 2 opens the program on its first task and closes it on
        // its last, as in every decoded multi-task H14 program. A single task
        // carries both. Setting both on every task of a multi-task program made the
        // firmware's next command never complete on the M2.
        storeLE32(task.taskStream, span.first + 8,
            taskHeader2(id == 0, id + 1 == taskCount));
        storeLE32(task.taskStream, span.first + 24, 0);
        storeLE32(task.taskStream, span.first + 28, taskHeader7());
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
        if (p.mobileRingActivation != Activation::None)
            throw std::invalid_argument("H14 native ring convolution already has a post-operation");
        p.mobileRingActivation = Activation::Relu;
        return;
    }
    for (std::uint32_t i = 0; i < p.taskCount; ++i) {
        const auto span = h14TaskSpan(p.taskStream, i);
        const auto at = h14RegisterOffset(p.taskStream, span.first,
            span.second, registers::macConfig);
        storeLE32(p.taskStream, at, applyMacActivation(
            loadLE32(p.taskStream, at), Activation::Relu));
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
    slot(clipUnity0Offset / sizeof(std::uint16_t), fp16One);
    slot(clipUnity1Offset / sizeof(std::uint16_t), fp16One);
    slot(clipScaleOffset / sizeof(std::uint16_t), clipDescriptorScale);
    slot(clipEnableOffset / sizeof(std::uint16_t), clipDescriptorEnable);
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
        const auto base = recordAddress.get(header) * 4;
        std::vector<std::uint32_t> addresses{base};
        if (header & scatterPresent.mask()) {
            for (unsigned bit = 0; bit != 16; ++bit)
                if ((header >> (15 + bit)) & 1)
                    addresses.push_back(base + (bit + 1) * 4);
        } else {
            for (std::uint32_t count = denseCountMinusOne.get(header);
                 count; --count)
                addresses.push_back(addresses.back() + 4);
        }
        const auto end = cursor + (addresses.size() + 1) * 4;
        if (end > task.size())
            throw std::invalid_argument(
                "h14.conv-clip-outside-envelope: truncated packed record");
        const bool touches = std::any_of(addresses.begin(), addresses.end(),
            [](std::uint32_t address) {
                return address >= registers::kernelLaneOffsetTail &&
                       address <= registers::kernelLaneSizeLast;
            });
        if (touches) {
            if (std::any_of(addresses.begin(), addresses.end(),
                    [](std::uint32_t address) {
                        return address < registers::kernelLaneOffsetTail ||
                               address > registers::kernelLaneSizeLast;
                    }))
                throw std::invalid_argument(
                    "h14.conv-clip-outside-envelope: kernel-DMA record "
                    "straddles the clip table");
            if (!inserted) {
                appendLE(out, recordDense(registers::kernelLaneOffsetTail,
                                          static_cast<std::uint32_t>(
                                              values.size())), 4);
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
    storeLE32(out, 0, taskHeader0(static_cast<std::uint32_t>(out.size() / 4),
                                 taskIdentifier.get(loadLE32(out, 0))));
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
        if (p.mobileRingActivation != Activation::None)
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
        p.mobileRingActivation = Activation::FiniteClip;
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
        std::size_t size = bytes.size();
        const auto selectorAt = h14RegisterOffset(task.taskStream, off, size,
                                                  registers::macConfig);
        const auto selector = loadLE32(task.taskStream, selectorAt);
        const auto noPost = fields::macConfig(Activation::None);
        if (selector != noPost && selector != (noPost & ~macSelector.mask()))
            throw std::invalid_argument(
                "h14.conv-clip-outside-envelope: convolution NE word is not "
                "a decoded no-post-operation form");
        storeLE32(task.taskStream, selectorAt,
                  applyMacActivation(selector, Activation::FiniteClip));
        const auto kernel = get(task, off, size, registers::kernelConfig);
        put(task, off, size, registers::kernelConfig,
            applyNeActivation(kernel, Activation::FiniteClip));
        const auto convolution = get(task, off, size, registers::convolution);
        const bool threeTapKernel =
            convolutionKernelWidth.get(convolution) == 3;
        storeLE32(task.taskStream, off + sizeof(std::uint32_t),
                  applyTaskActivation(loadLE32(task.taskStream,
                                               off + sizeof(std::uint32_t)),
                                      threeTapKernel,
                                      Activation::FiniteClip));

        const auto oldBase = get(task, off, size, registers::kernelDmaBase);
        if (constants.size() > UINT32_MAX)
            throw std::invalid_argument(
                "h14.conv-clip-outside-envelope: kernel base overflows");
        const auto newBase = static_cast<std::uint32_t>(constants.size());
        storeLE32(task.taskStream,
                  lastRegister(task, off, size, registers::kernelDmaBase), newBase);
        std::array<std::uint32_t, 16> newOffsets{}, newSizes{};
        for (std::uint32_t lane = 0; lane != 16; ++lane) {
            const std::size_t oldOffset = get(task, off, size,
                                              registers::kernelLaneOffset + 4 * lane);
            const std::size_t oldSize = get(task, off, size,
                                            registers::kernelLaneSize + 4 * lane);
            const std::size_t oldGlobal = oldBase + oldOffset;
            if (oldGlobal + oldSize > p.constants.size())
                throw std::invalid_argument(
                    "h14.conv-clip-outside-envelope: kernel lane exceeds the "
                    "constant section");
            if (oldOffset > UINT32_MAX - clipDescriptorBytes * lane ||
                oldSize > UINT32_MAX - clipDescriptorBytes)
                throw std::invalid_argument(
                    "h14.conv-clip-outside-envelope: kernel lane offset or "
                    "size overflows");
            newOffsets[lane] = static_cast<std::uint32_t>(
                oldOffset + clipDescriptorBytes * lane);
            newSizes[lane] = static_cast<std::uint32_t>(
                oldSize + clipDescriptorBytes);
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
                    lastRegister(task, off, size,
                                 registers::kernelLaneOffset + 4 * lane),
                    newOffsets[lane]);
            for (std::uint32_t lane = 0; lane != 16; ++lane)
                storeLE32(task.taskStream,
                    lastRegister(task, off, size,
                                 registers::kernelLaneSize + 4 * lane),
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
// records and no SourceCfg/depth-stride words. Its NE repeat fields take the
// NE class, and ResultCfg drops the external-DMA and DMA-loaded-result fields.
// The task must be the last one in the stream.
void reuseLayerInput(Program &p, std::size_t off, std::size_t &bytes) {
    if (off + bytes != p.taskStream.size())
        throw std::logic_error("H14 input reuse requires the final task");
    const auto ne = get(p, off, bytes, registers::schedule);
    const auto cls = scheduleIssueClass.get(ne);
    const auto resultCfg = get(p, off, bytes, registers::resultConfig) &
        ~(l2ExternalBinding.mask() | l2ExternalDma.mask() |
          l2DmaLoadedResult.mask());
    const std::size_t extra = (loadLE32(p.taskStream, off + 28) & 3) == 3 ? 4 : 0;
    std::vector<std::uint8_t> out(p.taskStream.begin() + static_cast<std::ptrdiff_t>(off),
        p.taskStream.begin() + static_cast<std::ptrdiff_t>(off + 32 + extra));
    for (std::size_t c = off + 32 + extra; c < off + bytes;) {
        const auto h = loadLE32(p.taskStream, c);
        const auto base = recordAddress.get(h) * 4;
        std::vector<std::uint32_t> addrs{base};
        if (h & scatterPresent.mask()) {
            for (unsigned b = 0; b < 16; ++b)
                if ((h >> (15 + b)) & 1) addrs.push_back(base + (b + 1) * 4);
        } else {
            for (std::uint32_t n = denseCountMinusOne.get(h); n; --n)
                addrs.push_back(addrs.back() + 4);
        }
        const auto end = c + (addrs.size() + 1) * 4;
        const bool source = base >= registers::sourceDmaConfig &&
                            base < registers::resultDmaConfig;
        if (source != (addrs.back() >= registers::sourceDmaConfig &&
                       addrs.back() < registers::resultDmaConfig))
            throw std::logic_error("H14 record straddles the source TileDMA block");
        if (source) { c = end; continue; }
        const bool sourceCfg = std::any_of(addrs.begin(), addrs.end(), [](std::uint32_t a) {
            return a == registers::sourceConfig ||
                   a == registers::sourceDepthStride ||
                   a == registers::sourcePlaneStride; });
        if (!sourceCfg) {
            out.insert(out.end(), p.taskStream.begin() + static_cast<std::ptrdiff_t>(c),
                       p.taskStream.begin() + static_cast<std::ptrdiff_t>(end));
            c = end;
            continue;
        }
        if (!(h & scatterPresent.mask()))
            throw std::logic_error("H14 input reuse expects the L2 words in a scatter record");
        std::vector<std::pair<std::uint32_t, std::uint32_t>> kept;
        for (std::size_t k = 0; k < addrs.size(); ++k) {
            const auto a = addrs[k];
            if (a == registers::sourceConfig || a == registers::sourceDepthStride ||
                a == registers::sourcePlaneStride) continue;
            kept.emplace_back(a, a == registers::resultConfig ? resultCfg
                : loadLE32(p.taskStream, c + 4 + 4 * k));
        }
        if (!kept.empty()) {
            const auto first = kept.front().first / 4;
            std::uint32_t mask = 0;
            for (std::size_t k = 1; k < kept.size(); ++k) {
                const auto bit = kept[k].first / 4 - first - 1;
                if (bit >= 16) throw std::logic_error("H14 L2 scatter record does not fit");
                mask |= 1u << bit;
            }
            appendLE(out, recordScatter(first * 4, mask), 4);
            for (const auto &kv : kept) appendLE(out, kv.second, 4);
        }
        c = end;
    }
    storeLE32(out, 0, taskHeader0(static_cast<std::uint32_t>(out.size() / 4),
                                 taskIdentifier.get(loadLE32(out, 0))));
    p.taskStream.resize(off);
    p.taskStream.insert(p.taskStream.end(), out.begin(), out.end());
    bytes = out.size();
    put(p, off, bytes, registers::schedule,
        (ne & ~(scheduleOuterRepeat.mask() | scheduleInnerRepeat.mask())) |
        scheduleOuterRepeat.value(cls) | scheduleInnerRepeat.value(cls));
}

Program linkConvChain(const std::vector<Program> &layers,
                      std::vector<LinkedLayer> *layout) {
    if (layers.empty())
        throw std::invalid_argument(
            "h14.linked-outside-envelope: need at least one conv layer");
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
            const auto readEnd = static_cast<std::uint64_t>(
                    r(registers::sourceDmaBase)) +
                static_cast<std::uint64_t>(r(registers::sourceDmaPlaneStride)) *
                    r(registers::inputChannels);
            const auto writeEnd = static_cast<std::uint64_t>(
                    r(registers::resultDmaBase)) +
                static_cast<std::uint64_t>(r(registers::resultDmaPlaneStride)) *
                    r(registers::outputChannels);
            if (readEnd > p.inputs[0].allocationBytes || writeEnd > p.output.allocationBytes ||
                r(registers::sourceDmaRowStride) != p.inputs[0].nchw[5] ||
                r(registers::sourceDmaPlaneStride) != p.inputs[0].nchw[4] ||
                r(registers::resultDmaRowStride) != p.output.nchw[5] ||
                r(registers::resultDmaPlaneStride) != p.output.nchw[4])
                throw std::invalid_argument("h14.linked-outside-envelope: DMA span exceeds its surface");
            for (unsigned j = 0; j < 16; ++j)
                if (static_cast<std::uint64_t>(r(registers::kernelDmaBase)) +
                    r(registers::kernelLaneOffset + 4 * j) +
                    r(registers::kernelLaneSize + 4 * j) > p.constants.size())
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
            set(registers::kernelDmaBase,
                get(p, src.first, src.second, registers::kernelDmaBase) +
                static_cast<std::uint32_t>(result.constants.size()));
            if (l && !reuse) {
                route(result, off, bytes, registers::sourceDmaBase, 3);
                set(registers::sourceDmaBase,
                    static_cast<std::uint32_t>(regions[l - 1]));
                // Native H14 scratch role, with a first-slice dependency.
                set(registers::sourceDmaConfig,
                    fields::sourceDmaConfig(false, 0, true, i == 0));
                set(registers::sourceDmaExtension,
                    fields::sourceDmaExtension(false));
            }
            if (l + 1 < layers.size()) {
                route(result, off, bytes, registers::resultDmaBase, 3);
                set(registers::resultDmaBase,
                    get(p, src.first, src.second, registers::resultDmaBase) +
                    static_cast<std::uint32_t>(regions[l]));
                set(registers::resultDmaConfig,
                    (get(p, src.first, src.second, registers::resultDmaConfig) &
                     ~dmaRole.mask()) | dmaRoleResultWrite.value(1) |
                     dmaRoleLinearSurface.value(1) |
                     dmaRoleScratchSurface.value(1) |
                     dmaRoleInterleavedResult.value(1));
            }
            if (result.taskCount >= 254)
                throw std::invalid_argument("h14.linked-outside-envelope: DMA task-tag limit exceeded");
            if (!reuse)
                set(registers::sourceDmaConfig,
                    (get(result, off, bytes, registers::sourceDmaConfig) &
                     ~dmaTaskTag.mask()) | dmaTaskTag.value(result.taskCount));
            set(registers::resultDmaConfig,
                (get(result, off, bytes, registers::resultDmaConfig) &
                 ~dmaTaskTag.mask()) | dmaTaskTag.value(result.taskCount));
            set(registers::kernelDmaConfig,
                (get(result, off, bytes, registers::kernelDmaConfig) &
                 ~kernelNextTaskTag.mask()) |
                 kernelNextTaskTag.value(result.taskCount + 1));
            for (unsigned j = 0; j < 16; ++j)
                set(registers::kernelLaneConfig + 4 * j,
                    (get(result, off, bytes,
                         registers::kernelLaneConfig + 4 * j) & ~dmaTaskTag.mask()) |
                     dmaTaskTag.value(result.taskCount));
            const bool last = l + 1 == layers.size() && i + 1 == p.taskCount;
            storeLE32(result.taskStream, off,
                taskHeader0(static_cast<std::uint32_t>(bytes / 4),
                            result.taskCount));
            storeLE32(result.taskStream, off + 8,
                taskHeader2(result.taskCount == 0, last));
            // LiveOuts: every task of a multi-task layer keeps L2 live, as
            // in Apple's linked programs, so later slices can reuse the
            // staged input; single tasks and the program's last task close
            // it. TSR remains set. H14 has no H13 header[6] TSE flag.
            storeLE32(result.taskStream, off + 24,
                      p.taskCount > 1 && !last ? liveBankMask : 0u);
            storeLE32(result.taskStream, off + 28, taskHeader7());
            if (!result.taskCount) result.firstTaskBytes = bytes;
            ++result.taskCount;
        }
        result.constants.insert(result.constants.end(), p.constants.begin(), p.constants.end());
    }
    if (result.taskCount >= (std::uint32_t(1) << taskIdentifier.width) ||
        result.constants.size() > UINT32_MAX)
        throw std::invalid_argument("h14.linked-outside-envelope: task or constant limit exceeded");
    result.constantOffsetBytes = alignUp(result.taskStream.size(), 64);
    return result;
}

namespace {

// CostHint is a count of one-MiMAC scheduler work quanta.  A fixed binary
// quantum makes the hint scale with all convolution geometry instead of a
// fixed layer/width class.  Terminal output adds one 64-KiB L2-bank drain
// quantum per fp16 result span; this is scheduling only and never arithmetic.
// The wide dependent terminal form switches from the compact direct-result
// class above the largest edge that class carries.  H13 uses the same bound;
// H14's 128x56x56 prefixes are the first native-ring outputs to cross it.

std::uint64_t align16(std::uint64_t value) {
    return (value + 15) & ~std::uint64_t(15);
}

bool mobileOverlap(std::uint64_t left, std::uint64_t leftBytes,
                   std::uint64_t right, std::uint64_t rightBytes) {
    if (leftBytes + rightBytes > mobileL2Bytes) return true;
    for (const auto l : {left % mobileL2Bytes,
                         left % mobileL2Bytes + mobileL2Bytes})
        for (const auto r : {right % mobileL2Bytes,
                             right % mobileL2Bytes + mobileL2Bytes})
            if (l < r + rightBytes && r < l + leftBytes) return true;
    return false;
}

std::uint32_t mobileSkew(std::uint32_t channels,
                         std::uint32_t channelStride) {
    const auto rowPayload = static_cast<std::uint64_t>(channels) *
        channelStride;
    // A resident row must move the next row start away from the preceding
    // row's offset inside a 64-KiB L2 bank.  Search positive execution-lane
    // quanta and take the smallest one that does so.  This is a layout-only
    // bank-conflict rule: it cannot change convolution arithmetic.
    for (std::uint32_t skew = mobileExecutionLaneBytes;
         skew < mobileL2BankBytes; skew += mobileExecutionLaneBytes) {
        const auto rowStride = rowPayload + skew;
        if (rowStride % mobileL2BankBytes != 0) return skew;
    }
    throw std::logic_error("H14 linked-ring row has no conflict-free L2 skew");
}

struct MobileGeometry {
    ConvShape shape{};
    bool depthwise = false;
    std::uint32_t sourceChannel = 0;
    std::uint32_t sourceRow = 0;
    std::uint64_t sourceSpan = 0;
};

// The single-task boundary form stages both complete external surfaces in the
// circular arena.  Keep every contributor named: a later, proven tiled form
// can reduce the surface spans without silently dropping scratch traffic or
// the largest simultaneously prefetched coefficient lane from its budget.
struct MobileTaskWorkingSpan {
    std::uint64_t source = 0;
    std::uint64_t result = 0;
    std::uint64_t scratch = 0;
    std::uint64_t coefficientPrefetch = 0;

    std::uint64_t total() const {
        const auto surface = source + result;
        if (surface < source || surface + scratch < surface ||
            surface + scratch + coefficientPrefetch < surface + scratch)
            throw std::overflow_error("H14 linked-ring working span overflows");
        return surface + scratch + coefficientPrefetch;
    }
};

std::uint64_t mobileCoefficientPrefetchSpan(const Program &layer) {
    return *std::max_element(layer.mobileRingKernelSizes.begin(),
                             layer.mobileRingKernelSizes.end());
}

void requireStandaloneArena(const MobileTaskWorkingSpan &span) {
    if (span.total() <= mobileL2Bytes) return;
    throw std::invalid_argument(
        "h14.linked-ring-outside-arena: standalone task needs " +
        std::to_string(span.total()) + " bytes (source=" +
        std::to_string(span.source) + ", result=" +
        std::to_string(span.result) + ", scratch=" +
        std::to_string(span.scratch) + ", coefficient-prefetch=" +
        std::to_string(span.coefficientPrefetch) + ") in the " +
        std::to_string(mobileL2Bytes) +
        "-byte ring arena; use --schedule linked");
}

std::uint32_t mobileCommonClass(const MobileGeometry &g) {
    // Issue class is the enabled issue-pipe mask.  The widest spatial stage
    // enables the first two pipes.  Every narrower stage uses its capped
    // geometry stage as both the issue class and the number of inner/outer
    // schedule replays; stage one is deliberately not a special case.
    constexpr unsigned wideSpatialIssuePipeCount = 2;
    constexpr std::uint32_t wideSpatialIssueClass =
        (std::uint32_t(1) << wideSpatialIssuePipeCount) - 1;
    constexpr std::uint32_t stageZeroPointwiseRepeatCount = 1;
    constexpr std::uint32_t maximumGeometryStage = 4;
    constexpr std::uint32_t widestSpatialLog2 = 7;
    const auto width = g.shape.output.width;
    std::uint32_t ceilLog = 0;
    for (auto value = width - 1; value; value >>= 1) ++ceilLog;
    const auto stage = ceilLog >= widestSpatialLog2
        ? 0u : widestSpatialLog2 - ceilLog;
    if (!stage) {
        const auto repeatCount = !g.depthwise && g.shape.kernel == 1
            ? stageZeroPointwiseRepeatCount : 0u;
        return scheduleConfig(wideSpatialIssueClass, repeatCount,
                              repeatCount, false);
    }
    const auto issueClass = std::min(stage, maximumGeometryStage);
    const auto repeatCount = std::min(stage, maximumGeometryStage);
    return scheduleConfig(issueClass, repeatCount, repeatCount,
                          repeatCount == maximumGeometryStage);
}

std::size_t compactTaskPaddingMinimum(
        const LinkedRingResearchOptions &options) {
    const auto value = options.compactTaskMinimumBytes.value_or(
        compactTaskMinimumBytes);
    for (const auto candidate : {std::size_t(0), std::size_t(128),
                                 std::size_t(256),
                                 compactTaskMinimumBytes})
        if (candidate == value) return candidate;
    throw std::invalid_argument(
        "H14 compact-task probe minimum must be 0, 128, 256, or 312");
}

std::uint32_t mobileCost(const MobileGeometry &g, bool terminal) {
    const auto ceilDiv = [](std::uint64_t value, std::uint64_t quantum) {
        return value / quantum + (value % quantum != 0);
    };
    // The linked-ring envelope is square TF-SAME convolution.  Re-derive the
    // scheduled output extent from input width/height and stride so stride is
    // an explicit part of the work rule rather than hidden in a layer class.
    const auto outputWidth = ceilDiv(g.shape.input.width, g.shape.stride);
    const auto outputHeight = ceilDiv(g.shape.input.height, g.shape.stride);
    const auto reductionChannels =
        g.shape.input.channels / g.shape.groups;
    const auto kernelArea =
        std::uint64_t(g.shape.kernel) * g.shape.kernel;
    const auto macs = outputWidth * outputHeight * g.shape.output.channels *
                      reductionChannels * kernelArea;
    auto quanta = ceilDiv(macs, mobileCostMacQuantum);
    if (terminal) {
        const auto resultBytes = outputWidth * outputHeight *
            g.shape.output.channels * sizeof(std::uint16_t);
        quanta += ceilDiv(resultBytes, mobileCostDrainQuantumBytes);
    }
    if (quanta > lowHalfMask)
        throw std::invalid_argument("H14 linked-ring scheduler cost exceeds task mode");
    return static_cast<std::uint32_t>(quanta);
}

void appendHalf(std::vector<std::uint8_t> &out, const std::uint8_t *value) {
    out.push_back(value[0]);
    out.push_back(value[1]);
}

bool nonzeroHalf(const std::uint8_t *value) {
    return value[0] || (value[1] & fp16MagnitudeMask);
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
    if (body > std::numeric_limits<std::uint16_t>::max())
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
    appendLE(task, recordScatter(base, mask), 4);
    for (const auto &word : words) appendLE(task, word.second, 4);
}

void ringDense(std::vector<std::uint8_t> &task, std::uint32_t address,
               const std::vector<std::uint32_t> &words,
               int barSlot = -1) {
    if (words.empty() || words.size() > 64 || address % 4)
        throw std::logic_error("invalid H14 ring dense record");
    const auto header = recordDense(address,
        static_cast<std::uint32_t>(words.size()), barSlot);
    appendLE(task, header, 4);
    for (const auto word : words) appendLE(task, word, 4);
}

std::uint32_t liveBanks(std::uint64_t base, std::uint64_t bytes) {
    std::uint32_t mask = 0;
    for (std::uint32_t bank = 0; bank != 32; ++bank)
        if (mobileOverlap(base, bytes,
                          static_cast<std::uint64_t>(bank) * mobileL2BankBytes,
                          mobileL2BankBytes))
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
        std::uint64_t coefficientBase, std::size_t paddingMinimum) {
    std::vector<std::uint8_t> task(
        compactTaskHeaderWords * sizeof(std::uint32_t), 0);
    const bool sparse = g.shape.kernel == 3 && g.shape.stride == 2;
    const auto activation = layer.mobileRingActivation;
    const auto mac = fields::macConfig(activation);

    storeLE32(task, 4, applyTaskActivation(mobileCost(g, last),
        g.shape.kernelWidth == 3, activation));
    storeLE32(task, 8, taskHeader2(first, last));
    storeLE32(task, 16, taskHeader4(first || last));
    std::uint32_t banks = 0;
    if (inputResident && !last) banks |= liveBanks(sourceBase, sourceSpan);
    if (outputResident) banks |= liveBanks(resultBase, resultSpan);
    storeLE32(task, 24, banks);
    storeLE32(task, 28, taskHeader7());

    // Coefficient DMA is emitted first so the BAR relocation is a later,
    // last-write-wins record.  Its tag and addend come solely from stream
    // position and the concatenated coefficient allocator.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> kernel = {
        {registers::kernelDmaConfig,
         fields::kernelDmaConfig(index + 1, sparse, last)},
    };
    for (unsigned lane = 0; lane != kernelLaneScatterCount; ++lane)
        kernel.emplace_back(registers::kernelLaneConfig +
                                sizeof(std::uint32_t) * lane,
                            fields::kernelLaneConfig(index));
    ringScatter(task, kernel);
    std::vector<std::uint32_t> tailTags(
        kernelLaneDenseCount, fields::kernelLaneConfig(index));
    ringDense(task, registers::kernelLaneConfigTail, tailTags);
    std::vector<std::uint32_t> laneWords;
    for (unsigned lane = 1; lane != kernelLaneCount; ++lane)
        laneWords.push_back(layer.mobileRingKernelOffsets[lane]);
    // A scratch-result task used to omit all lane sizes and inherit the prior
    // task's values.  That is harmless only when the sizes happen to match.
    // Finite clip makes the mismatch observable at the first spill: the stem
    // leaves its descriptor-extended size active while L01 owns a shorter
    // lane, so KernelDMA reads into the following lane's descriptor.  Keep
    // the compact ReLU byte stream unchanged, but make every finite-clip task
    // state its descriptor-inclusive kernelLaneSize fields explicitly.
    if (!writesScratch || activation == Activation::FiniteClip)
        for (const auto size : layer.mobileRingKernelSizes)
            laneWords.push_back(size);
    ringDense(task, registers::kernelLaneOffsetTail, laneWords);

    std::vector<std::pair<std::uint32_t, std::uint32_t>> common = {
        {registers::inputSurface, surface(g.shape.input.height, g.shape.input.width)},
        {registers::inputChannels, g.shape.input.channels},
        {registers::outputChannels, g.shape.output.channels},
        {registers::outputSurface, surface(g.shape.output.height, g.shape.output.width)},
    };
    const auto lanes = g.depthwise ? 1u : denseLanes(g.shape);
    const auto ocg = g.depthwise ? 0u : (lanes == 8 ? 0u : log2Floor(lanes));
    if (g.depthwise || ocg || sparse)
        common.emplace_back(registers::outputChannelGroup,
                            outputChannelGroup.value(ocg));
    if (g.shape.kernel != 1) {
        const auto pad = sparse ? 0u : static_cast<std::uint32_t>(padding(g.shape));
        common.emplace_back(registers::convolution,
            convolutionConfig(g.shape.kernelWidth, g.shape.kernel,
                              g.shape.stride, pad));
    }
    if (g.depthwise) common.emplace_back(registers::depthwise, depthwiseConfig());
    common.emplace_back(registers::tileHeight, g.shape.output.height);
    common.emplace_back(registers::schedule, mobileCommonClass(g));
    common.emplace_back(registers::macWork, macEnable.value(1));
    ringScatter(task, common);

    std::vector<std::pair<std::uint32_t, std::uint32_t>> l2 = {
        {registers::sourceConfig, sourceCfg},
    };
    if (sourceBase)
        l2.emplace_back(registers::sourceBase, static_cast<std::uint32_t>(sourceBase));
    l2.emplace_back(registers::sourceChannelStride, g.sourceChannel);
    l2.emplace_back(registers::sourceRowStride, g.sourceRow);
    if (!inputResident) {
        const auto depth = g.shape.input.channels * g.sourceChannel;
        l2.emplace_back(registers::sourceDepthStride, depth);
        l2.emplace_back(registers::sourcePlaneStride, depth);
    }
    l2.emplace_back(registers::resultConfig, resultCfg);
    if (resultBase)
        l2.emplace_back(registers::resultBase, static_cast<std::uint32_t>(resultBase));
    if (resultChannel || resultRow) {
        l2.emplace_back(registers::resultChannelStride, resultChannel);
        l2.emplace_back(registers::resultRowStride, resultRow);
    }
    ringScatter(task, l2);
    if (resultCfg == l2Config(L2Role::StagedDirectResult)) {
        const auto depth = g.shape.output.channels * resultChannel;
        ringDense(task, registers::resultDepthStride, {depth, depth});
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> ne;
    // The linked finite-clip path leaves KernelCfg's LUT-enable field clear.
    // Sparse geometry remains independent; dense reset state stays omitted.
    const auto kernelConfigWord = applyNeActivation(
        sparse ? neConfig(true, false) : 0u, activation);
    if (kernelConfigWord)
        ne.emplace_back(registers::kernelConfig, kernelConfigWord);
    ne.emplace_back(registers::macConfig, mac);
    ne.emplace_back(registers::unityScale, fp16One);
    ringScatter(task, ne);

    if (first || readsScratch) {
        const auto row = first ? layer.inputs[0].nchw[5]
                               : 16ull * g.shape.input.width;
        const auto plane = first ? layer.inputs[0].nchw[4]
                                 : row * g.shape.input.height;
        ringScatter(task, {
            {registers::sourceDmaConfig,
             fields::sourceDmaConfig(first, index, !first)},
            {registers::sourceDmaExtension, fields::sourceDmaExtension(first)},
            {registers::sourceDmaRowStride, static_cast<std::uint32_t>(row)},
            {registers::sourceDmaPlaneStride, static_cast<std::uint32_t>(plane)},
        });
        ringDense(task, registers::sourceDmaTile, {sourceDmaTile(first)});
    }

    if (writesScratch || last) {
        const auto row = writesScratch ? 16ull * g.shape.output.width
                                       : layer.output.nchw[5];
        const auto plane = writesScratch ? row * g.shape.output.height
                                         : layer.output.nchw[4];
        ringScatter(task, {
            {registers::resultDmaConfig,
             fields::resultDmaConfig(writesScratch,
                 resultCfg == l2Config(L2Role::StagedDirectResult), index)},
            {registers::resultDmaRowStride, static_cast<std::uint32_t>(row)},
            {registers::resultDmaPlaneStride, static_cast<std::uint32_t>(plane)},
            {registers::resultDmaTile, resultDmaTile(writesScratch)},
        });
        // These dead destination extensions are padding, not semantic DMA
        // controls.  The probe's smaller minima omit them; the default keeps
        // the device-tested byte stream unchanged.
        if (paddingMinimum == compactTaskMinimumBytes) {
            ringDense(task, registers::resultDmaDeadExtension,
                      {deadDestinationPadding});
            ringDense(task, registers::resultDmaDeadExtension,
                      {deadDestinationPadding});
        }
    }
    ringDense(task, registers::kernelDmaBase,
              {static_cast<std::uint32_t>(coefficientBase), 0}, 1);
    // BAR slots follow the H14 linked ABI that the ABI-2 runner and libane
    // bind: program input on slot 4, final output on slot 5, scratch on 3.
    if (first || readsScratch)
        ringDense(task, registers::sourceDmaBase, {0, 0}, first ? 4 : 3);
    if (writesScratch || last)
        ringDense(task, registers::resultDmaBase, {0, 0}, writesScratch ? 3 : 5);

    // Pad to the smallest size accepted by our device tests.  The explicit
    // research option varies only this lower bound and leaves semantic
    // records unchanged.
    while (task.size() < paddingMinimum)
        ringDense(task, registers::resultDmaDeadExtension,
                  {deadDestinationPadding});

    storeLE32(task, 0,
              taskHeader0(static_cast<std::uint32_t>(task.size() / 4), index));
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
    program.taskStream.assign(16, 0); // placeholder prefix; linker replaces it
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
                          std::vector<LinkedLayer> *layout,
                          LinkedRingResearchOptions options) {
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
            (index ? mobileSkew(g.shape.input.channels, g.sourceChannel) : 0u);
        g.sourceSpan = static_cast<std::uint64_t>(g.shape.input.height) *
            g.sourceRow;
        geometry.push_back(g);
    }
    const auto n = layers.size();
    const auto paddingMinimum = compactTaskPaddingMinimum(options);
    if (n >= 4) {
        // The bridge width is allocator-selected.  224 narrows to 64 pixels;
        // a 192 input fits at full width, proving this is not a layer table.
        auto &bridge = geometry[2];
        const auto result = geometry[3].sourceSpan;
        auto channel = bridge.sourceChannel;
        while (channel > 16 && result +
                static_cast<std::uint64_t>(bridge.shape.input.height) *
                    (bridge.shape.input.channels * channel +
                     mobileSkew(bridge.shape.input.channels, channel)) >
                         mobileL2Bytes)
            channel -= 16;
        bridge.sourceChannel = channel;
        bridge.sourceRow = bridge.shape.input.channels * channel +
            mobileSkew(bridge.shape.input.channels, channel);
        bridge.sourceSpan = static_cast<std::uint64_t>(bridge.shape.input.height) *
            bridge.sourceRow;
        if (result + bridge.sourceSpan > mobileL2Bytes)
            throw std::invalid_argument(
                "h14.linked-ring-outside-envelope: spill bridge exceeds L2");
    }

    std::vector<std::uint32_t> sourceCfg(
        n, l2Config(L2Role::ProgramInput));
    std::vector<std::uint32_t> resultCfg(
        n, l2Config(L2Role::DirectResult));
    std::vector<std::uint64_t> sourceBase(n), resultBase(n), resultSpan(n);
    std::vector<bool> readsScratch(n), writesScratch(n), outputResident(n);
    const auto standaloneResultSpan = [&](std::size_t i) {
        const auto channel = align16(2ull * geometry[i].shape.output.width);
        const auto row = geometry[i].shape.output.channels * channel +
            mobileSkew(geometry[i].shape.output.channels,
                       static_cast<std::uint32_t>(channel));
        return static_cast<std::uint64_t>(geometry[i].shape.output.height) * row;
    };
    if (n == 1) {
        const auto standaloneResult = standaloneResultSpan(0);
        requireStandaloneArena({geometry[0].sourceSpan, standaloneResult, 0,
                                mobileCoefficientPrefetchSpan(layers[0])});
        sourceBase[0] = align16(standaloneResult);
    }
    if (n >= 2) {
        outputResident[0] = true;
        resultCfg[0] = l2Config(L2Role::ProgramInputResult);
        resultSpan[0] = geometry[1].sourceSpan;
        sourceBase[0] = resultSpan[0];
        sourceCfg[1] = l2Config(L2Role::ResidentWideSource);
    }
    if (n == 2) {
        resultBase[1] = align16(geometry[1].sourceSpan) % mobileL2Bytes;
    } else if (n == 3) {
        outputResident[1] = true;
        resultCfg[1] = l2Config(L2Role::ResidentResult);
        resultSpan[1] = geometry[2].sourceSpan;
        resultBase[1] = align16(geometry[1].sourceSpan) % mobileL2Bytes;
        sourceBase[2] = resultBase[1];
        sourceCfg[2] = l2Config(L2Role::ResidentWideSource);
        resultBase[2] = align16(sourceBase[2] + geometry[2].sourceSpan) %
            mobileL2Bytes;
    } else if (n >= 4) {
        writesScratch[1] = true;
        readsScratch[2] = true;
        resultCfg[1] = l2Config(L2Role::ScratchResult);
        resultBase[1] = geometry[1].sourceSpan;
        sourceCfg[2] = l2Config(L2Role::ScratchSource);
        outputResident[2] = true;
        resultCfg[2] = l2Config(L2Role::ResidentResult);
        resultSpan[2] = geometry[3].sourceSpan;
        sourceBase[2] = resultSpan[2];
        sourceCfg[3] = l2Config(L2Role::ResidentWideSource);
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
            sourceCfg[i + 1] = l2Config(wide
                ? L2Role::ResidentWideSource
                : L2Role::ResidentCompactSource);
            outputResident[i] = true;
            resultCfg[i] = l2Config(L2Role::ResidentResult);
            resultSpan[i] = consumer.sourceSpan;
            auto candidate = align16(sourceBase[i] + geometry[i].sourceSpan) %
                mobileL2Bytes;
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
        // A wide resident consumer whose external result exceeds the compact
        // direct-result limit needs the complete terminal staging form.  Its
        // result strides are derived below from the external surface.  This
        // selects p05-p07 by geometry and edge role, not by layer number.
        resultCfg[last] =
            sourceCfg[last] == l2Config(L2Role::ResidentWideSource) &&
                layers[last].output.allocationBytes >
                    mobileDirectResultMaxBytes
            ? l2Config(L2Role::StagedDirectResult)
            : l2Config(L2Role::DirectResult);
        resultBase[last] = align16(sourceBase[last] +
                                   geometry[last].sourceSpan) % mobileL2Bytes;
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
        const bool stagedTerminal = !outputResident[i] &&
            resultCfg[i] == l2Config(L2Role::StagedDirectResult);
        const auto resultChannel = outputResident[i]
            ? geometry[i + 1].sourceChannel
            : stagedTerminal
                ? static_cast<std::uint32_t>(align16(
                    2ull * geometry[i].shape.output.width)) : 0u;
        const auto resultRow = outputResident[i]
            ? geometry[i + 1].sourceRow
            : stagedTerminal
                ? geometry[i].shape.output.channels * resultChannel +
                    mobileSkew(geometry[i].shape.output.channels,
                               resultChannel) : 0u;
        const auto task = mobileTask(layers[i], geometry[i],
            static_cast<std::uint32_t>(i), i == 0, i + 1 == n,
            readsScratch[i], writesScratch[i], inputResident,
            outputResident[i], sourceCfg[i], sourceBase[i],
            geometry[i].sourceSpan, resultCfg[i], resultBase[i],
            resultSpan[i], resultChannel, resultRow,
            spillBytes, result.constants.size(), paddingMinimum);
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
            item.inputEdge = inputResident
                ? (sourceCfg[i] == l2Config(L2Role::ResidentCompactSource)
                    ? 3 : 4) : 0;
            item.outputEdge = outputResident[i] && i + 1 < n
                ? (sourceCfg[i + 1] ==
                       l2Config(L2Role::ResidentCompactSource) ? 3 : 4) : 0;
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
