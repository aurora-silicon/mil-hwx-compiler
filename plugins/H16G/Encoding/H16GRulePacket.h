#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ane::h16::rule {

struct Packet {
    std::vector<std::uint8_t> bytes;
    std::vector<std::size_t> kernelRelocations;
    std::size_t recordCount = 0;
};

enum class BARSlot : std::uint32_t {
    TaskState = 4,
    Input = 6,
    Output = 7,
    Kernel = 8,
};

// Relocation record: bit 29 enables BAR addressing and bits 26:23 select it.
constexpr std::uint32_t relocationRecordFlags(BARSlot slot) {
    return (std::uint32_t(1) << 29) |
           (static_cast<std::uint32_t>(slot) << 23);
}

namespace fields {

enum class ElementFormat : std::uint32_t { FP16 = 2 };
enum class DescriptorFormat : std::uint32_t { FP16 = 1 };
enum class UnaryOpcode : std::uint32_t { RSqrt = 2, Direct = 3, Reciprocal = 2 };

// Common.surface_limit is the maximum resident surface span, in bytes.
constexpr std::uint32_t surfaceLimit(std::uint32_t bytes) { return bytes; }

// L2 transfer configuration: cache hint, enable, dependency class, format.
constexpr std::uint32_t l2Transfer(std::uint32_t cacheHint,
                                   std::uint32_t dependencyClass,
                                   std::uint32_t format) {
    return (cacheHint << 20) | (std::uint32_t(1) << 8) |
           (dependencyClass << 4) | format;
}

// NE input format: enable, unary dependency class, and fp16 element format.
constexpr std::uint32_t neuralInput(bool enable = true) {
    return (enable ? std::uint32_t(1) << 16 : 0) |
           (std::uint32_t(8) << 4) |
           static_cast<std::uint32_t>(ElementFormat::FP16);
}

// Unary NE command: execution-unit enable, numeric mode, opcode, sub-mode.
constexpr std::uint32_t unaryCommand(std::uint32_t unitEnable,
                                     UnaryOpcode opcode,
                                     bool reciprocalSubmode = false) {
    return (unitEnable << 16) | (std::uint32_t(15) << 10) |
           (static_cast<std::uint32_t>(opcode) << 2) |
           (reciprocalSubmode ? 1u : 0u);
}

// Reduction NE command: accumulator enable, numeric mode, reduction opcode.
constexpr std::uint32_t reductionCommand(std::uint32_t opcode) {
    return (std::uint32_t(1) << 16) | (std::uint32_t(15) << 6) |
           (opcode << 2);
}

// DMA endpoint selector: valid bit plus dependency class.
constexpr std::uint32_t endpoint(std::uint32_t dependencyClass) {
    return std::uint32_t(1) | (dependencyClass << 4);
}

// ResultDMA.endpoint also owns the engine-enable bit. SourceDMA is armed by
// its separate flags word, but ResultDMA never produces the completion event
// consumed by Store when this bit is clear.
constexpr std::uint32_t resultEndpoint(std::uint32_t dependencyClass) {
    return (std::uint32_t(1) << 24) | endpoint(dependencyClass);
}

// Source DMA flags: external-memory, cacheable, and seven-beat burst.
constexpr std::uint32_t sourceDMAFlags() {
    return (std::uint32_t(1) << 7) | (std::uint32_t(1) << 6) |
           (std::uint32_t(7) << 1);
}

// Surface descriptor: enabled, cache hint, dependency class, fp16 format.
constexpr std::uint32_t surfaceDescriptor(std::uint32_t cacheHint,
                                          std::uint32_t dependencyClass) {
    return (std::uint32_t(1) << 24) | (cacheHint << 20) |
           (dependencyClass << 12) | (std::uint32_t(3) << 4) |
           static_cast<std::uint32_t>(DescriptorFormat::FP16);
}

// Kernel DMA command: enable plus the lookup-table load opcode.
constexpr std::uint32_t lookupKernelCommand() {
    return (std::uint32_t(1) << 16) | (std::uint32_t(11) << 6);
}

// Store flags: store-enable plus fp16 element format.
constexpr std::uint32_t storeFlags() {
    return (std::uint32_t(1) << 4) |
           static_cast<std::uint32_t>(ElementFormat::FP16);
}

// Store geometry: ten input lanes and ten output lanes.
constexpr std::uint32_t storeGeometry() {
    return (std::uint32_t(10) << 20) | (std::uint32_t(10) << 4);
}

// Store completion mask: the low seven completion sources are enabled.
constexpr std::uint32_t storeCompletionMask() {
    return std::uint32_t(127) << 16;
}

// Output surface: external BAR, enabled, dependency class seven, format one.
constexpr std::uint32_t outputSurface() {
    return (std::uint32_t(1) << 29) | (std::uint32_t(1) << 16) |
           (std::uint32_t(7) << 8) | std::uint32_t(1);
}

// Task control: enabled engine count, packet format one, completion enable.
constexpr std::uint32_t taskControl(std::uint32_t enabledEngines,
                                    bool completesProgram) {
    return (enabledEngines << 16) | std::uint32_t(1) |
           (completesProgram ? std::uint32_t(1) << 3 : 0);
}

} // namespace fields

struct TaskSpec {
    std::map<std::uint32_t,std::uint32_t> registers;
    std::uint32_t enabledEngines = 1;
    bool kernelRelocation = false;
};

inline void appendWord(std::vector<std::uint8_t> &bytes,
                       std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes.push_back((std::uint8_t)(value >> shift));
}

inline void storeWord(std::vector<std::uint8_t> &bytes, std::size_t offset,
                      std::uint32_t value) {
    if (offset > bytes.size() || bytes.size() - offset < 4)
        throw std::logic_error("H16G packet word is outside the stream");
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes[offset + shift / 8] = (std::uint8_t)(value >> shift);
}

inline std::uint32_t loadWord(const std::uint8_t *bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        value |= (std::uint32_t)bytes[offset + shift / 8] << shift;
    return value;
}

inline std::size_t registerValueOffset(const std::uint8_t *bytes,
                                       std::size_t length,
                                       std::uint32_t wantedAddress) {
    if (length < 16 + 9 * 4) return SIZE_MAX;
    std::size_t position = 16 + 9 * 4;
    while (position + 4 <= length) {
        std::uint32_t header = loadWord(bytes,position);
        std::uint32_t base = (header & ((std::uint32_t(1)<<15)-1)) * 4;
        std::vector<std::uint32_t> addresses{base};
        if (header & (std::uint32_t(1) << 31)) {
            for (unsigned bit = 0; bit < 16; ++bit)
                if (header & (1u << (15 + bit)))
                    addresses.push_back(base + 4 * (bit + 1));
        } else {
            unsigned count = (header >> 15) & 63;
            for (unsigned index = 1; index <= count; ++index)
                addresses.push_back(base + 4 * index);
        }
        position += 4;
        if (position + 4 * addresses.size() > length) return SIZE_MAX;
        for (std::size_t index = 0; index < addresses.size(); ++index)
            if (addresses[index] == wantedAddress)
                return position + 4 * index;
        position += 4 * addresses.size();
    }
    return SIZE_MAX;
}

inline std::size_t appendRecord(
    std::vector<std::uint8_t> &bytes,
    const std::vector<std::pair<std::uint32_t,std::uint32_t>> &values,
    std::uint32_t flags = 0) {
    if (values.empty()) return 0;
    if (values.size() > 64) throw std::logic_error("H16G record is too long");
    std::uint32_t base = values.front().first;
    bool dense = true;
    for (std::size_t index = 0; index < values.size(); ++index)
        dense &= values[index].first == base + 4 * index;
    std::uint32_t header = base / 4 | flags;
    if (dense) {
        header |= (std::uint32_t)(values.size() - 1) << 15;
    } else {
        header |= std::uint32_t(1) << 31;
        for (std::size_t index = 1; index < values.size(); ++index) {
            std::uint32_t delta = (values[index].first - base) / 4;
            if (!delta || delta > 16)
                throw std::logic_error("H16G scatter record exceeds its mask");
            header |= 1u << (delta + 14);
        }
    }
    appendWord(bytes,header);
    std::size_t valuesOffset = bytes.size();
    for (const auto &[address,value] : values) {
        (void)address;
        appendWord(bytes,value);
    }
    return valuesOffset;
}

inline bool isRelocationOperand(std::uint32_t address) {
    // Source/output/kernel BAR selectors and their 64-bit relative addresses.
    switch (address) {
        case 0x4d00: case 0x4d10: case 0x4d14:
        case 0x5100: case 0x5108: case 0x510c:
        case 0x5510: case 0x5514:
            return true;
        default:
            return false;
    }
}

inline void appendTask(Packet &packet, const TaskSpec &task,
                       std::size_t taskIndex, std::size_t taskCount) {
    if (taskIndex) packet.bytes.resize((packet.bytes.size() + 15) & ~std::size_t(15),0);
    const bool last = taskIndex + 1 == taskCount;
    const std::size_t taskStart = packet.bytes.size();

    appendWord(packet.bytes,(std::uint32_t)taskIndex);
    appendWord(packet.bytes,0); // ExeCycles is a scheduler hint, not semantics.
    // Header word 2: record format 104 plus final-output enable in bit 26.
    appendWord(packet.bytes,std::uint32_t(104) |
        (last ? std::uint32_t(1) << 26 : 0));
    appendWord(packet.bytes,0); // No extended header for unary/reduction tasks.
    // Header word 4: dispatch mask, dependency class eight, record format.
    appendWord(packet.bytes,(((std::uint32_t(1) << 12) - 1) << 12) |
        (std::uint32_t(8) << 8) | std::uint32_t(last ? 104 : 32));
    appendWord(packet.bytes,0); // No firmware-private scheduling hint.
    appendWord(packet.bytes,0); // No firmware-private scheduling hint.
    // Every later task depends on all earlier rule-built phases.
    const unsigned dependencyBits = (unsigned)std::min<std::size_t>(taskIndex,31);
    appendWord(packet.bytes,dependencyBits ?
        (std::uint32_t(1) << dependencyBits) - 1 : 0);
    appendWord(packet.bytes,fields::taskControl(task.enabledEngines,last));

    const auto emitBlock = [&](std::uint32_t blockFirst,
                               std::uint32_t blockLast) {
        std::vector<std::pair<std::uint32_t,std::uint32_t>> run;
        for (const auto &[address,value] : task.registers) {
            if (address < blockFirst || address > blockLast ||
                isRelocationOperand(address)) continue;
            if (!run.empty() && address != run.back().first + 4) {
                appendRecord(packet.bytes,run);
                ++packet.recordCount;
                run.clear();
            }
            run.push_back({address,value});
            if (run.size() == 64) {
                appendRecord(packet.bytes,run);
                ++packet.recordCount;
                run.clear();
            }
        }
        if (!run.empty()) {
            appendRecord(packet.bytes,run);
            ++packet.recordCount;
        }
    };
    emitBlock(0x0000,0x0058);
    emitBlock(0x4100,0x41a0);
    emitBlock(0x44d0,0x44d0);
    emitBlock(0x4500,0x4538);
    emitBlock(0x4900,0x492c);
    emitBlock(0x4d00,0x4e40);
    emitBlock(0x5100,0x5150);
    emitBlock(0x5500,0x561c);
    emitBlock(0x5900,0x592c);

    const auto value = [&](std::uint32_t address) {
        auto found = task.registers.find(address);
        return found == task.registers.end() ? 0u : found->second;
    };
    if (task.registers.count(0x4d00)) {
        appendRecord(packet.bytes,{{0x4d00,value(0x4d00)}},
                     relocationRecordFlags(BARSlot::TaskState));
        ++packet.recordCount;
    }
    if (task.registers.count(0x5100)) {
        appendRecord(packet.bytes,{{0x5100,value(0x5100)}},
                     relocationRecordFlags(BARSlot::TaskState));
        ++packet.recordCount;
    }
    if (task.registers.count(0x4d10)) {
        appendRecord(packet.bytes,{{0x4d10,value(0x4d10)},
                                   {0x4d14,value(0x4d14)}},
                     relocationRecordFlags(BARSlot::Input));
        ++packet.recordCount;
    }
    if (task.registers.count(0x5108)) {
        appendRecord(packet.bytes,{{0x5108,value(0x5108)},
                                   {0x510c,value(0x510c)}},
                     relocationRecordFlags(BARSlot::Output));
        ++packet.recordCount;
    }
    if (task.kernelRelocation) {
        std::size_t relocation = appendRecord(packet.bytes,
            {{0x5510,value(0x5510)},{0x5514,value(0x5514)}},
            relocationRecordFlags(BARSlot::Kernel));
        packet.kernelRelocations.push_back(relocation);
        ++packet.recordCount;
    }
    std::size_t taskBytes = packet.bytes.size() - taskStart;
    if (taskBytes % 4 || taskBytes / 4 > (std::size_t(1)<<11)-1)
        throw std::logic_error("H16G task exceeds the record grammar");
    storeWord(packet.bytes,taskStart,
        ((std::uint32_t)(taskBytes / 4) << 16) | (std::uint32_t)taskIndex);
}

inline Packet build(const std::vector<TaskSpec> &tasks) {
    if (tasks.empty()) throw std::logic_error("H16G program has no tasks");
    Packet packet;
    packet.bytes.assign(16,0);
    packet.bytes[0] = 1; // Linear H16G task-stream prologue, format version 1.
    for (std::size_t index = 0; index < tasks.size(); ++index)
        appendTask(packet,tasks[index],index,tasks.size());
    return packet;
}

inline Packet build(const std::map<std::uint32_t,std::uint32_t> &registers,
                    std::uint32_t enabledEngines, bool kernelRelocation) {
    return build({TaskSpec{registers,enabledEngines,kernelRelocation}});
}

} // namespace ane::h16::rule
