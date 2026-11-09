#pragma once

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
        std::uint32_t base = (header & 0x7fffu) * 4;
        std::vector<std::uint32_t> addresses{base};
        if (header & 0x80000000u) {
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
    const std::vector<std::pair<std::uint32_t,std::uint32_t>> &fields,
    std::uint32_t flags = 0) {
    if (fields.empty()) return 0;
    if (fields.size() > 64) throw std::logic_error("H16G record is too long");
    std::uint32_t base = fields.front().first;
    bool dense = true;
    for (std::size_t index = 0; index < fields.size(); ++index)
        dense &= fields[index].first == base + 4 * index;
    std::uint32_t header = base / 4 | flags;
    if (dense) {
        header |= (std::uint32_t)(fields.size() - 1) << 15;
    } else {
        header |= 0x80000000u;
        for (std::size_t index = 1; index < fields.size(); ++index) {
            std::uint32_t delta = (fields[index].first - base) / 4;
            if (!delta || delta > 16)
                throw std::logic_error("H16G scatter record exceeds its mask");
            header |= 1u << (delta + 14);
        }
    }
    appendWord(bytes,header);
    std::size_t valuesOffset = bytes.size();
    for (const auto &[address,value] : fields) {
        (void)address;
        appendWord(bytes,value);
    }
    return valuesOffset;
}

inline Packet build(const std::map<std::uint32_t,std::uint32_t> &registers,
                    std::uint32_t taskControl, bool kernelRelocation) {
    Packet packet;
    packet.bytes.assign(16,0);
    packet.bytes[0] = 1;
    const std::size_t taskStart = packet.bytes.size();
    appendWord(packet.bytes,0);
    appendWord(packet.bytes,0);
    appendWord(packet.bytes,0x04000068);
    appendWord(packet.bytes,0);
    appendWord(packet.bytes,0x00fff868);
    appendWord(packet.bytes,0);
    appendWord(packet.bytes,0);
    appendWord(packet.bytes,0);
    appendWord(packet.bytes,taskControl);

    const auto special = [](std::uint32_t address) {
        return address == 0x4d00 || address == 0x4d10 || address == 0x4d14 ||
               address == 0x5100 || address == 0x5108 || address == 0x510c ||
               address == 0x5510 || address == 0x5514;
    };
    const auto emitBlock = [&](std::uint32_t first, std::uint32_t last) {
        std::vector<std::pair<std::uint32_t,std::uint32_t>> run;
        for (const auto &[address,value] : registers) {
            if (address < first || address > last || special(address)) continue;
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
        auto found = registers.find(address);
        return found == registers.end() ? 0u : found->second;
    };
    if (registers.count(0x4d00)) {
        appendRecord(packet.bytes,{{0x4d00,value(0x4d00)}},0x22000000);
        ++packet.recordCount;
    }
    if (registers.count(0x5100)) {
        appendRecord(packet.bytes,{{0x5100,value(0x5100)}},0x22000000);
        ++packet.recordCount;
    }
    if (registers.count(0x4d10)) {
        appendRecord(packet.bytes,{{0x4d10,value(0x4d10)},
                                   {0x4d14,value(0x4d14)}},0x23000000);
        ++packet.recordCount;
    }
    if (registers.count(0x5108)) {
        appendRecord(packet.bytes,{{0x5108,value(0x5108)},
                                   {0x510c,value(0x510c)}},0x23800000);
        ++packet.recordCount;
    }
    if (kernelRelocation) {
        std::size_t relocation = appendRecord(packet.bytes,
            {{0x5510,value(0x5510)},{0x5514,value(0x5514)}},0x24000000);
        packet.kernelRelocations.push_back(relocation);
        ++packet.recordCount;
    }
    std::size_t taskBytes = packet.bytes.size() - taskStart;
    if (taskBytes % 4 || taskBytes / 4 > 0x7ff)
        throw std::logic_error("H16G task exceeds the record grammar");
    storeWord(packet.bytes,taskStart,(std::uint32_t)(taskBytes / 4) << 16);
    return packet;
}

} // namespace ane::h16::rule
