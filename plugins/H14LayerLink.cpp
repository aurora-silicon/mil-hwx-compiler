#include "LayerOps.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace ane::layers {
namespace {
using Registers = std::map<std::uint32_t, std::uint32_t>;
struct Record {
    std::uint32_t header;
    std::vector<std::uint32_t> addresses;
};
struct Task {
    std::array<std::uint32_t, 8> header{};
    std::vector<Record> records;
    Registers local;
    Registers extra;
    Registers values;
    Registers bars;
    Registers barWords;
};

[[noreturn]] void refuse(const char *why) {
    throw std::invalid_argument(std::string("h14.linked-outside-envelope: ") + why);
}

std::uint32_t word(const std::vector<std::uint8_t> &bytes, std::size_t offset) {
    if (offset + 4 > bytes.size()) refuse("truncated task stream");
    std::uint32_t value = 0;
    for (unsigned i = 0; i != 4; ++i) value |= static_cast<std::uint32_t>(bytes[offset + i]) << (8 * i);
    return value;
}

void append(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
    for (unsigned i = 0; i != 4; ++i) bytes.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

std::vector<Task> decode(const h14::Program &program) {
    std::vector<Task> tasks;
    const auto sizes = h14::taskSizes(program.taskStream);
    if (sizes.size() != program.taskCount || sizes.front() != program.firstTaskBytes)
        refuse("native task metadata does not match its stream");
    std::size_t offset = 16;
    Registers state;
    while (offset < program.taskStream.size()) {
        const auto size = ((word(program.taskStream, offset) >> 16) & 0x7ff) * 4;
        if (size < 32 || offset + size > program.taskStream.size()) refuse("invalid native task length");
        Task task;
        Registers bars, barWords;
        for (unsigned i = 0; i != 8; ++i) task.header[i] = word(program.taskStream, offset + 4 * i);
        const auto extra = (task.header[7] & 3) == 3 ? 4 : 0;
        if (extra) refuse("extended native task header is not qualified");
        std::size_t cursor = offset + 32;
        while (cursor < offset + size) {
            const auto h = word(program.taskStream, cursor);
            const auto base = (h & 0x7fff) * 4;
            std::vector<std::uint32_t> addresses{base};
            if (h & 0x80000000u) {
                const auto mask = (h >> 15) & 0xffff;
                for (unsigned bit = 0; bit != 16; ++bit)
                    if (mask & (1u << bit)) addresses.push_back(base + 4 * (bit + 1));
            } else {
                const auto count = ((h >> 15) & 0x3f) + 1;
                for (unsigned i = 1; i < count; ++i) addresses.push_back(base + 4 * i);
                if (h & 0x20000000u) {
                    if (count != 1 && count != 2) refuse("BAR reference has an unqualified width");
                    const auto slot = (h >> 23) & 0x3f;
                    if (slot >= 32) refuse("BAR slot exceeds the qualified range");
                    bars[base] = slot;
                    barWords[base] = count;
                }
            }
            task.records.push_back({h, addresses});
            cursor += 4;
            if (cursor + addresses.size() * 4 > offset + size) refuse("truncated native register record");
            for (const auto address : addresses) {
                state[address] = word(program.taskStream, cursor);
                task.local[address] = state[address];
                cursor += 4;
            }
        }
        task.values = state;
        task.bars = bars;
        task.barWords = barWords;
        tasks.push_back(std::move(task));
        offset = std::min((offset + size + 15) / 16 * 16, program.taskStream.size());
    }
    if (tasks.size() != program.taskCount) refuse("native task count differs from its stream");
    return tasks;
}

std::vector<std::uint8_t> emit(Task task) {
    // An absent write stays absent. In particular, native L2-only tasks
    // must not inherit an enabled DMA port from the preceding task and
    // acquire invented BAR records. Keep Apple's record framing verbatim;
    // only the documented edge/cache fields may add a register write.
    std::vector<std::uint8_t> out;
    for (const auto h : task.header) append(out, h);
    std::set<std::uint32_t> present;
    for (const auto &record : task.records) {
        auto h = record.header;
        if ((h & 0xa0000000u) == 0x20000000u) {
            const auto base = record.addresses.front();
            if (!task.bars.count(base)) refuse("unrouted native BAR reference");
            h = (h & ~(63u << 23)) | (task.bars.at(base) << 23);
        }
        append(out,h);
        for (const auto address : record.addresses) {
            append(out,task.values.at(address));
            present.insert(address);
        }
    }
    for (const auto &field : task.extra)
        if (!present.count(field.first)) {
            append(out,field.first/4);
            append(out,field.second);
        }
    if (out.size()/4 > 0x7ff) refuse("expanded task exceeds the H14 length field");
    const auto first=(task.header[0]&0xffff)|static_cast<std::uint32_t>(out.size()/4)<<16;
    for (unsigned i=0;i!=4;++i) out[i]=static_cast<std::uint8_t>(first>>(8*i));
    return out;
}
} // namespace

h14::Program linkH14Graph(const std::vector<h14::Program> &layers,
    const std::vector<Connection> &connections,const std::vector<std::string> &boundaries,
    const std::string &result,GraphPlan *placement,std::vector<h13::LinkedLayer> *layout) {
    auto plan=planGraph(layers,connections,boundaries,result,"h14.linked-outside-envelope");
    std::vector<std::vector<Task>> tasks;
    std::uint32_t total = 0;
    for (std::size_t i = 0; i != layers.size(); ++i) {
        const auto &layer = layers[i];
        if (layer.inputs.empty() || layer.inputs.size()>2 || !layer.kernelTable.empty() ||
            layer.scratchAllocationBytes || layer.output.index!=4)
            refuse("native graph layer has no qualified DRAM routing");
        for (std::size_t j=0;j<layer.inputs.size();++j)
            if (layer.inputs[j].index!=5+j) refuse("invalid graph input role");
        tasks.push_back(decode(layer));
        if (!plan.zeroCopy[i]) total+=layer.taskCount;
    }
    if (!total || total > 256 || plan.scratchBytes > 0xffffffffu) refuse("task or scratch address budget exceeded");
    h14::Program out;
    out.inputs.resize(boundaries.size());
    for (std::size_t i=0;i<layers.size();++i)
        for (std::size_t j=0;j<plan.inputs[i].size();++j)
            if (plan.inputs[i][j].boundary) {
                const auto b=plan.inputs[i][j].index;
                out.inputs[b]=layers[i].inputs[j];out.inputs[b].index=static_cast<unsigned>(5+b);
            }
    out.output=layers[plan.result].output;
    // The Linux H14 reader sizes its automatic scratch BO as the largest
    // BAR offset plus the output allocation. Keep that declared allocation
    // large enough for every intermediate; logical output strides are fixed.
    for (const auto &layer : layers)
        out.output.allocationBytes = std::max(out.output.allocationBytes, layer.output.allocationBytes);
    out.taskStream.resize(16, 0);
    out.taskCount = total;
    out.scratchAllocationBytes = plan.scratchBytes;
    out.normalizedBarRoles = true;
    if (layout) layout->clear();
    unsigned global = 0;
    std::vector<std::int64_t> writer(layers.size(),-1);
    for (std::size_t i = 0; i != layers.size(); ++i) {
        const auto &layer = layers[i];
        const auto constantBase = out.constants.size();
        if (constantBase > 0xffffffffu - layer.constants.size()) refuse("constant address budget exceeded");
        const auto target=plan.outputs[i];
        bool readsScratch=false;
        std::uint64_t firstRead=0;
        for (const auto ref:plan.inputs[i])
            if (!ref.boundary) { readsScratch=true;firstRead=plan.outputs[ref.index].offset;break; }
        if (layout) layout->push_back({global,plan.zeroCopy[i]?0:layer.taskCount,
            readsScratch,firstRead,!target.output,target.offset});
        if (plan.zeroCopy[i]) continue;
        bool wrote=false;
        for (auto task : tasks[i]) {
            auto &regs = task.values;
            auto bars = task.bars;
            task.bars.clear();
            task.header[0] = (task.header[0] & 0xffff0000u) | global;
            task.header[2] = (task.header[2] & ~0x22u) | (global + 1 == total ? 0x22u : 0u);
            for (const auto cfg : {0x1100u,0x1104u,0x1500u})
                if (task.local.count(cfg) && (regs[cfg] & 1)) regs[cfg] = (regs[cfg] & ~0x00ff0000u) | (global << 16);
            for (unsigned source = 0; source != 2; ++source) {
                const auto cfg = 0x1100u + 4 * source;
                const auto base = 0x1110u + 0x18 * source;
                if (!bars.count(base)) continue;
                if (!task.local.count(cfg) || !(regs[cfg]&1) || regs[base+4]) refuse("source is not a qualified active 32-bit BAR reference");
                const auto slot = bars.at(base);
                if (slot <= 1) {
                    if (regs[base] >= layer.constants.size()) refuse("constant DMA exceeds its layer section");
                    regs[base] += static_cast<std::uint32_t>(constantBase);
                    task.bars[base] = 1;
                } else {
                    const auto inputSlot = layer.normalizedBarRoles ? 5u : 4u;
                    const auto input=slot==inputSlot?0u:slot==6?1u:3u;
                    if (source || input>=plan.inputs[i].size() || regs[base]) refuse("unknown native boundary source form");
                    const auto ref=plan.inputs[i][input];
                    if (ref.boundary) task.bars[base]=static_cast<unsigned>(5+ref.index);
                    else {
                        const auto route=plan.outputs[ref.index];
                        if (route.output || writer[route.owner]<0) refuse("graph reads an unavailable surface");
                        task.bars[base]=7;
                        regs[base]=static_cast<std::uint32_t>(route.offset);
                        // Native DRAM handoffs in env_mm_r2br_m1_k2048_n2048
                        // and sched_fused_lut_d512_s512: DependencyMode 1,
                        // distance-minus-one, and scratch cache hints.
                        const auto distance=static_cast<unsigned>(std::min<std::int64_t>(global-writer[route.owner]-1,15));
                        regs[cfg] = (regs[cfg] & ~0xff0000f0u) | 0x100000e0u | (distance << 24);
                        regs[0x1108 + 4 * source] = 0xee;
                        task.extra[0x1108 + 4 * source] = 0xee;
                    }
                }
            }
            if (bars.count(0x1508)) {
                if (!task.local.count(0x1500) || !(regs[0x1500]&1)) refuse("destination BAR has no active DMA port");
                const auto outputSlot = layer.normalizedBarRoles ? 4u : 5u;
                if (!bars.count(0x1508) || bars.at(0x1508) != outputSlot || regs[0x150c]) refuse("unknown native destination form");
                task.bars[0x1508]=target.output?4:7;
                if (static_cast<std::uint64_t>(regs[0x1508])+layer.output.nchw[4]*regs[0x10]>layer.output.allocationBytes ||
                    regs[0x1508]+target.offset>0xffffffffu) refuse("native destination exceeds its owning surface");
                regs[0x1508]+=static_cast<std::uint32_t>(target.offset);
                if (!target.output) regs[0x1500]=(regs[0x1500]&~0xf0u)|0xf0u;
                writer[target.owner]=global;wrote=true;
            }
            bool kernel = false;
            for (unsigned core = 0; core != 16; ++core) kernel = kernel || (regs[0x1918 + 4 * core] & 1);
            if (kernel && bars.count(0x1908)) {
                for (unsigned core = 0; core != 16; ++core)
                    if (regs[0x1918 + 4 * core] & 1)
                        if (static_cast<std::uint64_t>(regs[0x1908]) + regs[0x1958 + 4 * core] + regs[0x1998 + 4 * core] > layer.constants.size())
                            refuse("native kernel DMA exceeds its layer section");
            }
            // H14 LUT loads use BAR refs at 0x19f4 (one word) and 0x1a00
            // (two words), not the convolution coefficient table at 0x1908.
            // Preserve and relocate every decoded kernel/LUT BAR ref.
            for (const auto &bar : bars)
                if (bar.first >= 0x1900 && bar.first < 0x1a40) {
                    if (bar.second > 1 || regs[bar.first] >= layer.constants.size() ||
                        (task.barWords.at(bar.first) == 2 && regs[bar.first+4]))
                        refuse("unknown native kernel or LUT BAR form");
                    regs[bar.first] += static_cast<std::uint32_t>(constantBase);
                    task.bars[bar.first] = 1;
                }
            const auto encoded = emit(std::move(task));
            if (!global) out.firstTaskBytes = encoded.size();
            out.taskStream.insert(out.taskStream.end(), encoded.begin(), encoded.end());
            if (global + 1 != total) out.taskStream.resize((out.taskStream.size() + 15) / 16 * 16, 0);
            ++global;
        }
        if (!wrote) refuse("native graph layer does not write its declared output");
        out.constants.insert(out.constants.end(), layer.constants.begin(), layer.constants.end());
        out.constants.resize((out.constants.size() + 127) / 128 * 128, 0);
    }
    out.constantOffsetBytes = (out.taskStream.size() + 63) / 64 * 64;
    if (placement) *placement=std::move(plan);
    return out;
}

h14::Program linkH14(const std::vector<h14::Program> &layers,std::vector<h13::LinkedLayer> *layout) {
    if (layers.empty()) refuse("empty native chain");
    std::vector<std::string> boundaries;
    for (std::size_t i=0;i<layers.front().inputs.size();++i) boundaries.push_back("$input"+std::to_string(i));
    std::vector<Connection> connections;
    for (std::size_t i=0;i<layers.size();++i) {
        Connection c;
        c.inputs=i?std::vector<std::string>{connections.back().output}:boundaries;
        c.output="$layer"+std::to_string(i);
        connections.push_back(std::move(c));
    }
    return linkH14Graph(layers,connections,boundaries,connections.back().output,nullptr,layout);
}
} // namespace ane::layers
