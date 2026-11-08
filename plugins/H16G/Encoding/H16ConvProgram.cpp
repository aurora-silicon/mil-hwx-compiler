#include "H16ConvProgram.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ane::h16 {
namespace {
using U32 = std::uint32_t;
// Standalone H16G objects place input, output and __KERN_0 at BAR slots
// 6, 7 and 8. Formula tasks must use this object-writer BAR law rather than
// the one-slot-earlier numbering retained by sanitized captures.
struct Schedule {
    U32 cin, cout, width, kernel, stride, groups;
    bool bias;
    U32 tileHeight, sourceSkew, common;
    bool direct;
    // Output-channel group count; -1 derives log2(Cout/16) (depthwise: 0).
    int ocg;
};
// Scheduler outputs, not extrapolated shape formulas. See h16-h17-conv.md.
// Rows below the first block come from the round-2 corpus
// (H16-H17-CONV-FIELDS-r2.md): MobileNetV1 widths 256..1024 and bias.
constexpr Schedule schedules[] = {
    {64,64,64,1,1,1,false,64,0,0x210000,true,-1},
    {32,64,112,1,1,1,false,112,0x50,0x231100,false,-1},
    {64,128,112,1,1,1,false,80,0x50,0x231100,false,-1},
    {64,128,56,1,1,1,false,56,0x50,0x240000,false,-1},
    {3,32,224,3,2,1,false,112,0,0x230000,false,-1},
    {64,64,112,3,1,1,false,112,0,0x200000,true,-1},
    {32,32,112,3,1,32,false,112,0x50,0x230000,false,-1},
    {64,64,112,3,2,64,false,56,0,0x210000,true,-1},
    {128,128,56,3,2,128,false,28,0x30,0x220000,true,-1},
    // round 2
    {256,256,28,3,1,256,false,28,0x50,0x220000,true,-1},
    {512,512,14,3,1,512,false,14,0x30,0x230000,true,-1},
    {1024,1024,7,3,1,1024,false,7,0x30,0x244404,true,-1},
    {256,256,28,3,2,256,false,14,0x20,0x230000,true,-1},
    {512,512,14,3,2,512,false,7,0x20,0x244404,true,-1},
    {128,256,28,1,1,1,false,28,0x50,0x220000,true,4},
    {256,256,28,1,1,1,false,28,0x50,0x220000,true,4},
    {256,512,14,1,1,1,false,14,0x30,0x230000,true,4},
    {512,512,14,1,1,1,false,14,0x30,0x233300,true,4},
    {512,1024,14,1,1,1,false,14,0x30,0x233300,true,4},
    {512,1024,28,1,1,1,false,28,0x50,0x222200,true,4},
    {512,1024,7,1,1,1,false,7,0x30,0x233304,true,5},
    {512,1024,7,1,1,1,true,7,0x30,0x244404,true,5},
    {1024,1024,7,1,1,1,false,7,0x30,0x244404,true,5},
    // Round 3 (t-4235): the two MobileNetV1 layers at 56x56.
    {128,128,56,3,1,128,false,56,0,0x210000,true,-1},
    // Provisional: depthwise 64 at 112 (stride-1 form of the s2 layer), from
    // the measured depthwise 32 at 112.
    {64,64,112,3,1,64,false,112,0x50,0x230000,false,-1},
    {128,128,56,1,1,1,false,56,0x50,0x240000,false,-1},
};
const Schedule *schedule(ConvShape s) {
    const Schedule *plain = nullptr;
    for (const auto &p : schedules)
        if (s.input.channels == p.cin && s.output.channels == p.cout &&
            s.input.width == p.width && s.kernel == p.kernel &&
            s.stride == p.stride && s.groups == p.groups) {
            if (s.bias == p.bias) return &p;
            if (!p.bias) plain = &p;
        }
    // A biased layer without its own measured row takes the plain row's
    // schedule; only NE bit 0x10 and the kernel bytes change (provisional).
    return s.bias ? plain : nullptr;
}
U32 align64(U32 n) { return (n + 63) & ~U32(63); }
constexpr U32 taskBar(U32 slot) { return 0x20000000u | (slot<<23); }
void word(std::vector<std::uint8_t> &out, U32 value) {
    for (unsigned b = 0; b < 4; ++b) out.push_back(value >> (8*b));
}
void store(std::vector<std::uint8_t> &out, std::size_t at, U32 value) {
    if (at > out.size() || out.size()-at < 4)
        throw std::logic_error("H16/H17 word is outside task stream");
    for (unsigned b = 0; b < 4; ++b) out[at+b] = value >> (8*b);
}
// Rebuild packed dense/scatter headers from register addresses. The high
// relocation flags are part of the record grammar, not register values.
void record(Task &t, const std::vector<U32> &addresses, U32 flags = 0) {
    if (addresses.empty()) return;
    const U32 base = addresses.front();
    bool dense = true;
    for (std::size_t i = 0; i < addresses.size(); ++i)
        dense &= addresses[i] == base + 4*i;
    U32 header = base/4 | flags;
    if (dense) {
        if (addresses.size() > 64) throw std::logic_error("record too long");
        header |= (addresses.size()-1) << 15;
    } else {
        header |= 0x80000000u;
        for (std::size_t i = 1; i < addresses.size(); ++i) {
            U32 delta = (addresses[i]-base)/4;
            if (!delta || delta > 16) throw std::logic_error("scatter too wide");
            header |= 1u << (delta+14);
        }
    }
    word(t.bytes, header);
    for (U32 address : addresses) {
        if (address == 0x5510) t.kernelRelocation = t.bytes.size();
        word(t.bytes, t.registers.at(address));
    }
}
void range(Task &t, U32 lo, U32 hi, U32 flags = 0) {
    std::vector<U32> addresses;
    for (const auto &[a,v] : t.registers) {
        (void)v;
        if (a >= lo && a <= hi) addresses.push_back(a);
    }
    record(t, addresses, flags);
}
void pack(Task &t) {
    t.bytes.clear();
    for (U32 h : t.header) word(t.bytes, h);
    if (t.registers.count(0x5500)) {
        if (t.registers.count(0x5520)) {
            std::vector<U32> coefficients{0x5500};
            for (U32 a=0x5520;a<=0x5540;a+=4) coefficients.push_back(a);
            record(t,coefficients); range(t,0x5544,0x555c);
        } else record(t,{0x5500});
    }
    // Reset chunk size 64 is omitted, including mixed sparse tables.
    // Split across a default hole; contiguous offsets/sizes share a record.
    std::vector<U32> chunk;
    for (const auto &[a,v] : t.registers) {
        (void)v;
        if (a < 0x5564 || a > 0x55dc) continue;
        if (!chunk.empty() && a != chunk.back()+4) {
            record(t,chunk); chunk.clear();
        }
        chunk.push_back(a);
    }
    record(t,chunk);
    range(t,0x0004,0x000c); range(t,0x0014,0x004c);
    if (t.registers.count(0x4d08)) {
        std::vector<U32> sourceGeometry;
        for (U32 address : {0x4d08u,0x4d18u,0x4d1cu})
            if (t.registers.count(address)) sourceGeometry.push_back(address);
        record(t,sourceGeometry);
    }
    range(t,0x4d68,0x4d68);
    if (t.transformerStagingRecords) {
        range(t,0x4104,0x4108); range(t,0x4110,0x415c);
    } else if (t.transformerContinuationRecords) {
        range(t,0x4114,0x415c);
    } else if (t.registers.at(0x4148) == 0x50017a) {
        range(t,0x4104,0x4108); range(t,0x4114,0x414c);
    } else {
        range(t,0x4104,0x4120); range(t,0x4148,0x415c);
    }
    range(t,0x4900,0x4910); range(t,0x5110,0x5150);
    range(t,0x5900,0x592c);
    range(t,0x5510,0x5514,t.kernelSlotFlags);
    range(t,0x4d10,0x4d14,t.sourceSlotFlags);
    range(t,0x5108,0x510c,t.resultSlotFlags);
    range(t,0x4d00,0x4d00,0x22000000);
    range(t,0x5100,0x5100,0x22000000);
    // DMA address records are relocation operands, emitted separately from
    // the ordinary register records. They must not also appear in 4d08 range.
    if (t.bytes.size()/4 > 0x7ff) throw std::invalid_argument("task exceeds TaskSize");
    t.header[0] = U32(t.bytes.size()/4)<<16 | (t.header[0]&0xffff);
    store(t.bytes,0,t.header[0]);
}
// Kernel bytes per output lane. Dense sections (every stride-1 form, and
// the dense stride-2 form an executable package uses) follow the packers:
// depthwise lanes hold Cout/16 channels of taps (+1 bias row), regular
// lanes hold Cout/16 columns of Cin*taps rows (+1 bias row), padded to 64.
// The sparse stride-2 vectors reproduce Apple's zero-skipping sections for
// the campaign's BLOBFILE view only (descriptor parity, never packaging).
std::array<U32,16> chunks(ConvShape s, bool sparse) {
    std::array<U32,16> out{};
    const bool dw = s.groups == s.input.channels;
    const U32 taps = s.kernel*s.kernelWidth;
    const U32 rows = (dw ? 1 : s.input.channels)*taps + (s.bias ? 1 : 0);
    if (s.stride == 1 || !sparse) {
        out.fill(align64(2*rows*(s.output.channels/16)));
    } else if (dw && s.input.channels == 64) out.fill(128);
    else if (dw && s.input.channels == 128) {
        for (unsigned i = 0; i < 16; ++i) out[i] = i < 7 ? 192 : 256;
    } else if (dw && s.input.channels == 256) {
        out.fill(0x1c0);
        for (unsigned i : {1u,2u,4u,5u,6u}) out[i] = 0x180;
    } else if (dw && s.input.channels == 512) out.fill(0x3c0);
    else {
        out.fill(128); out[0] = 64;
    }
    return out;
}
Task task(Generation gen, Layer layer, bool sparse, unsigned stage, unsigned count,
          U32 srcBase, U32 resultBase, U32 kernelBase, U32 residentRow,
          U32 residentStride) {
    const auto s = layer.shape;
    const auto &p = *schedule(s);
    const bool first = stage == 0, last = stage+1 == count;
    const bool linked = count != 1;
    const bool dw = s.groups == s.input.channels;
    const bool h17 = gen == Generation::H17;
    Task t;
    // ExeCycles (header[1]) is unknown; zero is an explicit unmodelled hint.
    t.header = {stage,0, first ? (last ? 0x04000068u : 0x68u)
                               : (last ? 0x04000000u : 0u),
                0, first ? 0x00fff868u : (last ? 0x00fff860u
                                    : (h17 ? 0x00fff800u : 0x00fff820u)),
                0,0, last ? 0u : (count == 2 ? 0x7fu
                                  : (first ? 0x01ffffffu : 0x7fffffffu)),
                last ? 0x00050009u : 0x00050001u};
    auto &r = t.registers;
    r[4]=s.input.width; r[8]=s.input.height; r[12]=s.input.channels;
    r[0x14]=s.output.width; r[0x18]=s.output.height; r[0x1c]=s.output.channels;
    if (s.kernel != 1) {
        U32 pad = s.stride == 1 ? 1 : 0;
        r[0x28]=s.kernel | s.kernel<<6 | s.stride<<13 | s.stride<<15 |
                   pad<<17 | pad<<22 | 1u<<28 | 1u<<30;
    }
    if (dw) r[0x30]=0x14000;
    r[0x34]=p.tileHeight; r[0x3c]=p.common;
    // Measured generation delta (r2): standalone depthwise C256 28->14.
    if (!h17 && dw && s.stride==2 && s.input.channels==256) r[0x3c]=0x233300;
    if (linked && !first && !last) r[0x3c] |= 0x1100;
    if (h17 && s.kernel==3 && s.groups==1 && s.input.channels==64)
        r[0x3c] |= 1u<<27;
    U32 ocg=0;
    if (p.ocg >= 0) ocg=U32(p.ocg);
    else if (!dw) for (U32 n=s.output.channels/16; n>1; n>>=1) ++ocg;
    if (ocg != 3) r[0x40]=ocg;
    r[0x4c]=h17 ? 0x200000 : 0x100000;
    // The L2 source row is the row's bytes rounded up to 16 (7 -> 16,
    // 14 -> 32, 28 -> 64); 1x1 kernels cap the row at 64 pixels.
    const U32 srcRow=first ? ((2*(s.kernel==1 ? std::min(s.input.width,64u)
                                                : s.input.width)+15)&~15u) : residentRow;
    const U32 srcStride=first ? s.input.channels*srcRow+p.sourceSkew : residentStride;
    r[0x4104]=first ? 0x500172 : 0x120;
    if (first) r[0x4108]=0x500130;
    if (srcBase) r[0x4110]=srcBase;
    r[0x4114]=srcRow; r[0x4118]=srcStride;
    if (first) r[0x411c]=r[0x4120]=s.input.channels*srcRow;
    r[0x4148]=!linked ? (p.direct ? 0x50017a : 0x500172)
                                    : (last ? 0x142 : (first ? 0x500170 : 0x140));
    if (!linked) resultBase=std::min(s.input.height,p.tileHeight*s.stride+s.kernel-1)*srcStride;
    if (resultBase) r[0x414c]=resultBase;
    if (!p.direct || linked) {
        const U32 row=linked ? 2*s.output.width : 2*std::min(s.output.width,64u);
        // The stem and depthwise use the full internal width.
        const U32 dstRow=s.kernel==3 ? 2*s.output.width : row;
        U32 skew=4*(align64(2*s.output.width)-2*s.output.width);
        if (linked && !last && dw) skew=0x80;
        if (linked && first && count==3) skew=0x20;
        r[0x4150]=dstRow; r[0x4154]=s.output.channels*dstRow+skew;
        if (last) r[0x4158]=r[0x415c]=s.output.channels*dstRow;
    }
    const bool sparseKernel = s.stride==2 && sparse;
    if (h17 || sparseKernel) r[0x4900]=(h17 ? 0x10000082 : 0x82) | (sparseKernel ? 0x100 : 0);
    r[0x4904]=0x103c00 | (layer.relu ? 0x10000 : 0) | (s.bias ? 0x10 : 0);
    r[0x4910]=0x3c00;
    if (first) {
        r[0x4d00]=0x21; r[0x4d08]=0xce;
        r[0x4d10]=r[0x4d14]=0;
        r[0x4d18]=align64(2*s.input.width);
        r[0x4d1c]=r[0x4d18]*s.input.height; r[0x4d68]=0x1002031;
    }
    if (last) {
        r[0x5100]=(h17 ? 0x21 : 0x31) | (linked ? stage<<16 : (p.direct ? 1u<<24 : 0));
        r[0x5108]=r[0x510c]=0;
        r[0x5110]=align64(2*s.output.width);
        r[0x5114]=r[0x5110]*s.output.height; r[0x5138]=0x1302031;
        // These decoded CacheDMA writes ARE present in all final conv tasks,
        // despite the field report's sentence saying the block is unwritten.
        r[0x5900]=0x12; r[0x5904]=0xa000a0;
        if (linked) r[0x5908]=stage<<16 | 0xe0;
        r[0x5918]=0; r[0x5928]=0x7f0000; r[0x592c]=0x20010701;
    }
    r[0x5500]=(stage+1)<<16 | (last ? 0x2c0 : 0x240) | (sparseKernel ? 0x20 : 0);
    r[0x5510]=kernelBase; r[0x5514]=0;
    const auto sizes=layer.kernelChunks[0] ? layer.kernelChunks : chunks(s,sparse);
    U32 offset=0;
    for (unsigned i=0; i<16; ++i) {
        r[0x5520+4*i]=stage<<16 | 0x21;
        if (i) r[0x5560+4*i]=offset;
        if (sizes[i]!=64) r[0x55a0+4*i]=sizes[i];
        offset+=sizes[i];
    }
    return t;
}

U32 linearInternalRow(U32 tokens) {
    return align64(2*tokens)+16;
}

std::vector<U32> linearOutputSlices(U32 tokens, U32 cin, U32 cout,
                                    bool bias, bool gelu) {
    if (tokens != 1500) return {cout};
    const std::uint64_t budget = 1024*1024;
    std::vector<U32> slices;
    U32 remaining = cout;
    while (remaining) {
        const auto fits=[&](U32 candidate) {
            const std::uint64_t lanes = candidate/16;
            const std::uint64_t group = align64(2*(cin+(bias ? 1u : 0u))*lanes)
                                      + (gelu ? 128u : 0u);
            return 16*group <= budget;
        };
        U32 candidate=remaining;
        if (!fits(candidate)) {
            candidate=remaining/16*16;
            while (candidate && !fits(candidate)) candidate-=16;
            candidate=candidate/(candidate>=256 ? 256 : 32)
                     *(candidate>=256 ? 256 : 32);
        }
        if (!candidate) throw std::invalid_argument(
            "one transformer-linear output slice exceeds the 1 MiB kernel envelope");
        slices.push_back(candidate);
        remaining -= candidate;
    }
    return slices;
}

std::vector<U32> linearLaneWidths(U32 outputs) {
    // The scratch-backed narrow contraction treats a sub-256 slice as one
    // (possibly non-power-of-two) interleave. Wider slices use 256-channel
    // chunks and a power-of-two tail.
    if (outputs<256) return {outputs/16};
    std::vector<U32> widths;
    U32 remaining = outputs;
    while (remaining) {
        U32 lanes = 1;
        while (lanes < 16 && 32*lanes <= remaining) lanes *= 2;
        widths.push_back(lanes);
        remaining -= 16*lanes;
    }
    return widths;
}

std::vector<std::uint8_t> packLinearSlice(
    U32 cin, U32 firstOutput, U32 outputs, const std::uint8_t *weights,
    const std::uint8_t *bias, const std::uint8_t *table, std::size_t tableBytes) {
    const auto widths = linearLaneWidths(outputs);
    const std::size_t rows = cin+(bias ? 1 : 0);
    std::size_t groupBytes = 0;
    for (U32 width : widths) groupBytes += 2*rows*width;
    groupBytes = align64(U32(groupBytes));
    std::vector<std::uint8_t> packed(16*(groupBytes+tableBytes));
    U32 chunkFirst = 0;
    for (U32 group=0; group<16; ++group) {
        const std::size_t groupStart=group*(groupBytes+tableBytes);
        if (tableBytes)
            std::copy(table,table+tableBytes,packed.begin()+groupStart);
        std::size_t cursor=groupStart+tableBytes;
        chunkFirst=0;
        for (U32 width : widths) {
            if (bias) for (U32 lane=0; lane<width; ++lane) {
                const U32 output=firstOutput+chunkFirst+group*width+lane;
                packed[cursor++]=bias[2*output];
                packed[cursor++]=bias[2*output+1];
            }
            for (U32 input=0; input<cin; ++input)
                for (U32 lane=0; lane<width; ++lane) {
                    const U32 output=firstOutput+chunkFirst+group*width+lane;
                    const std::size_t source=2*(std::size_t(output)*cin+input);
                    packed[cursor++]=weights[source];
                    packed[cursor++]=weights[source+1];
                }
            chunkFirst += 16*width;
        }
    }
    return packed;
}

Task linearStagingTask(U32 tokens, U32 cin, bool wide) {
    Task t;
    t.transformerStagingRecords=true;
    // A standalone object binds its first external surface at BAR 6.  Wide
    // contractions stage through the fixed scratch aperture at BAR 3.
    t.sourceSlotFlags=taskBar(6);
    if (wide) t.resultSlotFlags=taskBar(3);
    const U32 row=linearInternalRow(tokens);
    U32 dependency=0;
    if (!wide) {
        const U32 dispatches=(cin*row+0xffff)/0x10000;
        if (dispatches>=32) throw std::invalid_argument(
            "transformer-linear staging exceeds the decoded dependency mask");
        dependency=(U32(1)<<dispatches)-1;
    }
    t.header={0,0,0x68,0,0x68,0,0,dependency,0x50001};
    auto &r=t.registers;
    r[4]=r[0x14]=cin; r[8]=r[0x18]=1; r[0xc]=r[0x1c]=tokens;
    r[0x30]=0x84000; r[0x34]=1; r[0x3c]=0x30210100; r[0x4c]=0x100000;
    r[0x4104]=0x500172; r[0x4108]=0x500130; r[0x4114]=0x100;
    r[0x4118]=r[0x411c]=r[0x4120]=tokens*0x100;
    if (wide) {
        r[0x4148]=0x50087a; r[0x414c]=tokens*0x100;
    } else {
        r[0x4110]=cin*row; r[0x4148]=0x500170; r[0x4150]=row;
    }
    r[0x4904]=0x103c0c; r[0x4910]=0x3c00;
    r[0x4d00]=0x21; r[0x4d08]=0xce; r[0x4d10]=r[0x4d14]=0;
    r[0x4d1c]=2*cin; r[0x4d68]=0x1002031;
    if (wide) {
        r[0x5100]=0x10000f1; r[0x5108]=r[0x510c]=0;
        r[0x5114]=tokens*16; r[0x5138]=0x8302031;
    }
    pack(t);
    return t;
}

Task linearComputeTask(U32 tokens, U32 cin, U32 cout, U32 slice,
                       U32 firstOutput, U32 taskIndex, U32 taskCount,
                       U32 kernelBase, bool wide, bool bias, bool gelu,
                       U32 outputScale) {
    Task t;
    // The object writer places input/output/__KERN_0 at BAR 6/7/8.  Bias and
    // post-op tables are packed into __KERN_0 alongside the weights, so every
    // coefficient slice uses BAR 8.  Wide contractions read staged data from
    // the fixed scratch aperture at BAR 3.
    t.kernelSlotFlags=taskBar(8);
    t.resultSlotFlags=taskBar(7);
    if (wide) t.sourceSlotFlags=taskBar(3);
    if (!wide && taskIndex>1) t.transformerContinuationRecords=true;
    const bool last=taskIndex+1==taskCount;
    U32 dependency=0;
    if (!last && !wide) {
        const U32 dispatches=(cin*linearInternalRow(tokens)+0xffff)/0x10000;
        dependency=(U32(1)<<dispatches)-1;
    }
    bool serialized = false;
    if (wide) serialized=!bias;
    else if (tokens==128 && cin>cout) serialized=true;
    else if (gelu && (tokens==128 || cin>=512)) serialized=true;
    else if (outputScale!=0x3c00 && cin!=512) serialized=true;
    else if (bias && ((tokens==128 && cout>cin) ||
                      (tokens==1500 && cin==384 && cout==384))) serialized=true;
    t.header={taskIndex,0,last ? 0x04000000u : 0u,0,
              serialized ? (last ? 0x00fff860u : 0x00fff820u) : 0x28u,
              0,0,dependency,last ? 0x50009u : 0x50001u};
    auto &r=t.registers;
    r[4]=r[0x14]=tokens; r[8]=r[0x18]=1; r[0xc]=cin; r[0x1c]=slice;
    r[0x34]=1;
    r[0x3c]=wide && slice<=64 ? 0x10220200
            : wide && slice<256 ? 0x10210100 : 0x10200000;
    if (!wide) r[0x40]=4;
    else {
        U32 ocg=0;
        for (U32 lanes=slice/16; lanes>1; lanes=(lanes+1)/2) ++ocg;
        if (ocg!=3) r[0x40]=ocg;
    }
    r[0x4c]=0x100000;
    if (wide) {
        r[0x4104]=taskIndex==1 ? 0x862 : 0x842;
        r[0x4114]=0x100;
        r[0x4118]=r[0x411c]=r[0x4120]=cin*0x100;
        r[0x4148]=0x14a; r[0x414c]=cin*0x100;
        r[0x4d00]=0x0f000000+taskIndex*0x01010000+(last ? 0x41 : 0xe1);
        r[0x4d08]=last ? 0x4e : 0xee;
        r[0x4d10]=r[0x4d14]=0; r[0x4d1c]=tokens*16;
        r[0x4d68]=0x8002031;
    } else {
        if (taskIndex==1) r[0x4104]=0x120;
        r[0x4114]=linearInternalRow(tokens); r[0x4148]=0x14a;
        r[0x414c]=cin*linearInternalRow(tokens);
    }
    r[0x4904]=0x103c00 | (bias ? 0x10 : 0) | (gelu ? 0x20000 : 0);
    r[0x4910]=outputScale;
    r[0x5100]=0x1000031 | taskIndex<<16; r[0x5108]=r[0x510c]=0;
    r[0x5114]=2*cout; r[0x5138]=0x1302031;
    if (firstOutput) r[0x5150]=firstOutput<<16;
    r[0x5500]=(taskIndex+1)<<16 | (last ? 0x2c0 : 0x240);
    r[0x5510]=kernelBase; r[0x5514]=0;
    const U32 lanes=slice/16;
    const U32 laneBytes=align64(2*(cin+(bias ? 1u : 0u))*lanes)
                      +(gelu ? 128u : 0u);
    U32 offset=0;
    for (U32 lane=0; lane<16; ++lane) {
        r[0x5520+4*lane]=taskIndex<<16 | 0x21;
        if (lane) r[0x5560+4*lane]=offset;
        if (laneBytes!=64) r[0x55a0+4*lane]=laneBytes;
        offset+=laneBytes;
    }
    if (last) {
        r[0x5900]=0x12; r[0x5904]=0xa000a0; r[0x5908]=taskIndex<<16|0xe0;
        r[0x5918]=0; r[0x5928]=0x7f0000; r[0x592c]=0x20010701;
    }
    pack(t);
    return t;
}

Task whisperStem1Task(U32 tokens, U32 cout, U32 kernelBytes, bool gelu) {
    Task t;
    t.kernelSlotFlags=taskBar(8);
    t.sourceSlotFlags=taskBar(6);
    t.resultSlotFlags=taskBar(7);
    const U32 externalRow=align64(2*tokens);
    t.header={0,0,0x04000068,0,
              (gelu || cout>=512) ? 0x00fff868u : 0x68u,
              0,0,0,0x50009};
    auto &r=t.registers;
    r[4]=r[0x14]=tokens; r[8]=r[0x18]=1; r[0xc]=80; r[0x1c]=cout;
    r[0x28]=0x5002a043; r[0x34]=1; r[0x3c]=0x200000;
    r[0x40]=4; r[0x4c]=0x100000;
    r[0x4104]=0x500172; r[0x4108]=0x500130;
    r[0x4114]=0x150; r[0x4118]=r[0x411c]=r[0x4120]=80*0x150;
    r[0x4148]=0x50017a; r[0x414c]=80*0x150;
    r[0x4904]=0x103c10 | (gelu ? 0x20000 : 0); r[0x4910]=0x3c00;
    r[0x4d00]=0x21; r[0x4d08]=0xce; r[0x4d10]=r[0x4d14]=0;
    r[0x4d1c]=externalRow; r[0x4d68]=0x1002031;
    r[0x5100]=0x1000031; r[0x5108]=r[0x510c]=0;
    r[0x5114]=externalRow; r[0x5138]=0x1302031;
    r[0x5500]=0x102c0; r[0x5510]=0; r[0x5514]=0;
    const U32 laneBytes=kernelBytes/16;
    U32 offset=0;
    for (U32 lane=0;lane<16;++lane) {
        r[0x5520+4*lane]=0x21;
        if (lane) r[0x5560+4*lane]=offset;
        if (laneBytes!=64) r[0x55a0+4*lane]=laneBytes;
        offset+=laneBytes;
    }
    r[0x5900]=0x12; r[0x5904]=0xa000a0; r[0x5918]=0;
    r[0x5928]=0x7f0000; r[0x592c]=0x20010701;
    pack(t);
    return t;
}

std::vector<U32> whisperStem2Slices(U32 channels, bool gelu) {
    if (channels==384) return gelu ? std::vector<U32>{128,256}
                                   : std::vector<U32>{384};
    return gelu ? std::vector<U32>{64,224,224}
                : std::vector<U32>{208,288,16};
}

std::vector<U32> whisperStem2FirstOutputs(U32 channels, bool gelu) {
    if (channels==384) return gelu ? std::vector<U32>{256,0}
                                   : std::vector<U32>{0};
    return gelu ? std::vector<U32>{448,224,0}
                : std::vector<U32>{304,0,288};
}

bool whisperStemNonzeroHalf(const std::uint8_t *value) {
    return value[0] || (value[1]&0x7f);
}
void whisperStemHalf(std::vector<std::uint8_t> &out,
                     const std::uint8_t *value) {
    out.push_back(value[0]); out.push_back(value[1]);
}

// One output-lane group of the phase-sparse 1x3 writer. Each input channel
// occupies four even-width phase rows: taps 0,2,1, then an all-zero row.
// Sixteen logical halfword slots share one little-endian nonzero mask.
// Signed zero is structural zero; every other fp16 bit pattern is retained.
std::vector<std::uint8_t> packWhisperStem2Group(
    U32 cin, U32 firstOutput, U32 outputs, U32 group,
    const std::uint8_t *weights, const std::uint8_t *bias,
    const std::uint8_t *table, std::size_t tableBytes) {
    std::vector<std::uint8_t> out;
    if (tableBytes) out.insert(out.end(),table,table+tableBytes);
    U32 chunkFirst=0;
    for (U32 width : linearLaneWidths(outputs)) {
        std::vector<U32> indices;
        for (U32 lane=0;lane<width;++lane)
            indices.push_back(firstOutput+chunkFirst+group*width+lane);
        for (U32 output : indices) whisperStemHalf(out,bias+2*output);
        U32 encoded=(13*width*cin+511)/512;
        encoded=(encoded+1)&~U32(1);
        out.push_back(0); out.push_back(std::uint8_t(encoded));
        out.push_back(std::uint8_t(encoded>>8));

        const U32 evenWidth=(width+1)&~U32(1);
        std::vector<const std::uint8_t *> slots;
        slots.reserve(std::size_t(cin)*4*evenWidth);
        for (U32 input=0;input<cin;++input) {
            for (U32 tap : {0u,2u,1u}) {
                for (U32 output : indices)
                    slots.push_back(weights+2*((std::size_t(output)*cin+input)*3+tap));
                for (U32 lane=width;lane<evenWidth;++lane) slots.push_back(nullptr);
            }
            for (U32 lane=0;lane<evenWidth;++lane) slots.push_back(nullptr);
        }
        if (slots.size()%16) throw std::logic_error(
            "Whisper sparse plane is not a whole mask vector");
        for (std::size_t at=0;at<slots.size();at+=16) {
            U32 mask=0;
            for (U32 lane=0;lane<16;++lane)
                if (slots[at+lane] && whisperStemNonzeroHalf(slots[at+lane]))
                    mask|=1u<<lane;
            out.push_back(std::uint8_t(mask)); out.push_back(std::uint8_t(mask>>8));
            for (U32 lane=0;lane<16;++lane)
                if (mask&(1u<<lane)) whisperStemHalf(out,slots[at+lane]);
        }
        chunkFirst+=16*width;
    }
    while (out.size()%64) out.push_back(0);
    return out;
}

Task whisperStem2Task(U32 cin, U32 slice, U32 firstOutput,
                      U32 taskIndex, U32 taskCount, U32 kernelBase,
                      const std::array<U32,16> &chunkBytes, bool gelu) {
    Task t;
    t.kernelSlotFlags=taskBar(8);
    t.sourceSlotFlags=taskBar(6);
    t.resultSlotFlags=taskBar(7);
    const bool first=taskIndex==0,last=taskIndex+1==taskCount;
    t.header={taskIndex,0,first ? (last ? 0x04000068u : 0x68u)
                               : (last ? 0x04000000u : 0u),
              0,first ? 0x00fff868u : (last ? 0x00fff860u : 0x00fff820u),
              0,0,0,last ? 0x50009u : 0x50001u};
    auto &r=t.registers;
    r[4]=3000; r[8]=1; r[0xc]=cin;
    r[0x14]=1500; r[0x18]=1;
    // The one-lane tail uses the measured zero code for sixteen outputs.
    if (slice!=16) r[0x1c]=slice;
    r[0x28]=0x5000c043; r[0x34]=1; r[0x3c]=0x200000;
    if (slice==64) r[0x40]=2;
    else if (slice!=128) r[0x40]=4;
    r[0x4c]=0x100000;
    r[0x4104]=first ? 0x500172 : 0x142;
    if (first) r[0x4108]=0x500130;
    r[0x4114]=0x240;
    r[0x4118]=r[0x411c]=r[0x4120]=cin*0x240;
    r[0x4148]=first ? 0x50017a : 0x14a;
    r[0x414c]=cin*0x240;
    r[0x4900]=0x182; r[0x4904]=0x103c10|(gelu ? 0x20000 : 0);
    r[0x4910]=0x3c00;
    r[0x4d00]=taskIndex<<16|(last ? (first ? 0x21 : 0xc1) : 0xe1);
    r[0x4d08]=last ? 0xce : 0xee;
    r[0x4d10]=r[0x4d14]=0; r[0x4d1c]=align64(6000);
    r[0x4d68]=0x1002031;
    r[0x5100]=0x1000031|taskIndex<<16;
    r[0x5108]=firstOutput*align64(3000); r[0x510c]=0;
    r[0x5114]=align64(3000); r[0x5138]=0x1302031;
    r[0x5500]=(taskIndex+1)<<16|(last ? 0x2e0 : 0x260);
    r[0x5510]=kernelBase; r[0x5514]=0;
    U32 offset=0;
    for (U32 lane=0;lane<16;++lane) {
        r[0x5520+4*lane]=taskIndex<<16|0x21;
        if (lane) r[0x5560+4*lane]=offset;
        if (chunkBytes[lane]!=64) r[0x55a0+4*lane]=chunkBytes[lane];
        offset+=chunkBytes[lane];
    }
    if (last) {
        r[0x5900]=0x12; r[0x5904]=0xa000a0;
        if (!first) r[0x5908]=taskIndex<<16|0xe0;
        r[0x5918]=0; r[0x5928]=0x7f0000; r[0x592c]=0x20010701;
    }
    pack(t);
    return t;
}

Task whisperTransposeTask(U32 channels) {
    Task t;
    t.sourceSlotFlags=taskBar(6);
    t.resultSlotFlags=taskBar(7);
    t.header={0,0,0x04000068,0,0x68,0,0,0,0x50009};
    auto &r=t.registers;
    r[0x5500]=0x80;
    r[4]=r[0x14]=1500; r[8]=r[0x18]=1;
    r[0xc]=r[0x1c]=channels;
    r[0x30]=0x84000; r[0x34]=1; r[0x3c]=0x30210100; r[0x4c]=0x100000;
    r[0x4104]=0x500172; r[0x4108]=0x500130; r[0x4114]=0x100;
    r[0x4118]=r[0x411c]=r[0x4120]=channels*0x100;
    r[0x4148]=0x50017a; r[0x414c]=channels*0x100;
    r[0x4904]=0x103c0c; r[0x4910]=0x3c00;
    r[0x4d00]=0x21; r[0x4d08]=0xce; r[0x4d10]=r[0x4d14]=0;
    r[0x4d1c]=align64(3000); r[0x4d68]=0x1002031;
    r[0x5100]=0x1000031; r[0x5108]=r[0x510c]=0;
    r[0x5114]=2*channels; r[0x5138]=0x1302031;
    r[0x5900]=0x12; r[0x5904]=0xa000a0; r[0x5918]=0;
    r[0x5928]=0x7f0000; r[0x592c]=0x20010701;
    pack(t);
    return t;
}

U32 fp32Bits(float value) {
    U32 bits=0;
    static_assert(sizeof(bits)==sizeof(value));
    std::memcpy(&bits,&value,sizeof(bits));
    return bits;
}

void layerNormHeader(Task &task,U32 index,U32 dependency,U32 mode=1) {
    task.header={index,0,0,0,0x28,0,0,dependency,mode};
}

void packLayerNormLongLoad(Task &task,U32 index,bool noResultBase) {
    task.bytes.clear();for(U32 value:task.header)word(task.bytes,value);
    record(task,{4,8,12});record(task,{0x14,0x18,0x1c,0x34,0x3c,0x44,0x4c});
    record(task,{0x4d08,0x4d1c});record(task,{0x4d68});
    if(index==0) {
        record(task,{0x4104,0x4108});
        record(task,{0x4110,0x4114,0x4118,0x411c,0x4120,0x4148,0x414c,0x4150});
    } else {
        record(task,{0x4104,0x4110,0x4114,0x4118,0x411c,0x4120});
        if(noResultBase) record(task,{0x4150}); else record(task,{0x414c,0x4150});
    }
    record(task,{0x4514});record(task,{0x4d10,0x4d14},taskBar(6));
    record(task,{0x4d00},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|index;store(task.bytes,0,task.header[0]);
}

void packLayerNormLongVariance(Task &task,U32 index,bool noSourceBase,
                               bool edge) {
    task.bytes.clear();for(U32 value:task.header)word(task.bytes,value);
    record(task,{4,8,12});record(task,{0x14,0x18,0x1c,0x34,0x3c,0x44,0x48,0x4c});
    record(task,{0x4d0c,0x4d34});record(task,{0x4d6c});
    if(index==4) {
        record(task,{0x4104,0x4108});
        record(task,{0x4110,0x4114,0x4124,0x4128,0x412c,0x4130,0x4134,0x414c,0x4150});
    } else if(noSourceBase) {
        record(task,{0x4104,0x4108});
        record(task,{0x4114,0x4124,0x4128,0x412c,0x4130,0x4134,0x414c,0x4150});
    } else if(edge) {
        record(task,{0x4108,0x4110,0x4114,0x4124,0x4128,0x412c,0x4130,0x4134});
        record(task,{0x414c,0x4150});
    } else {
        record(task,{0x4104,0x4108});
        record(task,{0x4110,0x4114,0x4124,0x4128,0x412c,0x4130,0x4134,0x414c,0x4150});
    }
    record(task,{0x4500,0x450c,0x4514});
    record(task,{0x4d28,0x4d2c},taskBar(6));record(task,{0x4d04},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|index;store(task.bytes,0,task.header[0]);
}

Task layerNormLongLoadTask(U32 channels,U32 index) {
    const U32 tile=384,edge=1500-3*tile,row=2*channels;
    const bool edgeTile=index==0;Task task;
    task.header={index,0,index==0?0x68u:0u,0,index==0?0x68u:0x28u,0,0,1,1};
    auto &r=task.registers;r[4]=r[0x14]=channels;r[8]=r[0x18]=1;
    r[12]=r[0x1c]=edgeTile?edge:tile;r[0x34]=1;r[0x3c]=0x50;r[0x44]=9;r[0x4c]=0;
    r[0x4104]=index?0x142:0x500172;if(!index)r[0x4108]=0x500130;
    r[0x4110]=1500*32;r[0x4114]=row;
    const U32 bytes=(edgeTile?edge:tile)*row;
    r[0x4118]=r[0x411c]=r[0x4120]=bytes;
    if(!index)r[0x4148]=0x500130;
    const U32 position=3-index;if(position)r[0x414c]=position*tile*16;
    r[0x4150]=16;r[0x4514]=fp32Bits(1.0f/float(channels));
    r[0x4d00]=index<<16|0xe1;r[0x4d08]=0xee;
    r[0x4d10]=position*tile*row;r[0x4d14]=0;r[0x4d1c]=row;r[0x4d68]=0x1002031;
    packLayerNormLongLoad(task,index,index==3);return task;
}

Task layerNormLongVarianceTask(U32 channels,U32 index) {
    const U32 tile=384,edge=1500-3*tile,row=2*channels;
    const U32 position=index==7?3:6-index;const bool last=index==7;
    Task task;task.header={index,0,0,0,0x28,0,0,1,1};auto &r=task.registers;
    r[4]=r[0x14]=channels;r[8]=r[0x18]=1;r[12]=r[0x1c]=last?edge:tile;
    r[0x34]=1;r[0x3c]=0x60;r[0x44]=9;r[0x48]=1;r[0x4c]=0;
    if(index!=7)r[0x4104]=0x400100;r[0x4108]=0x142;
    if(position)r[0x4110]=position*tile*16;r[0x4114]=16;r[0x4124]=1500*32;
    r[0x4128]=row;const U32 bytes=(last?edge:tile)*row;
    r[0x412c]=r[0x4130]=r[0x4134]=bytes;
    r[0x414c]=1500*16+position*tile*16;r[0x4150]=16;
    r[0x4500]=0x92010;r[0x450c]=0x37280000;r[0x4514]=fp32Bits(1.0f/float(channels));
    r[0x4d04]=index<<16|0xe1;r[0x4d0c]=0xee;
    r[0x4d28]=position*tile*row;r[0x4d2c]=0;r[0x4d34]=row;r[0x4d6c]=0x1002031;
    packLayerNormLongVariance(task,index,index==6,last);return task;
}

Task layerNormLongReduceTask(U32 channels) {
    Task task;layerNormHeader(task,8,channels==512?0x1ffffff:0,
                             channels==512?1:0);auto &r=task.registers;
    r[4]=r[0x14]=channels;r[8]=r[0x18]=1;r[12]=r[0x1c]=1500;
    r[0x34]=1;r[0x3c]=0x40;r[0x44]=7;r[0x4c]=0;
    r[0x4104]=0x142;r[0x4108]=0x400100;
    r[0x4110]=channels==384?0x184f80:0x182b80;r[0x4114]=0x100;
    r[0x4118]=r[0x411c]=r[0x4120]=1500*0x100;r[0x4128]=16;r[0x412c]=1500*16;
    if(channels==384){r[0x4148]=0x103;r[0x414c]=0x127380;r[0x4150]=0x100;
        r[0x4154]=r[0x4158]=r[0x415c]=1500*0x100;}
    else {r[0x414c]=1500*32;r[0x4150]=2*channels;}
    r[0x4500]=0xc0000;r[0x4d00]=0x800c1;r[0x4d08]=0xce;
    r[0x4d10]=r[0x4d14]=0;r[0x4d1c]=2*channels;r[0x4d68]=0x1002031;
    task.bytes.clear();for(U32 value:task.header)word(task.bytes,value);
    record(task,{4,8,12});record(task,{0x14,0x18,0x1c,0x34,0x3c,0x44,0x4c});
    record(task,{0x4d08,0x4d1c});record(task,{0x4d68});
    if(channels==384){record(task,{0x4104,0x4108,0x4110,0x4114,0x4118,0x411c,0x4120,0x4128,0x412c});
        record(task,{0x4148,0x414c,0x4150,0x4154,0x4158,0x415c});}
    else {record(task,{0x4104,0x4108});record(task,{0x4110,0x4114,0x4118,0x411c,0x4120,0x4128,0x412c,0x414c,0x4150});}
    record(task,{0x4500});record(task,{0x4d10,0x4d14},taskBar(6));
    record(task,{0x4d00},0x22000000);task.header[0]=U32(task.bytes.size()/4)<<16|8;
    store(task.bytes,0,task.header[0]);return task;
}

Task layerNormShortLoadTask() {
    Task task;task.header={0,0,0x68,0,0x68,0,0,0x3c0,1};auto &r=task.registers;
    r[4]=r[0x14]=768;r[8]=r[0x18]=1;r[12]=r[0x1c]=128;
    r[0x34]=1;r[0x3c]=0x50;r[0x44]=9;r[0x48]=0x400;r[0x4c]=0;
    r[0x4104]=0x500171;r[0x4108]=0x500130;r[0x4110]=0x63200;
    r[0x4114]=1536;r[0x4118]=r[0x411c]=r[0x4120]=128*1536;
    r[0x4148]=0x500130;r[0x414c]=0x63000;r[0x4514]=fp32Bits(1.0f/768.0f);
    r[0x4d00]=0x21;r[0x4d08]=0xce;r[0x4d10]=r[0x4d14]=0;
    r[0x4d1c]=1536;r[0x4d68]=0x1002031;
    task.bytes.clear();for(U32 value:task.header)word(task.bytes,value);
    record(task,{4,8,12});record(task,{0x14,0x18,0x1c,0x34,0x3c,0x44,0x48,0x4c});
    record(task,{0x4d08,0x4d1c});record(task,{0x4d68});record(task,{0x4104,0x4108});
    record(task,{0x4110,0x4114,0x4118,0x411c,0x4120,0x4148,0x414c});
    record(task,{0x4514});record(task,{0x4d10,0x4d14},taskBar(6));
    record(task,{0x4d00},0x22000000);task.header[0]=U32(task.bytes.size()/4)<<16;
    store(task.bytes,0,task.header[0]);return task;
}

Task layerNormShortMiddleTask(U32 index) {
    Task task;task.header={index,index==1?3u:1u,0,0,0x28,0,0,index==1?0x3c0u:0x3ffu,1};
    task.header[1]=0;auto &r=task.registers;r[4]=r[0x14]=768;r[8]=r[0x18]=1;
    r[12]=r[0x1c]=128;r[0x34]=1;r[0x3c]=index==1?0x60:0x40;
    r[0x44]=index==1?9:7;r[0x48]=index==1?0x501:0x200;r[0x4c]=0;
    if(index==1){r[0x4104]=0x400100;r[0x4110]=0x63000;r[0x4124]=0x63200;
        r[0x4128]=1536;r[0x414c]=0x63100;r[0x4500]=0x92010;r[0x450c]=0x37280000;
        r[0x4514]=fp32Bits(1.0f/768.0f);}
    else {r[0x4108]=0x400100;r[0x4110]=0x63200;r[0x4114]=1536;
        r[0x4124]=0x63000;r[0x412c]=0x100;r[0x414c]=0x33000;r[0x4150]=1536;
        r[0x4500]=0xc0000;}
    task.bytes.clear();for(U32 value:task.header)word(task.bytes,value);
    record(task,{4,8,12});record(task,{0x14,0x18,0x1c,0x34,0x3c,0x44,0x48,0x4c});
    if(index==1){record(task,{0x4104,0x4110,0x4124,0x4128});record(task,{0x414c});
        record(task,{0x4500,0x450c,0x4514});}
    else {record(task,{0x4108,0x4110,0x4114,0x4124,0x412c});record(task,{0x414c,0x4150});
        record(task,{0x4500});}
    task.header[0]=U32(task.bytes.size()/4)<<16|index;store(task.bytes,0,task.header[0]);return task;
}

Task layerNormBoundaryTask(U32 tokens,U32 channels,U32 index) {
    Task task;layerNormHeader(task,index,0x3ffff,1);auto &r=task.registers;
    const U32 row=2*channels;r[4]=r[0x14]=channels;r[8]=r[0x18]=1;
    r[12]=r[0x1c]=tokens;r[0x34]=1;r[0x3c]=0x30;r[0x44]=9;r[0x48]=0x80;r[0x4c]=0;
    r[0x4104]=0x400100;r[0x4108]=0x142;r[0x4114]=row;r[0x4124]=0;
    r[0x4128]=r[0x412c]=r[0x4130]=r[0x4134]=row;r[0x4150]=row;r[0x4500]=0x80004;
    r[0x4d04]=index<<16|0x21;r[0x4d0c]=0xce;r[0x4d28]=0x3000;r[0x4d2c]=0;
    r[0x4d6c]=0x1002031;
    task.bytes.clear();for(U32 value:task.header)word(task.bytes,value);
    record(task,{4,8,12});record(task,{0x14,0x18,0x1c,0x34,0x3c,0x44,0x48,0x4c});
    record(task,{0x4d0c});record(task,{0x4d6c});record(task,{0x4104,0x4108});
    record(task,{0x4114,0x4124,0x4128,0x412c,0x4130,0x4134,0x4150});
    record(task,{0x4500});record(task,{0x4d28,0x4d2c},taskBar(3));
    record(task,{0x4d04},0x22000000);task.header[0]=U32(task.bytes.size()/4)<<16|index;
    store(task.bytes,0,task.header[0]);return task;
}

Task layerNormOutputTask(U32 tokens,U32 channels,U32 index) {
    Task task;task.header={index,0,0x04000000,0,0x28,0,0,0,9};auto &r=task.registers;
    const U32 row=2*channels;r[0x5500]=0x80;r[4]=r[0x14]=channels;
    r[8]=r[0x18]=tokens;r[12]=r[0x1c]=1;r[0x34]=tokens;r[0x3c]=0x30;
    r[0x44]=9;r[0x48]=0x20;r[0x4c]=0;r[0x4104]=0x400100;r[0x4108]=0x142;
    r[0x4118]=row;r[0x4124]=0x40;r[0x4128]=r[0x412c]=r[0x4130]=r[0x4134]=row;
    r[0x4148]=0x14a;r[0x414c]=row+0x40;r[0x4500]=0x80000;
    r[0x4d04]=index<<16|0x21;r[0x4d0c]=0xce;r[0x4d28]=0x3300;r[0x4d2c]=0;
    r[0x4d6c]=0x1002031;r[0x5100]=0x01000031|index<<16;
    r[0x5108]=r[0x510c]=0;r[0x5110]=row;r[0x5138]=0x1302031;
    r[0x5900]=0x12;r[0x5904]=0xa000a0;r[0x5908]=index<<16|0xe0;
    r[0x5918]=0;r[0x5928]=0x7f0000;r[0x592c]=0x20010701;
    task.bytes.clear();for(U32 value:task.header)word(task.bytes,value);
    record(task,{0x5500});record(task,{4,8,12});
    record(task,{0x14,0x18,0x1c,0x34,0x3c,0x44,0x48,0x4c});record(task,{0x4d0c});
    record(task,{0x4d6c});record(task,{0x4104,0x4108});
    record(task,{0x4118,0x4124,0x4128,0x412c,0x4130,0x4134,0x4148,0x414c});
    record(task,{0x4500});record(task,{0x5110,0x5138});
    record(task,{0x5900,0x5904,0x5908,0x5918,0x5928,0x592c});
    record(task,{0x4d28,0x4d2c},taskBar(3));record(task,{0x5108,0x510c},taskBar(7));
    record(task,{0x4d04},0x22000000);record(task,{0x5100},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|index;store(task.bytes,0,task.header[0]);return task;
}
}

std::string transformerLayerNormRefusal(U32 tokens,U32 channels) {
    if((tokens==128&&channels==768)||(tokens==1500&&(channels==384||channels==512)))
        return {};
    return "transformer layer_norm requires BERT [1,128,768] or Whisper [1,1500,384/512]";
}

TransformerLayerNormProgram encodeTransformerLayerNorm(U32 tokens,U32 channels) {
    if(auto why=transformerLayerNormRefusal(tokens,channels);!why.empty())
        throw std::invalid_argument(why);
    TransformerLayerNormProgram out;auto &program=out.program;
    program.task.assign(16,0);program.task[0]=1;
    std::vector<Task> tasks;
    if(tokens==1500) {
        for(U32 index=0;index<4;++index)tasks.push_back(layerNormLongLoadTask(channels,index));
        for(U32 index=4;index<8;++index)tasks.push_back(layerNormLongVarianceTask(channels,index));
        tasks.push_back(layerNormLongReduceTask(channels));
    } else {
        tasks.push_back(layerNormShortLoadTask());
        tasks.push_back(layerNormShortMiddleTask(1));
        tasks.push_back(layerNormShortMiddleTask(2));
    }
    tasks.push_back(layerNormBoundaryTask(tokens,channels,U32(tasks.size())));
    tasks.push_back(layerNormOutputTask(tokens,channels,U32(tasks.size())));
    for(Task &task:tasks) {
        program.task.resize((program.task.size()+15)&~std::size_t(15),0);
        program.task.insert(program.task.end(),task.bytes.begin(),task.bytes.end());
    }
    program.taskCount=U32(tasks.size());program.firstTaskBytes=tasks.front().bytes.size();
    program.constantOffsetBytes=(program.task.size()+63)&~std::size_t(63);
    const std::size_t tensorBytes=std::size_t(tokens)*channels*2;
    program.scratchAllocationBytes=tensorBytes+0x3300;program.formulaBuilt=true;
    const std::uint64_t row=2*channels,plane=row*tokens;
    program.inputs={{5,{1,1,tokens,channels,plane,row},(plane+0x3fff)&~std::uint64_t(0x3fff)}};
    program.output={4,{1,1,tokens,channels,plane,row},(plane+0x3fff)&~std::uint64_t(0x3fff)};
    // Standalone object-writer metadata uses the admitted mixed-program code;
    // the captured whole-graph descriptor counters include neighboring ops
    // and are not portable to this isolated object.
    out.programRecordCount=0x10;
    out.programFormatCode=20;
    out.scratchBytes=program.scratchAllocationBytes;
    out.revealedInputPrefix=true;
    out.revealedOutputBoundary=tokens==1500&&channels==384;
    return out;
}

std::string transformerLinearRefusal(U32 tokens, U32 cin, U32 cout,
                                     TransformerLinearOptions options) {
    if (!tokens || !cin || !cout || cin%16 || cout%16)
        return "transformer linear requires nonzero S and channel multiples of 16";
    if (options.gelu && (!options.bias || !options.geluTable ||
                         options.geluTableBytes!=128))
        return "fused GELU requires a bias and the decoded 128-byte H16G table";
    if (tokens==1500 && cin>1024 && !(
                      ((cin==1536 && cout==384) ||
                       (cin==2048 && cout==512))))
        return "scratch-backed contraction is measured only for the two Whisper projections";
    if (tokens==1500) return {};
    if (tokens==64 || tokens==128 || tokens==256 || tokens==512) return {};
    return "transformer linear token count is outside the measured/predicted envelope";
}

TransformerLinearProgram encodeTransformerLinear(
    U32 tokens, U32 cin, U32 cout, const std::uint8_t *weights,
    std::size_t weightBytes, const std::uint8_t *bias,
    std::size_t biasBytes, TransformerLinearOptions options) {
    if (auto why=transformerLinearRefusal(tokens,cin,cout,options); !why.empty())
        throw std::invalid_argument(why);
    if (!weights || weightBytes!=std::size_t(cin)*cout*2)
        throw std::invalid_argument("transformer linear needs Cout*Cin fp16 weights");
    if (options.bias!=(bias!=nullptr) ||
        (bias && biasBytes!=std::size_t(cout)*2))
        throw std::invalid_argument("transformer linear bias must contain Cout fp16 values");
    const bool wide=tokens==1500 && cin>1024;
    const auto slices=linearOutputSlices(tokens,cin,cout,options.bias,options.gelu);
    TransformerLinearProgram result;
    auto &program=result.program;
    program.task.assign(16,0);
    program.task[0]=1;
    Task staging=linearStagingTask(tokens,cin,wide);
    program.task.insert(program.task.end(),staging.bytes.begin(),staging.bytes.end());
    U32 firstOutput=0,kernelBase=0;
    const U32 taskCount=U32(slices.size()+1);
    for (U32 index=0; index<slices.size(); ++index) {
        const U32 slice=slices[index];
        auto packed=packLinearSlice(cin,firstOutput,slice,weights,
            options.bias ? bias : nullptr,options.gelu ? options.geluTable : nullptr,
            options.gelu ? options.geluTableBytes : 0);
        program.constants.insert(program.constants.end(),packed.begin(),packed.end());
        program.task.resize((program.task.size()+15)&~std::size_t(15),0);
        Task compute=linearComputeTask(tokens,cin,cout,slice,firstOutput,
            index+1,taskCount,kernelBase,wide,options.bias,options.gelu,
            options.outputScale);
        program.kernelRelocations.push_back(program.task.size()+compute.kernelRelocation);
        program.task.insert(program.task.end(),compute.bytes.begin(),compute.bytes.end());
        kernelBase+=U32(packed.size()); firstOutput+=slice;
    }
    program.taskCount=taskCount;
    program.firstTaskBytes=staging.bytes.size();
    program.constantOffsetBytes=(program.task.size()+63)&~std::size_t(63);
    program.scratchAllocationBytes=wide ? std::size_t(tokens)*cin*2 : 0;
    const auto tensor=[](U32 tokens_,U32 channels,U32 index) {
        const std::uint64_t row=2*channels,plane=row*tokens_;
        return h13::TensorLayout{index,{1,1,tokens_,channels,plane,row},
                                 (plane+0x3fff)&~std::uint64_t(0x3fff)};
    };
    program.inputs={tensor(tokens,cin,5)}; program.output=tensor(tokens,cout,4);
    program.formulaBuilt=true;
    result.programRecordCount=wide ? 0x20 : 0x10;
    result.programFormatCode=0;
    result.scratchBacked=wide;
    result.scratchBytes=program.scratchAllocationBytes;
    return result;
}

TransformerLinearProgram encodeWhisperStem1(
    U32 tokens, U32 cout, const std::uint8_t *weights,
    std::size_t weightBytes, const std::uint8_t *bias,
    std::size_t biasBytes, TransformerLinearOptions options) {
    if (!tokens || (cout!=384 && cout!=512))
        throw std::invalid_argument(
            "Whisper stem1 requires T>0 and the measured C384/C512 width");
    if (!options.bias || !bias || biasBytes!=std::size_t(cout)*2 ||
        !weights || weightBytes!=std::size_t(cout)*80*3*2)
        throw std::invalid_argument(
            "Whisper stem1 requires fp16 [C,80,3] weights and [C] bias");
    if (options.gelu && (!options.geluTable || options.geluTableBytes!=128))
        throw std::invalid_argument(
            "Whisper stem1 GELU requires the decoded 128-byte table");
    auto packed=packLinearSlice(80*3,0,cout,weights,bias,
        options.gelu ? options.geluTable : nullptr,
        options.gelu ? options.geluTableBytes : 0);
    Task task=whisperStem1Task(tokens,cout,U32(packed.size()),options.gelu);
    TransformerLinearProgram result;
    auto &program=result.program;
    program.task.assign(16,0); program.task[0]=1;
    program.kernelRelocations={program.task.size()+task.kernelRelocation};
    program.task.insert(program.task.end(),task.bytes.begin(),task.bytes.end());
    program.constants=std::move(packed); program.taskCount=1;
    program.firstTaskBytes=task.bytes.size();
    program.constantOffsetBytes=(program.task.size()+63)&~std::size_t(63);
    const auto tensor=[&](U32 height,U32 index) {
        const std::uint64_t row=2*tokens,plane=row*height;
        return h13::TensorLayout{index,{1,1,height,tokens,plane,row},
                                 (plane+0x3fff)&~std::uint64_t(0x3fff)};
    };
    program.inputs={tensor(80,5)}; program.output=tensor(cout,4);
    program.formulaBuilt=true;
    result.programRecordCount=0x10; result.programFormatCode=0;
    return result;
}

TransformerLinearProgram encodeWhisperStem2(
    U32 channels, const std::uint8_t *weights, std::size_t weightBytes,
    const std::uint8_t *bias, std::size_t biasBytes,
    TransformerLinearOptions options) {
    if (channels!=384 && channels!=512)
        throw std::invalid_argument("Whisper stem2 requires C384 or C512");
    if (!options.bias || !weights || !bias ||
        weightBytes!=std::size_t(channels)*channels*3*2 ||
        biasBytes!=std::size_t(channels)*2)
        throw std::invalid_argument(
            "Whisper stem2 requires fp16 [C,C,3] weights and [C] bias");
    if (options.gelu && (!options.geluTable || options.geluTableBytes!=128))
        throw std::invalid_argument(
            "Whisper stem2 GELU requires the decoded 128-byte table");
    const auto slices=whisperStem2Slices(channels,options.gelu);
    const auto firstOutputs=whisperStem2FirstOutputs(channels,options.gelu);
    TransformerLinearProgram result;
    auto &program=result.program;
    program.task.assign(16,0); program.task[0]=1;
    U32 kernelBase=0;
    std::size_t firstTaskBytes=0;
    for (U32 index=0;index<slices.size();++index) {
        std::array<U32,16> chunkBytes{};
        for (U32 group=0;group<16;++group) {
            auto packed=packWhisperStem2Group(channels,firstOutputs[index],
                slices[index],group,weights,bias,
                options.gelu ? options.geluTable : nullptr,
                options.gelu ? options.geluTableBytes : 0);
            chunkBytes[group]=U32(packed.size());
            program.constants.insert(program.constants.end(),packed.begin(),packed.end());
        }
        Task task=whisperStem2Task(channels,slices[index],firstOutputs[index],
            index,U32(slices.size()),kernelBase,chunkBytes,options.gelu);
        if (!index) firstTaskBytes=task.bytes.size();
        program.task.resize((program.task.size()+15)&~std::size_t(15),0);
        program.kernelRelocations.push_back(program.task.size()+task.kernelRelocation);
        program.task.insert(program.task.end(),task.bytes.begin(),task.bytes.end());
        for (U32 bytes : chunkBytes) kernelBase+=bytes;
    }
    program.taskCount=U32(slices.size());
    program.firstTaskBytes=firstTaskBytes;
    program.constantOffsetBytes=(program.task.size()+63)&~std::size_t(63);
    const auto tensor=[](U32 height,U32 width,U32 index) {
        const std::uint64_t row=2*width,plane=row*height;
        return h13::TensorLayout{index,{1,1,height,width,plane,row},
                                 (plane+0x3fff)&~std::uint64_t(0x3fff)};
    };
    program.inputs={tensor(channels,3000,5)};
    program.output=tensor(channels,1500,4); program.formulaBuilt=true;
    result.programRecordCount=slices.size()==1 ? 0x10 :
                              slices.size()==2 ? 0x1f : 0x2d;
    result.programFormatCode=0;
    return result;
}

TransformerLinearProgram encodeWhisperTranspose(U32 channels) {
    if (channels!=384 && channels!=512)
        throw std::invalid_argument("Whisper transpose requires C384 or C512");
    Task task=whisperTransposeTask(channels);
    TransformerLinearProgram result;
    auto &program=result.program;
    program.task.assign(16,0); program.task[0]=1;
    program.task.insert(program.task.end(),task.bytes.begin(),task.bytes.end());
    program.taskCount=1; program.firstTaskBytes=task.bytes.size();
    program.constantOffsetBytes=(program.task.size()+63)&~std::size_t(63);
    const auto tensor=[](U32 height,U32 width,U32 index) {
        const std::uint64_t row=2*width,plane=row*height;
        return h13::TensorLayout{index,{1,1,height,width,plane,row},
                                 (plane+0x3fff)&~std::uint64_t(0x3fff)};
    };
    program.inputs={tensor(channels,1500,5)};
    program.output=tensor(1500,channels,4); program.formulaBuilt=true;
    result.programRecordCount=3; result.programFormatCode=0;
    return result;
}

std::string convRefusal(ConvShape s) {
    if (s.kernel!=s.kernelWidth || (s.kernel!=1 && s.kernel!=3))
        return "only measured square 1x1/3x3 kernels are supported";
    if (s.input.width!=s.input.height || s.output.width!=s.output.height ||
        !s.stride || s.output.width!=(s.input.width+s.stride-1)/s.stride)
        return "only the measured square SAME surfaces are supported";
    if (!schedule(s)) return "geometry has no measured H16G/H17 schedule (TileHeight, skew, result mode)";
    return {};
}
std::string executableRefusal(ConvShape s) {
    if (auto why=convRefusal(s); !why.empty()) return why;
    if (s.stride==2) return sparseStride2Refusal(s);
    return {};
}
// The H16 task stream opens with u32 1 and twelve zero bytes, the prologue
// the H16G validator and the H16 driver require (and Apple's H16G output
// carries). H17 keeps sixteen zero bytes, the form the J700 runs.
static std::vector<std::uint8_t> streamPrologue(Generation gen) {
    std::vector<std::uint8_t> stream(16,0);
    if (gen==Generation::H16G) stream[0]=1;
    return stream;
}
Description describe(Generation gen, const std::vector<Layer> &layers, bool sparseStride2) {
    if (gen!=Generation::H16G && gen!=Generation::H17)
        throw std::invalid_argument("unsupported convolution generation");
    if (layers.empty() || layers.size()>3) throw std::invalid_argument("only measured one/two/three-task schedules are supported");
    for (const auto &l:layers) if (auto why=convRefusal(l.shape); !why.empty()) throw std::invalid_argument(why);
    const unsigned count=static_cast<unsigned>(layers.size());
    if (count>1) {
        const unsigned pw=count==3 ? 1 : 0;
        const auto &d=layers[pw].shape;
        const auto &f=layers.back().shape;
        if (d.kernel!=3 || d.stride!=2 || d.input.channels!=64 ||
            d.groups!=64 || d.input.width!=112 || f.kernel!=1 ||
            f.input.channels!=64 || f.output.channels!=128 || f.input.width!=56 ||
            (count==3 && (layers[0].shape.kernel!=1 || layers[0].shape.input.channels!=32 ||
                          layers[0].shape.output.channels!=64 || layers[0].shape.input.width!=112)))
            throw std::invalid_argument("L2-resident linked schedule is outside the two measured resident chains (DRAM-chained programs use describeDramChain)");
    }
    Description out;
    out.stream=streamPrologue(gen);
    U32 residentRow=0,residentStride=0;
    U32 source=count==2 ? 0x63c00 : (count==3 ? 0x188e00 : 0);
    U32 kernelBase=0;
    for (unsigned i=0; i<count; ++i) {
        const auto s=layers[i].shape;
        U32 result=0;
        if (i+1==count && i) result=source+s.input.height*residentStride;
        else if (i && count==3) result=0x188e00;
        Task t=task(gen,layers[i],sparseStride2,i,count,source,result,kernelBase,residentRow,residentStride);
        pack(t);
        // Capacity bound of the measured allocations, not a new scheduler.
        const auto &r=t.registers;
        if (r.count(0x4154) && std::uint64_t(result)+s.output.height*r.at(0x4154)>4*1024*1024)
            throw std::invalid_argument("measured L2 allocation exceeds 4 MiB envelope");
        out.stream.resize((out.stream.size()+15)&~std::size_t(15),0);
        out.kernelRelocations.push_back(out.stream.size()+t.kernelRelocation);
        out.stream.insert(out.stream.end(),t.bytes.begin(),t.bytes.end());
        U32 length=0; for (U32 n:layers[i].kernelChunks[0] ? layers[i].kernelChunks : chunks(s,sparseStride2)) length+=n;
        kernelBase+=length;
        if (r.count(0x4150)) {residentRow=r.at(0x4150);residentStride=r.at(0x4154);}
        source=result;
        out.tasks.push_back(std::move(t));
    }
    out.kernelBytes=kernelBase;
    return out;
}
namespace {
// One phase-mask byte: (bit, tap) entries in ascending bit order.
using MaskByte = std::vector<std::pair<unsigned,unsigned>>;
using PhaseRow = std::array<MaskByte,2>;
// Stride-2 phase rows (H16-H17 stride-2 sparse kernel facts, t-4235).
const std::vector<PhaseRow> depthwiseRows = {
    {MaskByte{{0,0},{2,2},{4,6},{6,8}}, MaskByte{{0,1},{4,7}}},
    {MaskByte{{0,3},{2,5}}, MaskByte{{0,4}}},
};
const std::vector<PhaseRow> depthwiseRows32 = {
    {MaskByte{{0,0},{4,2}}, MaskByte{{0,6},{4,8}}},
    {MaskByte{{0,1}}, MaskByte{{0,7}}},
    {MaskByte{{0,3},{4,5}}, MaskByte{}},
    {MaskByte{{0,4}}, MaskByte{}},
};
bool nonzeroHalf(const std::uint8_t *h) { return h[0] || (h[1]&0x7f); }
void half(std::vector<std::uint8_t> &out, const std::uint8_t *h) { out.push_back(h[0]); out.push_back(h[1]); }
// Append one coefficient plane: u16 n, a structural zero, then per phase row
// the two mask bytes followed by the selected halfwords. `lanes` weights
// share each mask entry (adjacent bits, lane order); weight(lane, tap).
template <class Weight>
void plane(std::vector<std::uint8_t> &out, const std::vector<PhaseRow> &rows,
           unsigned repeats, unsigned lanes, Weight weight) {
    const std::size_t at=out.size();
    out.push_back(0); out.push_back(0); out.push_back(0);
    for (unsigned r=0;r<repeats;++r)
        for (const auto &row:rows) {
            std::uint8_t masks[2]={0,0};
            std::vector<std::uint8_t> values;
            for (unsigned m=0;m<2;++m)
                for (const auto &[bit,tap]:row[m])
                    for (unsigned lane=0;lane<lanes;++lane) {
                        const std::uint8_t *h=weight(r,lane,tap);
                        if (!nonzeroHalf(h)) continue;
                        masks[m]|=std::uint8_t(1u<<(bit+lane));
                        half(values,h);
                    }
            out.push_back(masks[0]); out.push_back(masks[1]);
            out.insert(out.end(),values.begin(),values.end());
        }
    const std::size_t n=out.size()-at-3;
    if (n>0xffff) throw std::logic_error("sparse plane exceeds u16 length");
    out[at]=std::uint8_t(n); out[at+1]=std::uint8_t(n>>8);
}
}
std::string sparseStride2Refusal(ConvShape s) {
    const bool dw=s.groups==s.input.channels && s.groups==s.output.channels;
    if (s.stride!=2 || s.kernel!=3 || s.kernelWidth!=3)
        return "sparse kernels are decoded for 3x3 stride 2 only";
    if (dw && (s.input.channels==64 || s.input.channels==128 ||
               s.input.channels==256 || s.input.channels==512)) return {};
    if (s.groups==1 && s.input.channels==3 && s.output.channels==32) return {};
    return "sparse stride-2 grammar is decoded for depthwise C64..C512 and the 3->32 stem only";
}
std::vector<std::uint8_t> packSparseStride2(ConvShape s, const std::uint8_t *weights,
                                            const std::uint8_t *bias,
                                            std::array<std::uint32_t,16> &chunkBytes) {
    if (auto why=sparseStride2Refusal(s); !why.empty()) throw std::invalid_argument(why);
    std::vector<std::uint8_t> out;
    const bool dw=s.groups!=1;
    for (unsigned chunk=0;chunk<16;++chunk) {
        const std::size_t start=out.size();
        if (dw) {
            // Striped: channel c lives in chunk c mod 16, ascending.
            const unsigned perChunk=s.input.channels/16;
            const auto &rows=perChunk>=32 ? depthwiseRows32 : depthwiseRows;
            for (unsigned c=chunk;c<s.input.channels;c+=16) {
                if (bias) half(out,bias+2*c);
                plane(out,rows,1,1,[&](unsigned,unsigned,unsigned tap) {
                    return weights+2*(std::size_t(c)*9+tap);
                });
            }
            (void)perChunk;
        } else {
            // Stem: outputs 2*chunk and 2*chunk+1 share one plane whose two
            // phase rows repeat for input channels 0..2.
            const unsigned o=2*chunk;
            if (bias) { half(out,bias+2*o); half(out,bias+2*(o+1)); }
            plane(out,depthwiseRows,s.input.channels,2,[&](unsigned c,unsigned lane,unsigned tap) {
                return weights+2*((std::size_t(o+lane)*s.input.channels+c)*9+tap);
            });
        }
        while (out.size()%64) out.push_back(0);
        chunkBytes[chunk]=U32(out.size()-start);
    }
    return out;
}
Description describeDramChain(Generation gen, const std::vector<Layer> &layers,
                              DramEdgeForm form, U32 scratchRegionBytes) {
    if (gen!=Generation::H16G && gen!=Generation::H17)
        throw std::invalid_argument("unsupported convolution generation");
    if (layers.size()<2) throw std::invalid_argument("a DRAM chain needs two or more layers");
    for (const auto &l:layers) if (auto why=convRefusal(l.shape); !why.empty()) throw std::invalid_argument(why);
    const bool h17=gen==Generation::H17;
    const unsigned count=static_cast<unsigned>(layers.size());
    Description out;
    out.stream=streamPrologue(gen);
    U32 kernelBase=0;
    for (unsigned i=0;i<count;++i) {
        const auto s=layers[i].shape;
        const bool first=i==0,last=i+1==count;
        // Each task keeps its measured standalone form (dense kernels) and
        // gains the chain's ID, header and DRAM edge fields (r2, DRAM chain).
        Task t=task(gen,layers[i],true,0,1,0,0,kernelBase,0,0);
        auto &r=t.registers;
        t.header={i,0,(first ? 0x68u : 0u) | (last ? 0x04000000u : 0u),0,
                  first ? 0x00fff868u : (last ? 0x00fff860u : (h17 ? 0x00fff800u : 0x00fff820u)),
                  0,0,last ? 0u : 0x1fffffffu,last ? 0x00050009u : 0x00050001u};
        r[0x5500]=(i+1)<<16 | (last ? 0x2c0 : 0x240);
        for (unsigned k=0;k<16;++k) r[0x5520+4*k]=i<<16 | 0x21;
        const U32 region=scratchRegionBytes;
        if (!first) {
            const U32 low=form==DramEdgeForm::Measured ? 0x41 : 0x21;
            r[0x4d00]=(form==DramEdgeForm::None ? 0u : 0x20000000u) | i<<16 | low;
            if (form==DramEdgeForm::Measured) r[0x4d08]=0x4e;
            r[0x4d10]=((i-1)%2)*region;
            t.sourceSlotFlags=0x21800000;
        }
        r[0x5100]=(r[0x5100]&0xff00ffffu) | i<<16;
        if (!last) {
            if (form!=DramEdgeForm::None) r[0x5100]=(r[0x5100]&~0xffu) | (h17 ? 0xe1 : 0xf1);
            r[0x5108]=(i%2)*region;
            t.resultSlotFlags=0x21800000;
            for (U32 a:{0x5900u,0x5904u,0x5908u,0x5918u,0x5928u,0x592cu}) r.erase(a);
        } else {
            r[0x5908]=i<<16 | 0xe0;
        }
        pack(t);
        out.stream.resize((out.stream.size()+15)&~std::size_t(15),0);
        out.kernelRelocations.push_back(out.stream.size()+t.kernelRelocation);
        out.stream.insert(out.stream.end(),t.bytes.begin(),t.bytes.end());
        U32 length=0; for (U32 n:layers[i].kernelChunks[0] ? layers[i].kernelChunks : chunks(s,true)) length+=n;
        kernelBase+=length;
        out.tasks.push_back(std::move(t));
    }
    out.kernelBytes=kernelBase;
    return out;
}

namespace {
struct MobileLayerShape {
    U32 cin, cout, width, kernel, stride, groups;
};

constexpr std::array<MobileLayerShape,28> mobileNetV1 = {{
    {3,32,224,3,2,1}, {32,32,112,3,1,32}, {32,64,112,1,1,1},
    {64,64,112,3,2,64}, {64,128,56,1,1,1}, {128,128,56,3,1,128},
    {128,128,56,1,1,1}, {128,128,56,3,2,128}, {128,256,28,1,1,1},
    {256,256,28,3,1,256}, {256,256,28,1,1,1}, {256,256,28,3,2,256},
    {256,512,14,1,1,1}, {512,512,14,3,1,512}, {512,512,14,1,1,1},
    {512,512,14,3,1,512}, {512,512,14,1,1,1}, {512,512,14,3,1,512},
    {512,512,14,1,1,1}, {512,512,14,3,1,512}, {512,512,14,1,1,1},
    {512,512,14,3,1,512}, {512,512,14,1,1,1}, {512,512,14,3,2,512},
    {512,1024,7,1,1,1}, {1024,1024,7,3,1,1024},
    {1024,1024,7,1,1,1}, {1024,1024,7,1,1,1},
}};

// Result R strides and bases from the decoded H16G full-edge form.  A zero
// base is deliberately omitted by the encoder: zero is the task-local
// default.  Edge 1 is the sole DRAM spill and therefore has no ring base.
constexpr std::array<U32,27> mobileResultR = {{
    0x1c30,0x1c00,0x3820,0x1c80,0x3830,0x3830,0x3830,
    0x2050,0x4050,0x4050,0x4040,0x2030,
    0x4030,0x4030,0x4030,0x4030,0x4030,0x4030,0x4030,0x4030,0x4030,
    0x4030,0x4020,0x2030,0x4030,0x4030,0x4030,
}};
constexpr std::array<U32,27> mobileResultBase = {{
    0x000000,0x000000,0x000000,0x188e00,0x000000,0x0c4a80,0x189500,
    0x0708c0,0x000000,0x0708c0,0x0e1180,0x151880,
    0x000000,0x0382a0,0x070540,0x0a87e0,0x0e0a80,0x118d20,0x150fc0,
    0x189260,0x1c1500,0x1f97a0,0x031a40,0x069c00,0x000000,0x01c150,
    0x0382a0,
}};
constexpr std::array<U32,28> mobileSchedule = {{
    0x230000,0x230000,0x231100,0x211100,0x243300,0x243300,0x243300,
    0x222200,0x222200,0x222200,0x222200,0x233300,0x233300,0x233300,
    0x233300,0x233300,0x233300,0x233300,0x233300,0x233300,0x233300,
    0x233300,0x233300,0x244404,0x244404,0x244404,0x244404,0x244404,
}};
constexpr std::array<U32,28> mobileCost = {{
    0x04,0x0b,0x0b,0x0c,0x02,0x07,0x05,0x06,0x03,0x03,0x06,0x02,0x03,0x01,
    0x07,0x01,0x07,0x01,0x07,0x01,0x07,0x01,0x07,0x01,0x0f,0x01,0x1e,0x25,
}};
constexpr std::array<U32,28> mobileMask = {{
    0x00001fff,0x00001fff,0x01ffffff,0x7fffffff,0x7f001fff,0x01ffffff,0xfffff01f,
    0xff00079f,0x000007ff,0x00007fff,0x003fff80,0x007fc000,0x0060000f,0x000000ff,
    0x000007f8,0x00007f80,0x0003fc00,0x003fc000,0x01fe0000,0x1fe00000,0xff000000,
    0xf000000f,0x8000007f,0x000000f8,0x000000c3,0x0000000f,0x0000003e,0x00000000,
}};

void requireMobileNet(const std::vector<Layer> &layers) {
    if (layers.size()<2 || layers.size()>mobileNetV1.size())
        throw std::invalid_argument("H16G linked-L2 policies require a 2..28-task prefix of MobileNetV1 224");
    for (std::size_t i=0;i<layers.size();++i) {
        const auto &s=layers[i].shape;
        const auto &want=mobileNetV1[i];
        if (s.input.channels!=want.cin || s.output.channels!=want.cout ||
            s.input.width!=want.width || s.input.height!=want.width ||
            s.kernel!=want.kernel || s.kernelWidth!=want.kernel ||
            s.stride!=want.stride || s.groups!=want.groups || !s.bias ||
            layers[i].relu!=(i+1!=mobileNetV1.size()))
            throw std::invalid_argument("H16G linked-L2 policies require an exact biased MobileNetV1 224 prefix and original ReLU sequence");
    }
}

void eraseRange(std::map<U32,U32> &r, U32 first, U32 last) {
    for (auto it=r.lower_bound(first);it!=r.end() && it->first<=last;)
        it=r.erase(it);
}

// BAR relocation operands are little-endian 64-bit addends.  The old
// ping-pong path copied a nonzero 32-bit offset into both words, turning
// 0x1c0000 into 0x001c0000001c0000.  Zero-offset sentinels could not expose
// that error.  Keep all address-pair writes behind this helper.
void setAddressAddend(std::map<U32,U32> &r, U32 address,
                      std::uint64_t addend) {
    r[address]=static_cast<U32>(addend);
    r[address+4]=static_cast<U32>(addend>>32);
}

U32 mobileSourceCfg(unsigned edge) {
    if (edge==1) return 0x842;
    if (edge<=5 || edge==7 || edge==8) return 0x120;
    return 0x104;
}

bool mobileResident(unsigned edge, unsigned prefix) {
    if (edge==1 || prefix>26) return false;
    const unsigned rank=edge==0 ? 0 : edge-1;
    return rank<prefix;
}

// A DRAM edge produced by a task whose input is resident is not an ordinary
// standalone DRAM edge.  The ordinary 0x500172/0x01302031 form leaves the
// resident source's TileDMA/BAR state live across executions.  Edge 1 is the
// decoded instance of this transition; prefix packages can create one more
// at the end of their resident run.  Use the same explicit spill handshake
// for every such boundary.
bool mobileSpillBridge(unsigned edge, MobileNetL2Variant variant) {
    if (!variant.measuredSpill || mobileResident(edge,variant.residentPrefix))
        return false;
    if (edge==1) return true;
    return edge && mobileResident(edge-1,variant.residentPrefix);
}

// DRAM scratch offset of a non-resident edge.  Edge 1 is the decoded spill
// at offset 0.  Other DRAM edges take the ping-pong region of their index
// parity, except that a task must never write the scratch region it is
// reading: when the previous edge is also in DRAM at that offset, the edge
// takes the other region.  Without this, a prefix package whose resident
// run ends at edge 0 wrote edge 2 over its own edge-1 input.
U32 mobileDramOffset(unsigned edge, MobileNetL2Variant variant) {
    const U32 region=variant.scratchRegionBytes;
    U32 previous=~0u;
    for (unsigned e=0;e<=edge;++e) {
        if (mobileResident(e,variant.residentPrefix)) { previous=~0u; continue; }
        U32 offset=variant.measuredSpill && e==1 ? 0 : U32(e%2)*region;
        if (offset==previous) offset=offset ? 0 : region;
        if (e==edge) return offset;
        previous=offset;
    }
    return 0;
}

std::uint64_t mobileSpan(unsigned edge) {
    const auto &s=mobileNetV1[edge];
    return std::uint64_t((s.width+s.stride-1)/s.stride)*mobileResultR[edge];
}

U32 mobileLinearFirstBase() {
    const auto &s=mobileNetV1[0];
    return align64(2*s.width)*s.cin*s.width;
}

U32 mobileLinearFirstR() {
    const auto &s=mobileNetV1[0];
    const U32 width=(s.width+s.stride-1)/s.stride;
    return s.cout*(2*width)+4*(align64(2*width)-2*width);
}

bool mobileRingOverlap(std::uint64_t a, std::uint64_t an,
                       std::uint64_t b, std::uint64_t bn) {
    constexpr std::uint64_t ring=0x200000;
    if (an+bn>ring) return true;
    for (std::uint64_t aa:{a%ring,a%ring+ring})
        for (std::uint64_t bb:{b%ring,b%ring+ring})
            if (aa<bb+bn && bb<aa+an) return true;
    return false;
}

void proveMobilePlacement() {
    // Task 0 stages its external input immediately after its result.  The
    // spill source of task 2 is not resident in the ring.  Every remaining
    // task with resident input and output must keep both full tensors apart.
    constexpr std::uint64_t inputBytes=224ull*3*448;
    if (mobileRingOverlap(0,mobileSpan(0),mobileSpan(0),inputBytes))
        throw std::logic_error("MobileNet task 0 input overlaps its L2 result");
    for (unsigned taskIndex=3;taskIndex<27;++taskIndex)
        if (mobileRingOverlap(mobileResultBase[taskIndex-1],mobileSpan(taskIndex-1),
                              mobileResultBase[taskIndex],mobileSpan(taskIndex)))
            throw std::logic_error("MobileNet L2 ring placement overlaps a live input");
}

void setSourceTuple(std::map<U32,U32> &r, U32 cfg, U32 base,
                    U32 channels, U32 width, U32 rowStride) {
    r[0x4104]=cfg; r.erase(0x4108);
    if (base) r[0x4110]=base; else r.erase(0x4110);
    const U32 c=(2*width+15)&~U32(15),d=channels*c;
    r[0x4114]=c; r[0x4118]=rowStride; r[0x411c]=d; r[0x4120]=d;
}

void setResultTuple(std::map<U32,U32> &r, U32 cfg, U32 base,
                    U32 channels, U32 width, U32 rowStride) {
    r[0x4148]=cfg;
    if (base) r[0x414c]=base; else r.erase(0x414c);
    const U32 c=(2*width+15)&~U32(15),d=channels*c;
    r[0x4150]=c; r[0x4154]=rowStride; r[0x4158]=d; r[0x415c]=d;
}

void compactMobileNetDefaults(std::map<U32,U32> &r, unsigned taskIndex) {
    // The rung-3 decoded write set relies on task-local defaults for tuple
    // members that are derivable from the shape.  Keep the complete values
    // in the placement plan above; omit only the measured default writes.
    if (taskIndex==0) {
        r.erase(0x4158); r.erase(0x415c);
    } else if (taskIndex==1) {
        r.erase(0x411c); r.erase(0x4120);
        r.erase(0x4150); r.erase(0x4154); r.erase(0x4158); r.erase(0x415c);
    } else if (taskIndex==2) {
        r.erase(0x4158); r.erase(0x415c);
    } else if (taskIndex+1==mobileNetV1.size()) {
        r.erase(0x411c); r.erase(0x4120);
    } else {
        r.erase(0x411c); r.erase(0x4120);
        r.erase(0x4158); r.erase(0x415c);
    }
}
} // namespace

Description describeMobileNetL2(Generation gen, const std::vector<Layer> &layers,
                                MobileNetL2Policy policy) {
    return describeMobileNetL2(gen,layers,MobileNetL2Variant{policy,26,true,0x1c0000});
}

Description describeMobileNetL2(Generation gen, const std::vector<Layer> &layers,
                                MobileNetL2Variant variant) {
    if (gen!=Generation::H16G)
        throw std::invalid_argument("the measured MobileNet linked-L2 policies are H16G-only");
    if (variant.residentPrefix>26)
        throw std::invalid_argument("the MobileNet linked-L2 resident prefix must be 0..26");
    requireMobileNet(layers);
    if (variant.fieldSwap!=MobileNetL2FieldSwap::None && layers.size()!=2)
        throw std::invalid_argument("MobileNet first-edge field swaps require exactly two tasks");
    if (variant.residentPrefix==26) proveMobilePlacement();
    const auto policy=variant.policy;
    Description out;
    out.stream=streamPrologue(gen);
    U32 kernelBase=0;
    for (unsigned i=0;i<layers.size();++i) {
        const bool first=i==0,last=i+1==layers.size();
        Task t=task(gen,layers[i],true,0,1,0,0,kernelBase,0,0);
        auto &r=t.registers;
        t.header={i,0,(first ? 0x68u : 0u) | (last ? 0x04000000u : 0u),0,
                  first ? 0x00fff868u : (last ? 0x00fff860u : 0x00fff820u),
                  0,0,last ? 0u : 0x1fffffffu,last ? 0x00050009u : 0x00050001u};
        if (policy!=MobileNetL2Policy::Placement) r[0x3c]=mobileSchedule[i];
        if (policy==MobileNetL2Policy::Header) {
            t.header[1]=mobileCost[i];
            t.header[4]=first ? 0x68u : 0x28u;
            t.header[7]=mobileMask[i];
        }
        // Keep KernelDMA's sparse bit on every stride-2 task.  The first
        // linked-L2 implementation rebuilt this word from the task id and
        // terminal class and accidentally cleared bit 0x20, while leaving
        // the coefficient section sparse.  The resulting descriptor ran but
        // interpreted the stem and four downsampling kernels incorrectly.
        r[0x5500]=(i+1)<<16 | (last ? 0x2c0 : 0x240) |
                  (layers[i].shape.stride==2 ? 0x20 : 0);
        if (variant.fieldSwap==MobileNetL2FieldSwap::LinearKernelDMA && i==0)
            r[0x5500]&=~0x20u;
        for (unsigned k=0;k<16;++k) r[0x5520+4*k]=i<<16 | 0x21;

        // Source edge: external input, the sole spill read, or an L2 tuple.
        if (first) {
            // Keep the ordinary external source DMA emitted by task().
            if (variant.fieldSwap==MobileNetL2FieldSwap::SentinelFirstEdgeGeometry)
                r.erase(0x4110);
            else
                r[0x4110]=static_cast<U32>(mobileSpan(0));
        } else if (mobileResident(i-1,variant.residentPrefix)) {
            const unsigned edge=i-1;
            const bool linearFirst = edge==0 &&
                variant.fieldSwap==MobileNetL2FieldSwap::SentinelFirstEdgeGeometry;
            setSourceTuple(r,mobileSourceCfg(edge),
                           linearFirst ? mobileLinearFirstBase() : mobileResultBase[edge],
                           mobileNetV1[i].cin,mobileNetV1[i].width,
                           linearFirst ? mobileLinearFirstR() : mobileResultR[edge]);
            eraseRange(r,0x4d00,0x4dff);
        } else if (mobileSpillBridge(i-1,variant)) {
            // A spill bridge stages into a fresh task-local L2 region.  If
            // this task produces a resident result, put the source after it;
            // otherwise the standalone task's result staging already starts
            // after the source tile and base zero is the local default.
            r[0x4104]=0x842; r.erase(0x4108);
            const U32 base=mobileResident(i,variant.residentPrefix)
                ? mobileResultBase[i]+static_cast<U32>(mobileSpan(i)) : 0;
            if (base) r[0x4110]=base; else r.erase(0x4110);
            const auto &edge=mobileNetV1[i-1];
            const U32 width=(edge.width+edge.stride-1)/edge.stride;
            const U32 row=16*width,plane=row*width;
            r[0x4d00]=0x20000041 | i<<16; r[0x4d08]=0x4e;
            setAddressAddend(r,0x4d10,mobileDramOffset(i-1,variant)); r[0x4d18]=row; r[0x4d1c]=plane;
            r[0x4d68]=0x08002031; t.sourceSlotFlags=0x21800000;
        } else {
            if (i<27 && mobileResident(i,variant.residentPrefix))
                r[0x4110]=mobileResultBase[i]+static_cast<U32>(mobileSpan(i));
            r[0x4d00]=0x20000021 | i<<16;
            setAddressAddend(r,0x4d10,mobileDramOffset(i-1,variant));
            t.sourceSlotFlags=0x21800000;
        }

        // Result edge: one spill, 26 resident edges, or the external output.
        if (last) {
            // Preserve the terminal task's measured direct/staged result
            // class.  The complete MobileNet ends in a direct 1x1 task and
            // therefore uses 0x14a.  A leading prefix can instead end in a
            // staged task (notably dw32 and pw32->64); forcing the full
            // graph's direct bit onto those tasks produces plausible output
            // once, but leaves result state armed and wedges the following
            // EXECUTE.  This is the linked equivalent of task()'s
            // 0x50017a/0x500172 distinction.
            r[0x4148]=variant.fieldSwap==MobileNetL2FieldSwap::LegacyTerminalDirect
                ? 0x14a : (schedule(layers[i].shape)->direct ? 0x14a : 0x142);
            if (i==1 && variant.fieldSwap==MobileNetL2FieldSwap::SentinelFirstEdgeGeometry) {
                const U32 width=(mobileNetV1[0].width+mobileNetV1[0].stride-1)/
                                mobileNetV1[0].stride;
                r[0x414c]=mobileLinearFirstBase()+width*mobileLinearFirstR();
            } else {
                r[0x414c]=mobileResultBase[i-1]+mobileSpan(i-1);
            }
            r[0x5100]=(r[0x5100]&0xff00ffffu) | i<<16;
            r[0x5908]=i<<16 | 0xe0;
        } else if (mobileResident(i,variant.residentPrefix)) {
            const U32 cfg=i==0 ? 0x500170 : 0x140;
            const U32 outWidth=(mobileNetV1[i].width+mobileNetV1[i].stride-1)/mobileNetV1[i].stride;
            const bool linearFirst = i==0 &&
                variant.fieldSwap==MobileNetL2FieldSwap::SentinelFirstEdgeGeometry;
            setResultTuple(r,cfg,linearFirst ? mobileLinearFirstBase() : mobileResultBase[i],
                           mobileNetV1[i].cout,outWidth,
                           linearFirst ? mobileLinearFirstR() : mobileResultR[i]);
            eraseRange(r,0x5100,0x51ff);
            eraseRange(r,0x5900,0x59ff);
        } else if (mobileSpillBridge(i,variant)) {
            // A spill result is staged after its resident input.  The
            // 0x84a/0x842 pair is also the state boundary that makes another
            // EXECUTE independent of TileDMA state left by the first one.
            const U32 inputBase=mobileResultBase[i-1];
            setResultTuple(r,0x84a,inputBase+static_cast<U32>(mobileSpan(i-1)),
                           mobileNetV1[i].cout,
                           (mobileNetV1[i].width+mobileNetV1[i].stride-1)/mobileNetV1[i].stride,
                           mobileResultR[i]);
            const U32 width=(mobileNetV1[i].width+mobileNetV1[i].stride-1)/mobileNetV1[i].stride;
            const U32 row=16*width,plane=row*width;
            r[0x5100]=0x010000f1 | i<<16; setAddressAddend(r,0x5108,mobileDramOffset(i,variant));
            r[0x5110]=row; r[0x5114]=plane; r[0x5138]=0x08302031;
            t.resultSlotFlags=0x21800000;
            eraseRange(r,0x5900,0x59ff);
        } else {
            if (i && mobileResident(i-1,variant.residentPrefix))
                r[0x414c]=mobileResultBase[i-1]+static_cast<U32>(mobileSpan(i-1));
            r[0x5100]=(r[0x5100]&0xff00ff00u) | i<<16 | 0xf1;
            setAddressAddend(r,0x5108,mobileDramOffset(i,variant));
            t.resultSlotFlags=0x21800000;
            eraseRange(r,0x5900,0x59ff);
        }

        if (policy==MobileNetL2Policy::Header && variant.residentPrefix==26 &&
            variant.measuredSpill) compactMobileNetDefaults(r,i);
        pack(t);
        out.stream.resize((out.stream.size()+15)&~std::size_t(15),0);
        out.kernelRelocations.push_back(out.stream.size()+t.kernelRelocation);
        out.stream.insert(out.stream.end(),t.bytes.begin(),t.bytes.end());
        U32 length=0;
        for (U32 n:layers[i].kernelChunks[0] ? layers[i].kernelChunks : chunks(layers[i].shape,true))
            length+=n;
        kernelBase+=length;
        out.tasks.push_back(std::move(t));
    }
    // Tasks start on 16-byte boundaries, but the stream ends exactly at the
    // last task's final word, as in every decoded program. Zero bytes after
    // the last task inside the declared __text leave the engine armed: the
    // next EXECUTE of any program times out.
    out.kernelBytes=kernelBase;
    return out;
}
h13::Program encodeConv(Generation gen, ConvShape s, const std::uint8_t *weights, std::size_t size,
                        const std::uint8_t *bias, std::size_t biasBytes, Layer *layerOut) {
    if (auto why=executableRefusal(s); !why.empty()) throw std::invalid_argument(why);
    Layer layer{s,false};
    std::vector<std::uint8_t> constants;
    if (s.stride==2) {
        const std::size_t want=std::size_t(s.output.channels)*s.input.channels/s.groups*9*2;
        if (!weights || size!=want || (s.bias != (bias!=nullptr)) ||
            (bias && biasBytes!=std::size_t(s.output.channels)*2))
            throw std::invalid_argument("stride-2 conv needs Cout*Cin/groups*9 fp16 weights (and Cout bias halfwords)");
        constants=packSparseStride2(s,weights,bias,layer.kernelChunks);
    } else {
        constants=h13::packConvWeights(s,weights,size,bias,biasBytes);
    }
    auto d=describe(gen,{layer},true);
    if (constants.size()!=d.kernelBytes) throw std::logic_error("packed kernel disagrees with the chunk table");
    if (layerOut) *layerOut=layer;
    h13::Program p;
    p.task=std::move(d.stream); p.constants=std::move(constants);
    p.firstTaskBytes=p.task.size(); p.taskCount=1;
    p.constantOffsetBytes=(p.task.size()+63)&~std::size_t(63);
    p.kernelRelocations=std::move(d.kernelRelocations);
    const auto tensor=[](h13::ElementwiseShape shape, U32 index) {
        const std::uint64_t row=align64(2*shape.width),plane=row*shape.height;
        const auto bytes=(plane*shape.channels+0x3fff)&~std::uint64_t(0x3fff);
        return h13::TensorLayout{index,{1,shape.channels,shape.height,shape.width,plane,row},bytes};
    };
    p.inputs={tensor(s.input,5)}; p.output=tensor(s.output,4);
    p.formulaBuilt=true; p.depthwiseConv=s.groups==s.input.channels;
    return p;
}
void fuseRelu(h13::Program &p) {
    // Repack through the descriptor's register record; do not scan arbitrary
    // payload words for a MAC literal.
    std::size_t cursor=16+9*4;
    while (cursor<p.task.size()) {
        U32 h=0; for (unsigned b=0;b<4;++b) h|=U32(p.task[cursor+b])<<(8*b);
        cursor+=4;
        const U32 base=(h&0x7fff)*4;
        std::vector<U32> addresses{base};
        if (h&0x80000000) {
            for (unsigned bit=0;bit<16;++bit) if (h&(1u<<(15+bit))) addresses.push_back(base+4*(bit+1));
        } else for (unsigned i=1;i<=((h>>15)&0x3f);++i) addresses.push_back(base+4*i);
        for (U32 a:addresses) {
            if (a==0x4904) {store(p.task,cursor,0x113c00); return;}
            cursor+=4;
        }
    }
    throw std::logic_error("H16/H17 task has no NE MAC configuration");
}
h13::Program linkDramChain(Generation gen, const std::vector<Layer> &layers,
                           const std::vector<h13::Program> &programs,
                           std::vector<h13::LinkedLayer> *layout) {
    if (layers.size()!=programs.size() || layers.size()<2)
        throw std::invalid_argument("a DRAM chain needs one encoded program per layer, two or more");
    // Intermediates ping-pong between two scratch regions, each sized for the
    // largest intermediate surface.
    std::uint64_t region=0;
    for (std::size_t i=0;i+1<programs.size();++i)
        region=std::max<std::uint64_t>(region,programs[i].output.allocationBytes);
    region=(region+0x3fff)&~std::uint64_t(0x3fff);
    if (region>0x80000000ull) throw std::invalid_argument("intermediate surface exceeds the scratch envelope");
    const auto d=describeDramChain(gen,layers,DramEdgeForm::DependencyOnly,U32(region));
    h13::Program out;
    out.task=d.stream;
    for (std::size_t i=0;i<programs.size();++i) {
        // Each standalone section is packed back to back; the chain's kernel
        // bases are the running sum of the same chunk tables.
        out.constants.insert(out.constants.end(),programs[i].constants.begin(),programs[i].constants.end());
        if (layout) layout->push_back({U32(i),1,i!=0,(i-1)%2*region,i+1!=programs.size(),i%2*region});
    }
    if (out.constants.size()!=d.kernelBytes) throw std::logic_error("chain kernel bytes disagree with the descriptors");
    out.firstTaskBytes=out.task.size(); out.taskCount=U32(programs.size());
    out.constantOffsetBytes=(out.task.size()+63)&~std::size_t(63);
    out.kernelRelocations=d.kernelRelocations;
    out.scratchAllocationBytes=2*region;
    out.inputs=programs.front().inputs; out.output=programs.back().output;
    out.formulaBuilt=true;
    return out;
}

h13::Program linkMobileNetL2(Generation gen, const std::vector<Layer> &layers,
                             const std::vector<h13::Program> &programs,
                             MobileNetL2Policy policy,
                             std::vector<h13::LinkedLayer> *layout) {
    return linkMobileNetL2(gen,layers,programs,
                           MobileNetL2Variant{policy,26,true,0x1c0000},layout);
}

h13::Program linkMobileNetL2(Generation gen, const std::vector<Layer> &layers,
                             const std::vector<h13::Program> &programs,
                             MobileNetL2Variant variant,
                             std::vector<h13::LinkedLayer> *layout) {
    requireMobileNet(layers);
    if (gen!=Generation::H16G || programs.size()!=layers.size())
        throw std::invalid_argument("the MobileNet linked-L2 policy needs one H16G layer program per prefix task");
    for (std::size_t i=0;i<programs.size();++i) {
        if (!programs[i].formulaBuilt || programs[i].taskCount!=1 ||
            programs[i].inputs.size()!=1)
            throw std::invalid_argument("every MobileNet linked-L2 layer must be one formula-built convolution task");
        if (i && (programs[i-1].output.nchw!=programs[i].inputs[0].nchw ||
                  programs[i-1].output.elementSize!=programs[i].inputs[0].elementSize))
            throw std::invalid_argument("a MobileNet linked-L2 layer input differs from its producer output");
    }
    const auto d=describeMobileNetL2(gen,layers,variant);
    h13::Program out;
    out.task=d.stream;
    if (layout) layout->clear();
    for (std::size_t i=0;i<programs.size();++i) {
        out.constants.insert(out.constants.end(),programs[i].constants.begin(),programs[i].constants.end());
        if (layout) {
            const auto edgeCode=[&](std::size_t edge) {
                if (!mobileResident(static_cast<unsigned>(edge),variant.residentPrefix)) return 0;
                return mobileSourceCfg(static_cast<unsigned>(edge))==0x104 ? 3 : 4;
            };
            const bool readsScratch=i && !mobileResident(U32(i-1),variant.residentPrefix);
            const bool writesScratch=i+1<programs.size() &&
                                     !mobileResident(U32(i),variant.residentPrefix);
            h13::LinkedLayer entry{U32(i),1,readsScratch,
                readsScratch ? mobileDramOffset(unsigned(i-1),variant) : 0u,
                writesScratch,
                writesScratch ? mobileDramOffset(unsigned(i),variant) : 0u};
            entry.inputEdge=i ? edgeCode(i-1) : 0;
            entry.outputEdge=i+1<programs.size() ? edgeCode(i) : 0;
            const bool linearFirst=variant.fieldSwap==
                MobileNetL2FieldSwap::SentinelFirstEdgeGeometry;
            if (entry.inputEdge)
                entry.l2Source=linearFirst && i==1
                    ? mobileLinearFirstBase() : mobileResultBase[i-1];
            if (entry.outputEdge) {
                entry.l2Result=linearFirst && i==0
                    ? mobileLinearFirstBase() : mobileResultBase[i];
                const U32 outHeight=(mobileNetV1[i].width+mobileNetV1[i].stride-1)/mobileNetV1[i].stride;
                entry.l2ResultBytes=std::uint64_t(outHeight)*
                    (linearFirst && i==0 ? mobileLinearFirstR() : mobileResultR[i]);
            }
            layout->push_back(entry);
        }
    }
    if (out.constants.size()!=d.kernelBytes)
        throw std::logic_error("MobileNet linked-L2 kernel bytes disagree with the descriptors");
    out.firstTaskBytes=out.task.size(); out.taskCount=U32(programs.size());
    out.constantOffsetBytes=(out.task.size()+63)&~std::size_t(63);
    out.kernelRelocations=d.kernelRelocations;
    unsigned dramEdges=0;
    bool onlyMeasuredEdge1=true;
    for (unsigned edge=0;edge+1<layers.size();++edge)
        if (!mobileResident(edge,variant.residentPrefix)) {
            ++dramEdges;
            onlyMeasuredEdge1 &= variant.measuredSpill && edge==1;
        }
    out.scratchAllocationBytes=!dramEdges ? 0
        : onlyMeasuredEdge1 ? 0xc4000 : 2*variant.scratchRegionBytes;
    out.inputs=programs.front().inputs; out.output=programs.back().output;
    out.formulaBuilt=true;
    return out;
}

namespace {
U32 attentionAlign64(U32 value) { return (value+63u)&~63u; }
constexpr U32 attentionBar(U32 slot) { return taskBar(slot); }

void packAttentionLayoutTask(Task &task) {
    task.bytes.clear();
    for (U32 value : task.header) word(task.bytes,value);
    record(task,{0x5500});
    record(task,{0x0004,0x0008,0x000c});
    std::vector<U32> common{0x0014,0x0018,0x001c,0x0030,0x0034,0x003c};
    if (task.registers.count(0x0040)) common.push_back(0x0040);
    common.push_back(0x004c);
    record(task,common);
    record(task,{0x4d08,0x4d18,0x4d1c});
    record(task,{0x4d68});
    record(task,{0x4104,0x4108});
    record(task,{0x4114,0x4118,0x411c,0x4120,0x4148,0x414c});
    record(task,{0x4904,0x4910});
    record(task,{0x5110,0x5114,0x5138});
    record(task,{0x5900,0x5904,0x5918,0x5928,0x592c});
    record(task,{0x4d10,0x4d14},attentionBar(6));
    record(task,{0x5108,0x510c},attentionBar(7));
    record(task,{0x4d00},0x22000000);
    record(task,{0x5100},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16;
    store(task.bytes,0,task.header[0]);
}

Task attentionLayoutTask(U32 tokens,U32 heads,U32 headDimension,
                         AttentionLayout operation) {
    const bool split=operation==AttentionLayout::Split;
    const bool splitTranspose=operation==AttentionLayout::SplitTranspose;
    const U32 width=heads*headDimension;
    const U32 widthBytes=2*width;
    const U32 headRowBytes=2*headDimension;
    const U32 totalBytes=tokens*widthBytes;
    const U32 sourceSkew=widthBytes>=1024 ? 0x50 : 0;
    Task task;
    task.header={0,0,0x04000068,0,0x68,0,0,0,0x00050009};
    auto &r=task.registers;
    r[0x0004]=headDimension;
    r[0x0008]=split ? tokens : heads;
    r[0x000c]=split ? heads : tokens;
    r[0x0014]=headDimension;
    r[0x0018]=split ? tokens : heads;
    r[0x001c]=split ? heads : tokens;
    r[0x0030]=splitTranspose ? 0x84000 : 0x14000;
    r[0x0034]=split ? 4 : heads;
    if (split) r[0x003c]=0x160000+(heads/2u)*0x10000;
    else r[0x003c]=splitTranspose ? 0x30211100 : 0x210000;
    if (split) {
        const U32 exponent=heads<=2 ? 6 : heads<=3 ? 6 : heads<=5 ? 5
                           : heads<=6 ? 4 : 3;
        r[0x0040]=1u<<exponent;
    } else if (!splitTranspose) r[0x0040]=0;
    r[0x004c]=0x100000;
    r[0x4104]=0x500172; r[0x4108]=0x500130;
    r[0x4114]=headRowBytes;
    if (split) {
        r[0x4118]=widthBytes+sourceSkew;
        r[0x411c]=widthBytes; r[0x4120]=widthBytes;
        r[0x414c]=4*r[0x4118];
    } else {
        r[0x4118]=tokens*headRowBytes;
        r[0x411c]=r[0x4118]; r[0x4120]=r[0x4118];
        r[0x414c]=totalBytes;
    }
    r[0x4148]=0x50017a;
    r[0x4904]=0x103c0c; r[0x4910]=0x3c00;
    r[0x4d00]=0x21; r[0x4d08]=0xce;
    r[0x4d10]=0; r[0x4d14]=0;
    r[0x4d18]=split ? widthBytes
                  : splitTranspose ? headRowBytes : tokens*headRowBytes;
    r[0x4d1c]=split ? headRowBytes
                  : splitTranspose ? widthBytes : headRowBytes;
    r[0x4d68]=0x1002031;
    r[0x5100]=0x01000031; r[0x5108]=0; r[0x510c]=0;
    if (splitTranspose) {
        r[0x5110]=attentionAlign64(2*tokens)*headDimension;
        r[0x5114]=attentionAlign64(2*tokens);
    } else if (split) {
        r[0x5110]=headRowBytes; r[0x5114]=tokens*headRowBytes;
    } else {
        r[0x5110]=headRowBytes; r[0x5114]=widthBytes;
    }
    r[0x5138]=0x1302031;
    r[0x5500]=headRowBytes;
    r[0x5900]=0x12; r[0x5904]=0x00a000a0;
    r[0x5918]=0; r[0x5928]=0x007f0000; r[0x592c]=0x20010701;
    packAttentionLayoutTask(task);
    return task;
}
}

std::string attentionLayoutRefusal(U32 tokens,U32 heads,U32 headDimension,
                                   AttentionLayout operation) {
    (void)operation;
    if (!tokens || !heads || !headDimension) return "attention layout dimensions must be nonzero";
    if (std::uint64_t(tokens)*heads*headDimension*2>std::numeric_limits<U32>::max())
        return "attention layout surface exceeds the 32-bit packet envelope";
    if (headDimension%16) return "attention layout head dimension must be a multiple of 16";
    if (tokens!=128 && tokens!=1500 && tokens!=64 && tokens!=256 && tokens!=512)
        return "attention layout sequence has neither measured nor held-out coverage";
    return {};
}

TransformerAttentionProgram encodeAttentionLayout(
    U32 tokens,U32 heads,U32 headDimension,AttentionLayout operation) {
    if (auto why=attentionLayoutRefusal(tokens,heads,headDimension,operation);
        !why.empty()) throw std::invalid_argument(why);
    Task task=attentionLayoutTask(tokens,heads,headDimension,operation);
    TransformerAttentionProgram out;
    out.program.task=streamPrologue(Generation::H16G);
    out.program.task.insert(out.program.task.end(),task.bytes.begin(),task.bytes.end());
    out.program.taskCount=1;
    out.program.firstTaskBytes=task.bytes.size();
    out.program.constantOffsetBytes=(out.program.task.size()+63)&~std::size_t(63);
    out.program.formulaBuilt=true;
    out.programRecordCount=3;
    const U32 baseFormat=heads*11u/2u;
    out.programFormatCode=baseFormat+
        (operation==AttentionLayout::SplitTranspose && heads>=8 ? 1u : 0u);
    out.revealed=tokens==1500 && headDimension==64 && (heads==6 || heads==8);
    return out;
}

namespace {
std::size_t appendAttentionTask(std::vector<std::uint8_t> &stream,Task &task) {
    stream.resize((stream.size()+15)&~std::size_t(15),0);
    const std::size_t start=stream.size();
    stream.insert(stream.end(),task.bytes.begin(),task.bytes.end());
    return start;
}

// A standalone PV stages probabilities through fp16 scratch before the matrix
// engine consumes them.  At S=1500 the normalized values are small enough
// that the unit-scale staging path loses useful accumulator precision.  A
// power-of-two scale is exact in fp16: stage P*128 and fold 1/128 into the PV
// output scale.  The fused path keeps its softmax intermediate resident and
// therefore retains the captured unit-scale PV tail.
constexpr U32 pvProbabilityStageScale=0x5800; // fp16 128
constexpr U32 pvResultScale=0x2000;           // fp16 1/128

void packQKLoad(Task &task) {
    task.bytes.clear(); for (U32 value:task.header) word(task.bytes,value);
    record(task,{0x0004,0x0008,0x000c});
    record(task,{0x0014,0x0018,0x001c,0x0024,0x0034,0x003c,0x0044,0x004c});
    record(task,{0x4d08,0x4d1c,0x4d24}); record(task,{0x4d68});
    record(task,{0x4104,0x4108});
    record(task,{0x4114,0x4118,0x411c,0x4120,0x4148,0x414c});
    record(task,{0x5114,0x511c,0x5138});
    record(task,{0x4d10,0x4d14},attentionBar(6));
    record(task,{0x5108,0x510c},attentionBar(3));
    record(task,{0x4d00},0x22000000); record(task,{0x5100},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16; store(task.bytes,0,task.header[0]);
}

void packQKCompute(Task &task,bool final,U32 resultFlags=attentionBar(8)) {
    task.bytes.clear(); for (U32 value:task.header) word(task.bytes,value);
    word(task.bytes,0); // extended H16 runtime/runtime matmul header
    record(task,{0x5500,0x5518,0x551c});
    std::vector<U32> tileTable;
    for (U32 address=0x5564;address<=0x55dc;address+=4) tileTable.push_back(address);
    record(task,tileTable);
    record(task,{0x0004,0x0008,0x000c});
    record(task,{0x0014,0x0018,0x001c,0x0024,0x0034,0x003c,0x004c});
    record(task,{0x4d08,0x4d1c,0x4d24}); record(task,{0x4d68});
    record(task,{0x4104,0x4114,0x4118,0x411c,0x4120});
    record(task,{0x4148,0x414c});
    record(task,{0x4900,0x4904,0x4910});
    record(task,{0x5114,0x511c,0x5138});
    if (final) record(task,{0x5900,0x5904,0x5908,0x5918,0x5928,0x592c});
    record(task,{0x5510,0x5514},attentionBar(3));
    record(task,{0x4d10,0x4d14},attentionBar(7));
    record(task,{0x5108,0x510c},resultFlags);
    record(task,{0x4d00},0x22000000); record(task,{0x5100},0x22000000);
    for (U32 lane=0;lane<16;++lane)
        record(task,{0x5520+4*lane},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16 | (task.header[0]&0xffff);
    store(task.bytes,0,task.header[0]);
}

Task qkLoadTask(U32 tokens,U32 heads,U32 headDimension) {
    Task task; task.header={0,0,0x68,0,0x00fff868,0,0,0,1};
    auto &r=task.registers; const U32 rowBytes=2*headDimension;
    const U32 planeBytes=tokens*rowBytes;
    const U32 paddedRow=attentionAlign64(2*tokens);
    r[4]=headDimension;r[8]=1;r[12]=tokens;
    r[0x14]=headDimension;r[0x18]=1;r[0x1c]=tokens;
    r[0x24]=heads;r[0x34]=1;r[0x3c]=0x20;r[0x44]=6;r[0x4c]=0;
    r[0x4104]=0x500172;r[0x4108]=0x500130;r[0x4114]=rowBytes;
    r[0x4118]=planeBytes;r[0x411c]=planeBytes;r[0x4120]=planeBytes;
    r[0x4148]=0x50087a;r[0x414c]=planeBytes;
    r[0x4d00]=0x21;r[0x4d08]=0xce;r[0x4d10]=0;r[0x4d14]=0;
    r[0x4d1c]=rowBytes;r[0x4d24]=planeBytes;r[0x4d68]=0x1002031;
    r[0x5100]=0x10000f1;r[0x5108]=0;r[0x510c]=0;
    r[0x5114]=0x400;r[0x511c]=paddedRow*headDimension;
    r[0x5138]=0x8302031;packQKLoad(task);return task;
}

Task qkComputeTask(U32 tokens,U32 heads,U32 headDimension,U32 stage,bool final) {
    Task task; task.header={stage,0,final?0x04000000u:0u,0,
        final?0x00fff860u:0x00fff820u,0,0,0,
        final?0x0005000bu:0x00050003u};
    auto &r=task.registers; const U32 headPairs=(heads+1)/2;
    const U32 paddedRow=attentionAlign64(2*tokens);
    const U32 halfPaddedPlane=paddedRow*headDimension/2;
    const U32 scorePlane=tokens*paddedRow;
    r[4]=tokens;r[8]=1;r[12]=headDimension;
    r[0x14]=tokens;r[0x18]=1;r[0x1c]=tokens;
    r[0x24]=headPairs;r[0x34]=1;r[0x3c]=0x20200000;r[0x4c]=0x100000;
    r[0x4104]=final?0x142:0x100142;r[0x4114]=0x100;
    r[0x4118]=0x4000;r[0x411c]=0x4000;r[0x4120]=0x4000;
    r[0x4148]=0x14a;r[0x414c]=0x4000;
    r[0x4900]=0x10082;r[0x4904]=0x103c00;r[0x4910]=0x3c00;
    r[0x4d00]=(stage<<16)|0x21;r[0x4d08]=0xce;
    r[0x4d10]=final?0:heads*halfPaddedPlane;r[0x4d14]=0;
    r[0x4d1c]=paddedRow;r[0x4d24]=paddedRow*headDimension;
    r[0x4d68]=0x1002031;
    r[0x5100]=0x01000031|(stage<<16);r[0x5108]=final?0:heads*scorePlane/2;
    r[0x510c]=0;r[0x5114]=paddedRow;r[0x511c]=scorePlane;
    r[0x5138]=0x1302031;
    r[0x5500]=final?0x302c0:0x20240;r[0x5510]=final?0:heads*halfPaddedPlane;
    r[0x5514]=0;r[0x5518]=paddedRow*headDimension;r[0x551c]=0x4000;
    for (U32 lane=0;lane<15;++lane) r[0x5564+4*lane]=(lane+1)*0x400;
    for (U32 lane=0;lane<16;++lane) r[0x55a0+4*lane]=0x400;
    for (U32 lane=0;lane<16;++lane)
        r[0x5520+4*lane]=(stage<<16)|(final?0x41:0xc1);
    if (final) {
        r[0x5900]=0x12;r[0x5904]=0x00a000a0;r[0x5908]=0x200e0;
        r[0x5918]=0;r[0x5928]=0x007f0000;r[0x592c]=0x20010701;
    }
    packQKCompute(task,final);return task;
}

Task qkTransposeStageTask(U32 tokens,U32 heads,U32 headDimension) {
    const bool shortSchedule=heads==6;
    Task task;task.header={1,0,0,0,shortSchedule?0x28u:0x20u,0,0,
        shortSchedule?0x0003ffffu:0x00ffffffu,0x00050001};
    auto &r=task.registers;const U32 rowBytes=2*headDimension;
    const U32 planeBytes=tokens*rowBytes;
    const U32 paddedRow=attentionAlign64(2*tokens);
    const U32 paddedPlane=paddedRow*headDimension;
    r[4]=headDimension;r[8]=1;r[12]=tokens;r[0x14]=headDimension;
    r[0x18]=1;r[0x1c]=tokens;r[0x24]=heads;r[0x30]=0x84000;
    r[0x34]=1;r[0x3c]=0x30211100;r[0x4c]=0x100000;
    r[0x4104]=0x142;r[0x4110]=heads*paddedPlane;r[0x4114]=rowBytes;
    r[0x4118]=planeBytes;r[0x411c]=planeBytes;r[0x4120]=planeBytes;
    r[0x4148]=0x400140;r[0x4150]=paddedRow;r[0x415c]=paddedPlane;
    r[0x4904]=0x103c0c;r[0x4910]=0x3c00;
    r[0x4d00]=0x10021;r[0x4d08]=0xce;r[0x4d10]=0;r[0x4d14]=0;
    r[0x4d1c]=rowBytes;r[0x4d24]=planeBytes;r[0x4d68]=0x1002031;
    task.bytes.clear();for(U32 value:task.header)word(task.bytes,value);
    record(task,{0x0004,0x0008,0x000c});
    record(task,{0x0014,0x0018,0x001c,0x0024,0x0030,0x0034,0x003c,0x004c});
    record(task,{0x4d08,0x4d1c,0x4d24});record(task,{0x4d68});
    record(task,{0x4104,0x4110,0x4114,0x4118,0x411c,0x4120});
    record(task,{0x4148,0x4150,0x415c});record(task,{0x4904,0x4910});
    record(task,{0x4d10,0x4d14},attentionBar(7));record(task,{0x4d00},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|1;store(task.bytes,0,task.header[0]);
    return task;
}

Task qkTransposeComputeTask(U32 tokens,U32 heads,U32 headDimension,
                            U32 stage,bool final) {
    const bool shortSchedule=heads==6;
    Task task;task.header={stage,0,final?0x04000000u:0u,0,
        shortSchedule?0x28u:(final?0x00fff860u:0x00fff820u),0,0,
        final?0u:(shortSchedule?0x0003ffffu:0x00ffffffu),
        final?0x0005000bu:0x00050003u};
    auto &r=task.registers;const U32 pairs=(heads+1)/2;
    const U32 paddedRow=attentionAlign64(2*tokens);
    const U32 paddedPlane=paddedRow*headDimension;
    const U32 scorePlane=tokens*paddedRow;
    r[4]=tokens;r[8]=1;r[12]=headDimension;r[0x14]=tokens;
    r[0x18]=1;r[0x1c]=tokens;r[0x24]=pairs;r[0x34]=1;
    r[0x3c]=0x20200000;r[0x4c]=0x100000;
    if (!final) {r[0x4104]=0x120;r[0x4110]=heads*paddedPlane/2;}
    r[0x4114]=paddedRow;r[0x4120]=paddedPlane;
    r[0x4148]=0x14a;r[0x414c]=heads*paddedPlane;
    r[0x4900]=0x10082;r[0x4904]=0x103c00;r[0x4910]=0x3c00;
    r[0x5100]=0x01000031|(stage<<16);
    r[0x5108]=final?0:heads*scorePlane/2;r[0x510c]=0;
    r[0x5114]=paddedRow;r[0x511c]=scorePlane;r[0x5138]=0x1302031;
    r[0x5500]=final?0x402c0:0x30240;r[0x5510]=final?0:heads*paddedPlane/2;
    r[0x5514]=0;r[0x5518]=paddedPlane;r[0x551c]=0x4000;
    for(U32 lane=0;lane<15;++lane)r[0x5564+4*lane]=(lane+1)*0x400;
    for(U32 lane=0;lane<16;++lane)r[0x55a0+4*lane]=0x400;
    for(U32 lane=0;lane<16;++lane)
        r[0x5520+4*lane]=(stage<<16)|(final?0x41:0xc1);
    if(final){r[0x5900]=0x12;r[0x5904]=0x00a000a0;r[0x5908]=0x300e0;
        r[0x5918]=0;r[0x5928]=0x007f0000;r[0x592c]=0x20010701;}
    task.bytes.clear();for(U32 value:task.header)word(task.bytes,value);word(task.bytes,0);
    record(task,{0x5500,0x5518,0x551c});
    std::vector<U32> table;for(U32 a=0x5564;a<=0x55dc;a+=4)table.push_back(a);record(task,table);
    record(task,{0x0004,0x0008,0x000c});
    record(task,{0x0014,0x0018,0x001c,0x0024,0x0034,0x003c,0x004c});
    if(!final)record(task,{0x4104,0x4110,0x4114,0x4120});
    else record(task,{0x4114,0x4120,0x4148,0x414c});
    if(!final)record(task,{0x4148,0x414c});
    record(task,{0x4900,0x4904,0x4910});record(task,{0x5114,0x511c,0x5138});
    if(final)record(task,{0x5900,0x5904,0x5908,0x5918,0x5928,0x592c});
    record(task,{0x5510,0x5514},attentionBar(3));
    record(task,{0x5108,0x510c},attentionBar(8));record(task,{0x5100},0x22000000);
    for(U32 lane=0;lane<16;++lane)record(task,{0x5520+4*lane},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|stage;store(task.bytes,0,task.header[0]);
    return task;
}

Task pvLoadTask(U32 tokens,U32 heads) {
    Task task;task.header={0,0,0x68,0,0x68,0,0,0,0x00050001};
    auto &r=task.registers;const U32 paddedRow=attentionAlign64(2*tokens);
    const U32 scorePlane=tokens*paddedRow;
    r[4]=tokens;r[8]=1;r[12]=tokens;r[0x14]=tokens;r[0x18]=1;r[0x1c]=tokens;
    r[0x24]=heads;r[0x30]=0x14000;r[0x34]=1;r[0x3c]=0x200000;
    r[0x40]=0;r[0x4c]=0x100000;
    r[0x4104]=0x500172;r[0x4108]=0x500130;r[0x4114]=0x100;
    r[0x4118]=tokens*0x100;r[0x411c]=tokens*0x100;r[0x4120]=tokens*0x100;
    r[0x4148]=0x50087a;r[0x414c]=tokens*0x100;
    r[0x4904]=0x103c0c;r[0x4910]=pvProbabilityStageScale;
    r[0x4d00]=0x21;r[0x4d08]=0xce;r[0x4d10]=0;r[0x4d14]=0;
    r[0x4d1c]=paddedRow;r[0x4d24]=scorePlane;r[0x4d68]=0x1002031;
    r[0x5100]=0x10000f1;r[0x5108]=0;r[0x510c]=0;
    r[0x5114]=tokens*16;r[0x511c]=scorePlane;r[0x5138]=0x8302031;
    task.bytes.clear();for(U32 value:task.header)word(task.bytes,value);
    record(task,{0x0004,0x0008,0x000c});
    record(task,{0x0014,0x0018,0x001c,0x0024,0x0030,0x0034,0x003c,0x0040,0x004c});
    record(task,{0x4d08,0x4d1c,0x4d24});record(task,{0x4d68});
    record(task,{0x4104,0x4108});
    record(task,{0x4114,0x4118,0x411c,0x4120,0x4148,0x414c});
    record(task,{0x4904,0x4910});record(task,{0x5114,0x511c,0x5138});
    record(task,{0x4d10,0x4d14},attentionBar(6));record(task,{0x5108,0x510c},attentionBar(3));
    record(task,{0x4d00},0x22000000);record(task,{0x5100},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16;store(task.bytes,0,task.header[0]);return task;
}

Task pvHeadTask(U32 tokens,U32 heads,U32 headDimension,U32 stage,
                bool fused=false) {
    const bool final=stage==heads;const U32 rowBytes=2*headDimension;
    const U32 taskIndex=fused?10+stage:stage;
    const U32 planeBytes=tokens*rowBytes;const U32 paddedRow=attentionAlign64(2*tokens);
    const U32 scorePlane=tokens*paddedRow;const U32 remaining=heads-stage;
    Task task;task.header={taskIndex,0,final?0x04000000u:0u,0,
        heads==6?0x28u:(final?0x00fff860u:0x00fff820u),0,0,0,
        final?0x0005000bu:0x00050003u};auto &r=task.registers;
    r[4]=headDimension;r[8]=1;r[12]=tokens;r[0x14]=headDimension;
    r[0x18]=1;r[0x1c]=tokens;r[0x34]=1;r[0x3c]=0x20200004;r[0x4c]=0x100000;
    r[0x4104]=0x142|(fused&&stage==1?0x100000:0);r[0x4114]=rowBytes;r[0x4118]=planeBytes;
    r[0x411c]=planeBytes;r[0x4120]=planeBytes;
    r[0x4148]=0x14a|(fused&&stage==1&&heads==8?0x100000:0);r[0x414c]=planeBytes;
    r[0x4900]=0x10082;r[0x4904]=0x103c00;
    r[0x4910]=fused ? 0x3c00 : pvResultScale;
    r[0x4d00]=(taskIndex<<16)|0x21;r[0x4d08]=0xce;r[0x4d10]=remaining*planeBytes;
    r[0x4d14]=0;r[0x4d1c]=rowBytes;r[0x4d68]=0x1002031;
    r[0x5100]=0x01000031|(taskIndex<<16);r[0x5108]=remaining*planeBytes;
    r[0x510c]=0;r[0x5114]=rowBytes;r[0x5138]=0x1302031;
    r[0x5500]=((taskIndex+1)<<16)|(final?0x2c0:0x240);
    r[0x5510]=remaining*scorePlane;r[0x5514]=0;r[0x551c]=2*planeBytes;
    const U32 tileBytes=tokens*16;
    for(U32 lane=0;lane<15;++lane)r[0x5564+4*lane]=(lane+1)*tileBytes;
    for(U32 lane=0;lane<16;++lane)r[0x55a0+4*lane]=tileBytes;
    for(U32 lane=0;lane<16;++lane)r[0x5520+4*lane]=(taskIndex<<16)|(final?0x41:0xc1);
    if(final){r[0x5900]=0x12;r[0x5904]=0x00a000a0;r[0x5908]=(taskIndex<<16)|0xe0;
        r[0x5918]=0;r[0x5928]=0x007f0000;r[0x592c]=0x20010701;}
    task.bytes.clear();for(U32 value:task.header)word(task.bytes,value);word(task.bytes,fused?0x0a:0);
    record(task,{0x5500,0x551c});std::vector<U32> table;
    for(U32 a=0x5564;a<=0x55dc;a+=4)table.push_back(a);record(task,table);
    record(task,{0x0004,0x0008,0x000c});
    record(task,{0x0014,0x0018,0x001c,0x0034,0x003c,0x004c});
    record(task,{0x4d08,0x4d1c});record(task,{0x4d68});
    record(task,{0x4104,0x4114,0x4118,0x411c,0x4120});record(task,{0x4148,0x414c});
    record(task,{0x4900,0x4904,0x4910});record(task,{0x5114,0x5138});
    if(final)record(task,{0x5900,0x5904,0x5908,0x5918,0x5928,0x592c});
    record(task,{0x5510,0x5514},attentionBar(3));
    record(task,{0x4d10,0x4d14},attentionBar(fused ? 8 : 7));
    record(task,{0x5108,0x510c},attentionBar(fused ? 9 : 8));
    record(task,{0x4d00},0x22000000);
    record(task,{0x5100},0x22000000);for(U32 lane=0;lane<16;++lane)
        record(task,{0x5520+4*lane},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|taskIndex;store(task.bytes,0,task.header[0]);return task;
}

Task fusedQKLoadTask(U32 tokens,U32 heads,U32 headDimension) {
    Task task=qkLoadTask(tokens,heads,headDimension);auto &r=task.registers;
    const U32 scorePlane=tokens*attentionAlign64(2*tokens);
    task.header[4]=heads==6?0x68:0x00fff868;r[0x4508]=0x3e000000;
    r[0x5108]=heads*scorePlane;
    task.bytes.clear();for(U32 v:task.header)word(task.bytes,v);
    record(task,{4,8,12});record(task,{0x14,0x18,0x1c,0x24,0x34,0x3c,0x44,0x4c});
    record(task,{0x4d08,0x4d1c,0x4d24});record(task,{0x4d68});
    record(task,{0x4104,0x4108});record(task,{0x4114,0x4118,0x411c,0x4120,0x4148,0x414c});
    record(task,{0x4508});record(task,{0x5114,0x511c,0x5138});
    record(task,{0x4d10,0x4d14},attentionBar(6));record(task,{0x5108,0x510c},attentionBar(3));
    record(task,{0x4d00},0x22000000);record(task,{0x5100},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16;store(task.bytes,0,task.header[0]);return task;
}

Task fusedQKComputeTask(U32 tokens,U32 heads,U32 headDimension,U32 stage) {
    Task task=qkComputeTask(tokens,heads,headDimension,stage,false);auto &r=task.registers;
    const U32 paddedRow=attentionAlign64(2*tokens),scorePlane=tokens*paddedRow;
    const U32 halfPaddedPlane=paddedRow*headDimension/2;
    task.header[4]=heads==6?0x28:0x00fff820;
    r[0x4148]=0x44a;r[0x5100]=0x010000f1|(stage<<16);
    r[0x5114]=4*paddedRow;r[0x5138]=0x04302031;
    r[0x5500]=((stage+1)<<16)|0x240;
    r[0x5510]=heads*(scorePlane+(stage==1?halfPaddedPlane:0));
    if(stage==2){r[0x4104]=0x142;r[0x4d10]=0;r[0x5108]=0;
        for(U32 lane=0;lane<16;++lane)r[0x5520+4*lane]=(stage<<16)|0x41;}
    packQKCompute(task,false,attentionBar(3));return task;
}

Task fusedSoftmaxTileTask(U32 tokens,U32 heads,U32 index) {
    Task task;task.header={index,0,0,0,heads==6?0x28u:0x00fff820u,0,0,7,1};
    auto &r=task.registers;const U32 channels=index==3?348:384;
    const U32 paddedRow=attentionAlign64(2*tokens),scorePlane=tokens*paddedRow;
    r[4]=tokens;r[8]=1;r[12]=channels;r[0x14]=tokens;r[0x18]=1;r[0x1c]=channels;
    r[0x24]=heads;r[0x34]=1;r[0x3c]=0x50;r[0x44]=9;r[0x4c]=0;
    r[0x4100]=4;r[0x4104]=index==3?0x400442:0x442;r[0x4110]=heads*tokens*16;
    r[0x4114]=1024;r[0x4118]=index==3?0x57000:0x60000;
    r[0x411c]=r[0x4118];r[0x4120]=r[0x4118];
    if(index==3)r[0x4148]=0x120;
    if(index<=5)r[0x414c]=index==3?0x4800:index==4?0x3000:0x1800;
    r[0x4150]=16;r[0x415c]=tokens*16;r[0x4500]=2;
    const U32 begins[]={0,0,0,0x100300c1,0x11040021,0x12050021,0x130600e1};
    const U32 offsets[]={0,0,0,0x34e000,0x234000,0x11a000,0};
    r[0x4d00]=begins[index];r[0x4d08]=index==6?0xee:0xce;
    r[0x4d10]=offsets[index];r[0x4d14]=0;r[0x4d1c]=4*paddedRow;
    r[0x4d24]=scorePlane;r[0x4d68]=0x4002031;
    task.bytes.clear();for(U32 v:task.header)word(task.bytes,v);record(task,{4,8,12});
    record(task,{0x14,0x18,0x1c,0x24,0x34,0x3c,0x44,0x4c});
    record(task,{0x4d08,0x4d1c,0x4d24});record(task,{0x4d68});
    record(task,{0x4100,0x4104,0x4110,0x4114,0x4118,0x411c,0x4120});
    std::vector<U32> l2;
    if(r.count(0x4148))l2.push_back(0x4148);if(r.count(0x414c))l2.push_back(0x414c);
    l2.push_back(0x4150);l2.push_back(0x415c);record(task,l2);record(task,{0x4500});
    record(task,{0x4d10,0x4d14},attentionBar(3));record(task,{0x4d00},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|index;store(task.bytes,0,task.header[0]);return task;
}

Task fusedSoftmaxMaxTask(U32 tokens,U32 heads) {
    Task task;task.header={7,0,0,0,heads==6?0x28u:0x00fff820u,0,0,0,0};auto &r=task.registers;
    const U32 paddedRow=attentionAlign64(2*tokens),scorePlane=tokens*paddedRow;
    r[4]=tokens;r[8]=heads;r[12]=tokens;r[0x14]=tokens;r[0x18]=heads;r[0x1c]=tokens;
    r[0x34]=heads;r[0x3c]=0x30;r[0x44]=0x34;r[0x48]=16;r[0x4c]=0;
    r[0x4104]=0x442;r[0x4108]=0x400100;r[0x4110]=heads*tokens*48;
    r[0x4114]=32;r[0x4118]=tokens*32;r[0x411c]=tokens*32;r[0x4120]=tokens*32;
    r[0x4128]=16;r[0x412c]=tokens*16;r[0x4148]=0x103;r[0x414c]=heads*tokens*16;
    r[0x4150]=32;r[0x4154]=tokens*32;r[0x4158]=tokens*32;r[0x415c]=tokens*32;
    r[0x4500]=0xc0000;r[0x4508]=0x3fb8a000;r[0x4d00]=0x14070041;
    r[0x4d08]=0x4e;r[0x4d10]=0;r[0x4d14]=0;r[0x4d18]=scorePlane;
    r[0x4d1c]=4*paddedRow;r[0x4d68]=0x4002031;
    task.bytes.clear();for(U32 v:task.header)word(task.bytes,v);record(task,{4,8,12});
    record(task,{0x14,0x18,0x1c,0x34,0x3c,0x44,0x48,0x4c});
    record(task,{0x4d08,0x4d18,0x4d1c});record(task,{0x4d68});
    record(task,{0x4104,0x4108,0x4110,0x4114,0x4118,0x411c,0x4120,0x4128,0x412c});
    record(task,{0x4148,0x414c,0x4150,0x4154,0x4158,0x415c});record(task,{0x4500,0x4508});
    record(task,{0x4d10,0x4d14},attentionBar(3));record(task,{0x4d00},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|7;store(task.bytes,0,task.header[0]);return task;
}

Task fusedSoftmaxExpTask(U32 tokens,U32 heads) {
    Task task;task.header={8,0,0,0,heads==6?0x28u:0x00fff820u,0,0,
        (1u<<(heads+1))-1,0x00050001};auto &r=task.registers;
    const U32 paddedRow=attentionAlign64(2*tokens),scorePlane=tokens*paddedRow;
    r[4]=tokens;r[8]=heads;r[12]=tokens;r[0x14]=tokens;r[0x18]=heads;r[0x1c]=tokens;
    r[0x30]=0x14000;r[0x34]=heads;r[0x3c]=0x233300;r[0x40]=0;r[0x4c]=0x100000;
    r[0x4104]=0x103;r[0x4110]=heads*tokens*16;r[0x4114]=32;
    r[0x4118]=tokens*32;r[0x411c]=tokens*32;r[0x4120]=tokens*32;
    r[0x4148]=0x142;r[0x414c]=heads*tokens*80;r[0x4150]=64;
    r[0x4154]=tokens*64+16;r[0x4158]=tokens*64;r[0x415c]=tokens*64;
    r[0x4900]=0x10082;r[0x4904]=0x123c0c;r[0x4910]=0x3c00;
    r[0x5100]=0x800f1;r[0x5104]=0x2c000033;r[0x5108]=heads*scorePlane;
    r[0x510c]=0;r[0x5110]=paddedRow;r[0x5114]=heads*paddedRow;r[0x5138]=0x1302031;
    r[0x5500]=0x90240;for(U32 lane=0;lane<16;++lane)r[0x5520+4*lane]=0x80020;
    r[0x5610]=0x80021;r[0x5614]=128;r[0x5618]=0;r[0x561c]=0;
    task.bytes.clear();for(U32 v:task.header)word(task.bytes,v);
    std::vector<U32> first{0x5500};for(U32 a=0x5520;a<=0x5540;a+=4)first.push_back(a);record(task,first);
    std::vector<U32> second;for(U32 a=0x5544;a<=0x555c;a+=4)second.push_back(a);record(task,second);
    record(task,{0x5610,0x5614});record(task,{4,8,12});
    record(task,{0x14,0x18,0x1c,0x30,0x34,0x3c,0x40,0x4c});
    record(task,{0x4104,0x4110,0x4114,0x4118,0x411c,0x4120});
    record(task,{0x4148,0x414c,0x4150,0x4154,0x4158,0x415c});
    record(task,{0x4900,0x4904,0x4910});record(task,{0x5104,0x5110,0x5114,0x5138});
    task.kernelRelocation=task.bytes.size()+4;record(task,{0x5618,0x561c},attentionBar(10));
    record(task,{0x5108,0x510c},attentionBar(3));record(task,{0x5100},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|8;store(task.bytes,0,task.header[0]);return task;
}

Task fusedSoftmaxSumTask(U32 tokens,U32 heads) {
    Task task;task.header={9,0,0,0,heads==6?0x28u:0x00fff820u,0,0,7,1};auto &r=task.registers;
    const U32 paddedRow=attentionAlign64(2*tokens),scorePlane=tokens*paddedRow;
    r[4]=tokens;r[8]=1;r[12]=heads;r[0x14]=tokens;r[0x18]=1;r[0x1c]=heads;
    r[0x24]=tokens;r[0x34]=1;r[0x3c]=0x50;r[0x44]=9;r[0x4c]=0;
    r[0x4104]=0x152;r[0x4110]=heads*tokens*16;r[0x4114]=1024;
    r[0x4118]=heads*1024;r[0x411c]=heads*1024;r[0x4120]=heads*1024;
    r[0x4150]=16;r[0x415c]=heads*16;r[0x4500]=0x1000;
    r[0x4d00]=0x10090021;r[0x4d08]=0xce;r[0x4d10]=heads*scorePlane;
    r[0x4d14]=0;r[0x4d1c]=paddedRow;r[0x4d24]=heads*paddedRow;r[0x4d68]=0x1002031;
    task.bytes.clear();for(U32 v:task.header)word(task.bytes,v);record(task,{4,8,12});
    record(task,{0x14,0x18,0x1c,0x24,0x34,0x3c,0x44,0x4c});
    record(task,{0x4d08,0x4d1c,0x4d24});record(task,{0x4d68});
    record(task,{0x4104,0x4110,0x4114,0x4118,0x411c,0x4120});
    record(task,{0x4150,0x415c});record(task,{0x4500});
    record(task,{0x4d10,0x4d14},attentionBar(3));record(task,{0x4d00},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|9;store(task.bytes,0,task.header[0]);return task;
}

Task fusedSoftmaxNormalizeTask(U32 tokens,U32 heads) {
    Task task;task.header={10,0,0,0,heads==6?0x28u:0x00fff820u,0,0,7,1};auto &r=task.registers;
    const U32 paddedRow=attentionAlign64(2*tokens),scorePlane=tokens*paddedRow;
    r[4]=tokens;r[8]=1;r[12]=tokens;r[0x14]=tokens;r[0x18]=1;r[0x1c]=tokens;
    r[0x24]=heads;r[0x34]=1;r[0x3c]=0x30;r[0x44]=9;r[0x48]=1;r[0x4c]=0;
    r[0x4104]=0x400100;r[0x4108]=0x162;r[0x4114]=heads*16;r[0x4120]=16;
    r[0x4124]=heads*tokens*16;r[0x4128]=1024;r[0x412c]=tokens*1024;
    r[0x4130]=tokens*1024;r[0x4134]=tokens*1024;r[0x4148]=0x84a;
    r[0x414c]=tokens*1024+heads*tokens*16;r[0x4500]=0x80004;
    r[0x4d04]=0x110a0041;r[0x4d0c]=0x4e;r[0x4d28]=heads*scorePlane;
    r[0x4d2c]=0;r[0x4d34]=heads*paddedRow;r[0x4d3c]=paddedRow;r[0x4d6c]=0x1002031;
    r[0x5100]=0x010a00f1;r[0x5104]=0x11000033;r[0x5108]=0;r[0x510c]=0;
    r[0x5114]=tokens*16;r[0x511c]=scorePlane;r[0x5138]=0x8302031;
    task.bytes.clear();for(U32 v:task.header)word(task.bytes,v);record(task,{4,8,12});
    record(task,{0x14,0x18,0x1c,0x24,0x34,0x3c,0x44,0x48,0x4c});
    record(task,{0x4d0c,0x4d34,0x4d3c});record(task,{0x4d6c});
    record(task,{0x4104,0x4108});
    record(task,{0x4114,0x4120,0x4124,0x4128,0x412c,0x4130,0x4134,0x4148,0x414c});
    record(task,{0x4500});record(task,{0x5104,0x5114,0x511c,0x5138});
    record(task,{0x4d28,0x4d2c},attentionBar(3));record(task,{0x5108,0x510c},attentionBar(3));
    record(task,{0x4d04},0x22000000);record(task,{0x5100},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|10;store(task.bytes,0,task.header[0]);return task;
}

Task softmaxSetupTask(U32 tokens,U32 heads) {
    // The five-task standalone program keeps the captured serialized schedule
    // for every head count.  The 0x68/0x28 form belongs to softmax inside the
    // much larger fused/full-model schedule; using it here wedges EXECUTE.
    Task task;task.header={0,0,0x68,0,0x00fff868,0,0,7,1};auto &r=task.registers;
    const U32 paddedRow=attentionAlign64(2*tokens),scorePlane=tokens*paddedRow;
    r[4]=tokens;r[8]=1;r[12]=heads;r[0x14]=tokens;r[0x18]=1;r[0x1c]=heads;
    r[0x24]=tokens;r[0x34]=1;r[0x3c]=0x50;r[0x44]=heads+1;r[0x4c]=0;
    r[0x4100]=4;r[0x4104]=0x500172;r[0x4108]=0x500130;
    r[0x4110]=heads*tokens*16;r[0x4114]=heads*128;
    r[0x4118]=heads*1024;r[0x411c]=heads*1024;r[0x4120]=heads*1024;
    r[0x4148]=0x500130;r[0x4150]=2*heads;r[0x415c]=16*heads;r[0x4500]=2;
    r[0x4d00]=0x21;r[0x4d08]=0xce;r[0x4d10]=0;r[0x4d14]=0;
    r[0x4d1c]=scorePlane;r[0x4d24]=paddedRow;r[0x4d68]=0x1002031;
    task.bytes.clear();for(U32 v:task.header)word(task.bytes,v);
    record(task,{4,8,12});record(task,{0x14,0x18,0x1c,0x24,0x34,0x3c,0x44,0x4c});
    record(task,{0x4d08,0x4d1c,0x4d24});record(task,{0x4d68});
    record(task,{0x4100,0x4104,0x4108,0x4110,0x4114,0x4118,0x411c,0x4120});
    record(task,{0x4148,0x4150,0x415c});record(task,{0x4500});
    record(task,{0x4d10,0x4d14},attentionBar(6));record(task,{0x4d00},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16;store(task.bytes,0,task.header[0]);return task;
}

Task softmaxMaxTask(U32 tokens,U32 heads) {
    Task task;task.header={1,0,0,0,0x00fff820,0,0,0,0};auto &r=task.registers;
    const U32 paddedRow=attentionAlign64(2*tokens),scorePlane=tokens*paddedRow;
    r[4]=tokens;r[8]=tokens;r[12]=heads;r[0x14]=tokens;r[0x18]=tokens;r[0x1c]=heads;
    r[0x34]=tokens;r[0x3c]=0x30;r[0x44]=0x44;r[0x48]=2*heads;r[0x4c]=0;
    r[0x4104]=0x142;r[0x4108]=0x400100;r[0x4110]=heads*tokens*50;
    r[0x4114]=8*heads;r[0x4118]=66*heads;r[0x411c]=64*heads;r[0x4120]=64*heads;
    r[0x4128]=2*heads;r[0x412c]=16*heads;r[0x4148]=0x103;
    r[0x414c]=heads*tokens*16;r[0x4150]=4*heads;r[0x4154]=34*heads;
    r[0x4158]=32*heads;r[0x415c]=32*heads;r[0x4500]=heads*0x18000;
    r[0x4508]=0x3fb8a000;r[0x4d00]=0x100c1;r[0x4d08]=0xce;
    r[0x4d10]=0;r[0x4d14]=0;r[0x4d18]=paddedRow;r[0x4d1c]=scorePlane;
    r[0x4d68]=0x1002031;
    task.bytes.clear();for(U32 v:task.header)word(task.bytes,v);
    record(task,{4,8,12});record(task,{0x14,0x18,0x1c,0x34,0x3c,0x44,0x48,0x4c});
    record(task,{0x4d08,0x4d18,0x4d1c});record(task,{0x4d68});
    record(task,{0x4104,0x4108,0x4110,0x4114,0x4118,0x411c,0x4120,0x4128,0x412c});
    record(task,{0x4148,0x414c,0x4150,0x4154,0x4158,0x415c});
    record(task,{0x4500,0x4508});record(task,{0x4d10,0x4d14},attentionBar(6));
    record(task,{0x4d00},0x22000000);task.header[0]=U32(task.bytes.size()/4)<<16|1;
    store(task.bytes,0,task.header[0]);return task;
}

Task softmaxExpTask(U32 tokens,U32 heads) {
    // The captured H=8 dependency mask is 0x3ff: two fixed predecessors plus
    // one completion per head.  Keeping that literal for H=6 waits for two
    // completions which can never be produced and wedges EXECUTE.
    const U32 dependencyMask=(U32(1)<<(heads+2))-1;
    Task task;task.header={2,0,0,0,0x00fff820,0,0,
        dependencyMask,0x00050001};auto &r=task.registers;
    r[4]=tokens;r[8]=tokens;r[12]=heads;r[0x14]=tokens;r[0x18]=tokens;r[0x1c]=heads;
    r[0x30]=0x14000;r[0x34]=tokens;r[0x3c]=0x1c4400;r[0x40]=heads;r[0x4c]=0x100000;
    r[0x4104]=0x103;r[0x4110]=heads*tokens*16;r[0x4114]=4*heads;
    r[0x4118]=34*heads;r[0x411c]=32*heads;r[0x4120]=32*heads;
    r[0x4148]=0x84a;r[0x414c]=heads*174000;
    r[0x4900]=0x10082;r[0x4904]=0x123c0c;r[0x4910]=0x3c00;
    r[0x5100]=0x010200f1;r[0x5108]=0;r[0x510c]=0;
    r[0x5110]=tokens*16;r[0x5138]=0x8302031;
    r[0x5500]=0x30240;for(U32 lane=0;lane<16;++lane)r[0x5520+4*lane]=0x20020;
    r[0x5610]=0x20021;r[0x5614]=128;r[0x5618]=0;r[0x561c]=0;
    task.bytes.clear();for(U32 v:task.header)word(task.bytes,v);
    std::vector<U32> first{0x5500};for(U32 a=0x5520;a<=0x5540;a+=4)first.push_back(a);record(task,first);
    std::vector<U32> second;for(U32 a=0x5544;a<=0x555c;a+=4)second.push_back(a);record(task,second);
    record(task,{0x5610,0x5614});record(task,{4,8,12});
    record(task,{0x14,0x18,0x1c,0x30,0x34,0x3c,0x40,0x4c});
    record(task,{0x4104,0x4110,0x4114,0x4118,0x411c,0x4120});record(task,{0x4148,0x414c});
    record(task,{0x4900,0x4904,0x4910});record(task,{0x5110,0x5138});
    task.kernelRelocation=task.bytes.size()+4;record(task,{0x5618,0x561c},attentionBar(8));
    record(task,{0x5108,0x510c},attentionBar(3));record(task,{0x5100},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|2;store(task.bytes,0,task.header[0]);return task;
}

Task softmaxSumTask(U32 tokens,U32 heads) {
    Task task;task.header={3,0,0,0,0x00fff820,0,0,7,1};auto &r=task.registers;
    r[4]=tokens;r[8]=1;r[12]=heads;r[0x14]=tokens;r[0x18]=1;r[0x1c]=heads;
    r[0x24]=tokens;r[0x34]=1;r[0x3c]=0x50;r[0x44]=heads+1;r[0x4c]=0;
    r[0x4104]=0x852;r[0x4110]=heads*tokens*16;r[0x4114]=heads*128;
    r[0x4118]=heads*1024;r[0x411c]=heads*1024;r[0x4120]=heads*1024;
    r[0x4150]=2*heads;r[0x415c]=16*heads;r[0x4500]=0x1000;
    r[0x4d00]=0x10030021;r[0x4d08]=0xce;r[0x4d10]=0;r[0x4d14]=0;
    r[0x4d24]=tokens*16;r[0x4d68]=0x8002031;
    task.bytes.clear();for(U32 v:task.header)word(task.bytes,v);
    record(task,{4,8,12});record(task,{0x14,0x18,0x1c,0x24,0x34,0x3c,0x44,0x4c});
    record(task,{0x4d08,0x4d24});record(task,{0x4d68});
    record(task,{0x4104,0x4110,0x4114,0x4118,0x411c,0x4120});
    record(task,{0x4150,0x415c});record(task,{0x4500});
    record(task,{0x4d10,0x4d14},attentionBar(3));record(task,{0x4d00},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|3;store(task.bytes,0,task.header[0]);return task;
}

Task softmaxNormalizeTask(U32 tokens,U32 heads) {
    Task task;task.header={4,0,0x04000000,0,0x00fff860,0,0,0,9};auto &r=task.registers;
    const U32 paddedRow=attentionAlign64(2*tokens),scorePlane=tokens*paddedRow;
    r[4]=tokens;r[8]=tokens;r[12]=heads;r[0x14]=tokens;r[0x18]=tokens;r[0x1c]=heads;
    r[0x34]=4*heads;r[0x3c]=0x30;r[0x44]=heads+1;r[0x48]=1;r[0x4c]=0;
    r[0x4104]=0x400100;r[0x4108]=0x842;r[0x4114]=2*heads;r[0x4118]=16*heads;
    r[0x4124]=heads*tokens*16;r[0x4128]=heads*128;r[0x412c]=heads*1024;
    r[0x4130]=heads*1024;r[0x4134]=heads*1024;r[0x4148]=0x14a;
    r[0x414c]=heads*56768;r[0x4500]=0x80004;
    r[0x4d04]=0x21040041;r[0x4d0c]=0x4e;r[0x4d28]=0;r[0x4d2c]=0;
    r[0x4d30]=tokens*16;r[0x4d6c]=0x8002031;
    r[0x5100]=0x01040031;r[0x5108]=0;r[0x510c]=0;r[0x5110]=paddedRow;
    r[0x5114]=scorePlane;r[0x5138]=0x1302031;r[0x5500]=128;
    r[0x5900]=0x12;r[0x5904]=0x00a000a0;r[0x5908]=0x400e0;
    r[0x5918]=0;r[0x5928]=0x007f0000;r[0x592c]=0x20010701;
    task.bytes.clear();for(U32 v:task.header)word(task.bytes,v);record(task,{0x5500});
    record(task,{4,8,12});record(task,{0x14,0x18,0x1c,0x34,0x3c,0x44,0x48,0x4c});
    record(task,{0x4d0c,0x4d30});record(task,{0x4d6c});record(task,{0x4104,0x4108});
    record(task,{0x4114,0x4118,0x4124,0x4128,0x412c,0x4130,0x4134,0x4148,0x414c});
    record(task,{0x4500});record(task,{0x5110,0x5114,0x5138});
    record(task,{0x5900,0x5904,0x5908,0x5918,0x5928,0x592c});
    record(task,{0x4d28,0x4d2c},attentionBar(3));record(task,{0x5108,0x510c},attentionBar(7));
    record(task,{0x4d04},0x22000000);record(task,{0x5100},0x22000000);
    task.header[0]=U32(task.bytes.size()/4)<<16|4;store(task.bytes,0,task.header[0]);return task;
}
}

std::string attentionMatmulRefusal(U32 tokens,U32 heads,U32 headDimension,
                                   AttentionMatmul operation) {
    (void)operation;
    if (auto why=attentionLayoutRefusal(tokens,heads,headDimension,
        AttentionLayout::Split);!why.empty()) return why;
    return {};
}

TransformerAttentionProgram encodeAttentionMatmul(
    U32 tokens,U32 heads,U32 headDimension,AttentionMatmul operation) {
    if (auto why=attentionMatmulRefusal(tokens,heads,headDimension,operation);
        !why.empty()) throw std::invalid_argument(why);
    TransformerAttentionProgram out;out.program.task=streamPrologue(Generation::H16G);
    Task load=operation==AttentionMatmul::PV ? pvLoadTask(tokens,heads)
                                             : qkLoadTask(tokens,heads,headDimension);
    appendAttentionTask(out.program.task,load);
    if(operation==AttentionMatmul::QK){
        Task compute=qkComputeTask(tokens,heads,headDimension,1,false);
        Task store=qkComputeTask(tokens,heads,headDimension,2,true);
        appendAttentionTask(out.program.task,compute);appendAttentionTask(out.program.task,store);
        out.program.taskCount=3;out.programRecordCount=7;
        out.programFormatCode=heads==6?209:heads==8?278:35*heads-1;
    }else if(operation==AttentionMatmul::QKTransposeY){
        if(heads==6){load.header[4]=0x68;store(load.bytes,16,0x68);}
        out.program.task=streamPrologue(Generation::H16G);appendAttentionTask(out.program.task,load);
        Task transpose=qkTransposeStageTask(tokens,heads,headDimension);
        Task compute=qkTransposeComputeTask(tokens,heads,headDimension,2,false);
        Task store=qkTransposeComputeTask(tokens,heads,headDimension,3,true);
        appendAttentionTask(out.program.task,transpose);appendAttentionTask(out.program.task,compute);
        appendAttentionTask(out.program.task,store);out.program.taskCount=4;
        out.programRecordCount=9;out.programFormatCode=heads==6?200:heads==8?266:33*heads+2;
    }else{
        for(U32 stage=1;stage<=heads;++stage){Task head=pvHeadTask(tokens,heads,headDimension,stage);
            appendAttentionTask(out.program.task,head);}
        out.program.taskCount=heads+1;out.programRecordCount=2*heads+3;
        out.programFormatCode=heads==6?791:heads==8?1054:132*heads-1;
    }
    out.program.firstTaskBytes=load.bytes.size();
    out.program.constantOffsetBytes=(out.program.task.size()+63)&~std::size_t(63);
    out.program.scratchAllocationBytes=operation==AttentionMatmul::PV
        ? heads*tokens*attentionAlign64(2*tokens)
        : heads*attentionAlign64(2*tokens)*headDimension;
    out.program.formulaBuilt=true;
    out.revealed=tokens==1500&&headDimension==64&&(heads==6||heads==8);
    return out;
}

std::string attentionSoftmaxRefusal(U32 tokens,U32 heads) {
    if(!heads)return "softmax head count must be nonzero";
    if(heads>29)return "softmax head count exceeds the dependency mask";
    if(tokens!=1500&&tokens!=128)return "softmax sequence has no measured or predicted law";
    return {};
}

TransformerAttentionProgram encodeAttentionSoftmax(U32 tokens,U32 heads) {
    if(auto why=attentionSoftmaxRefusal(tokens,heads);!why.empty())
        throw std::invalid_argument(why);
    TransformerAttentionProgram out;out.program.task=streamPrologue(Generation::H16G);
    Task tasks[]={softmaxSetupTask(tokens,heads),softmaxMaxTask(tokens,heads),
        softmaxExpTask(tokens,heads),softmaxSumTask(tokens,heads),
        softmaxNormalizeTask(tokens,heads)};
    for(Task &task:tasks){const std::size_t start=appendAttentionTask(out.program.task,task);
        if(task.kernelRelocation)out.program.kernelRelocations.push_back(start+task.kernelRelocation);}
    out.program.taskCount=5;out.program.firstTaskBytes=tasks[0].bytes.size();
    out.program.constantOffsetBytes=(out.program.task.size()+63)&~std::size_t(63);
    out.program.scratchAllocationBytes=std::uint64_t(heads)*tokens*tokens*2;
    out.program.formulaBuilt=true;out.programRecordCount=11;
    out.programFormatCode=heads==8?1054:132*heads-1;
    out.revealed=tokens==1500&&heads==8;return out;
}

TransformerAttentionProgram encodeFusedAttention(
    U32 tokens,U32 heads,U32 headDimension) {
    if(auto why=attentionMatmulRefusal(tokens,heads,headDimension,
        AttentionMatmul::PV);!why.empty())throw std::invalid_argument(why);
    if(tokens!=1500 || headDimension!=64 || (heads!=6 && heads!=8))
        throw std::invalid_argument("fused attention is limited to measured Whisper geometry");
    TransformerAttentionProgram out;out.program.task=streamPrologue(Generation::H16G);
    std::vector<Task> tasks;tasks.push_back(fusedQKLoadTask(tokens,heads,headDimension));
    tasks.push_back(fusedQKComputeTask(tokens,heads,headDimension,1));
    tasks.push_back(fusedQKComputeTask(tokens,heads,headDimension,2));
    for(U32 index=3;index<=6;++index)tasks.push_back(fusedSoftmaxTileTask(tokens,heads,index));
    tasks.push_back(fusedSoftmaxMaxTask(tokens,heads));
    tasks.push_back(fusedSoftmaxExpTask(tokens,heads));
    tasks.push_back(fusedSoftmaxSumTask(tokens,heads));
    tasks.push_back(fusedSoftmaxNormalizeTask(tokens,heads));
    for(U32 stage=1;stage<=heads;++stage)
        tasks.push_back(pvHeadTask(tokens,heads,headDimension,stage,true));
    for(std::size_t index=0;index<tasks.size();++index) {
        Task &task=tasks[index];const std::size_t start=appendAttentionTask(out.program.task,task);
        if(index==8)out.program.kernelRelocations.push_back(start+task.kernelRelocation);
    }
    out.program.taskCount=U32(tasks.size());out.program.firstTaskBytes=tasks[0].bytes.size();
    out.program.constantOffsetBytes=(out.program.task.size()+63)&~std::size_t(63);
    // The fused program keeps both the padded score surface and Q/K staging
    // planes in __DATA.  The old allocation stopped at the score surface even
    // though task 0 begins its staging result there, so the first load DMA and
    // later coefficient reads ran beyond the mapped scratch segment.
    const std::uint64_t paddedRow=attentionAlign64(2*tokens);
    out.program.scratchAllocationBytes=std::uint64_t(heads)*paddedRow*
        (tokens+headDimension);out.program.formulaBuilt=true;
    out.programRecordCount=2*heads+23;out.programFormatCode=heads==6?791:heads==8?1054:132*heads-1;
    out.revealed=headDimension==64&&(heads==6||heads==8);return out;
}
}
