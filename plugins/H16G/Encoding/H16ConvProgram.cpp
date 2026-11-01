#include "H16ConvProgram.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ane::h16 {
namespace {
using U32 = std::uint32_t;
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
    std::vector<U32> coefficients{0x5500};
    for (U32 a=0x5520;a<=0x5540;a+=4) coefficients.push_back(a);
    record(t,coefficients); range(t,0x5544,0x555c);
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
    if (t.registers.count(0x4d08)) record(t,{0x4d08,0x4d18,0x4d1c});
    range(t,0x4d68,0x4d68);
    if (t.registers.at(0x4148) == 0x50017a) {
        range(t,0x4104,0x4108); range(t,0x4114,0x414c);
    } else {
        range(t,0x4104,0x4120); range(t,0x4148,0x415c);
    }
    range(t,0x4900,0x4910); range(t,0x5110,0x5138);
    range(t,0x5900,0x592c);
    range(t,0x5510,0x5514,0x24000000);
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
            throw std::invalid_argument("linked schedule is outside the two measured resident chains; DRAM dependency protocol is unproven on H16G/H17");
    }
    Description out;
    out.stream.resize(16,0);
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
    out.stream.resize(16,0);
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
h13::Program encodeConv(Generation gen, ConvShape s, const std::uint8_t *weights, std::size_t size,
                        const std::uint8_t *bias, std::size_t biasBytes) {
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
}
