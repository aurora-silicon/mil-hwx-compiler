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
}
