#include "H16ConvProgram.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ane::h16 {
namespace {
using U32 = std::uint32_t;
struct Schedule {
    U32 cin, cout, width, kernel, stride, groups;
    U32 tileHeight, sourceSkew, common;
    bool direct;
};
// Scheduler outputs, not extrapolated shape formulas. See h16-h17-conv.md.
constexpr Schedule schedules[] = {
    {64,64,64,1,1,1,64,0,0x210000,true},
    {32,64,112,1,1,1,112,0x50,0x231100,false},
    {64,128,112,1,1,1,80,0x50,0x231100,false},
    {64,128,56,1,1,1,56,0x50,0x240000,false},
    {3,32,224,3,2,1,112,0,0x230000,false},
    {64,64,112,3,1,1,112,0,0x200000,true},
    {32,32,112,3,1,32,112,0x50,0x230000,false},
    {64,64,112,3,2,64,56,0,0x210000,true},
    {128,128,56,3,2,128,28,0x30,0x220000,true},
};
const Schedule *schedule(ConvShape s) {
    for (const auto &p : schedules)
        if (s.input.channels == p.cin && s.output.channels == p.cout &&
            s.input.width == p.width && s.kernel == p.kernel &&
            s.stride == p.stride && s.groups == p.groups) return &p;
    return nullptr;
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
    range(t,0x4d10,0x4d14,0x23000000);
    range(t,0x5108,0x510c,0x23800000);
    range(t,0x4d00,0x4d00,0x22000000);
    range(t,0x5100,0x5100,0x22000000);
    // DMA address records are relocation operands, emitted separately from
    // the ordinary register records. They must not also appear in 4d08 range.
    if (t.bytes.size()/4 > 0x7ff) throw std::invalid_argument("task exceeds TaskSize");
    t.header[0] = U32(t.bytes.size()/4)<<16 | (t.header[0]&0xffff);
    store(t.bytes,0,t.header[0]);
}
std::array<U32,16> chunks(ConvShape s) {
    std::array<U32,16> out{};
    const bool dw = s.groups == s.input.channels;
    if (s.stride == 1) {
        out.fill(align64(2*(dw ? 1 : s.input.channels)*s.kernel*s.kernel*
                        (s.output.channels/16)));
    } else if (dw && s.input.channels == 64) out.fill(128);
    else if (dw && s.input.channels == 128) {
        for (unsigned i = 0; i < 16; ++i) out[i] = i < 7 ? 192 : 256;
    } else {
        out.fill(128); out[0] = 64;
    }
    return out;
}
Task task(Generation gen, Layer layer, unsigned stage, unsigned count,
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
    if (linked && !first && !last) r[0x3c] |= 0x1100;
    if (h17 && s.kernel==3 && s.groups==1 && s.input.channels==64)
        r[0x3c] |= 1u<<27;
    U32 ocg=0;
    if (!dw) for (U32 n=s.output.channels/16; n>1; n>>=1) ++ocg;
    if (ocg != 3) r[0x40]=ocg;
    r[0x4c]=h17 ? 0x200000 : 0x100000;
    const U32 srcRow=first ? 2*(s.kernel==1 ? std::min(s.input.width,64u)
                                                        : s.input.width) : residentRow;
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
    if (h17 || s.stride==2) r[0x4900]=(h17 ? 0x10000082 : 0x82) | (s.stride==2 ? 0x100 : 0);
    r[0x4904]=0x103c00 | (layer.relu ? 0x10000 : 0);
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
    r[0x5500]=(stage+1)<<16 | (last ? 0x2c0 : 0x240) | (s.stride==2 ? 0x20 : 0);
    r[0x5510]=kernelBase; r[0x5514]=0;
    const auto sizes=chunks(s);
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
    if (s.bias) return "bias packing and epilogue are not measured for H16G/H17";
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
    if (s.stride!=1) return "H16G/H17 sparse kernel byte grammar is unknown; H13 sparse bytes cannot be reused";
    return {};
}
Description describe(Generation gen, const std::vector<Layer> &layers) {
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
        Task t=task(gen,layers[i],i,count,source,result,kernelBase,residentRow,residentStride);
        pack(t);
        // Capacity bound of the measured allocations, not a new scheduler.
        const auto &r=t.registers;
        if (r.count(0x4154) && std::uint64_t(result)+s.output.height*r.at(0x4154)>4*1024*1024)
            throw std::invalid_argument("measured L2 allocation exceeds 4 MiB envelope");
        out.stream.resize((out.stream.size()+15)&~std::size_t(15),0);
        out.kernelRelocations.push_back(out.stream.size()+t.kernelRelocation);
        out.stream.insert(out.stream.end(),t.bytes.begin(),t.bytes.end());
        U32 length=0; for (U32 n:chunks(s)) length+=n;
        kernelBase+=length;
        if (r.count(0x4150)) {residentRow=r.at(0x4150);residentStride=r.at(0x4154);}
        source=result;
        out.tasks.push_back(std::move(t));
    }
    out.kernelBytes=kernelBase;
    return out;
}
h13::Program encodeConv(Generation gen, ConvShape s, const std::uint8_t *weights, std::size_t size) {
    if (auto why=executableRefusal(s); !why.empty()) throw std::invalid_argument(why);
    auto d=describe(gen,{{s,false}});
    auto constants=h13::packConvWeights(s,weights,size);
    if (constants.size()!=d.kernelBytes) throw std::logic_error("stride-1 packer disagrees with H16 chunk formulas");
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
