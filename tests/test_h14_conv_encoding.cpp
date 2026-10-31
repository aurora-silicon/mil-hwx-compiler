#include "H14Program.h"
#include <cassert>
#include <iostream>
#include <stdexcept>
using namespace ane::h14;

static std::uint32_t word(const std::vector<std::uint8_t> &v, std::size_t at) {
    assert(at + 4 <= v.size());
    return v[at] | std::uint32_t(v[at+1]) << 8 | std::uint32_t(v[at+2]) << 16 | std::uint32_t(v[at+3]) << 24;
}
static std::uint32_t reg(const Program &p, std::size_t off, std::uint32_t addr) {
    const auto end = off + ((word(p.taskStream, off) >> 16) & 0x7ff) * 4;
    std::uint32_t result = 0;
    for (auto at = off + 32; at < end;) {
        const auto h = word(p.taskStream, at), base = (h & 0x7fff) * 4;
        unsigned n = 0;
        if (h & 0x80000000u) {
            if (addr == base) result = word(p.taskStream, at+4);
            n = 1;
            for (unsigned b = 0; b < 16; ++b) if ((h >> (15+b)) & 1) {
                if (addr == base+(b+1)*4) result = word(p.taskStream, at+4*(n+1));
                ++n;
            }
        } else {
            n = ((h >> 15) & 63) + 1;
            if (addr >= base && addr < base+n*4) result = word(p.taskStream, at+4+addr-base);
        }
        at += (n+1)*4;
    }
    return result;
}
static unsigned bar(const Program &p, std::size_t off, unsigned address) {
    auto end = off + ((word(p.taskStream, off) >> 16) & 0x7ff)*4;
    for (auto at=off+32; at<end;) {
        const auto h=word(p.taskStream,at);
        unsigned n=1;
        if (h & 0x80000000u) for (unsigned b=0;b<16;++b) n+=(h>>(15+b))&1;
        else {
            n=((h>>15)&63)+1;
            if ((h&0x20000000u) && (h&0x7fff)*4==address) return (h>>23)&63;
        }
        at+=(n+1)*4;
    }
    throw std::logic_error("BAR absent");
}
static Program conv(unsigned ci, unsigned co, unsigned h, unsigned k=1,
                    unsigned stride=1, bool dw=false) {
    ConvShape s{k,k,stride,dw?ci:1,true,{ci,h,h},{co,(h+stride-1)/stride,(h+stride-1)/stride}};
    std::vector<std::uint8_t> w(co*(dw?1:ci)*k*k*2), b(co*2);
    for (std::size_t i=0;i<w.size();i+=2) { w[i]=i%251;w[i+1]=0x34; }
    return encodeConvGeneric(s,w.data(),w.size(),b.data(),b.size());
}
int main() {
    std::vector<Program> layers{conv(64,128,16),conv(128,128,16,3,1,true),
                                conv(128,256,16,1,2),conv(256,512,8)};
    assert(layers[2].taskCount==2);
    fuseConvRelu(layers[1]);
    assert(reg(layers[1],16,0xd04)&0x10000);
    std::vector<LinkedLayer> layout;
    const auto p=linkConvChain(layers,&layout);
    assert(p.taskCount==5 && layout.size()==4);
    assert(p.scratchAllocationBytes==3*128*16*64);
    const auto anec=encodeANEC(p);
    assert(word(anec,0x28+3*4)*tileBytes==p.scratchAllocationBytes);
    std::size_t off=16, kernel=0;
    unsigned global=0;
    for (unsigned l=0;l<layers.size();++l) {
        std::size_t own=16;
        for (unsigned i=0;i<layers[l].taskCount;++i,++global) {
            const auto size=((word(p.taskStream,off)>>16)&0x7ff)*4;
            assert((word(p.taskStream,off)&0xffff)==global);
            assert(word(p.taskStream,off+24)==0);
            assert(word(p.taskStream,off+28)==0x50001);
            assert(reg(p,off,0x1908)==kernel+reg(layers[l],own,0x1908));
            assert(bar(p,off,0x1908)==1);
            assert(bar(p,off,0x1110)==(l?3:4));
            assert(bar(p,off,0x1508)==(l+1<layers.size()?3:5));
            if(l) {
                assert(reg(p,off,0x1110)==layout[l].inputOffset);
                assert((reg(p,off,0x1100)>>28)==(i==0?1:0));
                assert(reg(p,off,0x1108)==0x4e);
            }
            if(l+1<layers.size()) {
                assert((reg(p,off,0x1500)&0xf0)==0xf0);
                assert(reg(p,off,0x1508)>=layout[l].outputOffset);
            }
            assert((reg(p,off,0x1c)<=3) || l!=2);
            off=(off+size+15)&~std::size_t(15);
            const auto ownsize=((word(layers[l].taskStream,own)>>16)&0x7ff)*4;
            own=(own+ownsize+15)&~std::size_t(15);
        }
        kernel+=layers[l].constants.size();
    }
    assert(word(p.taskStream,16+8)==8);
    assert(kernel==p.constants.size());
    auto bad=layers; bad[1].inputs[0].nchw[1]++;
    bool refused=false;
    try { (void)linkConvChain(bad); } catch(const std::invalid_argument &) { refused=true; }
    assert(refused);
    ConvShape invalid{3,3,2,1,false,{64,16,16},{64,8,8}};
    assert(!convGenericRefusal(invalid).empty());
    invalid={1,1,2,1,false,{64,16,16},{144,8,8}};
    assert(!convGenericRefusal(invalid).empty());
    std::cout << "H14 formula/linked encoding: PASS\n";
}
