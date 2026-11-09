#include "H14Program.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
using namespace ane::h14;

static std::uint32_t word(const std::vector<std::uint8_t> &v, std::size_t at) {
    assert(at + 4 <= v.size());
    return v[at] | std::uint32_t(v[at+1]) << 8 | std::uint32_t(v[at+2]) << 16 | std::uint32_t(v[at+3]) << 24;
}
static std::uint16_t half(const std::vector<std::uint8_t> &v, std::size_t at) {
    assert(at + 2 <= v.size());
    return v[at] | std::uint16_t(v[at+1]) << 8;
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
static bool hasReg(const Program &p, std::size_t off, std::uint32_t addr) {
    const auto end = off + ((word(p.taskStream, off) >> 16) & 0x7ff) * 4;
    for (auto at = off + 32; at < end;) {
        const auto h = word(p.taskStream, at), base = (h & 0x7fff) * 4;
        unsigned n = 1;
        if (h & 0x80000000u) {
            if (addr == base) return true;
            for (unsigned b = 0; b < 16; ++b) if ((h >> (15+b)) & 1) {
                if (addr == base+(b+1)*4) return true;
                ++n;
            }
        } else {
            n = ((h >> 15) & 63) + 1;
            if (addr >= base && addr < base+n*4) return true;
        }
        at += (n+1)*4;
    }
    return false;
}
static bool hasBar(const Program &p, std::size_t off, unsigned address) {
    try { (void)bar(p, off, address); return true; } catch (const std::logic_error &) { return false; }
}
static Program conv(unsigned ci, unsigned co, unsigned h, unsigned k=1,
                    unsigned stride=1, bool dw=false) {
    ConvShape s{k,k,stride,dw?ci:1,true,{ci,h,h},{co,(h+stride-1)/stride,(h+stride-1)/stride}};
    std::vector<std::uint8_t> w(co*(dw?1:ci)*k*k*2), b(co*2);
    for (std::size_t i=0;i<w.size();i+=2) { w[i]=i%251;w[i+1]=0x34; }
    return encodeConvGeneric(s,w.data(),w.size(),b.data(),b.size());
}
static Program ringConv(unsigned ci, unsigned co, unsigned h, unsigned k,
                        unsigned stride, bool dw) {
    ConvShape s{k,k,stride,dw?ci:1,true,{ci,h,h},
                {co,(h+stride-1)/stride,(h+stride-1)/stride}};
    std::vector<std::uint8_t> w(co*(dw?1:ci)*k*k*2), b(co*2);
    for (std::size_t i=0;i<w.size();i+=2) { w[i]=1+(i/2)%251;w[i+1]=0x34; }
    return encodeConvLinkedRing(s,w.data(),w.size(),b.data(),b.size());
}
static std::vector<Program> mobile(unsigned input) {
    static const unsigned co[] = {32,32,64,64,128,128,128,128,256,256,256,256,
        512,512,512,512,512,512,512,512,512,512,512,512,1024,1024,1024,1024};
    static const unsigned stride[] = {2,1,1,2,1,1,1,2,1,1,1,2,1,1,1,1,1,1,1,
        1,1,1,1,2,1,1,1,1};
    std::vector<Program> layers;
    unsigned ci=3,h=input;
    for(unsigned i=0;i<28;++i) {
        const bool dw=i&&i!=27&&(i&1);
        const auto k=(i==0||dw)?3u:1u;
        layers.push_back(ringConv(ci,co[i],h,k,stride[i],dw));
        if(i!=27) fuseConvRelu(layers.back());
        ci=co[i]; h=(h+stride[i]-1)/stride[i];
    }
    return layers;
}
int main() {
    std::vector<Program> layers{conv(64,128,16),conv(128,128,16,3,1,true),
                                conv(128,256,16,1,2),conv(256,512,8)};
    assert(layers[2].taskCount==2);
    // A standalone multi-task program opens on its first task (0x08) and
    // closes on its last (0x22), as Apple's decoded multi-task programs do;
    // one task carries both. 0x2a on every task wedged the M2 firmware.
    for (const auto &q : layers) {
        std::size_t at=16;
        for (unsigned i=0;i<q.taskCount;++i) {
            assert(word(q.taskStream,at+8)==((i==0?0x08u:0u)|(i+1==q.taskCount?0x22u:0u)));
            at=(at+((word(q.taskStream,at)>>16)&0x7ff)*4+15)&~std::size_t(15);
        }
    }
    assert(word(layers[0].taskStream,16+8)==0x2a);
    fuseConvRelu(layers[1]);
    assert(reg(layers[1],16,0xd04)&0x10000);
    for (const bool dw : {false, true}) {
        auto base = dw ? conv(32,32,16,3,1,true) : conv(64,64,16);
        const auto oldConstants = base.constants;
        const auto oldTaskBytes = ((word(base.taskStream,16)>>16)&0x7ff)*4;
        std::array<std::uint32_t,16> offsets{}, sizes{};
        for (unsigned lane=0;lane<16;++lane) {
            offsets[lane]=reg(base,16,0x1958+4*lane);
            sizes[lane]=reg(base,16,0x1998+4*lane);
        }
        auto relu=base;
        fuseConvRelu(relu);
        assert(reg(relu,16,0xd04)==0x00111c10);

        auto clip=base;
        fuseConvClip(clip,0x0000,0x4600);
        assert(reg(clip,16,0xd04)==0x00121c10);
        assert(clip.constants.size()==oldConstants.size()+16*128);
        const auto clipTaskBytes=((word(clip.taskStream,16)>>16)&0x7ff)*4;
        if (dw) {
            assert(clipTaskBytes==408);
            assert((word(clip.taskStream,20)&0xffff)==2);
            bool packed=false;
            for(std::size_t at=48;at<16+clipTaskBytes;) {
                const auto h=word(clip.taskStream,at);
                if(h==0x000f0657u) packed=true;
                unsigned n=1;
                if(h&0x80000000u) for(unsigned b=0;b<16;++b) n+=(h>>(15+b))&1;
                else n=((h>>15)&63)+1;
                at+=(n+1)*4;
            }
            assert(packed);
        } else {
            assert(clipTaskBytes==oldTaskBytes);
        }
        for(unsigned lane=0;lane<16;++lane) {
            const auto at=reg(clip,16,0x1908)+reg(clip,16,0x1958+4*lane);
            assert(reg(clip,16,0x1958+4*lane)==offsets[lane]+128*lane);
            assert(reg(clip,16,0x1998+4*lane)==sizes[lane]+128);
            assert(half(clip.constants,at)==0x0000);
            assert(half(clip.constants,at+2)==0x4600);
            assert(half(clip.constants,at+4)==0x0000);
            assert(half(clip.constants,at+6)==0x4600);
            assert(half(clip.constants,at+74)==0x3c00);
            assert(half(clip.constants,at+78)==0x3c00);
            assert(half(clip.constants,at+82)==0x0040);
            assert(half(clip.constants,at+84)==0x0001);
            assert(std::equal(oldConstants.begin()+offsets[lane],
                              oldConstants.begin()+offsets[lane]+sizes[lane],
                              clip.constants.begin()+at+128));
        }
    }
    {
        std::vector<Program> clipped{conv(64,64,16),conv(64,64,16)};
        for(auto &q:clipped) fuseConvClip(q,0x0000,0x4600);
        const auto linkedClip=linkConvChain(clipped);
        assert(linkedClip.taskCount==2);
        assert(linkedClip.constants.size()==
               clipped[0].constants.size()+clipped[1].constants.size());
        assert(reg(linkedClip,16,0xd04)==0x00121c10);
    }
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
            const bool multi=layers[l].taskCount>1, reuse=multi&&i>0;
            const bool last=l+1==layers.size()&&i+1==layers[l].taskCount;
            assert((word(p.taskStream,off)&0xffff)==global);
            // Multi-task layers keep L2 live (Apple's linked form).
            assert(word(p.taskStream,off+24)==(multi&&!last?0x1fffffffu:0u));
            assert(word(p.taskStream,off+28)==0x50001);
            assert(reg(p,off,0x1908)==kernel+reg(layers[l],own,0x1908));
            assert(bar(p,off,0x1908)==1);
            assert(bar(p,off,0x1508)==(l+1<layers.size()?3:5));
            if(reuse) {
                // Later slices reuse the first slice's L2 input.
                assert(!hasBar(p,off,0x1110));
                for (unsigned a=0x1100;a<0x1500;a+=4) assert(!hasReg(p,off,a));
                assert(!hasReg(p,off,0x504)&&!hasReg(p,off,0x514)&&!hasReg(p,off,0x518));
                const auto ne=reg(layers[l],own,0x38), cls=(ne>>16)&15;
                assert(reg(p,off,0x38)==((ne&~0xff00u)|(cls<<12)|(cls<<8)));
                assert(reg(p,off,0x530)==(reg(layers[l],own,0x530)&~0x500030u));
                assert(reg(p,off,0x534)==reg(layers[l],own,0x534));
                assert(reg(p,off,0x510)==reg(layers[l],own,0x510));
            } else {
                assert(bar(p,off,0x1110)==(l?3:4));
                assert(reg(p,off,0x504)==reg(layers[l],own,0x504));
            }
            if(l && !reuse) {
                assert(reg(p,off,0x1110)==layout[l].inputOffset);
                assert((reg(p,off,0x1100)>>28)==1);
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
    {
        const auto native=mobile(224);
        std::vector<LinkedLayer> ringLayout;
        const auto ring=linkMobileNetRing(native,&ringLayout);
        assert(ring.taskCount==28 && ringLayout.size()==28);
        assert(ring.scratchAllocationBytes==0x284000);
        assert(ringLayout[1].writesScratch && ringLayout[2].readsScratch);
        assert(ringLayout[2].l2Result==0);
        assert(ringLayout[2].l2ResultBytes==1609216);
        assert(ringLayout[2].l2ResultBytes+467712==2076928);
        assert(ring.constants.size()==8506368);
        const auto sizes=taskSizes(ring.taskStream);
        assert(*std::min_element(sizes.begin(),sizes.end())>=312);
        assert(*std::max_element(sizes.begin(),sizes.end())<=384);
        std::size_t at=16, constant=0;
        for(unsigned i=0;i<28;++i) {
            assert(reg(ring,at,0x2c)==native[i].mobileRingShape.output.height);
            assert(reg(ring,at,0x1908)==constant);
            assert(bar(ring,at,0x1908)==1);
            assert(((reg(ring,at,0x1900)>>16)&255)==i+1);
            assert(((reg(ring,at,0x1918)>>16)&255)==i);
            const auto expectedSource=i==0?0x500172u:i==2?0x842u:
                (i>=10||i==7)?0x104u:0x120u;
            if(reg(ring,at,0x504)!=expectedSource)
                std::cerr << "native ring source role mismatch at " << i
                          << ": 0x" << std::hex << reg(ring,at,0x504)
                          << " != 0x" << expectedSource << std::dec << "\n";
            assert(reg(ring,at,0x504)==expectedSource);
            if(i>2 && i<27) {
                for(unsigned a=0x1100;a<0x1500;a+=4) assert(!hasReg(ring,at,a));
                assert(!hasReg(ring,at,0x1500));
            }
            constant+=native[i].constants.size();
            at=(at+sizes[i]+15)&~std::size_t(15);
        }
        assert(constant==ring.constants.size());
        const auto bytes=encodeANEC(ring);
        assert(word(bytes,0x28+3*4)*tileBytes==ring.scratchAllocationBytes);
        assert(word(bytes,0x28+4*4)*tileBytes>=ring.output.allocationBytes);
        assert(word(bytes,0x28+5*4)*tileBytes>=ring.inputs[0].allocationBytes);

        // Same rules, allocator and record builder must cover another frame.
        const auto smallerLayers=mobile(192);
        const auto smaller=linkMobileNetRing(smallerLayers);
        assert(smaller.taskCount==28);
        assert(reg(smaller,16,0x2c)==96);
        std::size_t bridge=16;
        const auto smallerSizes=taskSizes(smaller.taskStream);
        bridge=(bridge+smallerSizes[0]+15)&~std::size_t(15);
        bridge=(bridge+smallerSizes[1]+15)&~std::size_t(15);
        assert(reg(smaller,bridge,0x50c)==192); // no 224-only 64px narrowing
    }
    std::cout << "H14 formula/linked encoding: PASS\n";
}
