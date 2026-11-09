#include "H14Program.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
using namespace ane::h14;
using namespace ane::h14::fields;

static std::uint32_t word(const std::vector<std::uint8_t> &v, std::size_t at) {
    assert(at + 4 <= v.size());
    return v[at] | std::uint32_t(v[at+1]) << 8 | std::uint32_t(v[at+2]) << 16 | std::uint32_t(v[at+3]) << 24;
}
static std::uint16_t half(const std::vector<std::uint8_t> &v, std::size_t at) {
    assert(at + 2 <= v.size());
    return v[at] | std::uint16_t(v[at+1]) << 8;
}
static std::uint32_t reg(const Program &p, std::size_t off, std::uint32_t addr) {
    const auto end = off + taskWordCount.get(word(p.taskStream, off)) * 4;
    std::uint32_t result = 0;
    for (auto at = off + 32; at < end;) {
        const auto h = word(p.taskStream, at), base = recordAddress.get(h) * 4;
        unsigned n = 0;
        if (h & scatterPresent.mask()) {
            if (addr == base) result = word(p.taskStream, at+4);
            n = 1;
            for (unsigned b = 0; b < 16; ++b) if ((h >> (15+b)) & 1) {
                if (addr == base+(b+1)*4) result = word(p.taskStream, at+4*(n+1));
                ++n;
            }
        } else {
            n = denseCountMinusOne.get(h) + 1;
            if (addr >= base && addr < base+n*4) result = word(p.taskStream, at+4+addr-base);
        }
        at += (n+1)*4;
    }
    return result;
}
static unsigned bar(const Program &p, std::size_t off, unsigned address) {
    auto end = off + taskWordCount.get(word(p.taskStream, off))*4;
    for (auto at=off+32; at<end;) {
        const auto h=word(p.taskStream,at);
        unsigned n=1;
        if (h & scatterPresent.mask()) for (unsigned b=0;b<16;++b) n+=(h>>(15+b))&1;
        else {
            n=denseCountMinusOne.get(h)+1;
            if ((h&barPresent.mask()) && recordAddress.get(h)*4==address)
                return barSlot.get(h);
        }
        at+=(n+1)*4;
    }
    throw std::logic_error("BAR absent");
}
static bool hasReg(const Program &p, std::size_t off, std::uint32_t addr) {
    const auto end = off + taskWordCount.get(word(p.taskStream, off)) * 4;
    for (auto at = off + 32; at < end;) {
        const auto h = word(p.taskStream, at), base = recordAddress.get(h) * 4;
        unsigned n = 1;
        if (h & scatterPresent.mask()) {
            if (addr == base) return true;
            for (unsigned b = 0; b < 16; ++b) if ((h >> (15+b)) & 1) {
                if (addr == base+(b+1)*4) return true;
                ++n;
            }
        } else {
            n = denseCountMinusOne.get(h) + 1;
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
    for (std::size_t i=0;i<w.size();i+=2) {
        w[i]=1+(i/2)%251; w[i+1]=52;
    }
    return encodeConvLinkedRing(s,w.data(),w.size(),b.data(),b.size());
}
static std::vector<Program> mobile(unsigned input, bool relu=true) {
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
        if(relu && i!=27) fuseConvRelu(layers.back());
        ci=co[i]; h=(h+stride[i]-1)/stride[i];
    }
    return layers;
}
static std::vector<Program> linkedMobileReference(bool clip) {
    // The accurate 45-task emitter consumes the mirrored 112 frame: k3
    // depthwise layers stay stride one and their following k1 projection
    // carries each downsample.  These are the same ten operation forms used
    // by the native ring diagnostic kit.
    static const unsigned co[] = {32,32,64,64,128,128,128,128,256,256,256,256,
        512,512,512,512,512,512,512,512,512,512,512,512,1024,1024,1024,1024};
    static const unsigned nativeStride[] = {2,1,1,2,1,1,1,2,1,1,1,2,1,1,1,1,
        1,1,1,1,1,1,1,2,1,1,1,1};
    std::vector<Program> layers;
    unsigned ci=16,h=112;
    for(unsigned i=0;i<28;++i) {
        const bool dw=i&&i!=27&&(i&1);
        const auto k=(i==0||dw)?3u:1u;
        const auto stride=i==0||dw?1u:
            (i&&!(i&1)?nativeStride[i-1]:1u);
        layers.push_back(conv(ci,co[i],h,k,stride,dw));
        if(clip&&i!=27) fuseConvClip(layers.back(),0,fp16Six);
        ci=co[i]; h=(h+stride-1)/stride;
    }
    return layers;
}
int main() {
    std::vector<Program> layers{conv(64,128,16),conv(128,128,16,3,1,true),
                                conv(128,256,16,1,2),conv(256,512,8)};
    assert(layers[2].taskCount==2);
    // A standalone multi-task program opens only its first task and closes
    // only its last; a single task carries both named field groups.
    for (const auto &q : layers) {
        std::size_t at=16;
        for (unsigned i=0;i<q.taskCount;++i) {
            assert(word(q.taskStream,at+8)==taskHeader2(i==0,i+1==q.taskCount));
            at=(at+taskWordCount.get(word(q.taskStream,at))*4+15)&~std::size_t(15);
        }
    }
    assert(word(layers[0].taskStream,16+8)==taskHeader2(true,true));
    fuseConvRelu(layers[1]);
    assert(reg(layers[1],16,registers::macConfig)&macRelu.mask());
    for (const bool dw : {false, true}) {
        auto base = dw ? conv(32,32,16,3,1,true) : conv(64,64,16);
        const auto oldConstants = base.constants;
        const auto oldTaskBytes = taskWordCount.get(word(base.taskStream,16))*4;
        std::array<std::uint32_t,16> offsets{}, sizes{};
        for (unsigned lane=0;lane<16;++lane) {
            offsets[lane]=reg(base,16,registers::kernelLaneOffset+4*lane);
            sizes[lane]=reg(base,16,registers::kernelLaneSize+4*lane);
        }
        auto relu=base;
        fuseConvRelu(relu);
        assert(reg(relu,16,registers::macConfig)==macConfig(Activation::Relu));

        auto clip=base;
        fuseConvClip(clip,0,fp16Six);
        assert(reg(clip,16,registers::macConfig)==macConfig(Activation::FiniteClip));
        assert(clip.constants.size()==oldConstants.size()+
               16*clipDescriptorBytes);
        const auto clipTaskBytes=taskWordCount.get(word(clip.taskStream,16))*4;
        if (dw) {
            assert(clipTaskBytes==408);
            assert((word(clip.taskStream,20)&taskMode.mask())==
                   taskModeKernelClipDescriptor.value(1));
            bool packed=false;
            for(std::size_t at=48;at<16+clipTaskBytes;) {
                const auto h=word(clip.taskStream,at);
                if(recordAddress.get(h)*4==registers::kernelLaneOffsetTail)
                    packed=true;
                unsigned n=1;
                if(h&scatterPresent.mask()) for(unsigned b=0;b<16;++b) n+=(h>>(15+b))&1;
                else n=denseCountMinusOne.get(h)+1;
                at+=(n+1)*4;
            }
            assert(packed);
        } else {
            assert(clipTaskBytes==oldTaskBytes);
        }
        for(unsigned lane=0;lane<16;++lane) {
            const auto at=reg(clip,16,registers::kernelDmaBase)+
                reg(clip,16,registers::kernelLaneOffset+4*lane);
            assert(reg(clip,16,registers::kernelLaneOffset+4*lane)==
                   offsets[lane]+clipDescriptorBytes*lane);
            assert(reg(clip,16,registers::kernelLaneSize+4*lane)==
                   sizes[lane]+clipDescriptorBytes);
            assert(half(clip.constants,at+clipLower0Offset)==0);
            assert(half(clip.constants,at+clipUpper0Offset)==fp16Six);
            assert(half(clip.constants,at+clipLower1Offset)==0);
            assert(half(clip.constants,at+clipUpper1Offset)==fp16Six);
            assert(half(clip.constants,at+clipUnity0Offset)==fp16One);
            assert(half(clip.constants,at+clipUnity1Offset)==fp16One);
            assert(half(clip.constants,at+clipScaleOffset)==clipDescriptorScale);
            assert(half(clip.constants,at+clipEnableOffset)==clipDescriptorEnable);
            assert(std::equal(oldConstants.begin()+offsets[lane],
                              oldConstants.begin()+offsets[lane]+sizes[lane],
                              clip.constants.begin()+at+clipDescriptorBytes));
        }
    }
    {
        std::vector<Program> clipped{conv(64,64,16),conv(64,64,16)};
        for(auto &q:clipped) fuseConvClip(q,0,fp16Six);
        const auto linkedClip=linkConvChain(clipped);
        assert(linkedClip.taskCount==2);
        assert(linkedClip.constants.size()==
               clipped[0].constants.size()+clipped[1].constants.size());
        assert(reg(linkedClip,16,registers::macConfig)==
               macConfig(Activation::FiniteClip));
    }
    std::vector<LinkedLayer> layout;
    const auto p=linkConvChain(layers,&layout);
    assert(p.taskCount==5 && layout.size()==4);
    assert(p.scratchAllocationBytes==3*128*16*64);
    const auto anec=encodeANEC(p);
    constexpr std::size_t anecTileTable = 40; // ANEC channel tile counts
    assert(word(anec,anecTileTable+3*4)*tileBytes==p.scratchAllocationBytes);
    std::size_t off=16, kernel=0;
    unsigned global=0;
    for (unsigned l=0;l<layers.size();++l) {
        std::size_t own=16;
        for (unsigned i=0;i<layers[l].taskCount;++i,++global) {
            const auto size=taskWordCount.get(word(p.taskStream,off))*4;
            const bool multi=layers[l].taskCount>1, reuse=multi&&i>0;
            const bool last=l+1==layers.size()&&i+1==layers[l].taskCount;
            assert(taskIdentifier.get(word(p.taskStream,off))==global);
            // Multi-task layers keep L2 live (Apple's linked form).
            assert(word(p.taskStream,off+24)==(multi&&!last?liveBankMask:0u));
            assert(word(p.taskStream,off+28)==taskHeader7());
            assert(reg(p,off,registers::kernelDmaBase)==
                   kernel+reg(layers[l],own,registers::kernelDmaBase));
            assert(bar(p,off,registers::kernelDmaBase)==1);
            assert(bar(p,off,registers::resultDmaBase)==
                   (l+1<layers.size()?3:5));
            if(reuse) {
                // Later slices reuse the first slice's L2 input.
                assert(!hasBar(p,off,registers::sourceDmaBase));
                for (unsigned a=registers::sourceDmaConfig;
                     a<registers::resultDmaConfig;a+=4) assert(!hasReg(p,off,a));
                assert(!hasReg(p,off,registers::sourceConfig)&&
                       !hasReg(p,off,registers::sourceDepthStride)&&
                       !hasReg(p,off,registers::sourcePlaneStride));
                const auto ne=reg(layers[l],own,registers::schedule);
                const auto cls=scheduleIssueClass.get(ne);
                assert(reg(p,off,registers::schedule)==
                       ((ne&~(scheduleInnerRepeat.mask()|
                              scheduleOuterRepeat.mask()))|
                        scheduleOuterRepeat.value(cls)|
                        scheduleInnerRepeat.value(cls)));
                assert(reg(p,off,registers::resultConfig)==
                       (reg(layers[l],own,registers::resultConfig)&
                        ~(l2ExternalBinding.mask()|l2ExternalDma.mask()|
                          l2DmaLoadedResult.mask())));
                assert(reg(p,off,registers::resultBase)==
                       reg(layers[l],own,registers::resultBase));
                assert(reg(p,off,registers::sourceRowStride)==
                       reg(layers[l],own,registers::sourceRowStride));
            } else {
                assert(bar(p,off,registers::sourceDmaBase)==(l?3:4));
                assert(reg(p,off,registers::sourceConfig)==
                       reg(layers[l],own,registers::sourceConfig));
            }
            if(l && !reuse) {
                assert(reg(p,off,registers::sourceDmaBase)==layout[l].inputOffset);
                assert(dmaScratchDependency.get(
                       reg(p,off,registers::sourceDmaConfig))==1);
                assert(reg(p,off,registers::sourceDmaExtension)==
                       sourceDmaExtension(false));
            }
            if(l+1<layers.size()) {
                const auto resultDma=reg(p,off,registers::resultDmaConfig);
                assert(dmaRoleResultWrite.get(resultDma));
                assert(dmaRoleLinearSurface.get(resultDma));
                assert(dmaRoleScratchSurface.get(resultDma));
                assert(dmaRoleInterleavedResult.get(resultDma));
                assert(reg(p,off,registers::resultDmaBase)>=layout[l].outputOffset);
            }
            assert((reg(p,off,registers::outputChannelGroup)<=3) || l!=2);
            off=(off+size+15)&~std::size_t(15);
            const auto ownsize=taskWordCount.get(word(layers[l].taskStream,own))*4;
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
        const auto spillRow = 16ull * native[1].mobileRingShape.output.width;
        const auto spillPlane = spillRow *
            native[1].mobileRingShape.output.height;
        const auto expectedScratch = 2 * native[1].output.allocationBytes +
            spillPlane * (native[1].mobileRingShape.output.channels / 8);
        assert(ring.scratchAllocationBytes==expectedScratch);
        assert(ringLayout[1].writesScratch && ringLayout[2].readsScratch);
        assert(ringLayout[2].l2Result==0);
        assert(ringLayout[2].l2ResultBytes>0 &&
               ringLayout[2].l2ResultBytes<mobileL2Bytes);
        const auto sizes=taskSizes(ring.taskStream);
        std::size_t at=16, constant=0;
        for(unsigned i=0;i<28;++i) {
            assert(sizes[i]>=compactTaskMinimumBytes);
            const auto width=native[i].mobileRingShape.output.width;
            unsigned ceilLog=0;
            for(auto value=width-1;value;value>>=1) ++ceilLog;
            const auto stage=ceilLog>=7 ? 0u : 7u-ceilLog;
            const auto issue=stage ? std::min(stage,4u) : 3u;
            const auto repeat=stage ? std::min(stage,4u) :
                (native[i].mobileRingShape.kernel==1 ? 1u : 0u);
            const auto schedule=reg(ring,at,registers::schedule);
            assert(scheduleIssueClass.get(schedule)==issue);
            assert(scheduleInnerRepeat.get(schedule)==repeat);
            assert(scheduleOuterRepeat.get(schedule)==repeat);
            assert(reg(ring,at,registers::tileHeight)==
                   native[i].mobileRingShape.output.height);
            assert(reg(ring,at,registers::kernelDmaBase)==constant);
            assert(bar(ring,at,registers::kernelDmaBase)==1);
            assert(kernelNextTaskTag.get(
                       reg(ring,at,registers::kernelDmaConfig))==i+1);
            assert(dmaTaskTag.get(
                       reg(ring,at,registers::kernelLaneConfig))==i);
            const auto expectedSource=i==0
                ? l2Config(L2Role::ProgramInput)
                : ringLayout[i].readsScratch
                    ? l2Config(L2Role::ScratchSource)
                : ringLayout[i].inputEdge==3
                    ? l2Config(L2Role::ResidentCompactSource)
                    : l2Config(L2Role::ResidentWideSource);
            if(reg(ring,at,registers::sourceConfig)!=expectedSource)
                std::cerr << "native ring source role mismatch at " << i
                          << ": " << reg(ring,at,registers::sourceConfig)
                          << " != " << expectedSource << "\n";
            assert(reg(ring,at,registers::sourceConfig)==expectedSource);
            if(i) {
                const auto channel =
                    reg(ring,at,registers::sourceChannelStride);
                const auto payload = static_cast<std::uint64_t>(
                    native[i].mobileRingShape.input.channels) * channel;
                const auto row = reg(ring,at,registers::sourceRowStride);
                std::uint32_t expectedSkew = mobileExecutionLaneBytes;
                while ((payload + expectedSkew) % mobileL2BankBytes == 0)
                    expectedSkew += mobileExecutionLaneBytes;
                assert(row == payload + expectedSkew);
            }
            if(i && !ringLayout[i].readsScratch) {
                for(unsigned a=registers::sourceDmaConfig;
                    a<registers::resultDmaConfig;a+=4) assert(!hasReg(ring,at,a));
            }
            if(!ringLayout[i].writesScratch && i+1<ring.taskCount) {
                assert(!hasReg(ring,at,registers::resultDmaConfig));
            }
            constant+=native[i].constants.size();
            at=(at+sizes[i]+15)&~std::size_t(15);
        }
        assert(constant==ring.constants.size());

        // The explicit research option varies only the padding lower bound.
        // With no padding, at least one compact resident task becomes shorter.
        LinkedRingResearchOptions unpaddedOptions;
        unpaddedOptions.compactTaskMinimumBytes=0;
        const auto unpadded=linkMobileNetRing(native,nullptr,unpaddedOptions);
        const auto unpaddedSizes=taskSizes(unpadded.taskStream);
        assert(unpaddedSizes.size()==sizes.size());
        assert(unpaddedSizes[1]<sizes[1]);
        bool invalidMinimumRefused=false;
        try {
            LinkedRingResearchOptions invalidOptions;
            invalidOptions.compactTaskMinimumBytes=64;
            (void)linkMobileNetRing(native,nullptr,invalidOptions);
        } catch(const std::invalid_argument &) {
            invalidMinimumRefused=true;
        }
        assert(invalidMinimumRefused);
        const auto bytes=encodeANEC(ring);
        constexpr std::size_t anecTileTable = 40; // ANEC channel tile counts
        assert(word(bytes,anecTileTable+3*4)*tileBytes==
               ring.scratchAllocationBytes);
        assert(word(bytes,anecTileTable+4*4)*tileBytes>=
               ring.output.allocationBytes);
        assert(word(bytes,anecTileTable+5*4)*tileBytes>=
               ring.inputs[0].allocationBytes);

        // Same rules, allocator and record builder must cover another frame.
        const auto smallerLayers=mobile(192);
        const auto smaller=linkMobileNetRing(smallerLayers);
        assert(smaller.taskCount==28);
        assert(reg(smaller,16,registers::tileHeight)==
               smallerLayers[0].mobileRingShape.output.height);
        std::size_t bridge=16;
        const auto smallerSizes=taskSizes(smaller.taskStream);
        bridge=(bridge+smallerSizes[0]+15)&~std::size_t(15);
        bridge=(bridge+smallerSizes[1]+15)&~std::size_t(15);
        assert(reg(smaller,bridge,registers::sourceChannelStride)==
               2*smallerLayers[2].mobileRingShape.input.width);

        // A terminal wide-dependent result crosses the compact direct-result
        // limit at 128x56x56.  The boundary is geometric: p04 and p08 keep
        // compact direct output, while all three operation kinds ending at
        // that surface use the named staged-direct form and external strides.
        for(unsigned prefix=4;prefix<=8;++prefix) {
            const std::vector<Program> p(native.begin(),native.begin()+prefix);
            const auto q=linkMobileNetRing(p);
            const auto qSizes=taskSizes(q.taskStream);
            std::size_t last=16;
            for(unsigned i=1;i<prefix;++i)
                last=(last+qSizes[i-1]+15)&~std::size_t(15);
            const bool wide=
                reg(q,last,registers::sourceConfig)==
                    l2Config(L2Role::ResidentWideSource) &&
                p.back().output.allocationBytes>mobileDirectResultMaxBytes;
            assert(reg(q,last,registers::resultConfig)==l2Config(wide
                ? L2Role::StagedDirectResult : L2Role::DirectResult));
            assert(hasReg(q,last,registers::resultChannelStride)==wide);
            assert(hasReg(q,last,registers::resultRowStride)==wide);
            assert(hasReg(q,last,registers::resultDepthStride)==wide);
            assert(hasReg(q,last,registers::resultDepthStride+4)==wide);
            if(wide) {
                const auto &shape=p.back().mobileRingShape.output;
                const auto channel=static_cast<std::uint32_t>(
                    (2ull*shape.width+15)&~15ull);
                const auto payload=static_cast<std::uint64_t>(
                    shape.channels)*channel;
                std::uint32_t skew=mobileExecutionLaneBytes;
                while((payload+skew)%mobileL2BankBytes==0)
                    skew+=mobileExecutionLaneBytes;
                const auto row=payload+skew;
                const auto depth=shape.channels*channel;
                assert(reg(q,last,registers::resultChannelStride)==channel);
                assert(reg(q,last,registers::resultRowStride)==row);
                assert(reg(q,last,registers::resultDepthStride)==depth);
                assert(reg(q,last,registers::resultDepthStride+4)==depth);
                assert(dmaUpperByte.get(
                    reg(q,last,registers::resultDmaConfig))==0);
            }
        }
        std::size_t fullLast=16;
        for(unsigned i=1;i<28;++i)
            fullLast=(fullLast+sizes[i-1]+15)&~std::size_t(15);
        assert(reg(ring,fullLast,registers::resultConfig)==
               l2Config(L2Role::DirectResult));
        assert(!hasReg(ring,fullLast,registers::resultChannelStride));
        assert(!hasReg(ring,fullLast,registers::resultRowStride));

        // Finite clips use the linked emitter's per-kernel task mode and keep
        // KernelCfg's LUT enable clear. ReLU retains its scheduler cost.
        auto clippedLayers=mobile(224,false);
        for(unsigned i=0;i+1<clippedLayers.size();++i)
            fuseConvClip(clippedLayers[i],0,fp16Six);
        const auto clipped=linkMobileNetRing(clippedLayers);
        const auto clippedSizes=taskSizes(clipped.taskStream);
        const auto clippedPrefix4=linkMobileNetRing({
            clippedLayers[0],clippedLayers[1],clippedLayers[2],
            clippedLayers[3]});
        const auto clippedPrefix4Sizes=taskSizes(clippedPrefix4.taskStream);
        const auto clippedPrefix4L01=
            (std::size_t(16)+clippedPrefix4Sizes[0]+15)&~std::size_t(15);
        // L01 is the scratch writer once p04 introduces the spill bridge.  A
        // finite clip must not inherit the stem's descriptor-extended sizes.
        for(unsigned lane=0;lane<kernelLaneCount;++lane) {
            const auto address=registers::kernelLaneSize+
                sizeof(std::uint32_t)*lane;
            assert(hasReg(clippedPrefix4,clippedPrefix4L01,address));
            assert(reg(clippedPrefix4,clippedPrefix4L01,address)==
                   clippedLayers[1].mobileRingKernelSizes[lane]);
        }
        const auto reluPrefix4=linkMobileNetRing({
            native[0],native[1],native[2],native[3]});
        const auto reluPrefix4Sizes=taskSizes(reluPrefix4.taskStream);
        const auto reluPrefix4L01=
            (std::size_t(16)+reluPrefix4Sizes[0]+15)&~std::size_t(15);
        for(unsigned lane=0;lane<kernelLaneCount;++lane)
            assert(!hasReg(reluPrefix4,reluPrefix4L01,
                registers::kernelLaneSize+sizeof(std::uint32_t)*lane));
        std::size_t clippedAt=16, reluAt=16;
        for(unsigned i=0;i<28;++i) {
            const auto clipCost=word(clipped.taskStream,clippedAt+4);
            const auto reluCost=word(ring.taskStream,reluAt+4);
            const auto &shape=clippedLayers[i].mobileRingShape;
            assert(clipCost==applyTaskActivation(
                reluCost,shape.kernelWidth==3,
                i+1<28?Activation::FiniteClip:Activation::None));
            if(i+1<28) {
                const bool sparse=shape.kernel==3&&shape.stride==2;
                const auto ne=reg(clipped,clippedAt,registers::kernelConfig);
                assert(!neLutEnable.get(ne));
                assert(ne==(sparse?neConfig(true,false):0u));
            }
            clippedAt=(clippedAt+clippedSizes[i]+15)&~std::size_t(15);
            reluAt=(reluAt+sizes[i]+15)&~std::size_t(15);
        }

        // Field-by-field activation parity with the accurate linked emitter
        // for every single-layer form in the device diagnostic kit.
        const auto linkedReference=linkedMobileReference(true);
        constexpr unsigned forms[] = {0,1,2,3,5,7,13,25,26,27};
        for(const auto index:forms) {
            const auto single=linkMobileNetRing({clippedLayers[index]});
            const auto &linked=linkedReference[index];
            const auto singleMac=reg(single,16,registers::macConfig);
            std::size_t linkedAt=16;
            for(unsigned task=0;task<linked.taskCount;++task) {
                const auto linkedMac=reg(linked,linkedAt,registers::macConfig);
                assert(macRelu.get(singleMac)==macRelu.get(linkedMac));
                assert(macFiniteClip.get(singleMac)==
                       macFiniteClip.get(linkedMac));
                assert(neLutEnable.get(
                           reg(single,16,registers::kernelConfig))==
                       neLutEnable.get(
                           reg(linked,linkedAt,registers::kernelConfig)));
                assert(kernelDescriptorMode.get(
                           reg(single,16,registers::kernelDmaConfig))==
                       kernelDescriptorMode.get(
                           reg(linked,linkedAt,registers::kernelDmaConfig)));
                if(index+1<28) {
                    assert(taskMode.get(word(single.taskStream,20))==
                           taskMode.get(word(linked.taskStream,linkedAt+4)));
                    for(unsigned lane=0;lane<kernelLaneCount;++lane) {
                        const auto singleOffset=
                            reg(single,16,registers::kernelDmaBase)+
                            reg(single,16,registers::kernelLaneOffset+
                                sizeof(std::uint32_t)*lane);
                        const auto linkedOffset=
                            reg(linked,linkedAt,registers::kernelDmaBase)+
                            reg(linked,linkedAt,registers::kernelLaneOffset+
                                sizeof(std::uint32_t)*lane);
                        assert(reg(single,16,registers::kernelLaneSize+
                                   sizeof(std::uint32_t)*lane)>=
                               clipDescriptorBytes);
                        assert(reg(linked,linkedAt,registers::kernelLaneSize+
                                   sizeof(std::uint32_t)*lane)>=
                               clipDescriptorBytes);
                        assert(std::equal(
                            single.constants.begin()+singleOffset,
                            single.constants.begin()+singleOffset+
                                clipDescriptorBytes,
                            linked.constants.begin()+linkedOffset));
                    }
                }
                // Where the linked reference uses the identical one-task
                // geometry, pin the whole descriptor+coefficient lane rather
                // than only its descriptor prefix.
                // L02 and L05 are the two failing standalone forms.  Their
                // native and mirrored linked geometries are identical.
                const bool exactGeometry=index==2 || index==5;
                if(linked.taskCount==1 && exactGeometry) {
                    for(unsigned lane=0;lane<kernelLaneCount;++lane) {
                        const auto address=sizeof(std::uint32_t)*lane;
                        assert(reg(single,16,registers::kernelLaneOffset+
                                   address)==
                               reg(linked,linkedAt,
                                   registers::kernelLaneOffset+address));
                        assert(reg(single,16,registers::kernelLaneSize+
                                   address)==
                               reg(linked,linkedAt,
                                   registers::kernelLaneSize+address));
                    }
                }
                const auto linkedBytes=taskWordCount.get(
                    word(linked.taskStream,linkedAt))*sizeof(std::uint32_t);
                linkedAt=(linkedAt+linkedBytes+15)&~std::size_t(15);
            }
        }

    }
    std::cout << "H14 formula/linked encoding: PASS\n";
}
