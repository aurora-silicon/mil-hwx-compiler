#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ane::layers {
struct Connection {
    std::vector<std::string> inputs;
    std::string output;
    bool concat = false;
};
struct Reference { bool boundary; std::size_t index; };
struct Placement {
    std::size_t owner;
    std::uint64_t offset;
    bool output;
};
struct GraphPlan {
    std::vector<std::vector<Reference>> inputs;
    std::vector<Placement> outputs;
    std::vector<bool> zeroCopy;
    std::uint64_t scratchBytes = 0;
    std::size_t result = 0;
};

template<class Layout>
inline bool sameSurface(const Layout &a,const Layout &b) {
    return a.nchw==b.nchw && a.allocationBytes==b.allocationBytes;
}

/// Allocate whole surfaces without reusing live storage. An internal concat
/// may alias two single-task producers when each producer has exactly one
/// consumer, and its entire allocation is its written channel span. Their
/// writes then occupy disjoint adjacent slices of the concat surface.
template<class Program>
GraphPlan planGraph(const std::vector<Program> &programs,
    const std::vector<Connection> &connections,const std::vector<std::string> &boundaries,
    const std::string &result,const char *prefix) {
    auto fail=[&](const char *why)->void { throw std::invalid_argument(std::string(prefix)+": "+why); };
    if (programs.size()!=connections.size() || programs.size()<2 || boundaries.empty() || boundaries.size()>2)
        fail("invalid native graph boundaries or layer count");
    GraphPlan plan;
    const auto n=programs.size();
    plan.inputs.resize(n);plan.outputs.resize(n);plan.zeroCopy.assign(n,false);
    std::map<std::string,Reference> values;
    for (std::size_t i=0;i<boundaries.size();++i)
        if (!values.emplace(boundaries[i],Reference{true,i}).second) fail("duplicate boundary name");
    std::vector<std::size_t> uses(n,0),parent(n);
    std::vector<std::uint64_t> delta(n,0),base(n,0);
    std::vector<const typename std::decay_t<decltype(programs[0].output)> *> boundaryLayouts(boundaries.size(),nullptr);
    for (std::size_t i=0;i<n;++i) {
        parent[i]=i;
        if (connections[i].inputs.size()!=programs[i].inputs.size() || programs[i].inputs.size()>2)
            fail("layer runtime inputs do not match graph edges");
        for (std::size_t j=0;j<connections[i].inputs.size();++j) {
            const auto source=values.find(connections[i].inputs[j]);
            if (source==values.end()) fail("forward or missing graph input");
            const auto ref=source->second;
            plan.inputs[i].push_back(ref);
            if (ref.boundary) {
                auto &layout=boundaryLayouts[ref.index];
                if (layout && !sameSurface(*layout,programs[i].inputs[j])) fail("inconsistent boundary layout");
                layout=&programs[i].inputs[j];
            } else {
                if (!sameSurface(programs[ref.index].output,programs[i].inputs[j])) fail("incompatible native graph edge");
                ++uses[ref.index];
            }
        }
        if (!values.emplace(connections[i].output,Reference{false,i}).second) fail("duplicate graph result");
    }
    const auto final=values.find(result);
    if (final==values.end() || final->second.boundary) fail("graph result must be produced by a layer");
    plan.result=final->second.index;
    ++uses[plan.result];
    for (const auto *layout:boundaryLayouts) if (!layout) fail("unused graph boundary");
    for (std::size_t i=0;i<n;++i) {
        if (!uses[i]) fail("unused native graph layer");
        if (!connections[i].concat || plan.inputs[i].size()!=2) continue;
        const auto a=plan.inputs[i][0],b=plan.inputs[i][1];
        if (a.boundary || b.boundary || a.index==b.index || uses[a.index]!=1 || uses[b.index]!=1 ||
            programs[a.index].taskCount!=1 || programs[b.index].taskCount!=1) continue;
        const auto &x=programs[a.index].output;
        const auto &z=programs[b.index].output;
        const auto &y=programs[i].output;
        const auto span=x.nchw[1]*x.nchw[4];
        if (x.nchw[0]!=1 || z.nchw[0]!=1 || y.nchw[0]!=1 ||
            x.nchw[2]!=z.nchw[2] || x.nchw[3]!=z.nchw[3] || x.nchw[4]!=z.nchw[4] || x.nchw[5]!=z.nchw[5] ||
            y.nchw[4]!=x.nchw[4] || y.nchw[5]!=x.nchw[5] || y.nchw[1]!=x.nchw[1]+z.nchw[1] ||
            x.allocationBytes!=span || z.allocationBytes!=z.nchw[1]*z.nchw[4] ||
            span+z.allocationBytes>y.allocationBytes) continue;
        parent[a.index]=i;parent[b.index]=i;delta[b.index]=span;plan.zeroCopy[i]=true;
    }
    for (std::size_t i=0;i<n;++i)
        if (parent[i]==i && i!=plan.result) {
            base[i]=plan.scratchBytes;
            if (programs[i].output.allocationBytes>0xffffffffull-plan.scratchBytes) fail("scratch arena exceeds 32 bits");
            plan.scratchBytes+=programs[i].output.allocationBytes;
        }
    for (std::size_t i=0;i<n;++i) {
        auto owner=i;
        std::uint64_t offset=0;
        for (std::size_t hops=0;parent[owner]!=owner;++hops) {
            if (hops>=n) fail("cyclic native graph placement");
            offset+=delta[owner];owner=parent[owner];
        }
        if (offset+programs[i].output.allocationBytes>programs[owner].output.allocationBytes)
            fail("concat view exceeds its owning surface");
        plan.outputs[i]={owner,offset+(owner==plan.result?0:base[owner]),owner==plan.result};
    }
    return plan;
}
} // namespace ane::layers
