#pragma once

#include "H13/H13Program.h"
#include "H14/H14Program.h"
#include "LayerGraph.h"

namespace ane::layers {

enum class Operation { Nearest2x, BilinearHalf2x, Sigmoid, HardSwish, Concat };
struct Shape {
    std::uint32_t channels, height, width;
    std::uint32_t otherChannels = 0;
};

/// Exact decoded points only. Resize doubles both spatial axes; concat is
/// channel-axis, two operands with equal N/H/W and no interleaving.
bool supports(bool h14, Operation operation, Shape shape);
h13::Program encodeH13(Operation operation, Shape shape);
/// No task is emitted. Linking must prove a zero-copy producer placement.
h13::Program placementConcatH13(Shape shape);
h14::Program encodeH14(Operation operation, Shape shape);
/// DRAM-linked, one-input native layers. Internal L2 tasks remain inside
/// their layer. The BAR scratch slot follows the decoded H14 resource form.
h14::Program linkH14(const std::vector<h14::Program> &layers,
                     std::vector<h13::LinkedLayer> *layout = nullptr);
/// Native ANEC BAR refs use role slots 5/6 for inputs and 4 for output.
/// HWX retains the decoded Apple slots; old H14 encoders stay unchanged.
std::vector<std::uint8_t> h14HWXStream(const h14::Program &program);
h14::Program linkH14Graph(const std::vector<h14::Program> &programs,
    const std::vector<Connection> &connections,const std::vector<std::string> &boundaries,
    const std::string &result,GraphPlan *placement=nullptr,
    std::vector<h13::LinkedLayer> *layout=nullptr);

} // namespace ane::layers
