#pragma once

#include "layer.cuh"

#include <vector>

namespace nnv2 {

class Flatten : public Layer {
public:
    Flatten() : Layer() {}

    void forward() override;
    void backward() override;

    Array *get_grad() { return next->get_grad(); }
    Array *get_output() { return prev->get_output(); }

private:
    std::vector<int> in_shape;
};

} // namespace nnv2