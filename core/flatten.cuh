#pragma once

#include "layer.cuh"

#include <vector>

namespace nnv2 {

class Flatten : public Layer {
public:
    Flatten() : Layer() {}

    void forward() override;
    void backward() override;

    Array *get_output() { return prev->get_output(); }
    virtual const Array *get_output() const { return prev->get_output(); }

    Array *get_grad() { return next->get_grad(); }
    virtual const Array *get_output() const { return next->get_grad(); }

private:
    std::vector<int> in_shape;
};

} // namespace nnv2