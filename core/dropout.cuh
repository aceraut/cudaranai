#pragma once

#include "common.cuh"
#include "layer.cuh"

#include <vector>

namespace nnv2 {

class Dropout : public Layer {
public:
    Dropout(float p) : Layer(), drop_rate(p) {}

    void forward() override;
    void backward() override;
    void forward_test() override;

    Array *get_output() { return prev->get_output(); }
    const Array *get_output() const { return prev->get_output(); }

    Array *get_grad() { return next->get_grad(); }
    const Array *get_grad() const { return next->get_grad(); }

private:
    float drop_rate;
    thrust::device_vector<char> mask;
};

void dropout_forward(Array *output, const Array *input, float drop_rate,
                     thrust::device_vector<char> mask);
void dropout_backward(Array *input_grad, const Array *output_grad,
                      float drop_rate, const thrust::device_vector<char> mask);

} // namespace nnv2