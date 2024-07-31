#pragma once

#include "common.cuh"
#include "initializer.cuh"
#include "layer.cuh"

#include <memory>
#include <vector>

namespace nnv2 {

class Linear : public Layer {
public:
    Linear(int in_feats, int out_feats, const Initializer *init);

    std::vector<Param> get_parameters() override;

    void forward() override;
    void backward() override;

private:
    int in_feats;
    int out_feats;

    std::unique_ptr<Array> filter;
    std::unique_ptr<Array> filter_grad;
    std::unique_ptr<Array> bias;
    std::unique_ptr<Array> bias_grad;

    ArrayMap cache;
};

// for testing purpose, local functions are defined here
void linear_forward(Array *output, const Array *input, const Array *filter);
void linear_forward_bias(Array *output, const Array *bias);

void linear_backward(Array *input_grad, Array *filter_grad, const Array *input,
                     const Array *filter, const Array *output_grad,
                     ArrayMap &cache);
void linear_backward_bias(Array *bias_grad, const Array *output_grad);

} // namespace nnv2