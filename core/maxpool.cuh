#pragma once

#include "common.cuh"
#include "layer.cuh"

#include <vector>

namespace nnv2 {

class MaxPool2D : public Layer {
public:
    MaxPool2D(int pad_h, int pad_w, int kernel_h, int kernel_w, int stride_h,
              int stride_w);

    void forward() override;
    void backward() override;

private:
    int pad_h;
    int pad_w;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;

    std::unique_ptr<Array> indices;
};

void maxpool_forward(Array *output, const Array *input, Array *indices,
                     int pad_h, int pad_w, int kernel_h, int kernel_w,
                     int stride_h, int stride_w);

void maxpool_backward(Array *input_grad, const Array *output_grad,
                      const Array *indices, int pad_h, int pad_w, int filter_h,
                      int filter_w, int stride_h, int stride_w);

} // namespace nnv2