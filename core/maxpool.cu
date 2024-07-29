// This implements the MaxPool2D layer, a variant of pooling layer.
//
// This layer is typically used after the convolution layer and is used to
// reduce the spatial dimensions of the input data.
// Similar to the Conv2D layer, it uses kernels to perform pooling operation
// on section that intersects with the input data, take the max value in each
// section and generate a reduced data map for the rest of the network.

#include "maxpool.h"
#include "common.h"

#include <cfloat>
#include <vector>

namespace nnv2 {

static void maxpool_im_chmax(const Array *im, int feat_idx, int r, int c,
                             int pad_h, int pad_w, float &max_val,
                             int &max_idx) {
    int h = im->get_shape()[2];
    int w = im->get_shape()[3];

    r -= pad_h;
    c -= pad_w;

    if (r < 0 || r >= h || c < 0 || c >= w) {
        return;
    }

    int im_idx = (feat_idx * h + r) * w + c;
    int value = im->get_vec()[im_idx];

    if (value > max_val) {
        max_val = value;
        max_idx = im_idx;
    }
}

void maxpool_forward(Array *output, const Array *input,
                     std::vector<int> &indices, int pad_h, int pad_w,
                     int kernel_h, int kernel_w, int stride_h, int stride_w) {
    CHECK_EQ(output->get_shape().size(), 4,
             "maxpool_forward: output shape error");
    CHECK_EQ(input->get_shape().size(), 4,
             "maxpool_forward: input shape error");

    int batch_size = input->get_shape()[0];
    int in_feats = input->get_shape()[1];

    CHECK_EQ(output->get_shape()[0], batch_size,
             "maxpool_forward: batch size error");
    CHECK_EQ(output->get_shape()[1], in_feats,
             "maxpool_forward: batch size error");

    int out_h = output->get_shape()[2];
    int out_w = output->get_shape()[3];

    CHECK_EQ(indices.size(), output->get_vec().size(),
             "maxpool_forward: indices size not equal to output size");

    int images_size = batch_size * in_feats;

    for (int i = 0; i < images_size; i++) {
        for (int r = 0; r < out_h; r++) {
            for (int c = 0; c < out_w; c++) {
                int out_idx = (i * out_h + r) * out_w + c;
                float max_val = -FLT_MAX;
                int max_idx = -1;

                for (int kr = 0; kr < kernel_h; kr++) {
                    for (int kc = 0; kc < kernel_w; kc++) {
                        int im_r = kr + stride_h * r;
                        int im_c = kc + stride_w * c;
                        maxpool_im_chmax(input, i, im_r, im_c, pad_h, pad_w,
                                         max_val, max_idx);
                    }
                }

                output->get_vec()[out_idx] = max_val;
                indices[out_idx] = max_idx;
            }
        }
    }
}

void maxpool_backward(Array *input_grad, const Array *output_grad,
                      const std::vector<int> &indices) {
    CHECK_EQ(input_grad->get_shape().size(), 4,
             "maxpool_backward: input gradient shape error");
    CHECK_EQ(output_grad->get_shape().size(), 4,
             "maxpool_backward: output gradient shape error");

    int batch_size = input_grad->get_shape()[0];
    int in_feats = input_grad->get_shape()[1];

    CHECK_EQ(output_grad->get_shape()[0], batch_size,
             "maxpool_backward: batch size error");
    CHECK_EQ(output_grad->get_shape()[1], in_feats,
             "maxpool_backward: batch size error");

    CHECK_EQ(
        indices.size(), output_grad->get_vec().size(),
        "maxpool_backward: indices size not equal to output gradient size");

    const std::vector<float> &output_grad_ref = output_grad->get_vec();
    for (int i = 0; i < output_grad_ref.size(); i++) {
        int max_idx = indices[i];
        input_grad->get_vec()[max_idx] += output_grad_ref[i];
    }
}

MaxPool2D::MaxPool2D(int pad_h, int pad_w, int kernel_h, int kernel_w,
                     int stride_h, int stride_w)
    : pad_h(pad_h), pad_w(pad_w), kernel_h(kernel_h), kernel_w(kernel_w),
      stride_h(stride_h), stride_w(stride_w) {}

void MaxPool2D::forward() {
    const Array *input = prev->get_output();

    int batch_size = input->get_shape()[0];
    int in_feats = input->get_shape()[1];
    int in_h = input->get_shape()[2];
    int in_w = input->get_shape()[3];

    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    set_array_ptr(output, {batch_size, in_feats, out_h, out_w});
    indices.resize(batch_size * in_feats * out_h * out_w);

    maxpool_forward(output.get(), input, indices, pad_h, pad_w, kernel_h,
                    kernel_w, stride_h, stride_w);
}

void MaxPool2D::backward() {
    const Array *input = prev->get_output();
    const Array *output_grad = next->get_grad();
    set_array_ptr(grad, input->get_shape());
    maxpool_backward(grad.get(), output_grad, indices);
}

} // namespace nnv2