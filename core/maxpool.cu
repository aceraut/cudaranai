// This implements the MaxPool2D layer, a variant of pooling layer.
//
// This layer is typically used after the convolution layer and is used to
// reduce the spatial dimensions of the input data, by taking pixels with the
// maximum value in smaller grids of the input image to the output image.

#include "common.cuh"
#include "maxpool.cuh"

#include <cfloat>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

namespace nnv2 {

__global__ void maxpool_forward_kernel(int size, float *output,
                                       float *max_indices, const float *input,
                                       int in_h, int in_w, int pad_h, int pad_w,
                                       int filter_h, int filter_w, int stride_h,
                                       int stride_w, int out_h, int out_w,
                                       int in_stride, int out_stride) {
    // each thread handles a pixel in the output image
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
         idx += blockIdx.x * blockDim.x) {
        int out_x = (idx / out_w) % out_h;
        int out_y = idx % out_w;
        int feat_idx = idx / out_w / out_h;

        input += feat_idx * in_stride;
        output += feat_idx * out_stride;
        max_indices += feat_idx * out_stride;

        int in_x_start = out_x * stride_h - pad_h;
        int in_y_start = out_y * stride_w - pad_w;
        int in_x_end = fminf(in_x_start + filter_h, in_h);
        int in_y_end = fminf(in_y_start + filter_w, in_w);
        in_x_start = fmaxf(in_x_start, 0);
        in_y_start = fmaxf(in_y_start, 0);

        float max_val = -FLT_MAX;
        float max_idx = -1;
        for (int in_x = in_x_start; in_x < in_x_end; in_x++) {
            for (int in_y = in_y_start; in_y < in_y_end; in_y++) {
                int in_idx = in_x * in_w + in_y;
                if (input[in_idx] > max_val) {
                    max_idx = in_idx;
                    max_val = input[in_idx];
                }
            }
        }

        output[idx] = max_val;
        max_indices[idx] = max_idx;
    }
}

void maxpool_forward(Array *output, const Array *input, Array *indices,
                     int pad_h, int pad_w, int filter_h, int filter_w,
                     int stride_h, int stride_w) {
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
    CHECK_EQ(indices->get_vec().size(), output->get_vec().size(),
             "maxpool_forward: size mismatch beetween indices and output");

    int in_h = input->get_shape()[2];
    int in_w = input->get_shape()[3];
    int in_stride = in_h * in_w;
    int out_h = output->get_shape()[2];
    int out_w = output->get_shape()[3];
    int out_stride = out_h * out_w;

    float *output_raw = RAW_PTR(output->get_vec());
    float *indices_raw = RAW_PTR(indices->get_vec());
    const float *input_raw = RAW_PTR(input->get_vec());

    int size = batch_size * in_feats * out_h * out_w;
    int grid_size = ceil((float)size / BLOCK_SIZE);

    maxpool_forward_kernel<<<grid_size, BLOCK_SIZE>>>(
        size, output_raw, indices_raw, input_raw, in_h, in_w, pad_h, pad_w,
        filter_h, filter_w, stride_h, stride_w, out_h, out_w, in_stride,
        out_stride);
    CUDA_POST_KERNEL_CHECK;
}

__global__ void
maxpool_backward_kernel(int size, float *input_grad, const float *output_grad,
                        const float *max_indices, int in_h, int in_w, int pad_h,
                        int pad_w, int filter_h, int filter_w, int stride_h,
                        int stride_w, int out_h, int out_w, int in_stride,
                        int out_stride) {
    // each thread handles a pixel in the input image
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
         idx += blockIdx.x * blockDim.x) {
        // coord in input image
        int in_x = (idx / in_w) % in_h + pad_h;
        int in_y = idx % in_w + pad_w;
        int feat_idx = idx / out_w / out_h;

        input_grad += feat_idx * in_stride;
        output_grad += feat_idx * out_stride;
        max_indices += feat_idx * out_stride;

        // locate the base coords of the section in the output image that
        // depends on pixel (in_x, in_y) of the input image
        int out_x_start =
            (in_x < filter_h) ? 0 : (in_x - filter_h) / stride_h + 1;
        int out_x_end = fminf(out_h, in_x / stride_h + 1);
        int out_y_start =
            (in_y < filter_w) ? 0 : (in_y - filter_w) / stride_w + 1;
        int out_y_end = fminf(out_w, in_y / stride_w + 1);

        float value = 0;
        in_x -= pad_h;
        in_y -= pad_w; // since max_indices are unpadded

        for (int out_x = out_x_start; out_x < out_x_end; out_x++) {
            for (int out_y = out_y_start; out_y < out_y_end; out_y++) {
                if (max_indices[out_x * out_w + out_y] == in_x * in_w + in_y) {
                    value += output_grad[out_x * out_w + out_y];
                }
            }
        }
        input_grad[idx] = value;
    }
}

void maxpool_backward(Array *input_grad, const Array *output_grad,
                      const Array *indices, int pad_h, int pad_w, int filter_h,
                      int filter_w, int stride_h, int stride_w) {
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

    CHECK_EQ(indices->get_vec().size(), output_grad->get_vec().size(),
             "maxpool_backward: size mismatch between indices and output grad");

    int in_h = input_grad->get_shape()[2];
    int in_w = input_grad->get_shape()[3];
    int in_stride = in_h * in_w; // is also in_stride

    int out_h = output_grad->get_shape()[2];
    int out_w = output_grad->get_shape()[3];
    int out_stride = out_h * out_w;

    float *input_grad_raw = RAW_PTR(input_grad->get_vec());
    const float *output_grad_raw = RAW_PTR(output_grad->get_vec());
    const float *indices_raw = RAW_PTR(indices->get_vec());

    int size = batch_size * in_feats * in_h * in_w;
    int grid_size = ceil((float)size / BLOCK_SIZE);

    maxpool_backward_kernel<<<grid_size, BLOCK_SIZE>>>(
        size, input_grad_raw, output_grad_raw, indices_raw, in_h, in_w, pad_h,
        pad_w, filter_h, filter_w, stride_h, stride_w, out_h, out_w, in_stride,
        out_stride);
    CUDA_POST_KERNEL_CHECK;
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
    set_array_ptr(indices, output->get_shape());

    maxpool_forward(output.get(), input, indices.get(), pad_h, pad_w, kernel_h,
                    kernel_w, stride_h, stride_w);
}

void MaxPool2D::backward() {
    const Array *input = prev->get_output();
    const Array *output_grad = next->get_grad();
    set_array_ptr(grad, input->get_shape());
    maxpool_backward(grad.get(), output_grad, indices.get(), pad_h, pad_w,
                     kernel_h, kernel_w, stride_h, stride_w);
}

} // namespace nnv2