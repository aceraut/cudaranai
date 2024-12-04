#include "common.cuh"
#include "maxpool.cuh"

#include <cfloat>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

namespace nnv2 {

// The forward phase of max-pooling layer involves selecting the pixel with the
// largest value in every smaller local patches of the input feature maps and
// place them in the output feature maps.
__global__ void maxpool_forward_kernel(int size, float *output,
                                       int *max_indices, const float *input,
                                       int in_h, int in_w, int pad_h, int pad_w,
                                       int filter_h, int filter_w, int stride_h,
                                       int stride_w, int out_h, int out_w,
                                       int in_stride, int out_stride) {
    // Each thread handles a pixel in the output image
    CUDA_GRID_STRIDE_LOOP(idx, size) {
        // Point to input and output images that this thread handles
        int feat_idx = blockIdx.y;
        input += feat_idx * in_stride;
        output += feat_idx * out_stride;
        max_indices += feat_idx * out_stride;

        // Coord in output image
        int out_x = (idx / out_w) % out_h;
        int out_y = idx % out_w;

        // Locate the base coords of the section in the input image that
        // affects (out_x, out_y) of the output image
        int in_x_start = out_x * stride_h - pad_h;
        int in_y_start = out_y * stride_w - pad_w;
        int in_x_end = fminf(in_x_start + filter_h, in_h);
        int in_y_end = fminf(in_y_start + filter_w, in_w);
        in_x_start = fmaxf(in_x_start, 0);
        in_y_start = fmaxf(in_y_start, 0);

        // Loop over the local patch and select the pixel with largest value
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

void maxpool_forward(Array *output, const Array *input,
                     thrust::device_vector<int> &indices, int pad_h, int pad_w,
                     int filter_h, int filter_w, int stride_h, int stride_w) {
    const ShapeType &output_shape = output->get_shape();
    const ShapeType &input_shape = input->get_shape();

    CHECK_EQ(output_shape.size(), 4, "maxpool_forward: output shape error");
    CHECK_EQ(input_shape.size(), 4, "maxpool_forward: input shape error");
    CHECK_EQ(indices.size(), output->get_vec().size(),
             "maxpool_forward: size mismatch beetween indices and output");

    int batch_size = input_shape[0];
    int in_feats = input_shape[1];

    CHECK_EQ(output_shape[0], batch_size, "maxpool_forward: batch size error");
    CHECK_EQ(output_shape[1], in_feats, "maxpool_forward: feature size error");

    int in_h = input_shape[2];
    int in_w = input_shape[3];
    int in_stride = in_h * in_w;
    int out_h = output_shape[2];
    int out_w = output_shape[3];
    int size = out_h * out_w; // is also out_stride
    dim3 grid_dim(utils::div_ceil(size, BLOCK_SIZE), batch_size * in_feats);

    float *output_raw = RAW_PTR(output->get_vec());
    int *indices_raw = RAW_PTR(indices);
    const float *input_raw = RAW_PTR(input->get_vec());

    maxpool_forward_kernel<<<grid_dim, BLOCK_SIZE>>>(
        size, output_raw, indices_raw, input_raw, in_h, in_w, pad_h, pad_w,
        filter_h, filter_w, stride_h, stride_w, out_h, out_w, in_stride, size);
    CUDA_POST_KERNEL_CHECK;
}

// Each output gradient is propagated back to its pooled position in the input.
__global__ void maxpool_backward_kernel(int size, float *input_grad,
                                        const float *output_grad,
                                        const int *max_indices, int in_stride,
                                        int out_stride) {
    CUDA_GRID_STRIDE_LOOP(idx, size) {
        int feat_idx = blockIdx.y;
        input_grad += feat_idx * in_stride;
        output_grad += feat_idx * out_stride;
        max_indices += feat_idx * out_stride;

        int pooled_index = max_indices[idx];
        float grad_value = output_grad[idx];

        atomicAdd(&input_grad[pooled_index], grad_value);
    }
}

void maxpool_backward(Array *input_grad, const Array *output_grad,
                      const thrust::device_vector<int> &indices, int pad_h,
                      int pad_w, int filter_h, int filter_w, int stride_h,
                      int stride_w) {
    const ShapeType &input_grad_shape = input_grad->get_shape();
    const ShapeType &output_grad_shape = output_grad->get_shape();

    CHECK_EQ(input_grad_shape.size(), 4,
             "maxpool_backward: input gradient shape error");
    CHECK_EQ(output_grad_shape.size(), 4,
             "maxpool_backward: output gradient shape error");
    CHECK_EQ(indices.size(), output_grad->get_vec().size(),
             "maxpool_backward: size mismatch between indices and output grad");

    int batch_size = input_grad_shape[0];
    int in_feats = input_grad_shape[1];

    CHECK_EQ(output_grad_shape[0], batch_size,
             "maxpool_backward: batch size error");
    CHECK_EQ(output_grad_shape[1], in_feats,
             "maxpool_backward: feature size error");

    int in_stride = input_grad_shape[2] * input_grad_shape[3];
    int out_stride = output_grad_shape[2] * output_grad_shape[3];
    dim3 grid_dim(utils::div_ceil(out_stride, BLOCK_SIZE),
                  batch_size * in_feats);

    float *input_grad_raw = RAW_PTR(input_grad->get_vec());
    const float *output_grad_raw = RAW_PTR(output_grad->get_vec());
    const int *indices_raw = RAW_PTR(indices);

    maxpool_backward_kernel<<<grid_dim, BLOCK_SIZE>>>(
        out_stride, input_grad_raw, output_grad_raw, indices_raw, in_stride,
        out_stride);
    CUDA_POST_KERNEL_CHECK;
}

MaxPool2D::MaxPool2D(int pad_h, int pad_w, int kernel_h, int kernel_w,
                     int stride_h, int stride_w)
    : pad_h(pad_h), pad_w(pad_w), kernel_h(kernel_h), kernel_w(kernel_w),
      stride_h(stride_h), stride_w(stride_w) {}

void MaxPool2D::forward() {
    const Array *input = prev->get_output();
    const ShapeType &input_shape = input->get_shape();

    int batch_size = input_shape[0];
    int in_feats = input_shape[1];
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    utils::set_array_ptr(output, {batch_size, in_feats, out_h, out_w});
    indices.resize(output->get_vec().size());

    maxpool_forward(output.get(), input, indices, pad_h, pad_w, kernel_h,
                    kernel_w, stride_h, stride_w);
}

void MaxPool2D::backward() {
    const Array *input = prev->get_output();
    const Array *output_grad = next->get_grad();

    utils::set_array_ptr(grad, input->get_shape());

    maxpool_backward(grad.get(), output_grad, indices, pad_h, pad_w, kernel_h,
                     kernel_w, stride_h, stride_w);
}

} // namespace nnv2