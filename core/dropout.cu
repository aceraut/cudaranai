#include "dropout.cuh"

#include <chrono>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>

namespace nnv2 {

__global__ void dropout_forward_kernel(int size, float *output,
                                       const float *input, char *mask,
                                       float drop_rate, unsigned seed) {
    CUDA_GRID_STRIDE_LOOP(idx, size) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        char keep = curand_uniform(&state) >= drop_rate;
        mask[idx] = keep;
        output[idx] = keep ? input[idx] / (1 - drop_rate) : 0;
    }
}

void dropout_forward(Array *output, const Array *input, float drop_rate,
                     thrust::device_vector<char> &mask) {
    const ShapeType &output_shape = output->get_shape();
    const ShapeType &input_shape = input->get_shape();

    CHECK_EQ(output_shape, input_shape,
             "dropout_forward: shape mismatch between input and output");

    float *output_raw = RAW_PTR(output->get_vec());
    const float *input_raw = RAW_PTR(input->get_vec());
    char *mask_raw = RAW_PTR(mask);

    unsigned seed =
        (unsigned)std::chrono::steady_clock::now().time_since_epoch().count();

    int size = input->get_vec().size();
    int grid_size = utils::quotient_ceil(size, BLOCK_SIZE);

    dropout_forward_kernel<<<grid_size, BLOCK_SIZE>>>(
        size, output_raw, input_raw, mask_raw, drop_rate, seed);
    CUDA_POST_KERNEL_CHECK;
}

__global__ void dropout_backward_kernel(int size, float *input_grad,
                                        const float *output_grad,
                                        const char *mask, float drop_rate) {
    CUDA_GRID_STRIDE_LOOP(idx, size) {
        char keep = mask[idx];
        input_grad[idx] = keep ? output_grad[idx] / (1 - drop_rate) : 0;
    }
}

void dropout_backward(Array *input_grad, const Array *output_grad,
                      float drop_rate,
                      const thrust::device_vector<char> &mask) {
    const ShapeType &input_grad_shape = input_grad->get_shape();
    const ShapeType &output_grad_shape = output_grad->get_shape();

    CHECK_EQ(input_grad_shape, output_grad_shape,
             "dropout_backward: shape mismatch between input grad and output "
             "grad");

    float *input_grad_raw = RAW_PTR(input_grad->get_vec());
    const float *output_grad_raw = RAW_PTR(output_grad->get_vec());
    const char *mask_raw = RAW_PTR(mask);

    int size = output_grad->get_vec().size();
    int grid_size = utils::quotient_ceil(size, BLOCK_SIZE);

    dropout_backward_kernel<<<grid_size, BLOCK_SIZE>>>(
        size, input_grad_raw, output_grad_raw, mask_raw, drop_rate);
    CUDA_POST_KERNEL_CHECK;
}

void Dropout::forward() {
    Array *input = prev->get_output();
    mask.resize(input->get_vec().size());
    dropout_forward(input, input, drop_rate, mask);
}

// On test mode, the layer simply passes the input forward
void Dropout::forward_test() {}

void Dropout::backward() {
    Array *output_grad = next->get_grad();
    dropout_backward(output_grad, output_grad, drop_rate, mask);
}

} // namespace nnv2