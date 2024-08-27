#include "common.cuh"
#include "optimizer.cuh"

#include <cmath>
#include <memory>

#include <cuda_runtime.h>

namespace nnv2 {

// Stochastic gradient descent (SGD) with momentum
void SGD::add_parameters(std::vector<Param> params) {
    for (const auto &[weight, grad] : params) {
        const ShapeType &grad_shape = grad->get_shape();

        CHECK_EQ(weight->get_shape(), grad_shape,
                 "shape mismatch between weight and gradient");

        weights.push_back(weight);
        grads.push_back(grad);

        if (momentum != 0) {
            velocities.push_back(std::make_unique<Array>(grad_shape, 0));
        }
    }
}

__global__ void sgd_kernel(int size, float *weight, const float *grad, float lr,
                           float decay) {
    CUDA_GRID_STRIDE_LOOP(idx, size) {
        float g = grad[idx] + decay * weight[idx];
        weight[idx] -= lr * g;
    }
}

static void sgd_single(Array *weight, const Array *grad, float lr,
                       float decay) {
    int size = weight->get_vec().size();
    int grid_size = ceil((float)size / BLOCK_SIZE);

    float *weight_raw = RAW_PTR(weight->get_vec());
    const float *grad_raw = RAW_PTR(grad->get_vec());

    sgd_kernel<<<grid_size, BLOCK_SIZE>>>(size, weight_raw, grad_raw, lr,
                                          decay);
    CUDA_POST_KERNEL_CHECK;
}

__global__ void sgd_momentum_kernel(int size, float *weight, const float *grad,
                                    float *velocity, float lr, float decay,
                                    float momentum) {
    CUDA_GRID_STRIDE_LOOP(idx, size) {
        float g = grad[idx] + decay * weight[idx];
        velocity[idx] = momentum * velocity[idx] + g;
        weight[idx] -= lr * velocity[idx];
    }
}

static void sgd_single(Array *weight, const Array *grad, Array *velocity,
                       float lr, float decay, float momentum) {
    int size = weight->get_vec().size();
    int grid_size = ceil((float)size / BLOCK_SIZE);

    float *weight_raw = RAW_PTR(weight->get_vec());
    float *velocity_raw = RAW_PTR(velocity->get_vec());
    const float *grad_raw = RAW_PTR(grad->get_vec());

    sgd_momentum_kernel<<<grid_size, BLOCK_SIZE>>>(
        size, weight_raw, grad_raw, velocity_raw, lr, decay, momentum);
    CUDA_POST_KERNEL_CHECK;
}

void SGD::update_parameters() {
    int param_size = weights.size();
    for (int i = 0; i < param_size; i++) {
        if (momentum == 0) {
            sgd_single(weights[i], grads[i], lr, decay);
        } else {
            sgd_single(weights[i], grads[i], velocities[i].get(), lr, decay,
                       momentum);
        }
    }
}

// Root-mean-squared propagation (RMSProp)
void RMSProp::add_parameters(std::vector<Param> params) {
    for (const auto &[weight, grad] : params) {
        const ShapeType &grad_shape = grad->get_shape();

        CHECK_EQ(weight->get_shape(), grad_shape,
                 "shape mismatch between weight and gradient");

        weights.push_back(weight);
        grads.push_back(grad);
        mean_sqr_grads.push_back(std::make_unique<Array>(grad_shape, 0));
    }
}

__global__ void rmsprop_kernel(int size, float *weight, const float *grad,
                               float *mean_sqr_grad, float lr, float decay,
                               float beta) {
    CUDA_GRID_STRIDE_LOOP(idx, size) {
        float g = grad[idx] + decay * weight[idx];
        mean_sqr_grad[idx] = beta * mean_sqr_grad[idx] + (1 - beta) * g * g;
        weight[idx] -= g * lr / (sqrtf(mean_sqr_grad[idx]) + EPS);
    }
}

static void rmsprop_single(Array *weight, const Array *grad,
                           Array *mean_sqr_grad, float lr, float decay,
                           float beta) {
    int size = weight->get_vec().size();
    int grid_size = ceil((float)size / BLOCK_SIZE);

    float *weight_raw = RAW_PTR(weight->get_vec());
    float *mean_sqr_grad_raw = RAW_PTR(mean_sqr_grad->get_vec());
    const float *grad_raw = RAW_PTR(grad->get_vec());

    rmsprop_kernel<<<grid_size, BLOCK_SIZE>>>(
        size, weight_raw, grad_raw, mean_sqr_grad_raw, lr, decay, beta);
    CUDA_POST_KERNEL_CHECK;
}

void RMSProp::update_parameters() {
    int param_size = weights.size();
    for (int i = 0; i < param_size; i++) {
        rmsprop_single(weights[i], grads[i], mean_sqr_grads[i].get(), lr, decay,
                       beta);
    }
}

// Adaptive moment estimation (Adam)
void Adam::add_parameters(std::vector<Param> params) {
    for (const auto &[weight, grad] : params) {
        const ShapeType &grad_shape = grad->get_shape();

        CHECK_EQ(weight->get_shape(), grad_shape,
                 "shape mismatch between weight and gradient");

        weights.push_back(weight);
        grads.push_back(grad);
        mean_grads.push_back(std::make_unique<Array>(grad_shape, 0));
        mean_sqr_grads.push_back(std::make_unique<Array>(grad_shape, 0));
    }
}

__global__ void adam_kernel(int size, float *weight, const float *grad,
                            float *mean_grad, float *mean_sqr_grad, float lr,
                            float decay, float beta1, float beta2,
                            float beta1_pow, float beta2_pow) {
    CUDA_GRID_STRIDE_LOOP(idx, size) {
        float g = grad[idx] + decay * weight[idx];
        mean_grad[idx] = beta1 * mean_grad[idx] + (1 - beta1) * g;
        mean_sqr_grad[idx] = beta2 * mean_sqr_grad[idx] + (1 - beta2) * g * g;

        float m1_norm = mean_grad[idx] / (1 - beta1_pow);
        float m2_norm = mean_sqr_grad[idx] / (1 - beta2_pow);
        weight[idx] -= lr * m1_norm / (sqrtf(m2_norm) + EPS);
    }
}

// Poor Adam :(
static void adam_single(Array *weight, const Array *grad, Array *mean_grad,
                        Array *mean_sqr_grad, float lr, float decay,
                        float beta1, float beta2, float beta1_pow,
                        float beta2_pow) {
    int size = weight->get_vec().size();
    int grid_size = ceil((float)size / BLOCK_SIZE);

    float *weight_raw = RAW_PTR(weight->get_vec());
    float *mean_grad_raw = RAW_PTR(mean_grad->get_vec());
    float *mean_sqr_grad_raw = RAW_PTR(mean_sqr_grad->get_vec());
    const float *grad_raw = RAW_PTR(grad->get_vec());

    adam_kernel<<<grid_size, BLOCK_SIZE>>>(
        size, weight_raw, grad_raw, mean_grad_raw, mean_sqr_grad_raw, lr, decay,
        beta1, beta2, beta1_pow, beta2_pow);

    CUDA_POST_KERNEL_CHECK;
}

void Adam::update_parameters() {
    // Update beta^t and beta_sqr^t in each optimization iteration
    beta1_pow *= beta1;
    beta2_pow *= beta2;

    int param_size = weights.size();
    for (int i = 0; i < param_size; i++) {
        adam_single(weights[i], grads[i], mean_grads[i].get(),
                    mean_sqr_grads[i].get(), lr, decay, beta1, beta2, beta1_pow,
                    beta2_pow);
    }
}

} // namespace nnv2