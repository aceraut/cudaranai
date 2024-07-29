#include "optimizer.h"
#include "common.h"

#include <cmath>
#include <memory>

namespace nnv2 {

//
// Stochastic gradient descent (SGD) with momentum
//
void SGD::add_parameters(std::vector<Param> params) {
    for (const auto &[weight, grad] : params) {
        CHECK_EQ(weight->get_shape(), grad->get_shape(),
                 "weight shape in parameter mismatched with its gradient");

        weights.push_back(weight);
        grads.push_back(grad);
        if (momentum != 0) {
            velocities.push_back(std::make_unique<Array>(grad->get_shape(), 0));
        }
    }
}

static void sgd_single(Array *weight, const Array *grad, float lr,
                       float decay) {
    for (int i = 0; i < grad->get_vec().size(); i++) {
        float &w = weight->get_vec()[i];
        float g = grad->get_vec()[i] + decay * w;
        w -= lr * g;
    }
}

static void sgd_single(Array *weight, const Array *grad, Array *velocity,
                       float lr, float decay, float momentum) {
    for (int i = 0; i < grad->get_vec().size(); i++) {
        float &w = weight->get_vec()[i];
        float &v = velocity->get_vec()[i];
        float g = grad->get_vec()[i] + decay * w;

        // update velocity
        v = momentum * v + g;
        // update weight
        w -= lr * v;
    }
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

//
// Root-mean-squared propagation (RMSProp)
//
void RMSProp::add_parameters(std::vector<Param> params) {
    for (const auto &[weight, grad] : params) {
        CHECK_EQ(weight->get_shape(), grad->get_shape(),
                 "weight shape in parameter mismatched with its gradient");

        weights.push_back(weight);
        grads.push_back(grad);
        mean_sqr_grads.push_back(std::make_unique<Array>(grad->get_shape(), 0));
    }
}

static void rmsprop_single(Array *weight, const Array *grad,
                           Array *mean_sqr_grad, float lr, float decay,
                           float beta) {
    for (int i = 0; i < grad->get_vec().size(); i++) {
        float &w = weight->get_vec()[i];
        float &e = mean_sqr_grad->get_vec()[i];
        float g = grad->get_vec()[i] + decay * w;

        // Update accumulated squared gradient
        e = beta * e + (1 - beta) * g * g;
        // Update weight
        w -= g * lr / (sqrtf(e) + EPS);
    }
}

void RMSProp::update_parameters() {
    int param_size = weights.size();
    for (int i = 0; i < param_size; i++) {
        rmsprop_single(weights[i], grads[i], mean_sqr_grads[i].get(), lr, decay,
                       beta);
    }
}

//
// Adaptive moment estimation (Adam)
//
void Adam::add_parameters(std::vector<Param> params) {
    for (const auto &[weight, grad] : params) {
        CHECK_EQ(weight->get_shape(), grad->get_shape(),
                 "weight shape in parameter mismatched with its gradient");

        weights.push_back(weight);
        grads.push_back(grad);
        mean_grads.push_back(std::make_unique<Array>(grad->get_shape(), 0));
        mean_sqr_grads.push_back(std::make_unique<Array>(grad->get_shape(), 0));
    }
}

// poor Adam :(
static void adam_single(Array *weight, const Array *grad, Array *mean_grad,
                        Array *mean_sqr_grad, float lr, float decay,
                        float beta1, float beta2, float beta1_pow,
                        float beta2_pow) {
    for (int i = 0; i < grad->get_vec().size(); i++) {
        float &w = weight->get_vec()[i];
        float &m1 = mean_grad->get_vec()[i];
        float &m2 = mean_sqr_grad->get_vec()[i];
        float g = grad->get_vec()[i] + decay * w;

        // Update accumulated gradient
        m1 = beta1 * m1 + (1 - beta1) * g;
        // Update accumulated squared gradient
        m2 = beta2 * m2 + (1 - beta2) * g * g;
        // Get bias corrected m1 and m2
        float m1_norm = m1 / (1 - beta1_pow);
        float m2_norm = m2 / (1 - beta2_pow);
        // Update weight
        w -= lr * m1_norm / (sqrtf(m2_norm) + EPS);
    }
}

void Adam::update_parameters() {
    // update beta^t and beta_sqr^t in each optimization iteration
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