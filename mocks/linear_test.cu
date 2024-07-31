#include "common.cuh"
#include "linear.cuh"
#include "test_utils.cuh"

#include <iostream>
#include <vector>

using namespace nnv2;

void test_linear_forward(void) {
    Array input({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Array weights({4, 2}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
    Array output({3, 2});

    linear_forward(&output, &input, &weights);
    check_equal_vecs(output.get_vec(), {5, 6, 11.4, 14, 17.8, 22});

    std::cout << "test_linear_forward: Passed" << std::endl;
}

void test_linear_forward_bias(void) {
    Array bias({1, 2}, {1, 2});
    Array output({3, 2}, {5, 6, 11, 14, 17, 22});

    linear_forward_bias(&output, &bias);
    check_equal_vecs(output.get_vec(), {6, 8, 12, 16, 18, 24});

    std::cout << "test_linear_forward_bias: Passed" << std::endl;
}

void test_linear_backward(void) {
    Array output_grad({2, 3}, {0, 1, 2, 3, 4, 5});
    Array input({2, 3}, {0, 1, 2, 3, 4, 5});
    Array weights({3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    Array input_grad({2, 3});
    Array weights_grad({3, 3});
    ArrayMap cache;

    linear_backward(&input_grad, &weights_grad, &input, &weights, &output_grad,
                    cache);

    check_equal_vecs(input_grad.get_vec(), {5, 14, 23, 14, 50, 86});
    check_equal_vecs(weights_grad.get_vec(),
                     {9, 12, 15, 12, 17, 22, 15, 22, 29});

    std::cout << "test_linear_backward: Passed" << std::endl;
}

void test_linear_backward_bias(void) {
    Array output_grad({2, 3}, {0, 1, 2, 3, 4, 5});
    Array bias_grad({1, 3});

    linear_backward_bias(&bias_grad, &output_grad);
    check_equal_vecs(bias_grad.get_vec(), {3, 5, 7});

    std::cout << "test_linear_backward_bias: Passed" << std::endl;
}

int main(void) {
    test_linear_forward();
    test_linear_forward_bias();
    test_linear_backward();
    test_linear_backward_bias();
}