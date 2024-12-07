#include "activation.cuh"
#include "common.cuh"
#include "loss.cuh"
#include "test_utils.cuh"

#include <iostream>
#include <vector>

using namespace nnv2;

void test_relu_forward() {
  Array input({1, 1, 4, 3}, {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12});
  Array output({1, 1, 4, 3});

  relu_forward(&output, &input);
  check_equal_vecs(output.get_vec(), {0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12});

  std::cout << "test_relu_forward: Passed" << std::endl;
}

void test_relu_backward() {
  Array input({1, 1, 4, 3}, {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12});
  Array output_grad({1, 1, 4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Array input_grad({1, 1, 4, 3});

  relu_backward(&input_grad, &output_grad, &input);
  check_equal_vecs(
      input_grad.get_vec(), {0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12});

  std::cout << "test_relu_backward: Passed" << std::endl;
}

void test_softmax_forward() {
  Array input({4, 3}, {1, 2, 3, 1, 2, -1, 0, 0, 0, -1, 2, 0});
  Array output({4, 3});

  softmax_forward(&output, &input);
  print_vec(output.get_vec());

  std::cout << "test_softmax_forward: DIY!" << std::endl;
}

void test_log_softmax_forward() {
  Array input({3, 3}, {1, 2, 3, 1, 2, 3, 1, 2, 3});
  Array output({3, 3});

  log_softmax_forward(&output, &input);
  print_vec(output.get_vec());

  std::cout << "test_logsoftmax_forward: DIY!" << std::endl;
}

void test_log_softmax_trip() {
  Array y({3, 3}, {0, 0, 1, 0, 1, 0, 1, 0, 0});
  Array input({3, 3}, {0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33});

  // forward
  Array output({3, 3});
  log_softmax_forward(&output, &input);
  Array loss({1});
  ArrayMap cache;
  nll_loss(&loss, &output, &y, cache);
  print_vec(loss.get_vec());

  // backward
  Array loss_grad({3, 3});
  nll_loss_backward(&loss_grad, &y);
  print_vec(loss_grad.get_vec());

  Array input_grad({3, 3});
  log_softmax_backward(&input_grad, &loss_grad, &input);
  print_vec(input_grad.get_vec());

  std::cout << "test_logsoftmax_trip: DIY!" << std::endl;
}

int main() {
  test_relu_forward();
  test_relu_backward();
  test_softmax_forward();
  test_log_softmax_forward();
  test_log_softmax_trip();
}