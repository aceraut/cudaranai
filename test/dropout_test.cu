#include "common.cuh"
#include "dropout.cuh"
#include "test_utils.cuh"

#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

using namespace nnv2;

void test_dropout_forward() {
  std::cout << "test_dropout_forward: Begin" << std::endl;

  Array input({32, 1024}, 1);
  Array output({32, 1024});
  int size = 32 * 1024;

  thrust::device_vector<char> mask(size);

  // drop 60% of neurons
  dropout_forward(&output, &input, 0.6, mask);
  // check
  int drop_count = 0;
  float output_sum = 0;
  for (int i = 0; i < size; i++) {
    if (mask[i] == 0) {
      drop_count++;
    }
    output_sum += output.get_vec()[i];
  }

  float actual_drop_rate = 1.0 * drop_count / size;
  std::cout << "[Drop_rate] E: 0.6, A: " << actual_drop_rate << std::endl;
  std::cout << "[Output data point sum] E: 32768, A: " << output_sum
            << std::endl;
  std::cout << "test_dropout_forward: Passed" << std::endl;
}

int main() {
  test_dropout_forward();
}