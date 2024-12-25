#include "dataloader.cuh"
#include "mnist.cuh"
#include "test_utils.cuh"

#include <iostream>
#include <string>

using namespace nnv2;

void test_mnist_dataloader(std::string data_path) {
  DataLoader loader(new Mnist(data_path), 32);

  int epoch = 30;
  for (int e = 0; e < epoch; e++) {
    std::cout << "Epoch " << e + 1 << std::endl;

    // preparation
    loader.reset(false);

    // load all train batches
    while (loader.has_next_train_batch()) {
      loader.load_train_batch();
    }
    // debug
    std::cout << "Final train image batch shape:";
    print_vec(loader.get_output()->get_shape());
    std::cout << std::endl;

    // load all test batches
    while (loader.has_next_test_batch()) {
      loader.load_test_batch();
    }
    // debug
    std::cout << "Final test image batch shape:";
    print_vec(loader.get_output()->get_shape());
    std::cout << std::endl;
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <dataset_path>" << std::endl;
    return 1;
  }

  std::string data_path = argv[1];
  test_mnist_dataloader(data_path);
}