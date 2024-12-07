// Pretty much a broken demo
// 3 convolution + 3 fully connected with dropout inbetween (~644k params)

#include "nnv2.cuh"

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char **argv) {
  using namespace nnv2;

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <dataset_path>" << std::endl;
    return 1;
  }

  std::string data_path = argv[1];

  Network net;
  std::unique_ptr<Initializer> init = std::make_unique<XavierUniform>();

  // First convolution layer
  // Computes 32 features using a 5x5 filter with ReLU activation.
  // Padding of (2, 2) is added to preserve width and height.
  // Input shape: [batch_size, 1, 28, 28]
  // Output shape: [batch_size, 32, 28, 28]
  // No. of params: 1*32*5*5 + 32 = 832
  net.add(new Conv2D(1, 32, 28, 28, 2, 2, 5, 5, 1, 1, init.get()));
  net.add(new ReLU);

  // First max pooling layer
  // Selects the maximum values over a 2x2 filter with stride of 2
  // Input shape: [batch_size, 32, 28, 28]
  // Output shape: [batch_size, 32, 14, 14]
  net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

  // Second convolution layer
  // Computes 64 features using a 5x5 filter with ReLU activation.
  // Padding of (2, 2) is added to preserve width and height.
  // Input shape: [batch_size, 32, 14, 14]
  // Output shape: [batch_size, 64, 14, 14]
  // No. of params: 32*64*5*5 + 64 = 51_264
  net.add(new Conv2D(32, 64, 14, 14, 2, 2, 5, 5, 1, 1, init.get()));
  net.add(new ReLU);

  // Second max pooling layer
  // Selects the maximum values over a 2x2 filter with stride of 2
  // Input shape: [batch_size, 64, 14, 14]
  // Output shape: [batch_size, 64, 7, 7]
  net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

  // Third convolution layer
  // Computes 128 features using a 2x2 filter with ReLU activation.
  // Padding of (2, 2) is added to prevent missing pixels during max pooling
  // Input shape: [batch_size, 64, 7, 7]
  // Output shape: [batch_size, 128, 8, 8]
  // No. of params: 64*128*2*2 + 128 = 32_896
  net.add(new Conv2D(64, 128, 7, 7, 1, 1, 2, 2, 1, 1, init.get()));
  net.add(new ReLU);

  // Third max pooling layer
  // Selects the maximum values over a 2x2 filter with stride of 2.
  // The rightmost column and bottom row of an image are not used.
  // Input shape: [batch_size, 128, 8, 8]
  // Output shape: [batch_size, 128, 4, 4]
  net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

  // Flatten inference batches from 2D image features to 1D neurons
  // Input shape: [batch_size, 128, 4, 4]
  // Output shape: [batch_size, 128*4*4]
  net.add(new Flatten);

  // First fully connected layer with 256 neurons
  // Input shape: [batch_size, 128*4*4]
  // Output shape: [batch_size, 256]
  // No. of params: 128*4*4*256 + 256 = 524_544
  net.add(new Linear(128 * 4 * 4, 256, init.get()));
  net.add(new ReLU);

  // Dropout layer
  // Randomly zeroes out half of the current neurons
  net.add(new Dropout(0.5));

  // Second fully connected layer with 128 neurons
  // Input shape: [batch_size, 256]
  // Output shape: [batch_size, 128]
  // No. of params: 256*128 + 128 = 32_896
  net.add(new Linear(256, 128, init.get()));
  net.add(new ReLU);

  // Dropout layer
  // Randomly zeroes out half of the current neurons
  net.add(new Dropout(0.5));

  // Logits layer
  // Input shape: [batch_size, 128]
  // Output shape: [batch_size, 10]
  // No. of params: 128*10 + 10 = 1290
  net.add(new Linear(128, 10, init.get()));
  net.add(new LogSoftmax);

  std::cout << "Network setup complete" << std::endl;

  std::unique_ptr<DataLoader> loader =
      std::make_unique<DataLoader>(new Mnist(data_path), 64);
  std::unique_ptr<Loss> loss = std::make_unique<NLLLoss>();
  std::unique_ptr<Optimizer> optim = std::make_unique<Adam>(0.0015);

  net.init(loader.get(), loss.get(), optim.get());
  net.train(30, true);

  return 0;
}