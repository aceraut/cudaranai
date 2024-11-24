// https://github.com/hardmaru/pytorch_notebooks/blob/master/mnist_es/pytorch_mnist_mini_adam.ipynb
// A small network with 11k parameters

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
    // Computes 8 features using a 5x5 filter with ReLU activation.
    // Padding of (2, 2) is added to preserve width and height.
    // Input shape: [batch_size, 1, 28, 28]
    // Output shape: [batch_size, 8, 28, 28]
    // No. of params: 1*8*5*5 + 8 = 208
    net.add(new Conv2D(1, 8, 28, 28, 2, 2, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);

    // First max pooling layer
    // Selects the maximum values over a 2x2 filter with stride of 2
    // Input shape: [batch_size, 8, 28, 28]
    // Output shape: [batch_size, 8, 14, 14]
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

    // Second convolution layer
    // Computes 16 features using a 5x5 filter with ReLU activation.
    // Padding of (2, 2) is added to preserve width and height.
    // Input shape: [batch_size, 8, 14, 14]
    // Output shape: [batch_size, 16, 14, 14]
    // No. of params: 8*16*5*5 + 64 = 3216
    net.add(new Conv2D(8, 16, 14, 14, 2, 2, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);

    // Second max pooling layer
    // Selects the maximum values over a 2x2 filter with stride of 2
    // Input shape: [batch_size, 16, 14, 14]
    // Output shape: [batch_size, 16, 7, 7]
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

    // Flatten inference batches from 2D image features to 1D neurons
    // Input shape: [batch_size, 16, 7, 7]
    // Output shape: [batch_size, 16*7*7]
    net.add(new Flatten);

    // Logits layer
    // Input shape: [batch_size, 16*7*7]
    // Output shape: [batch_size, 10]
    // No. of params: 16*7*7*10 + 10 = 7850
    net.add(new Linear(16 * 7 * 7, 10, init.get()));
    net.add(new LogSoftmax);

    std::cout << "Network setup complete" << std::endl;

    std::unique_ptr<DataLoader> loader =
        std::make_unique<DataLoader>(new Mnist(data_path), 1000);
    std::unique_ptr<Loss> loss = std::make_unique<NLLLoss>();
    std::unique_ptr<Optimizer> optim = std::make_unique<Adam>(0.002);

    net.init(loader.get(), loss.get(), optim.get());
    net.train(30, true);

    return 0;
}