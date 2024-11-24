// Custom network with 3 conv + 2 linear (~193k parameters)

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
    // Input shape: [batch_size, 1, 28, 28]
    // Output shape: [batch_size, 32, 24, 24]
    // No. of params: 1*32*5*5 + 32 = 832
    net.add(new Conv2D(1, 32, 28, 28, 0, 0, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);

    // First max pooling layer
    // Selects the maximum values over a 2x2 filter with stride of 2.
    // Input shape: [batch_size, 32, 24, 24]
    // Output shape: [batch_size, 32, 12, 12]
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

    // Second convolution layer
    // Computes 64 features using a 5x5 filter with ReLU activation.
    // Input shape: [batch_size, 32, 12, 12]
    // Output shape: [batch_size, 64, 8, 8]
    // No. of params: 32*64*5*5 + 64 = 51_264
    net.add(new Conv2D(32, 64, 12, 12, 0, 0, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);

    // Second max pooling layer
    // Selects the maximum values over a 2x2 filter with stride of 2.
    // Input shape: [batch_size, 64, 8, 8]
    // Output shape: [batch_size, 64, 4, 4]
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));

    // Third convolution layer
    // Computes 128 features using a 3x3 filter with ReLU activation.
    // Input shape: [batch_size, 64, 4, 4]
    // Output shape: [batch_size, 128, 2, 2]
    // No. of params: 64*128*3*3 + 128 = 73_856
    net.add(new Conv2D(64, 128, 4, 4, 0, 0, 3, 3, 1, 1, init.get()));
    net.add(new ReLU);

    // Flatten inference batches from 2D image features to 1D neurons
    // Input shape: [batch_size, 128, 2, 2]
    // Output shape: [batch_size, 512]
    net.add(new Flatten);

    // Fully connected layer with 128 neurons
    // Input shape: [batch_size, 512]
    // Output shape: [batch_size, 128]
    // No. of params: 512*128 + 128 = 65_664
    net.add(new Linear(512, 128, init.get()));
    net.add(new ReLU);

    // Logits layer
    // I don't know why ReLU is placed after this layer but it just works.
    // Input shape: [batch_size, 128]
    // Output shape: [batch_size, 10]
    // No. of params: 128*10 + 10 = 1290
    net.add(new Linear(128, 10, init.get()));
    net.add(new ReLU);
    net.add(new LogSoftmax);

    std::cout << "Network setup complete" << std::endl;

    std::unique_ptr<DataLoader> loader =
        std::make_unique<DataLoader>(new Mnist(data_path, false), 128);
    std::unique_ptr<Loss> loss = std::make_unique<NLLLoss>();
    std::unique_ptr<Optimizer> optim = std::make_unique<RMSProp>(0.003, 1e-4);

    net.init(loader.get(), loss.get(), optim.get());
    net.train(30, true);

    return 0;
}