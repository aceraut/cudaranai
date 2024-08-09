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

    net.add(new Conv2D(1, 32, 28, 28, 0, 0, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));
    net.add(new Conv2D(32, 64, 12, 12, 0, 0, 5, 5, 1, 1, init.get()));
    net.add(new ReLU);
    net.add(new MaxPool2D(0, 0, 2, 2, 2, 2));
    net.add(new Conv2D(64, 128, 4, 4, 0, 0, 3, 3, 1, 1, init.get()));
    net.add(new ReLU);
    net.add(new Flatten);
    net.add(new Linear(512, 128, init.get()));
    net.add(new ReLU);
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