#pragma once

#include "dataset.cuh"

#include <string>
#include <vector>

namespace nnv2 {

class Mnist : public Dataset {
public:
    explicit Mnist(std::string data_path);

private:
    void read_images(std::vector<std::vector<float>> &output,
                     std::string filename) override;
    void read_labels(std::vector<unsigned char> &output,
                     std::string filename) override;
};

} // namespace nnv2