#pragma once

#include "dataset.cuh"

#include <string>
#include <vector>

namespace nnv2 {

class Mnist : public Dataset {
public:
    Mnist(std::string data_path, bool prep = false);

private:
    void read_images(std::vector<std::vector<float>> &output,
                     std::string filename) override;
    void read_labels(std::vector<unsigned char> &output,
                     std::string filename) override;
    void preprocess_images(std::vector<std::vector<float>> &data);
};

} // namespace nnv2