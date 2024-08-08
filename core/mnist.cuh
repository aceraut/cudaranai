#pragma once

#include "dataset.cuh"

#include <string>
#include <vector>

namespace nnv2 {

class Mnist : public Dataset {
public:
    Mnist(std::string data_path, bool preprocess = false);

private:
    void read_images(std::vector<std::vector<float>> &output,
                     std::string filename) override;
    void read_labels(std::vector<unsigned char> &output,
                     std::string filename) override;
    void normalize(std::vector<std::vector<float>> &images);
};

} // namespace nnv2