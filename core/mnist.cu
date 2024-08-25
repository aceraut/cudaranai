// This file implements the Mnist dataset adapter.
//
// It provides functions for MNIST dataset import and preprocesing.
// You can check Wikipedia for more information about MNIST dataset, best I know
// so far from this thing is that there are 60000 samples for training and
// 10000 more for testing.

#include "common.cuh"
#include "mnist.cuh"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace nnv2 {

Mnist::Mnist(std::string data_path, bool standard, bool rescaled)
    : Dataset(data_path, 28, 28, 10) {
    // Read train data
    read_images(train_images, data_path + "/train-images-idx3-ubyte");
    read_labels(train_labels, data_path + "/train-labels-idx1-ubyte");

    // Read test data
    read_images(test_images, data_path + "/t10k-images-idx3-ubyte");
    read_labels(test_labels, data_path + "/t10k-labels-idx1-ubyte");

    if (rescaled) {
        rescale_images(train_images);
        rescale_images(test_images);
    }

    if (standard) {
        standardize_images(train_images);
        standardize_images(test_images);
    }
}

static unsigned reverse_int(unsigned i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;

    return ((unsigned)ch1 << 24) + ((unsigned)ch2 << 16) +
           ((unsigned)ch3 << 8) + (unsigned)ch4;
}

void Mnist::read_images(std::vector<std::vector<float>> &output,
                        std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        unsigned magic_number = 0;
        unsigned n_images = 0;
        unsigned n_rows = 0;
        unsigned n_cols = 0;

        // Read metadata
        file.read((char *)&magic_number, sizeof(magic_number));
        file.read((char *)&n_images, sizeof(n_images));
        file.read((char *)&n_rows, sizeof(n_rows));
        file.read((char *)&n_cols, sizeof(n_cols));
        magic_number = reverse_int(magic_number);
        n_images = reverse_int(n_images);
        n_rows = reverse_int(n_rows);
        n_cols = reverse_int(n_cols);

        std::cout << filename << ", ";
        std::cout << "magic number = " << magic_number << ", ";
        std::cout << "number of images = " << n_images << ", ";
        std::cout << "rows = " << n_rows << ", ";
        std::cout << "cols = " << n_cols << std::endl;

        CHECK_EQ(h, n_rows, "Mnist::read_images: row length must be 28");
        CHECK_EQ(w, n_cols, "Mnist::read_images: column length must be 28");

        // Read images
        std::vector<unsigned char> image(h * w);
        std::vector<float> scaled_image(h * w);

        for (int i = 0; i < n_images; i++) {
            file.read((char *)image.data(), sizeof(unsigned char) * h * w);

            for (int k = 0; k < h * w; k++) {
                scaled_image[k] = image[k];
            }
            output.push_back(scaled_image);
        }
    } else {
        std::cerr << filename << " not found" << std::endl;
        exit(1);
    }
}

void Mnist::read_labels(std::vector<unsigned char> &output,
                        std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        unsigned magic_number = 0;
        unsigned n_images = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        file.read((char *)&n_images, sizeof(n_images));
        magic_number = reverse_int(magic_number);
        n_images = reverse_int(n_images);

        std::cout << filename << ", ";
        std::cout << "magic number = " << magic_number << ", ";
        std::cout << "number of images = " << n_images << std::endl;

        for (int i = 0; i < n_images; i++) {
            unsigned char label = 0;
            file.read((char *)&label, sizeof(label));
            output.push_back(label);
        }
    } else {
        std::cerr << filename << " not found" << std::endl;
        exit(1);
    }
}

void Mnist::rescale_images(std::vector<std::vector<float>> &images) {
    for (std::vector<float> &im : images) {
        std::transform(im.begin(), im.end(), im.begin(),
                       [](float x) { return x / 255.0; });
    }
}

void Mnist::standardize_images(std::vector<std::vector<float>> &images) {
    // Calculate mean and standard deviation of all pixels in dataset
    int n_pixels = images.size() * h * w;
    float mean = 0.0;
    float stddev = 0.0;

    for (const std::vector<float> &im : images) {
        mean += std::accumulate(im.begin(), im.end(), 0.0);
    }
    mean /= n_pixels;

    for (const std::vector<float> &im : images) {
        stddev +=
            std::accumulate(im.begin(), im.end(), 0.0, [&](float s, float x) {
                return s + (x - mean) * (x - mean);
            });
    }
    stddev = sqrtf(stddev / n_pixels);

    // Normalize images
    for (std::vector<float> &im : images) {
        std::transform(im.begin(), im.end(), im.begin(),
                       [&](float x) { return (x - mean) / stddev; });
    }
}

} // namespace nnv2