#include "common.h"
#include "maxpool.h"
#include "test_utils.h"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace nnv2;

void test_maxpool_forward() {
    int batch_size = 2;
    int in_feats = 2;
    int in_h = 4;
    int in_w = 4;

    // test max pooling with no padding
    Array input({2, 2, 4, 4}, {1, 3, 2, 1, 4, 6, 5, 1, 1, 2, 1, 3, 0, 2, 4, 1,
                               0, 1, 1, 0, 1, 0, 2, 1, 2, 3, 1, 2, 1, 0, 1, 3,
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                               5, 4, 3, 2, 4, 5, 6, 1, 3, 6, 5, 4, 2, 1, 4, 3});

    Array output({2, 2, 2, 2});
    std::vector<int> indices(2 * 2 * 2 * 2);

    maxpool_forward(&output, &input, indices, 0, 0, 2, 2, 2, 2);
    check_equal_vecs(
        output.get_vec(),
        std::vector<float>({6, 5, 2, 4, 1, 2, 3, 3, 2, 2, 2, 2, 5, 6, 6, 5}));

    // main test
    int pad_h = 1;
    int pad_w = 1;
    int kernel_h = 2;
    int kernel_w = 2;
    int stride_h = 2;
    int stride_w = 2;

    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    output.resize({batch_size, in_feats, out_h, out_w});
    indices.resize(batch_size * in_feats * out_h * out_w);

    maxpool_forward(&output, &input, indices, pad_h, pad_w, kernel_h, kernel_w,
                    stride_h, stride_w);

    // let's just print the result out :)
    for (int i = 0; i < batch_size * in_feats; i++) {
        std::cout << "Image: " << i + 1 << "/" << batch_size * in_feats
                  << std::endl;
        std::cout << "input" << std::endl;
        for (int r = 0; r < in_h; r++) {
            for (int c = 0; c < in_w; c++) {
                int idx = (i * in_h + r) * in_w + c;
                std::cout << std::setw(3) << input.get_vec()[idx] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "output" << std::endl;
        for (int r = 0; r < out_h; r++) {
            for (int c = 0; c < out_w; c++) {
                int idx = (i * out_h + r) * out_w + c;
                std::cout << std::setw(3) << output.get_vec()[idx] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "test_maxpool_forward: DIY :)" << std::endl;
}

void test_maxpool_backward() {
    int batch_size = 2;
    int in_feats = 2;
    int in_h = 4;
    int in_w = 4;

    int pad_h = 1;
    int pad_w = 1;
    int kernel_h = 2;
    int kernel_w = 2;
    int stride_h = 2;
    int stride_w = 2;

    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    Array input({2, 2, 4, 4}, {1, 3, 2, 1, 4, 6, 5, 1, 1, 2, 1, 3, 0, 2, 4, 1,
                               0, 1, 1, 0, 1, 0, 2, 1, 2, 3, 1, 2, 1, 0, 1, 3,
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                               5, 4, 3, 2, 4, 5, 6, 1, 3, 6, 5, 4, 2, 1, 4, 3});

    Array output({2, 2, 3, 3});

    Array input_grad({2, 2, 4, 4});
    Array output_grad({2, 2, 3, 3},
                      {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    std::vector<int> indices(batch_size * in_feats * out_h * out_w);

    maxpool_forward(&output, &input, indices, 1, 1, 2, 2, 2, 2);
    maxpool_backward(&input_grad, &output_grad, indices);

    for (int i = 0; i < batch_size * in_feats; i++) {
        std::cout << "Image: " << i + 1 << "/" << batch_size * in_feats
                  << std::endl;

        std::cout << "input" << std::endl;
        for (int r = 0; r < in_h; r++) {
            for (int c = 0; c < in_w; c++) {
                int idx = (i * in_h + r) * in_w + c;
                std::cout << std::setw(3) << input.get_vec()[idx] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "input grad" << std::endl;
        for (int r = 0; r < in_h; r++) {
            for (int c = 0; c < in_w; c++) {
                int idx = (i * in_h + r) * in_w + c;
                std::cout << std::setw(3) << input_grad.get_vec()[idx] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
    }
    std::cout << "test_maxpool_back: DIY :)" << std::endl;
}

int main() {
    test_maxpool_forward();
    test_maxpool_backward();
}