#include "common.cuh"
#include "conv.cuh"
#include "test_utils.cuh"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace nnv2;

void test_im2col() {
  int batch_size = 1;
  int in_feats = 2;
  int in_w = 5;
  int in_h = 5;

  int kernel_w = 3;
  int kernel_h = 3;

  int pad_h = 1;
  int pad_w = 1;

  int stride_h = 1;
  int stride_w = 1;

  int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
  int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

  Array im({batch_size, in_feats, in_h, in_w},
           {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49});
  Array cols({batch_size, in_feats * kernel_h * kernel_w, out_h * out_w});

  im2col(&im, &cols, pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w);
  check_equal_vecs(
      cols.get_vec(),
      {0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  0,  5,  6,  7,  8,  0,  10, 11,
       12, 13, 0,  15, 16, 17, 18, 0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,
       6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0,  0,  0,  0,
       0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0,  11, 12, 13, 14, 0,  16, 17,
       18, 19, 0,  0,  0,  1,  2,  3,  0,  5,  6,  7,  8,  0,  10, 11, 12, 13,
       0,  15, 16, 17, 18, 0,  20, 21, 22, 23, 0,  1,  2,  3,  4,  5,  6,  7,
       8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1,
       2,  3,  4,  0,  6,  7,  8,  9,  0,  11, 12, 13, 14, 0,  16, 17, 18, 19,
       0,  21, 22, 23, 24, 0,  0,  5,  6,  7,  8,  0,  10, 11, 12, 13, 0,  15,
       16, 17, 18, 0,  20, 21, 22, 23, 0,  0,  0,  0,  0,  5,  6,  7,  8,  9,
       10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,  0,  0,
       0,  0,  6,  7,  8,  9,  0,  11, 12, 13, 14, 0,  16, 17, 18, 19, 0,  21,
       22, 23, 24, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  25, 26, 27,
       28, 0,  30, 31, 32, 33, 0,  35, 36, 37, 38, 0,  40, 41, 42, 43, 0,  0,
       0,  0,  0,  25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
       40, 41, 42, 43, 44, 0,  0,  0,  0,  0,  26, 27, 28, 29, 0,  31, 32, 33,
       34, 0,  36, 37, 38, 39, 0,  41, 42, 43, 44, 0,  0,  25, 26, 27, 28, 0,
       30, 31, 32, 33, 0,  35, 36, 37, 38, 0,  40, 41, 42, 43, 0,  45, 46, 47,
       48, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
       42, 43, 44, 45, 46, 47, 48, 49, 26, 27, 28, 29, 0,  31, 32, 33, 34, 0,
       36, 37, 38, 39, 0,  41, 42, 43, 44, 0,  46, 47, 48, 49, 0,  0,  30, 31,
       32, 33, 0,  35, 36, 37, 38, 0,  40, 41, 42, 43, 0,  45, 46, 47, 48, 0,
       0,  0,  0,  0,  30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
       44, 45, 46, 47, 48, 49, 0,  0,  0,  0,  0,  31, 32, 33, 34, 0,  36, 37,
       38, 39, 0,  41, 42, 43, 44, 0,  46, 47, 48, 49, 0,  0,  0,  0,  0,  0});

  std::cout << "test_im2col: Passed" << std::endl;
}

void test_col2im() {
  int batch_size = 1;
  int in_feats = 1;
  int in_w = 3;
  int in_h = 3;

  int kernel_w = 3;
  int kernel_h = 3;

  int pad_h = 1;
  int pad_w = 1;

  int stride_h = 1;
  int stride_w = 1;

  int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
  int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

  Array im({batch_size, in_feats, in_h, in_w});
  Array cols({batch_size, in_feats * kernel_h * kernel_w, out_h * out_w},
             {0, 0, 0, 0, 1, 2, 0, 4, 5, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0,
              2, 3, 0, 5, 6, 0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 1, 2, 3, 4, 5, 6,
              7, 8, 9, 2, 3, 0, 5, 6, 0, 8, 9, 0, 0, 4, 5, 0, 7, 8, 0, 0, 0,
              4, 5, 6, 7, 8, 9, 0, 0, 0, 5, 6, 0, 8, 9, 0, 0, 0, 0});

  col2im(&cols, &im, pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w);
  check_equal_vecs(im.get_vec(), {4, 12, 12, 24, 45, 36, 28, 48, 36});

  std::cout << "test_col2im: Passed" << std::endl;
}

void test_conv_forward() {
  int batch_size = 1;
  int in_feats = 2;
  int out_feats = 3;

  int in_h = 5;
  int in_w = 5;
  int kernel_w = 3;
  int kernel_h = 3;
  int pad_h = 1;
  int pad_w = 1;
  int stride_h = 1;
  int stride_w = 1;

  int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
  int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

  Array input({batch_size, in_feats, in_h, in_w},
              {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,
               1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
               14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  Array kernel({out_feats, in_feats, kernel_h, kernel_w},
               {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8});

  Array cols({batch_size, in_feats * kernel_h * kernel_w, out_h * out_w});
  Array output({batch_size, out_feats, out_h, out_w});

  conv_forward(&output, &input, &cols, &kernel, pad_h, pad_w, stride_h,
               stride_w);

  check_equal_vecs(
      output.get_vec(),
      {176,  284,  350,  416,  272,  420,  624, 696, 768, 480, 690,  984,  1056,
       1128, 690,  960,  1344, 1416, 1488, 900, 464, 608, 638, 668,  368,  176,
       284,  350,  416,  272,  420,  624,  696, 768, 480, 690, 984,  1056, 1128,
       690,  960,  1344, 1416, 1488, 900,  464, 608, 638, 668, 368,  176,  284,
       350,  416,  272,  420,  624,  696,  768, 480, 690, 984, 1056, 1128, 690,
       960,  1344, 1416, 1488, 900,  464,  608, 638, 668, 368});

  std::cout << "test_conv_forward: Passed" << std::endl;
}

void test_conv_backward() {
  int batch_size = 1;
  int in_feats = 2;
  int out_feats = 3;

  int in_h = 5;
  int in_w = 5;
  int kernel_w = 3;
  int kernel_h = 3;
  int pad_h = 0;
  int pad_w = 0;
  int stride_h = 1;
  int stride_w = 1;

  int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
  int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

  Array input({batch_size, in_feats, in_h, in_w},
              {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,
               1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
               14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  Array kernel({out_feats, in_feats, kernel_h, kernel_w},
               {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8});

  Array cols({batch_size, in_feats * kernel_h * kernel_w, out_h * out_w});
  im2col(&input, &cols, pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w);

  Array input_grad({batch_size, in_feats, in_h, in_w});
  Array kernel_grad({out_feats, in_feats, out_h, out_w});

  Array output_grad({batch_size, out_feats, out_h, out_w},
                    {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
                     5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8});
  ArrayMap cache;

  conv_backward(&input_grad, &kernel_grad, &output_grad, &input, &kernel, &cols,
                pad_h, pad_w, stride_h, stride_w, cache);

  check_equal_vecs(input_grad.get_vec(),
                   {0,   0,   3,   12,  12,  0,   18,  60,  78,  60,
                    27,  108, 252, 252, 171, 108, 270, 492, 402, 240,
                    108, 252, 435, 336, 192, 0,   0,   3,   12,  12,
                    0,   18,  60,  78,  60,  27,  108, 252, 252, 171,
                    108, 270, 492, 402, 240, 108, 252, 435, 336, 192});

  check_equal_vecs(kernel_grad.get_vec(),
                   {312, 348, 384, 492, 528, 564, 672, 708, 744, 312, 348,
                    384, 492, 528, 564, 672, 708, 744, 312, 348, 384, 492,
                    528, 564, 672, 708, 744, 312, 348, 384, 492, 528, 564,
                    672, 708, 744, 312, 348, 384, 492, 528, 564, 672, 708,
                    744, 312, 348, 384, 492, 528, 564, 672, 708, 744});

  std::cout << "test_conv_backward: Passed" << std::endl;
}

void test_conv_forward_bias() {
  Array output({1, 2, 3, 3},
               {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Array bias({1, 2}, {1, 2});

  conv_forward_bias(&output, &bias);
  check_equal_vecs(output.get_vec(), {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14,
                                      15, 16, 17, 18, 19});

  std::cout << "test_conv_forward_bias: Passed" << std::endl;
}

void test_conv_backward_bias() {
  Array output_grad({2, 2, 3, 3},
                    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                     12, 13, 14, 15, 16, 17, 0,  1,  2,  3,  4,  5,
                     6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17});
  Array bias_grad({1, 2});
  ArrayMap cache;

  conv_backward_bias(&bias_grad, &output_grad, cache);
  check_equal_vecs(bias_grad.get_vec(), {72, 234});

  std::cout << "test_conv_backward_bias: Passed" << std::endl;
}

int main() {
  test_im2col();
  test_col2im();
  test_conv_forward();
  test_conv_backward();
  test_conv_forward_bias();
  test_conv_backward_bias();
}