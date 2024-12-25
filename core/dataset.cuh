#pragma once

#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>

namespace nnv2 {

class Dataset {
public:
  Dataset(std::string data_path, int h, int w, int label_count)
      : data_path(data_path), h(h), w(w), label_count(label_count) {}

  virtual ~Dataset() {}

  void shuffle_train_data() {
    unsigned int seed =
        std::chrono::system_clock::now().time_since_epoch().count() % 1237;

    std::shuffle(train_images.begin(), train_images.end(),
                 std::default_random_engine(seed));
    std::shuffle(train_labels.begin(), train_labels.end(),
                 std::default_random_engine(seed));
  }

  int get_image_height() const { return h; }
  int get_image_width() const { return w; }
  int get_label_count() const { return label_count; }

  const std::vector<std::vector<float>> &get_train_images() const {
    return train_images;
  }
  const std::vector<std::vector<float>> &get_test_images() const {
    return test_images;
  }

  const std::vector<unsigned char> &get_train_labels() const {
    return train_labels;
  }
  const std::vector<unsigned char> &get_test_labels() const {
    return test_labels;
  }

protected:
  virtual void read_images(std::vector<std::vector<float>> &data,
                           std::string filename) = 0;
  virtual void read_labels(std::vector<unsigned char> &data,
                           std::string filename) = 0;

  std::string data_path;

  std::vector<std::vector<float>> train_images;
  std::vector<std::vector<float>> test_images;

  std::vector<unsigned char> train_labels;
  std::vector<unsigned char> test_labels;

  int h;
  int w;
  int label_count;
};

} // namespace nnv2