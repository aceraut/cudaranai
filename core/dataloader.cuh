#pragma once

#include "dataset.h"
#include "layer.h"

#include <memory>
#include <utility>

namespace nnv2 {

class DataLoader : public Layer {
public:
    explicit DataLoader(Dataset *dataset, int batch_size)
        : dataset(dataset), batch_size(batch_size), train_data_offset(0),
          test_data_offset(0) {}

    // load a single batch of images (assigned to `output`) and their labels
    // in one-hot encoding (assigned to `output_labels`)
    int load_train_batch();
    int load_test_batch();

    void reset(bool shuffle);

    // check if there's any batch left in train/test data
    bool has_next_train_batch();
    bool has_next_test_batch();

    Array *get_labels() { return output_labels.get(); }

private:
    // the dataset
    std::unique_ptr<Dataset> dataset;

    // number of images in a batch
    int batch_size;

    // current offset in the train data in the next batch process
    int train_data_offset;
    // current offset in the test data in the next batch process
    int test_data_offset;

    // extra container to store the label batch
    std::unique_ptr<Array> output_labels;
};

} // namespace nnv2