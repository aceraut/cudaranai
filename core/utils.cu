#include "common.cuh"

#include <memory>
#include <string>
#include <vector>

namespace nnv2 {
namespace utils {

// Helper function for initializing Array object inside smart pointer
void set_array_ptr(std::unique_ptr<Array> &ptr, const std::vector<int> &shape) {
    if (ptr.get() == nullptr) {
        ptr.reset(new Array(shape));
    } else {
        if (ptr->get_shape() != shape) {
            ptr->resize(shape);
        }
        ptr->zero();
    }
}

// Helper function for initializing Array object from an ArrayMap cache
void set_array_cache(ArrayMap &map, std::string key,
                     const std::vector<int> &shape) {
    if (map.find(key) == map.end()) {
        map[key] = std::make_unique<Array>(shape);
    }
    set_array_ptr(map[key], shape);
}

// Calculates rounded up decimal quotient of two integers
int div_ceil(int a, int b) {
    return a / b + (int)(a % b != 0);
}

} // namespace utils
} // namespace nnv2