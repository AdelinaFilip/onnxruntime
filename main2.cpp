#include <iostream>
#include <algorithm>
#include "include/onnxruntime/core/session/onnxruntime_cxx_api.h"
// #include "include/onnxruntime/core/session/onnxruntime_cxx_inline.h"

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "onnxruntime.lib")


Ort::Env env{ORT_LOGGING_LEVEL_WARNING, ""};

struct MNIST {
    MNIST() {
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
    }

    int Run() {
        const char* input_names[] = {"conv2d_input"};
        const char* output_names[] = {"dense_1"};

        session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);

        result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
        return result_;
    }

    static constexpr const int widSessionOptionsth_ = 28;
    static constexpr const int height_ = 28;

    std::array<float, 28 * height_> input_image_{};
    std::array<float, 10> results_{};
    int result_{0};

private:

    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 4> input_shape_{1, height_, 28, 1};

    Ort::Value output_tensor_{nullptr};
    std::array<int64_t, 2> output_shape_{1, 10};
};

MNIST mnist_;

// We need to convert the true-color data in the DIB into the model's floating point format
// TODO: (also scales down the image and smooths the values, but this is not working properly)
void ConvertDibToMnist() {
    float* output = mnist_.input_image_.data();
    std::fill(mnist_.input_image_.begin(), mnist_.input_image_.end(), 0.f);
    output += 28;
}


int main() {
    ConvertDibToMnist();
    std::cout << mnist_.Run() << std::endl;
    return 0;
}
