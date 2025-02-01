#include "src/turbomind/api/python/linear.h"
#include "src/turbomind/utils/tensor.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>

namespace turbomind {

class LinearTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化CUDA
        cudaStreamCreate(&stream_);
        
        // 创建随机数生成器
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // 初始化输入数据
        input_data_.resize(batch_size * input_dims);
        for (auto& val : input_data_) {
            val = __float2half(dist(gen));
        }

        // 初始化输出数据
        output_data_.resize(batch_size * output_dims);
        
        // 创建输入输出Tensor
        input_tensor_ = Tensor(MEMORY_GPU, TYPE_FP16, {batch_size, input_dims}, input_data_.data());
        output_tensor_ = Tensor(MEMORY_GPU, TYPE_FP16, {batch_size, output_dims}, output_data_.data());

        // 分配GPU内存
        cudaMalloc(&d_input_data_, input_data_.size() * sizeof(half));
        cudaMalloc(&d_output_data_, output_data_.size() * sizeof(half));
        
        // 拷贝数据到GPU
        cudaMemcpy(d_input_data_, input_data_.data(), input_data_.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_data_, output_data_.data(), output_data_.size() * sizeof(half), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_input_data_);
        cudaFree(d_output_data_);
        cudaStreamDestroy(stream_);
    }

    const size_t batch_size = 16;
    const size_t input_dims = 128;
    const size_t output_dims = 256;
    const int w_bit = 4;
    const int group_size = 64;

    cudaStream_t stream_;
    std::vector<half> input_data_;
    std::vector<half> output_data_;
    half* d_input_data_;
    half* d_output_data_;
    
    Tensor input_tensor_;
    Tensor output_tensor_;
};

TEST_F(LinearTest, ForwardPass) {
    // 初始化输入输出数据
    std::vector<half> input_data(input_dims * batch_size);
    std::vector<half> output_data(output_dims * batch_size);

    // 创建输入输出Tensor
    Tensor input_tensor(MEMORY_GPU, TYPE_FP16, {batch_size, input_dims}, input_data.data());
    Tensor output_tensor(MEMORY_GPU, TYPE_FP16, {batch_size, output_dims}, output_data.data());

    // 创建权重Tensor
    std::vector<uint8_t> qweight_data(input_dims * output_dims / 2);
    std::vector<half> scales_data(input_dims / group_size * output_dims);
    std::vector<half> qzeros_data(input_dims / group_size * output_dims);

    // 初始化权重数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (auto& val : qweight_data) {
        val = dist(gen);
    }
    for (auto& val : scales_data) {
        val = __float2half(1.0f);  // 初始化为 1.0
    }
    for (auto& val : qzeros_data) {
        val = __float2half(0.0f);  // 初始化为 0.0
    }

    Tensor qweight_tensor_obj(MEMORY_GPU, TYPE_UINT8, {input_dims, output_dims / 2}, qweight_data.data());
    auto qweight_tensor = std::make_shared<Tensor>(qweight_tensor_obj);
    Tensor scales_tensor(MEMORY_GPU, TYPE_FP16, {input_dims / group_size, output_dims}, scales_data.data());
    Tensor qzeros_tensor(MEMORY_GPU, TYPE_FP16, {input_dims / group_size, output_dims}, qzeros_data.data());

    // 创建Linear对象
    Linear linear(input_dims, output_dims, w_bit, group_size);

    // 初始化Linear对象
    linear.post_init(qweight_tensor, scales_tensor, qzeros_tensor, false);

    // 执行forward操作
    linear.forward(input_tensor, output_tensor, 0);

    // 检查输出数据是否合理
    bool all_zero = true;
    for (const auto& val : output_data) {
        if (__half2float(val) != 0.0f) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "Output data should not be all zeros";
}

}  // namespace turbomind