#include "src/turbomind/api/python/linear.h"
#include "src/turbomind/utils/tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

using namespace turbomind;

void test_linear_fp32() {
    size_t input_dims = 256;
    size_t output_dims = 512;
    int w_bit = 4;
    int group_size = 128;

    // step1: initialization
    Linear linear(input_dims, output_dims, w_bit, group_size);

    // step2: Prepare input
    turbomind::Tensor input;

    std::vector<size_t> shape = {2, 3};
    size_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    size_t bytes = num_elements * sizeof(float);

    float* gpu_data = nullptr;
    cudaError_t err = cudaMalloc(&gpu_data, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory");
    }
    turbomind::Tensor gpu_tensor(turbomind::MEMORY_GPU, turbomind::TYPE_FP16, shape, gpu_data);

    Tensor output({1, output_dims}, DataType::TYPE_FP16);

    // 初始化输入数据
    float* h_input = (float*)input.data();
    for (size_t i = 0; i < input_dims; ++i) {
        h_input[i] = 1.0f;
    }

    // 创建量化权重
    auto qweight = std::make_shared<turbomind::Tensor>();
    qweight->type = turbomind::TYPE_UINT8;
    qweight->shape = {input_dims, output_dims / 2};  // 4bit量化
    cudaMalloc(&qweight->data, input_dims * output_dims / 2);

    // 创建scales和qzeros
    turbomind::Tensor scales, qzeros;
    scales.type = turbomind::TYPE_FP16;
    qzeros.type = turbomind::TYPE_FP16;
    size_t scale_count = input_dims / group_size * output_dims;
    scales.shape = {scale_count};
    qzeros.shape = {scale_count};
    cudaMalloc(&scales.data, scale_count * sizeof(__half));
    cudaMalloc(&qzeros.data, scale_count * sizeof(__half));

    // step3: 初始化后处理
    linear.post_init(qweight, scales, qzeros, false);

    // step4: forward
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    linear.forward(input, output, stream);

    // 同步并检查结果
    cudaStreamSynchronize(stream);
    float* h_output = (float*)output.data();
    std::cout << "Linear output: " << h_output[0] << std::endl;

    cudaStreamDestroy(stream);
}

int main() {
    try {
        test_linear_fp32();
        std::cout << "Linear test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}