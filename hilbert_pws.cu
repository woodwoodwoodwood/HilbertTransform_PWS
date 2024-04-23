#include <iostream> // Standard I/O
#include <fstream>  // For file writing
#include <cmath>    // For the sin() function
#include <cufft.h>
#include <cuda_runtime.h>
using namespace std;

#ifndef __CUDA_CALL
#define __CUDA_CALL(call)                                                                                                            \
    do                                                                                                                               \
    {                                                                                                                                \
        cudaError_t cuda_error = call;                                                                                               \
        if (cuda_error != cudaSuccess)                                                                                               \
        {                                                                                                                            \
            std::cerr << "CUDA Error: " << cudaGetErrorString(cuda_error) << ", " << __FILE__ << ", line " << __LINE__ << std::endl; \
            return -1;                                                                                                               \
        }                                                                                                                            \
    } while (0)
#endif

const float PI = 3.141592653589793f;

    const int num_signals = 2; // 信号数量
    int signal_length = 16; // 每个信号的长度  

const int LENGTH = 1024,
              CPX_LEN = signal_length / 2 + 1,
              B_SIZE = 256,
              G_SIZE = (num_signals * signal_length + B_SIZE - 1) / B_SIZE;

__global__ void Print_Complex(cufftComplex *dev_complex, const int LENGTH)
{
    printf("on GPU: \n");
    for (int i = 0; i < LENGTH; i++)
        printf("Real:%f, Img:%f\n", dev_complex[i].x, dev_complex[i].y);
}

__global__ void Print_Float(float *num_float, int length)
{
    printf("on GPU: \n");
    if(length > 100)
        length = 100;
    for (int i = 0; i < length; i++)
        printf("float:%f\n", num_float[i]);
}

__global__ void Divide(cufftComplex *dev_complex, const int length, const int num_signals)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < length * num_signals)
    {
        int signal_idx = tid / length;
        int data_idx = tid % length;
        dev_complex[tid].x /= length;
        dev_complex[tid].y /= length;
    }
}

__global__ void Hilbert(cufftComplex *dev_complex, const int CPX_LEN, const int LENGTH, const int num_signals)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x + 1, // Skips the first element (DC component)
        offset = gridDim.x * blockDim.x;

    while (tid < LENGTH * num_signals)
    {
        int signal_idx = tid / LENGTH;
        int data_idx = tid % LENGTH;
        if (data_idx != 0 && data_idx < CPX_LEN - 1)
        {
            dev_complex[tid].x *= 2;
            dev_complex[tid].y *= 2;
        }
        if (data_idx > CPX_LEN - 1)
        {
            dev_complex[tid].x = 0.0f;
            dev_complex[tid].y = 0.0f;
        }
        tid += offset;
    }
}

int HibertTransform(cufftComplex *dev_complex, const int LENGTH, const int num_signals, dim3 G_SIZE, dim3 B_SIZE, cufftHandle c2c_plan)
{
    const int CPX_LEN = LENGTH / 2 + 1;

    cufftExecC2C(c2c_plan, dev_complex, dev_complex, CUFFT_FORWARD); // Forward FFT in-place
    __CUDA_CALL(cudaDeviceSynchronize());

    Hilbert<<<G_SIZE, B_SIZE>>>(dev_complex, CPX_LEN, LENGTH, num_signals);
    __CUDA_CALL(cudaDeviceSynchronize());

    cufftExecC2C(c2c_plan, dev_complex, dev_complex, CUFFT_INVERSE); // Inverse FFT in-place
    __CUDA_CALL(cudaDeviceSynchronize());

    Divide<<<G_SIZE, B_SIZE>>>(dev_complex, LENGTH, num_signals);
    __CUDA_CALL(cudaDeviceSynchronize());
}


// ----------------------------- pws stack -----------------------------
// calculate the mean of the hibert_complex
void calculateMeanReal(cufftComplex *hilbert_complex, size_t length, int num_signals, float *mean_amp) {
    for (int i = 0; i < length; ++i) {
        float sum_real = 0.0f;
        // 计算当前信号的实部总和
        for (size_t j = 0; j < num_signals; ++j) {
            sum_real += hilbert_complex[j * length + i].x;
        }
        // 计算当前信号的均值并存储在给定数组中
        mean_amp[i] = sum_real / num_signals;
    }
}

void calculateMeanReal_abs(cufftComplex *hilbert_complex, size_t length, int num_signals, float *mean_amp) {
    for (int i = 0; i < length; ++i) {
        float sum_real = 0.0f;
        // 计算当前信号的实部总和
        for (size_t j = 0; j < num_signals; ++j) {
            sum_real += abs(hilbert_complex[j * length + i].x);
        }
        // 计算当前信号的均值并存储在给定数组中
        mean_amp[i] = sum_real / num_signals;
    }
}

void calculateMeanImg(cufftComplex *hilbert_complex, size_t length, int num_signals, float *mean_img) {
    for (int i = 0; i < length; ++i) {
        float sum_img = 0.0f;
        // 计算当前信号的实部总和
        for (size_t j = 0; j < num_signals; ++j) {
            sum_img += hilbert_complex[j * length + i].y;
        }
        // 计算当前信号的均值并存储在给定数组中
        mean_img[i] = sum_img / num_signals;
    }
}

void calculateMean(cufftComplex *hilbert_complex, size_t length, int num_signals, cufftComplex *mean) {
    for (int i = 0; i < length; ++i) {
        float sum_real = 0.0f, sum_img = 0.0f;
        // 计算当前信号的实部总和
        for (size_t j = 0; j < num_signals; ++j) {
            sum_real += hilbert_complex[j * length + i].x;
            sum_img += hilbert_complex[j * length + i].y;
        }
        // 计算当前信号的均值并存储在给定数组中
        mean[i].x = sum_real / num_signals;
        mean[i].y = sum_img / num_signals;
    }
}

void calculateABS(cufftComplex *hilbert_complex, size_t length, int num_signals, float *abs) {
    for (int i = 0; i < num_signals * length; ++i) {
        abs[i] = sqrt(hilbert_complex[i].x * hilbert_complex[i].x + hilbert_complex[i].y * hilbert_complex[i].y);
    }
}

void calculateDivideMean(cufftComplex *hilbert_complex, size_t length, int num_signals, float *abs, cufftComplex *divide_mean) {
    for (int i = 0; i < num_signals * length; ++i) {
        hilbert_complex[i].x /= abs[i];
        hilbert_complex[i].y /= abs[i];
    }
    calculateMean(hilbert_complex, length, num_signals, divide_mean);
    
}

void calculateMultiply(float *num1, float *num2, float *res, size_t length) {
    for (int i = 0; i < length; ++i) {
        res[i] = num1[i] * num2[i];
    }
}

void normalize(float *num, size_t length) {
    float maxVal = 0;
    for (size_t i = 0; i < length; ++i) {
        if (num[i] > maxVal) {
            maxVal = num[i];
        }
    }

    if (maxVal == 0) return;

    // Step 2: Divide each element by the maximum value to normalize
    for (size_t i = 0; i < length; ++i) {
        num[i] /= maxVal;
    }
}


__global__ void cudaMeanReal(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_amp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        float sum_real = 0.0f;
        // 计算当前时间点上所有信号的实部总和
        for (int j = 0; j < num_signals; ++j) {
            sum_real += hilbert_complex[j * length + tid].x;
        }
        // 计算平均值
        mean_amp[tid] = sum_real / num_signals;
    }
}

__global__ void cudaMeanReal_abs(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_amp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        float sum_real = 0.0f;
        // 计算当前时间点上所有信号的实部总和
        for (int j = 0; j < num_signals; ++j) {
            sum_real += abs(hilbert_complex[j * length + tid].x);
        }
        // 计算平均值
        mean_amp[tid] = sum_real / num_signals;
    }
}

__global__ void cudaMeanImg(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_img) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        float sum_img = 0.0f;
        // 计算当前时间点上所有信号的虚部总和
        for (int j = 0; j < num_signals; ++j) {
            sum_img += hilbert_complex[j * length + tid].y;
        }
        // 计算平均值
        mean_img[tid] = sum_img / num_signals;
    }
}

__global__ void cudaMean(cufftComplex *hilbert_complex, int length, int num_signals, cufftComplex *mean) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        float sum_real = 0.0f, sum_img = 0.0f;
        // 计算当前时间点上所有信号的实部总和
        for (int j = 0; j < num_signals; ++j) {
            sum_real += hilbert_complex[j * length + tid].x;
            sum_img += hilbert_complex[j * length + tid].y;
        }
        // 计算平均值
        mean[tid].x = sum_real / num_signals;
        mean[tid].y = sum_img / num_signals;
    }
}

__global__ void cudaABS(cufftComplex *hilbert_complex, int length, int num_signals, float *abs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_signals * length) {
        abs[tid] = sqrt(hilbert_complex[tid].x * hilbert_complex[tid].x + hilbert_complex[tid].y * hilbert_complex[tid].y);
    }
}

__global__ void cudaDivide(cufftComplex *hilbert_complex, int length, int num_signals, float *abs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_signals * length) {
        hilbert_complex[tid].x /= abs[tid];
        hilbert_complex[tid].y /= abs[tid];
    }
}

__global__ void cudaMultiply(float *num1, float *num2, float *res, int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        res[tid] = num1[tid] * num2[tid];
    }
}

__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*) address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__global__ void findMaxValKernel(float *num, float *maxVal, int length) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < length) {
        atomicMaxFloat(maxVal, num[tid]);
    }
}


__global__ void cudaNormalize(float *num, int length, float *maxVal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        num[tid] /= *maxVal;
    }
}


void pws_stack(cufftComplex *hilbert_complex, const int length, const int num_signals, float *mean_amp, float *weighted_amp) {
    // mean_amp: 1024
    calculateMeanReal(hilbert_complex, length, num_signals, mean_amp);
    
    // abs_hilb_amp: 2 * 1024
    float *abs_hilb_amp = (float *)malloc(num_signals * length * sizeof(float));
    calculateABS(hilbert_complex, length, num_signals, abs_hilb_amp);

    // divede_mean: 1024
    cufftComplex *divide_mean = (cufftComplex *)malloc(length * sizeof(cufftComplex));
    calculateDivideMean(hilbert_complex, length, num_signals, abs_hilb_amp, divide_mean);

    // weight: 1024
    float *weight = (float *)malloc(length * sizeof(float));
    calculateABS(divide_mean, length, 1, weight);

    // weighted_amp: 1024
    calculateMultiply(mean_amp, weight, weighted_amp, length);

    // normalize mean_amp, weighted_amp
    normalize(mean_amp, length);
    normalize(weighted_amp, length);

    // free memory
    free(abs_hilb_amp);
    free(divide_mean);
    free(weight);

}

void cudaPwsStack(cufftComplex *hilbert_complex, int length, int num_signals, float *mean_amp, float *weighted_amp, dim3 G_SIZE, dim3 B_SIZE) {
    // mean_amp: 1024
    cudaMeanReal<<<G_SIZE, B_SIZE>>>(hilbert_complex, length, num_signals, mean_amp);
    cudaDeviceSynchronize();

    // abs_hilb_amp: 2 * 1024
    float *abs_hilb_amp;
    cudaMalloc(&abs_hilb_amp, num_signals * length * sizeof(float));
    cudaABS<<<G_SIZE, B_SIZE>>>(hilbert_complex, length, num_signals, abs_hilb_amp);
    cudaDeviceSynchronize();

    // divede_mean: 1024
    cufftComplex *divide, *divide_mean;
    cudaMalloc(&divide, num_signals * length * sizeof(cufftComplex));
    cudaMalloc(&divide_mean, length * sizeof(cufftComplex));
    cudaDivide<<<G_SIZE, B_SIZE>>>(hilbert_complex, length, num_signals, abs_hilb_amp);
    cudaDeviceSynchronize();
    cudaMean<<<G_SIZE, B_SIZE>>>(hilbert_complex, length, num_signals, divide_mean);
    cudaDeviceSynchronize();

    // weight: 1024
    float *weight;
    cudaMalloc(&weight, length * sizeof(float));
    cudaABS<<<G_SIZE, B_SIZE>>>(divide_mean, length, 1, weight);
    cudaDeviceSynchronize();

    // weighted_amp: 1024
    cudaMultiply<<<G_SIZE, B_SIZE>>>(mean_amp, weight, weighted_amp, length);
    cudaDeviceSynchronize();

    // normalize mean_amp, weighted_amp
    float *mean_amp_maxVal, *weighted_amp_maxVal;
    cudaMalloc(&mean_amp_maxVal, sizeof(float));
    cudaMalloc(&weighted_amp_maxVal, sizeof(float));
    findMaxValKernel<<<G_SIZE, B_SIZE>>>(mean_amp, mean_amp_maxVal, length);
    findMaxValKernel<<<G_SIZE, B_SIZE>>>(weighted_amp, weighted_amp_maxVal, length);
    cudaDeviceSynchronize();

    cudaNormalize<<<G_SIZE, B_SIZE>>>(mean_amp, length, mean_amp_maxVal);
    cudaNormalize<<<G_SIZE, B_SIZE>>>(weighted_amp, length, weighted_amp_maxVal);
    cudaDeviceSynchronize();

    // free memory
    cudaFree(abs_hilb_amp);
    cudaFree(divide);
    cudaFree(divide_mean);
    cudaFree(weight);
    cudaFree(mean_amp_maxVal);
    cudaFree(weighted_amp_maxVal);

    printf("[INFO]: ------------------------------------------------------\n");

}

// ----------------------------- pws stack -----------------------------

int main(void)
{
    float *input1 = (float *)malloc(signal_length * sizeof(float));
    float *input2 = (float *)malloc(signal_length * sizeof(float));

    cufftComplex *complex;
    cufftHandle c2c_plan;

    // 创建 CUFFT 计划
    cufftPlanMany(&c2c_plan, 1, &signal_length, NULL, 1, 1, NULL, 1, 1, CUFFT_C2C, num_signals);

    // Allocates and initializes the complex GPU array
    __CUDA_CALL(cudaMalloc(&complex, num_signals * signal_length * sizeof(cufftComplex)));
    __CUDA_CALL(cudaMemset(complex, 0, num_signals * signal_length * sizeof(cufftComplex)));

    for (int i = 0; i < signal_length; i++) { // Initializes the input array in CPU
        input1[i] = sin(i * PI * 2 * 0.01);
        input2[i] = sin(i * PI * 2 * 0.02);
        // input1[i] = sin(i * 1);
        // input2[i] = sin(i * 2);
    }

    // 将输入信号复制到复数型数组
    __CUDA_CALL(cudaMemcpy2D(complex, 2 * sizeof(float), input1, 1 * sizeof(float), sizeof(float), signal_length, cudaMemcpyHostToDevice));
    __CUDA_CALL(cudaMemcpy2D(complex + signal_length, 2 * sizeof(float), input2, 1 * sizeof(float), sizeof(float), signal_length, cudaMemcpyHostToDevice));

    printf("before hilbertTransform:\n");
    Print_Complex<<<1, 1>>>(complex, num_signals * signal_length);

    HibertTransform(complex, signal_length, num_signals, G_SIZE, B_SIZE, c2c_plan);
    __CUDA_CALL(cudaDeviceSynchronize());
    printf("after hilbertTransform:\n");
    Print_Complex<<<1, 1>>>(complex, num_signals * signal_length);

    // 创建主机端数组以接收结果
    cufftComplex *host_result = (cufftComplex *)malloc(num_signals * signal_length * sizeof(cufftComplex));

    // 从 GPU 复制结果到主机内存
    __CUDA_CALL(cudaMemcpy(host_result, complex, num_signals * signal_length * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    
    // test gpu time
    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu, 0);

    float *mean_amp_gpu, *weighted_amp_gpu;
    float *mean_amp_host = (float *)malloc(signal_length * sizeof(float));
    float *weighted_amp_host = (float *)malloc(signal_length * sizeof(float));
    __CUDA_CALL(cudaMalloc(&mean_amp_gpu, signal_length * sizeof(float)));
    __CUDA_CALL(cudaMalloc(&weighted_amp_gpu, signal_length * sizeof(float)));
    cudaPwsStack(complex, signal_length, num_signals, mean_amp_gpu, weighted_amp_gpu, G_SIZE, B_SIZE);
    // cudaMemcpy to host
    cudaMemcpy(mean_amp_host, mean_amp_gpu, signal_length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weighted_amp_host, weighted_amp_gpu, signal_length * sizeof(float), cudaMemcpyDeviceToHost);
    Print_Float<<<1, 1>>>(mean_amp_gpu, signal_length);

    cudaEventRecord(end_gpu, 0);
    cudaEventSynchronize(end_gpu);

    float time_gpu;
    cudaEventElapsedTime(&time_gpu, start_gpu, end_gpu);

    // test cpu time:
    clock_t start_host, end_host;
    start_host = clock();

    float *mean_amp = (float *)malloc(signal_length * sizeof(float));
    float *weighted_amp = (float *)malloc(signal_length * sizeof(float));
    pws_stack(host_result, signal_length, num_signals, mean_amp, weighted_amp);

    end_host = clock();
    double time_host = (double)(end_host - start_host) / CLOCKS_PER_SEC * 1000;

    // 打印时间
    printf("GPU time: %f ms\n", time_gpu);
    printf("CPU time: %f ms\n", time_host);

    // 释放资源
    cufftDestroy(c2c_plan);
    __CUDA_CALL(cudaFree(complex));

    return 0;

}

/*
$ nvcc -o t381 t381.cu -lcufft
$ ./t381
on GPU:
0.568072
-0.101907
-0.099911
-0.309304
-0.293624
On CPU:
0.568072
-0.101907
-0.099911
-0.309304
-0.293624
$
*/