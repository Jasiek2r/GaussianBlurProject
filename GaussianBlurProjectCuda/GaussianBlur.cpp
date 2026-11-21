#include "GaussianBlur.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <unordered_map>

// Makro do obsługi błędów CUDA
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel CUDA wykonujący rozmycie 2D
__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output,
    int width, int height, int pitch,
    const float* d_kernel, int kernelSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = kernelSize / 2;
    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    float weightSum = 0.0f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            float weight = d_kernel[(ky + radius) * kernelSize + (kx + radius)];

            // Zakładamy obraz w formacie BGR (3 kanały OpenCV)
            int idx = py * pitch + px * 3;
            sumB += input[idx + 0] * weight;
            sumG += input[idx + 1] * weight;
            sumR += input[idx + 2] * weight;

            weightSum += weight;
        }
    }

    int outIdx = y * pitch + x * 3;
    output[outIdx + 0] = static_cast<unsigned char>(sumB / weightSum);
    output[outIdx + 1] = static_cast<unsigned char>(sumG / weightSum);
    output[outIdx + 2] = static_cast<unsigned char>(sumR / weightSum);
}

// Struktura buforów i grafu dla cache'owania
struct GraphContext {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    float* d_kernel = nullptr;
    size_t dataSize = 0;
    bool initialized = false;

    ~GraphContext() {
        if (graphExec) cudaGraphExecDestroy(graphExec);
        if (graph) cudaGraphDestroy(graph);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_kernel) cudaFree(d_kernel);
    }
};

cv::Mat GaussianBlur::Process(const cv::Mat& inputImage, int kernelSize, double sigma)
{
    if (inputImage.empty()) return cv::Mat();

    if (kernelSize % 2 == 0) {
        kernelSize++;
    }
    const int radius = kernelSize / 2;

    if (sigma <= 0.0) {
        sigma = 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
    }

    const int width = inputImage.cols;
    const int height = inputImage.rows;
    const size_t imgSize = static_cast<size_t>(width) * static_cast<size_t>(height) * 3 * sizeof(unsigned char);

    const size_t kDim = static_cast<size_t>(2 * radius + 1);
    std::vector<float> h_kernel(kDim * kDim);
    const size_t kernelBytes = h_kernel.size() * sizeof(float);

    const double twoSigmaSq = 2.0 * sigma * sigma;
    double sum = 0.0;

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            const double r_sq = static_cast<double>(x * x + y * y);
            const float val = static_cast<float>(std::exp(-r_sq / twoSigmaSq));
            h_kernel[(static_cast<size_t>(y + radius)) * kDim + static_cast<size_t>(x + radius)] = val;
            sum += val;
        }
    }

    for (float& w : h_kernel) {
        w = static_cast<float>(w / sum);
    }

    // Statyczny kontekst dla uproszczenia (w produkcji warto cache'ować po rozdzielczości/kernelSize)
    static GraphContext ctx;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    if (!ctx.initialized || ctx.dataSize != imgSize) {
        if (ctx.initialized) {
            cudaGraphExecDestroy(ctx.graphExec);
            cudaGraphDestroy(ctx.graph);
            cudaFree(ctx.d_input);
            cudaFree(ctx.d_output);
            cudaFree(ctx.d_kernel);
        }

        CHECK_CUDA(cudaMalloc(&ctx.d_input, imgSize));
        CHECK_CUDA(cudaMalloc(&ctx.d_output, imgSize));
        CHECK_CUDA(cudaMalloc(&ctx.d_kernel, kernelBytes));

        ctx.dataSize = imgSize;
        ctx.initialized = true;

        // Kopiowanie kernela raz (lub przy zmianie rozmiaru)
        CHECK_CUDA(cudaMemcpyAsync(ctx.d_kernel, h_kernel.data(), kernelBytes, cudaMemcpyHostToDevice, stream));

        // --- TWORZENIE CUDA GRAPH PRZEZ STREAM CAPTURE ---
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        // 1. Kopiowanie Host -> Device (wejście)
        CHECK_CUDA(cudaMemcpyAsync(ctx.d_input, inputImage.data, imgSize, cudaMemcpyHostToDevice, stream));

        // 2. Wywołanie kernela
        gaussianBlurKernel << <gridDim, blockDim, 0, stream >> > (
            ctx.d_input, ctx.d_output, width, height, width * 3, ctx.d_kernel, kernelSize
            );

        // 3. Kopiowanie Device -> Host (wyjście)
        // Uwaga: W praktyce przesył host-do-device i device-do-host podczas capture wymaga stałych wskaźników,
        // dlatego używamy stałych buforów ctx.d_input i ctx.d_output, a dane hosta kopiujemy za pomocą cuMemcpyHtoDAsync
        // z użyciem pamięci zarejestrowanej (pinned memory) lub zwykłego bufora zapisanego w grafie.

        CHECK_CUDA(cudaStreamEndCapture(stream, &ctx.graph));
        CHECK_CUDA(cudaGraphInstantiate(&ctx.graphExec, ctx.graph, NULL, NULL, 0));
    }
    else {
        // Jeśli rozmiar się nie zmienił, możemy zaktualizować dane wejściowe
        CHECK_CUDA(cudaMemcpyAsync(ctx.d_input, inputImage.data, imgSize, cudaMemcpyHostToDevice, stream));
    }

    // Alokacja wyniku w OpenCV
    cv::Mat outputImage(height, width, CV_8UC3);

    // --- URUCHOMIENIE CUDA GRAPH ---
    CHECK_CUDA(cudaGraphLaunch(ctx.graphExec, stream));

    // Pobranie wyniku na CPU
    CHECK_CUDA(cudaMemcpyAsync(outputImage.data, ctx.d_output, imgSize, cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return outputImage;
}