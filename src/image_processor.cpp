#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <opencv2/opencv.hpp>
#include "../headers/image_processor.h"
#include "../headers/cuda_kernels.h"
#include <cuda_runtime.h>

namespace fs = std::filesystem;

ImageProcessor::ImageProcessor() {}
ImageProcessor::~ImageProcessor() {}

void ImageProcessor::processImages(const std::string& inputDir, const std::string& outputDir) {
    // Create output directory if it doesn't exist
    fs::create_directories(outputDir);

    // Collect image files
    std::vector<fs::path> files;
    for (auto& p : fs::directory_iterator(inputDir)) {
        if (p.is_regular_file()) {
            auto ext = p.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                files.push_back(p.path());
            }
        }
    }

    size_t total = files.size();
    for (size_t i = 0; i < total; ++i) {
        processSingleImage(files[i].string(), outputDir);
        float progress = (i + 1) * 100.0f / total;
        std::cout << "Progress: " << progress << "% (" << i+1 << "/" << total << ")" << std::endl;
    }
}

void ImageProcessor::processSingleImage(const std::string& filePath, const std::string& outputDir) {
    // Load image with OpenCV (BGR)
    cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << filePath << std::endl;
        return;
    }
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    size_t imgSize = width * height * channels;
    size_t singleSize = width * height;

    // Allocate host buffers
    unsigned char *h_input = img.data;
    std::vector<unsigned char> h_red(singleSize);
    std::vector<unsigned char> h_green(singleSize);
    std::vector<unsigned char> h_blue(singleSize);

    // Allocate device buffers
    unsigned char *d_input, *d_red, *d_green, *d_blue;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_red, singleSize);
    cudaMalloc(&d_green, singleSize);
    cudaMalloc(&d_blue, singleSize);

    // Copy input to device
    cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    separateChannels<<<grid, block>>>(d_input, d_red, d_green, d_blue, width, height, channels);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_red.data(), d_red, singleSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_green.data(), d_green, singleSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blue.data(), d_blue, singleSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);

    // Compute average channel values
    double sumR = 0, sumG = 0, sumB = 0;
    for (size_t idx = 0; idx < singleSize; ++idx) {
        sumR += h_red[idx];
        sumG += h_green[idx];
        sumB += h_blue[idx];
    }
    double avgR = sumR / singleSize;
    double avgG = sumG / singleSize;
    double avgB = sumB / singleSize;
    std::cout << "Image: " << filePath << " | Avg R: " << avgR << " G: " << avgG << " B: " << avgB << std::endl;

    // Save channel images
    std::string base = fs::path(filePath).stem().string();
    std::string outR = outputDir + "/" + base + "-Red.png";
    std::string outG = outputDir + "/" + base + "-Green.png";
    std::string outB = outputDir + "/" + base + "-Blue.png";

    cv::Mat matR(height, width, CV_8UC1, h_red.data());
    cv::Mat matG(height, width, CV_8UC1, h_green.data());
    cv::Mat matB(height, width, CV_8UC1, h_blue.data());

    cv::imwrite(outR, matR);
    cv::imwrite(outG, matG);
    cv::imwrite(outB, matB);
}
