#include "../headers/cuda_kernels.h"

extern "C" __global__ void separateChannels(const unsigned char* input,
                                              unsigned char* red,
                                              unsigned char* green,
                                              unsigned char* blue,
                                              int width,
                                              int height,
                                              int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    int grayIdx = y * width + x;
    // Assuming input in BGR order
    unsigned char b = input[idx + 0];
    unsigned char g = input[idx + 1];
    unsigned char r = input[idx + 2];

    red[grayIdx] = r;
    green[grayIdx] = g;
    blue[grayIdx] = b;
}
