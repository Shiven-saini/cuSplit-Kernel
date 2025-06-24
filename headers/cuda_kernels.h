#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

// Kernel to separate BGR channels from input image
// input: pointer to BGR image data (3 channels)
// red, green, blue: pointers to output single-channel data
// width, height: image dimensions
extern "C" __global__ void separateChannels(const unsigned char* input,
                                              unsigned char* red,
                                              unsigned char* green,
                                              unsigned char* blue,
                                              int width,
                                              int height,
                                              int channels);

#endif // CUDA_KERNELS_H
