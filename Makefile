# Makefile for CUDA Image Channel Separation Project

CXX := g++
NVCC := nvcc

# Use pkg-config to get OpenCV flags
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

CXXFLAGS := -std=c++17 -O2 $(OPENCV_CFLAGS)
NVCCFLAGS := -std=c++17 -O2 -Xcompiler "-fPIC" $(OPENCV_CFLAGS)
LDFLAGS := $(OPENCV_LIBS) -lcudart

SRC_CPP := src/main.cpp src/image_processor.cpp
SRC_CU := src/cuda_kernels.cu
OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU := $(SRC_CU:.cu=.o)

TARGET := bin/channel_separation

all: $(TARGET)

# Compile C++ sources
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA sources
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link
$(TARGET): $(OBJ_CPP) $(OBJ_CU)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) $(TARGET)

.PHONY: all clean
