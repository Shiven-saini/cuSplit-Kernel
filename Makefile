# Makefile for CUDA Image Channel Separation Project

CXX := g++
NVCC := nvcc

# Use pkg-config for compile flags only
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
# Link only essential OpenCV modules to avoid extra VTK/Protobuf dependencies
OPENCV_LIBS := -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio

CXXFLAGS := -std=c++17 -O2 $(OPENCV_CFLAGS)
NVCCFLAGS := -std=c++17 -O2 -rdc=true -Xcompiler "-fPIC" $(OPENCV_CFLAGS)
LDFLAGS := $(OPENCV_LIBS) -lcudart

SRC_CPP := src/main.cpp src/image_processor.cpp
SRC_CU := src/cuda_kernels.cu
OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU := $(SRC_CU:.cu=.o)

TARGET := bin/channel_separation

all: $(TARGET)

# Compile C++ sources
%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

# Compile CUDA sources
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link
$(TARGET): $(OBJ_CPP) $(OBJ_CU)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	# Remove object files and binary
	rm -f $(OBJ_CPP) $(OBJ_CU) $(TARGET)
	# Remove all compiled binaries
	rm -rf bin/*
	# Remove processed images
	rm -rf processed/*

.PHONY: all clean run

run: $(TARGET)
	@echo "Running channel separation..."
	@./$(TARGET) data processed
