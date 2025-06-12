# Makefile for building a CUDA program

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -O3 -arch=sm_89 -lineinfo

# Target executable
TARGET = tensor_cuda_filters

# Source files
SRCS = tensor_cuda_filters.cu

# Build the target
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Clean up
clean:
	rm -f $(TARGET)

.PHONY: clean
