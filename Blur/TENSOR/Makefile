# Makefile for building a CUDA program

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -O0 -arch=sm_89

# Target executable
TARGET = tensor_multiple_filters

# Source files
SRCS = tensor_multiple_filters.cu

# Build the target
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Clean up
clean:
	rm -f $(TARGET)

.PHONY: clean
