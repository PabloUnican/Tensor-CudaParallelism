# Makefile for building a CUDA program

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -O0 -arch=sm_89

# Target executable
TARGET = cuda

# Source files
SRCS = cuda.cu

# Build the target
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Clean up
clean:
	rm -f $(TARGET)

.PHONY: clean
