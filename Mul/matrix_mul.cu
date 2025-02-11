#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <time.h>

#define N 8192 // Size of the matrices

__global__ void matrixMulKernel(half* d_A, half* d_B, float* d_C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    half sum = 0;

    if (row < width && col < width) {
        for (int k = 0; k < width; k++) {
            sum += d_A[row * width + k] * d_B[k * width + col];
        }
        d_C[row * width + col] = __half2float(sum);
    }
}

__global__ void matrixMulTensorCore(half* d_A, half* d_B, float* d_C, int width) {
    // Leading dimensions. Packed with no transpositions.
    int lda = width;
    int ldb = width;
    int ldc = width;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y) / 8;

    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < width; i += 16) {
        int aRow = warpM * 16;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * 16;

        if (aRow < width && aCol < width && bRow < width && bCol < width) {
            // Load the inputs
            nvcuda::wmma::load_matrix_sync(a_frag, d_A + aRow * lda + aCol, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, d_B + bRow * ldb + bCol, ldb);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Load in the current value of d_C, scale it by beta, and add this our result scaled by alpha
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    // Store the output
    nvcuda::wmma::store_matrix_sync(d_C + cRow * ldc + cCol, c_frag, ldc, nvcuda::wmma::mem_row_major);
}

void matrixMul(half* h_A, half* h_B, float* h_C, int width) {
    int sizeIn = width * width * sizeof(half);
    int sizeOut = width * width * sizeof(float);
    half *d_A, *d_B;
    float *d_C;

    cudaMalloc((void**)&d_A, sizeIn);
    cudaMalloc((void**)&d_B, sizeIn);
    cudaMalloc((void**)&d_C, sizeOut);

    cudaMemcpy(d_A, h_A, sizeIn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeIn, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulTensorCore<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeOut, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    half* h_A = (half*)malloc(N * N * sizeof(half));
    half* h_B = (half*)malloc(N * N * sizeof(half));
    float* h_C = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = __float2half(static_cast<float>(rand() % 10));
        h_B[i] = __float2half(static_cast<float>(rand() % 10));;
    }
    clock_t t = clock();
    matrixMul(h_A, h_B, h_C, N);
    t = clock() - t;

    /*
    printf("Result matrix:\n");
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 10; ++j) {
            printf("%.2f ", __half2float(h_C[i * N + j]));
        }
        printf("\n");
    }
    */

    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("The program took %f seconds to execute\n", time_taken);

    return 0;
}