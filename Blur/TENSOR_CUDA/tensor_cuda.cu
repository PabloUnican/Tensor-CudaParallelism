#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include <stdio.h>
#include <time.h>

// librerias para CUDA
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

// max 23 warps por bloque (exceed shared memory)
#define NUM_WARPS 8 // numero de warps por bloque maximo 32 (1024 threads)

#define WMMA_M 32
#define WMMA_N 8
#define WMMA_K 16

half * createFilter(int width)
{
        const float sigma = 2.f; // Standard deviation of the Gaussian distribution.

        const int middle = width / 2;
        float sum = 0.f;


        // Create convolution matrix
        float * aux = (float *)malloc(width*width*sizeof(float));
        half * res=(half *)malloc(width*width*sizeof(half));


        // Calculate filter sum first
        for (int r = -middle; r <= middle; ++r)
        {
                for (int c = -middle; c <= middle; ++c)
                {
                        // e (natural logarithm base) to the power x, where x is what's in the brackets
                        float weight = expf(-static_cast<float>(c * c + r * r) / (2.f * sigma * sigma));
                        int idx = (r + middle) * width + c + middle;

                        aux[idx] = weight;
                        sum += weight;
                }
        }

        // Normalize weight: sum of weights must equal 1
        float normal = 0.f;
        for (int i = 0; i < width * width; i++) {
                normal += (aux[i]);
        }
        for (int i = 0; i < width * width; i++) {
                res[i] = __float2half(aux[i] / (normal));
        }

        return res;
}

/*
Kernel de CUDA para realizar el desenfoque gaussiano
Estructura unidimensional de bloques (x para posicion)
Estructura unidimensional de threads (x para posicion y canal)
*/ 
__global__ void GaussianBlurOnCUDA(uint8_t* const blurredImage, const uint8_t* const rawImage, int width, int height, int channels, const half* filter, int filterWidth)
{        
        // Identificadores de threads
        int temp = blockIdx.x * blockDim.x + threadIdx.x;

        // Identificar thread dentro del warp
        int indexWarp = (threadIdx.x % (warpSize));

        // Identificar warp dentro del bloque
        int warpId = (threadIdx.x / warpSize);

        temp = (temp / warpSize) * WMMA_M + indexWarp;
        // pixel y canal a tratar
        int x = (temp / channels) % width;
        int y = (temp / channels) / width;
        int canal = temp % channels;

        // Comprobar thread util
        if (x >= width || y >= height || canal >= channels){return;}

        // mitad ancho del filtro
        int halfFilterWidth = filterWidth / 2;

        // numero de valores de filtrado
        int filterSize = filterWidth * filterWidth;

        // reparto de warps
        float balancer = 1.0;
        int blockTensor = balancer * gridDim.x;
        
        // Implementacion TENSOR
        if (blockIdx.x < blockTensor) {
        
        // Definir estructura matrices
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> data;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> mask;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> result;
        
        // inicializar resultados a cero
        nvcuda::wmma::fill_fragment(result, 0.0f);

        // tamanho de cada matriz individual
        int offsetLocalMatrix = WMMA_M * WMMA_K * warpId;
        int offsetResultMatrix = WMMA_M * WMMA_N * warpId;

        // declarar matrices en memoria compartida
         // deben estar alineados
        extern __shared__ half sharedMemory[];
        
        half* localMatrix = (half*)&sharedMemory[offsetLocalMatrix];
        
        half* filterMatrix = (half*)&sharedMemory[WMMA_M * WMMA_K * NUM_WARPS];
        
        float* resultMatrix = (float*)&sharedMemory[WMMA_M * WMMA_K * NUM_WARPS + WMMA_K * WMMA_N + offsetResultMatrix];

        int pendingValues = filterSize;
        // Iterar por bloques en tamanho de warp
        for (int i = 0; i < filterSize; i+= WMMA_K) {
                //data matrix
                // el indice no excede el numero de pixeles a cargar
                if (indexWarp < WMMA_M) {
                        int temp = 0;

                        //iterar por todas las posiciones que se pueden cargar en la matriz
                        for (temp = 0; temp < min(WMMA_K, pendingValues); temp++) {
                                //comprobacion de limites
                                int filterX = ((temp + i) % filterWidth);
                                int filterY = ((temp + i) / filterWidth);
                                //obtener posicion pixel vecino
                                int imageX = min(max(x + filterX - halfFilterWidth, 0), width - 1);
                                int imageY = min(max(y + filterY - halfFilterWidth, 0), height - 1);

                                //agregar vecino a matriz
                                localMatrix[indexWarp * WMMA_K + temp] = 
                                        (half) rawImage[((imageY * width + imageX) * channels) + canal];
                        }
                        //rellenar con 0 en caso de necesitarlo
                        for (int j = temp; j < WMMA_K; j++) {
                                localMatrix[indexWarp * WMMA_K + j] = 0;
                                if (warpId == 0) {
                                        filterMatrix[j] = 0;
                                }
                        }
                }
                //agregar coeficiente en el filtro
                if (indexWarp < WMMA_K && warpId == 0) {
                        filterMatrix[indexWarp] = filter[indexWarp + i];
                }
                __syncthreads();

                // cargar en matriz data
                nvcuda::wmma::load_matrix_sync(data, localMatrix, WMMA_K);
                // cargar en matriz mask
                nvcuda::wmma::load_matrix_sync(mask, filterMatrix, WMMA_K);
                // ejecutar codigo en tensor
                nvcuda::wmma::mma_sync(result, data, mask, result);
                //reducir numero de valores faltantes
                pendingValues -= WMMA_K;
        }
        // cargar resultado a matriz
        nvcuda::wmma::store_matrix_sync(resultMatrix, result, WMMA_N, nvcuda::wmma::mem_col_major);

        // almacenar resultados de vuelta en la memoria global
        if (indexWarp < WMMA_M) {
                blurredImage[((y * width + x) * channels) + canal] = (uint8_t) resultMatrix[indexWarp];
        }
        } else {        
        //Implementacion CUDA
        // pixel desenfocado
        half blurredPixel = 0;
        // Calcular el pixel desenfocado
        for (int filterY = -halfFilterWidth; filterY <= halfFilterWidth; filterY++) {
                for (int filterX = -halfFilterWidth; filterX <= halfFilterWidth; filterX++) {
                        
                        //comprobacion de limites
                        int imageX = min(max(x + filterX, 0), width - 1);
                        int imageY = min(max(y + filterY, 0), height - 1);

                        // Calcular el indice del filtro
                        int filterIndex = (filterY + halfFilterWidth) * filterWidth + (filterX + halfFilterWidth);
                        
                        // Pixel de la imagen a tratar
                        half pixel = (half) rawImage[((imageY * width + imageX) * channels) + canal];
                        blurredPixel += pixel * filter[filterIndex];
                }
        }
        blurredImage[((y * width + x) * channels) + canal] = (uint8_t) __half2float(blurredPixel);
        }        
}

// Main entry into the application
int main(int argc, char** argv)
{

	char * imagePath;
	char * outputPath;
	
	int height, width, bpp, channels=4, filterWidth;
	uint8_t * originalImage, * blurredImage;

	if (argc > 3)
	{
		imagePath = argv[1];
		outputPath = argv[2];
                filterWidth = atoi(argv[3]);
	}
	else
	{
		printf("Please provide input and output image files and filter size as arguments to this application.\n");
		exit(1);
	}
        
        //PRINT TITLE
        printf(
                " ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓███████▓▒░▒▓███████▓▒░▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░ \n"  
                "░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░\n"
                "░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░\n" 
                "░▒▓█▓▒▒▓███▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓██████▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░\n" 
                "░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░     ░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░\n" 
                "░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░     ░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░\n" 
                " ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░▒▓███████▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░\n"
                );
        
        //crear filtro
	half * filter=createFilter(filterWidth);

	//Read the image
	originalImage = stbi_load(imagePath, &width, &height, &bpp, channels);
	
	if(originalImage==NULL) printf("Could not load image file: %s\n",imagePath);

        //time
        clock_t t = clock();
        
	blurredImage=(uint8_t *)malloc(width*height*channels*sizeof(uint8_t));
	printf("Width:%d, Height:%d Size(in Bytes):%lu\n", width, height, (long unsigned int) width*height*bpp*channels);

        // Definir punteros para la memoria de la GPU
        uint8_t *d_originalImage, *d_blurredImage;
        half *d_filter;

        // Reservar memoria en la GPU para la imagen original y la imagen final
        cudaMalloc((void**)&d_originalImage, width * height * channels * sizeof(uint8_t));
        cudaMalloc((void**)&d_blurredImage, width * height * channels * sizeof(uint8_t));
        cudaMalloc((void**)&d_filter, filterWidth * filterWidth * sizeof(half));

        // Copiar la imagen original y filtro desde la memoria del host a la memoria de la GPU
        cudaMemcpy(d_originalImage, originalImage, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, filter, filterWidth * filterWidth * sizeof(half), cudaMemcpyHostToDevice);

        //procedimiento
        int threadsPerBlock = NUM_WARPS * 32;
        dim3 blockDim(threadsPerBlock);
        dim3 gridDim((width * height) / ((threadsPerBlock) / channels) + 1);

        // espacio de memoria compartida
        size_t sharedMemorySize = (WMMA_M * WMMA_K) * NUM_WARPS * sizeof(half) + 
                                  (WMMA_N * WMMA_K) * sizeof(half) +
                                  (WMMA_M * WMMA_N) * NUM_WARPS * sizeof(float);

        GaussianBlurOnCUDA<<<gridDim, blockDim, sharedMemorySize>>>(d_blurredImage, d_originalImage, width, height, channels, d_filter, filterWidth);

        cudaDeviceSynchronize();

        // Copiar la imagen final desde la memoria de la GPU a la memoria del host
        cudaMemcpy(blurredImage, d_blurredImage, width * height * channels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        //time
        t = clock() - t;

	stbi_write_jpg(outputPath, width, height, 4, blurredImage, 100);

        // Liberar la memoria de la GPU
        cudaFree(d_originalImage);
        cudaFree(d_blurredImage);
        cudaFree(d_filter);
        // Liberar la memoria del host
        free(originalImage);
        free(blurredImage);
        free(filter);

	printf("Done!\n");

        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        printf("The program took %f seconds to execute\n", time_taken);
	return 0;
}
