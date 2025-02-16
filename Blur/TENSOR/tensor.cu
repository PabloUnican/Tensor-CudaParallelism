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

#define MAX_THREADS_PER_BLOCK 32

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

        // Identificar warp
        int warpId = temp / warpSize;

        // Identificar thread dentro del warp
        int indexWarp = (threadIdx.x % (warpSize));
        temp = warpId * WMMA_M + indexWarp;

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

        // Implementacion TENSOR
        // Definir estructura matrices
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> data;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> mask;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> result;
        
        // inicializar resultados a cero
        nvcuda::wmma::fill_fragment(result, 0.0f);

        // declarar matrices en memoria compartida
        alignas(128) // deben estar alineados
        __shared__ half localMatrix[WMMA_M * WMMA_N];
        __shared__ half filterMatrix[WMMA_N * WMMA_K];
        __shared__ float resultMatrix[WMMA_M * WMMA_K];

        int pendingValues = filterSize;
        // Iterar por bloques en tamanho de warp
        for (int i = 0; i < filterSize; i+= WMMA_N) {
        //data matrix
        // el indice no excede el numero de pixeles a cargar
        if (indexWarp < WMMA_M) {
                int temp = 0;

                //iterar por todas las posiciones que se pueden cargar en la matriz
                for (temp = 0; temp < min(WMMA_N, pendingValues); temp++) {
                        //comprobacion de limites
                        int filterX = ((temp + i) % filterWidth);
                        int filterY = ((temp + i) / filterWidth);
                        //obtener posicion pixel vecino
                        int imageX = min(max(x + filterX - halfFilterWidth, 0), width - 1);
                        int imageY = min(max(y + filterY - halfFilterWidth, 0), height - 1);

                        //agregar vecino a matriz
                        localMatrix[indexWarp * WMMA_N + temp] = 
                                (half) rawImage[((imageY * width + imageX) * channels) + canal];
                }
                //rellenar con 0 en caso de necesitarlo
                for (int j = temp; j < WMMA_N; j++) {
                        localMatrix[indexWarp * WMMA_N + j] = 0;
                        filterMatrix[j] = 0;
                }
        }
        //agregar coeficiente en el filtro
        if (indexWarp < WMMA_N) {
                filterMatrix[indexWarp] = filter[indexWarp + i];
        }
        __syncwarp();

        // cargar en matriz data
        nvcuda::wmma::load_matrix_sync(data, localMatrix, WMMA_N);
        // cargar en matriz mask
        nvcuda::wmma::load_matrix_sync(mask, filterMatrix, WMMA_N);
        // ejecutar codigo en tensor
        nvcuda::wmma::mma_sync(result, data, mask, result);

        //reducir numero de valores faltantes
        pendingValues -= WMMA_N;
        }
        // cargar resultado a matriz
        nvcuda::wmma::store_matrix_sync(resultMatrix, result, WMMA_K, nvcuda::wmma::mem_col_major);
        // almacenar resultados de vuelta en la memoria global
        if (indexWarp < WMMA_M) {
                blurredImage[((y * width + x) * channels) + canal] = (uint8_t) resultMatrix[indexWarp];
        }

        /*
        
        //Implementacion CUDA
        // pixel desenfocado
        float blurredPixel = 0;
        // Calcular el pixel desenfocado
        for (int filterY = -halfFilterWidth; filterY <= halfFilterWidth; filterY++) {
                for (int filterX = -halfFilterWidth; filterX <= halfFilterWidth; filterX++) {
                        
                        //comprobacion de limites
                        int imageX = min(max(x + filterX, 0), width - 1);
                        int imageY = min(max(y + filterY, 0), height - 1);

                        // Calcular el indice del filtro
                        int filterIndex = (filterY + halfFilterWidth) * filterWidth + (filterX + halfFilterWidth);
                        
                        // Pixel de la imagen a tratar
                        uint8_t pixel = rawImage[((imageY * width + imageX) * channels) + canal];
                        blurredPixel += ((half)pixel) * filter[filterIndex];
                }
        }
        blurredImage[((y * width + x) * channels) + canal] = (uint8_t)blurredPixel;

        */
        
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
        dim3 blockDim(MAX_THREADS_PER_BLOCK);
        dim3 gridDim((width * height) / ((MAX_THREADS_PER_BLOCK) / channels) + 1);
        GaussianBlurOnCUDA<<<gridDim, blockDim>>>(d_blurredImage, d_originalImage, width, height, channels, d_filter, filterWidth);

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
