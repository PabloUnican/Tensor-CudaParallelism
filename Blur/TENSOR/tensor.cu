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
#include <stdio.h>
#include <cuda_fp16.h>
#include <iostream>

#define MAX_THREADS_PER_BLOCK 32
#define SIZE_MATRIX 16

half * createFilter(int width)
{
        const float sigma = 2.f; // Standard deviation of the Gaussian distribution.

        const int middle = width / 2;
        float sum = 0.f;


        // Create convolution matrix
        half * res=(half *)malloc(width*width*sizeof(half));


        // Calculate filter sum first
        for (int r = -middle; r <= middle; ++r)
        {
                for (int c = -middle; c <= middle; ++c)
                {
                        // e (natural logarithm base) to the power x, where x is what's in the brackets
                        float weight = expf(-static_cast<float>(c * c + r * r) / (2.f * sigma * sigma));
                        int idx = (r + middle) * width + c + middle;

                        res[idx] = __float2half(weight);
                        sum += weight;
                }
        }

        // Normalize weight: sum of weights must equal 1
        float normal = 1.f / sum;

        for (int r = -middle; r <= middle; ++r)
        {
                for (int c = -middle; c <= middle; ++c)
                {
                        int idx = (r + middle) * width + c + middle;

                        //res[idx] = __float2half(normal);
                        res[idx] = 0.1111;
                }
        }
        return res;
}

/*
Kernel de CUDA para realizar el desenfoque gaussiano
Estructura unidimensional de bloques (x para posicion)
Estructura bidimensional de threads (x para posicion y canal)
*/ 
__global__ void GaussianBlurOnCUDA(uint8_t* const blurredImage, const uint8_t* const rawImage, int width, int height, int channels, const half* filter, int filterWidth)
{        
        // Calcular la posicion del thread en la imagen
        int temp = blockIdx.x * blockDim.x + threadIdx.x;
        int warpId = temp / warpSize; // obtener el ID del warp
        int indexWarp = (threadIdx.x % (warpSize)); // obtener el índice del hilo dentro del warp
        temp = warpId * SIZE_MATRIX + indexWarp;
        int x = (temp / channels) % width;
        int y = (temp / channels) / width;
        int canal = temp % channels;

        // Comprobar thread util
        if (x >= width || y >= height || canal >= channels){return;}
        // mitad ancho del filtro
        int halfFilterWidth = filterWidth / 2;
        // Implementacion TENSOR
        // Definir estructura matrices
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, SIZE_MATRIX, SIZE_MATRIX, SIZE_MATRIX, half, nvcuda::wmma::row_major> data;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, SIZE_MATRIX, SIZE_MATRIX, SIZE_MATRIX, half, nvcuda::wmma::col_major> mask;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, SIZE_MATRIX, SIZE_MATRIX, SIZE_MATRIX, float> result;
        // inicializar resultados a cero
        nvcuda::wmma::fill_fragment(result, 0.0f);

        // mapear matriz "a" (cada fila = vecinos de un pixel)
        alignas(128) // debe estar alineado
        __shared__ half localMatrix[SIZE_MATRIX * SIZE_MATRIX]; // 16x16 = 256
        __shared__ half filterMatrix[SIZE_MATRIX * SIZE_MATRIX];
        __shared__ float resultMatrix[SIZE_MATRIX * SIZE_MATRIX];

        //data matrix
        // el indice no excede el numero de pixeles a cargar
        if (indexWarp < SIZE_MATRIX) {
                int neighbour = 0;
                // Iterar por pixeles vecinos
                for (int dataY = -halfFilterWidth; dataY <= halfFilterWidth; dataY++) {
                        for (int dataX = -halfFilterWidth; dataX <= halfFilterWidth; dataX++) {
                                
                                //comprobacion de limites
                                int imageX = min(max(x + dataX, 0), width - 1);
                                int imageY = min(max(y + dataY, 0), height - 1);
                                //agregar vecino a matriz
                                localMatrix[indexWarp * SIZE_MATRIX + neighbour] = 
                                        (half) rawImage[((imageY * width + imageX) * channels) + canal];
                                neighbour++;
                        }
                }
                for (int i = neighbour; i < SIZE_MATRIX; i++) {
                        localMatrix[indexWarp * SIZE_MATRIX + i] = 0;
                }
        }
        
        // filter matrix
        if (indexWarp < filterWidth * filterWidth) {         
                //agregar valor del filtro a matriz
                filterMatrix[indexWarp] = filter[indexWarp];
        }
        //rellenar fila con 0
        else if (indexWarp < SIZE_MATRIX) {
                filterMatrix[indexWarp] = 0;
        }
        // rellenar resto de matriz con 0
        if (indexWarp >= 1 && indexWarp < SIZE_MATRIX) {
                for (int i = 0; i < SIZE_MATRIX; i++) {
                        filterMatrix[indexWarp * SIZE_MATRIX + i] = 0;
                }
        }
        __syncthreads();
        // cargar en matriz data
        nvcuda::wmma::load_matrix_sync(data, localMatrix, 16);
        // cargar en matriz mask
        nvcuda::wmma::load_matrix_sync(mask, filterMatrix, 16);
        // ejecutar codigo en tensor
        __syncthreads();
        nvcuda::wmma::mma_sync(result, data, mask, result);
        __syncthreads();
        nvcuda::wmma::store_matrix_sync(resultMatrix, result, 16, nvcuda::wmma::mem_col_major);
        __syncthreads();
        // almacenar resultados de vuelta en la memoria global
        if (indexWarp < 16) {
                blurredImage[((y * width + x) * channels) + canal] = (uint8_t) resultMatrix[indexWarp];
        }
        if (x == 12 && y == 12 && canal == 0) {
                        printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n", 
                                resultMatrix[0], resultMatrix[1], resultMatrix[2], resultMatrix[3], resultMatrix[4], resultMatrix[5],
                                resultMatrix[6], resultMatrix[7], resultMatrix[8], resultMatrix[9], resultMatrix[10], resultMatrix[11],
                                resultMatrix[12], resultMatrix[13], resultMatrix[14], resultMatrix[15]);
        }
        if (x == 12 && y == 12 && canal == 0) {
                printf("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n", 
                        __half2float(filterMatrix[0]), __half2float(filterMatrix[1]), __half2float(filterMatrix[2]), __half2float(filterMatrix[3]), __half2float(filterMatrix[4]), __half2float(filterMatrix[5]),
                        __half2float(filterMatrix[6]), __half2float(filterMatrix[7]), __half2float(filterMatrix[8]), __half2float(filterMatrix[9]), __half2float(filterMatrix[10]), __half2float(filterMatrix[11]),
                        __half2float(filterMatrix[12]), __half2float(filterMatrix[13]), __half2float(filterMatrix[14]), __half2float(filterMatrix[15]));
        }
        if (x == 12 && y == 12 && canal == 0) {
                printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n", 
                        __half2float(localMatrix[3 * SIZE_MATRIX + 0]), __half2float(localMatrix[3 * SIZE_MATRIX + 1]), __half2float(localMatrix[3 * SIZE_MATRIX + 2]), __half2float(localMatrix[3 * SIZE_MATRIX + 3]), __half2float(localMatrix[3 * SIZE_MATRIX + 4]), __half2float(localMatrix[3 * SIZE_MATRIX + 5]),
                        __half2float(localMatrix[3 * SIZE_MATRIX + 6]), __half2float(localMatrix[3 * SIZE_MATRIX + 7]), __half2float(localMatrix[3 * SIZE_MATRIX + 8]), __half2float(localMatrix[3 * SIZE_MATRIX + 9]), __half2float(localMatrix[3 * SIZE_MATRIX + 10]), __half2float(localMatrix[3 * SIZE_MATRIX + 11]),
                        __half2float(localMatrix[3 * SIZE_MATRIX + 12]), __half2float(localMatrix[3 * SIZE_MATRIX + 13]), __half2float(localMatrix[3 * SIZE_MATRIX + 14]), __half2float(localMatrix[3 * SIZE_MATRIX + 15]));
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

	char * imagePath;
	char * outputPath;
	
	int height, width, bpp, channels=4;
	uint8_t * originalImage, * blurredImage;

	int filterWidth=3;
	half * filter=createFilter(filterWidth);

	if (argc > 2)
	{
		imagePath = argv[1];
		outputPath = argv[2];
	}
	else
	{
		printf("Please provide input and output image files as arguments to this application.\n");
		exit(1);
	}

	//Read the image
	originalImage = stbi_load(imagePath, &width, &height, &bpp, channels);
	
	if(originalImage==NULL) printf("Could not load image file: %s\n",imagePath);

        //time
        clock_t t = clock();
        
	blurredImage=(uint8_t *)malloc(width*height*channels*sizeof(uint8_t));
	printf("Width:%d, Height:%d Size(in Bytes):%lu\n", width, height, (long unsigned int) width*height*bpp*channels);

	//Tu práctica empieza aquí
	//CUDA

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
        dim3 blockDim(MAX_THREADS_PER_BLOCK * 2); //actualmente solo usamos la mitad del warp
        dim3 gridDim((width * height) / (MAX_THREADS_PER_BLOCK / channels) + 1);
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
