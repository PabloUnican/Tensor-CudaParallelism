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

#define MAX_THREADS_PER_BLOCK 128

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

                        res[idx] = (half) weight;
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

                        res[idx] *= (half) normal;
                }
        }
        return res;
}

/*
Kernel de CUDA para realizar el desenfoque gaussiano
Estructura unidimensional de bloques (x para posicion)
Estructura bidimensional de threads (x para posicion, y para canal)
*/ 
__global__ void GaussianBlurOnCUDA(uint8_t* const blurredImage, const uint8_t* const rawImage, int width, int height, int channels, const half* filter, int filterWidth)
{
        // Calcular la posicion del thread en la imagen
        int temp = blockIdx.x * blockDim.x + threadIdx.x;
        int x = temp % width;
        int y = temp / width;
        int canal = threadIdx.y;

        // Comprobar thread util
        if (x >= width || y >= height || canal >= channels){return;}

        const int warpId = temp / warpSize; // obtener el ID del warp
        const int indexWarp = (threadIdx.x % (warpSize)); // obtener el índice del hilo dentro del warp

        // mitad ancho del filtro
        int halfFilterWidth = filterWidth / 2;
        // pixel desenfocado
        float blurredPixel = 0;

        // Implementacion TENSOR
        // Definir estructura matrices
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> data;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> mask;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> result;

        // mapear matriz "a" (cada fila = vecinos de un pixel)
        alignas(128) // debe estar alineado
        __shared__ half localMatrix[256]; // 16x16 = 256
        __shared__ half filterMatrix[256];

        //data matrix
        // el indice no excede el numero de pixeles a cargar
        if (indexWarp < 16) {
                int neighbour = 0;
                // Iterar por pixeles vecinos
                for (int dataY = -halfFilterWidth; dataY <= halfFilterWidth; dataY++) {
                        for (int dataX = -halfFilterWidth; dataX <= halfFilterWidth; dataX++) {
                                
                                //comprobacion de limites
                                int imageX = min(max(x + dataX, 0), width - 1);
                                int imageY = min(max(y + dataY, 0), height - 1);
                                //agregar vecino a matriz
                                localMatrix[indexWarp * 16 + neighbour] = (half) rawImage[((imageY * width + imageX) * channels) + canal];
                                neighbour++;
                        }
                }
                for (int i = neighbour; i < 16; i++) {
                        localMatrix[indexWarp * 16 + i] = 0;
                }
        }
        
        // filter matrix
        if (indexWarp < filterWidth) {
                // Iterar por pixeles vecinos
                for (int filterX = 0; filterX < filterWidth; filterX++) {
                                
                        //agregar valor del filtro a matriz
                        filterMatrix[indexWarp * 16 + filterX] = filter[indexWarp * filterWidth + filterX];
                }
                for (int i = filterWidth; i < 16; i++) {
                        filterMatrix[indexWarp * 16 + i] = 0;
                }
        }
        else if (indexWarp < 16) {
                for (int i = 0; i < 16; i++) {
                        filterMatrix[indexWarp * 16 + i] = 0;
                }
        }
        __syncthreads();

        // cargar en matriz data
        nvcuda::wmma::load_matrix_sync(data, localMatrix, 16);
        // cargar en matriz mask
        nvcuda::wmma::load_matrix_sync(mask, filter, 16);
        // inicializar resultados a cero
        nvcuda::wmma::fill_fragment(result, 0.0f);

        // ejecutar codigo en tensor
        nvcuda::wmma::mma_sync(result, data, mask, result);
        __syncthreads();
        // almacenar resultados de vuelta en la memoria compartida
        nvcuda::wmma::store_matrix_sync(localMatrix, result, 16, nvcuda::wmma::mem_row_major);
        __syncthreads();

        // almacenar resultados de vuelta en la memoria global
        if (indexWarp < 16) {
                // Iterar por el array
                for (int dataX = 0; dataX < filterWidth; dataX++) {
                        blurredImage[((y * width + x) * channels) + canal] = localMatrix[indexWarp * 16];
                }
        }
        /*
        
        //Implementacion CUDA
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


	/*if (argc > 2)
	{
		imagePath = argv[1];
		outputPath = argv[2];
	}
	else
	{
		printf("Please provide input and output image files as arguments to this application.\n");
		exit(1);
	}
        */
       imagePath = "grande.jpg";
       outputPath = "grande_res.jpg";


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
        dim3 blockDim(MAX_THREADS_PER_BLOCK / channels, channels);
        dim3 gridDim((width * height) / (MAX_THREADS_PER_BLOCK / channels) + 1);
        GaussianBlurOnCUDA<<<gridDim, blockDim>>>(d_blurredImage, d_originalImage, width, height, channels, d_filter, filterWidth);

        cudaThreadSynchronize();

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
