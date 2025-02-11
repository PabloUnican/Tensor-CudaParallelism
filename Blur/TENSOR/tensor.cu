#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include <stdio.h>
#include <time.h>

#define MAX_THREADS_PER_BLOCK 128

float * createFilter(int width)
{
        const float sigma = 2.f; // Standard deviation of the Gaussian distribution.

        const int half = width / 2;
        float sum = 0.f;


        // Create convolution matrix
        float * res=(float *)malloc(width*width*sizeof(float));


        // Calculate filter sum first
        for (int r = -half; r <= half; ++r)
        {
                for (int c = -half; c <= half; ++c)
                {
                        // e (natural logarithm base) to the power x, where x is what's in the brackets
                        float weight = expf(-static_cast<float>(c * c + r * r) / (2.f * sigma * sigma));
                        int idx = (r + half) * width + c + half;

                        res[idx] = weight;
                        sum += weight;
                }
        }

        // Normalize weight: sum of weights must equal 1
        float normal = 1.f / sum;

        for (int r = -half; r <= half; ++r)
        {
                for (int c = -half; c <= half; ++c)
                {
                        int idx = (r + half) * width + c + half;

                        res[idx] *= normal;
                }
        }
        return res;
}

/*
Kernel de CUDA para realizar el desenfoque gaussiano
Estructura unidimensional de bloques (x para posicion)
Estructura bidimensional de threads (x para posicion, y para canal)
*/ 
__global__ void GaussianBlurOnCUDA(uint8_t* const blurredImage, const uint8_t* const rawImage, int width, int height, int channels, const float* filter, int filterWidth)
{
        // Calcular la posicion del thread en la imagen
        int temp = blockIdx.x * blockDim.x + threadIdx.x;
        int x = temp % width;
        int y = temp / width;
        int canal = threadIdx.y;

        // Comprobar thread util
        if (x >= width || y >= height || canal >= channels){return;}

        // mitad ancho del filtro
        int halfFilterWidth = filterWidth / 2;
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
                        blurredPixel += ((float)pixel) * filter[filterIndex];
                }
        }
        blurredImage[((y * width + x) * channels) + canal] = (uint8_t)blurredPixel;
        
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

	int filterWidth=9;
	float * filter=createFilter(filterWidth);


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
       imagePath = "Blur/grande.jpg";
       outputPath = "Blur/grande_res.jpg";


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
        float *d_filter;

        // Reservar memoria en la GPU para la imagen original y la imagen final
        cudaMalloc((void**)&d_originalImage, width * height * channels * sizeof(uint8_t));
        cudaMalloc((void**)&d_blurredImage, width * height * channels * sizeof(uint8_t));
        cudaMalloc((void**)&d_filter, filterWidth * filterWidth * sizeof(float));

        // Copiar la imagen original y filtro desde la memoria del host a la memoria de la GPU
        cudaMemcpy(d_originalImage, originalImage, width * height * channels * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice);

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
