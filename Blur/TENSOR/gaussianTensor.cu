// desenfoque gaussiano con tensor-core para generar 16 píxeles desenfocados diferentes de una imagen a la vez
// por ejemplo, la entrada es una imagen, la salida son 16 imágenes cada una con una fuerza de desenfoque diferente
// la fila de la matriz-A del tensor es la información de los píxeles vecinos de 1 píxel
// la fila de la matriz-B del tensor está compuesta por multiplicadores de desenfoque gaussiano
// la primera fila del resultado tiene 16 versiones diferentes desenfocadas del primer píxel
// 15 filas más para un total de 16 píxeles. Cada warp calcula 16 píxeles a la vez, posiblemente teniendo 4x TFLOPS que la versión solo CUDA
// Por cada bloque CUDA que usa tensor, también hay 4 bloques que usan solo CUDA para calcular algoritmos de desenfoque gaussiano para agregar rendimiento de núcleos CUDA además del tensor
// por cada bloque CUDA que usa tensor, también hay 8 bloques que usan solo enteros en CUDA que se suman al rendimiento de flotantes + tensor
// posiblemente limitado por el ancho de banda
// posiblemente tiene más error debido al sesgo del trabajo mixto de enteros+flotantes+tensor y solo 16 bits de precisión
// pero debería ser más rápido que cualquiera de ellos

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <device_functions.h>
#include <mma.h>
// imagen cuadrada por simplicidad (1024 x 1024)
//constexpr = const en tiempo de compilación
constexpr int imageSize = 1024;

// matriz de blur cuadrado por simplicidad (4 x 4)
constexpr int tileSize = 4;

constexpr int imagePixels = imageSize * imageSize;
constexpr int tilePixels = tileSize * tileSize;
constexpr int tilesPerDimension = imageSize / tileSize;

// tamaño de datos de vecinos para cargar al usar memoria compartida (4x4 requiere una región de 6x6)
// por simplicidad, solo se calcula el interior de la imagen. los bordes se dejan solos por ahora
constexpr int tileAccessSize = tileSize+2;
constexpr int tileAccessPixels = tileAccessSize * tileAccessSize;

// 1 warp por bloque por ahora
__global__ void superGaussPipeline(half* image, half* gaussianMultipliers, half* outputImages)
{
    // cada warp trabaja independientemente en un tile diferente. entonces id del tile = id del warp
    const int indexWarp = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    const int indexTileX = indexWarp & (tilesPerDimension - 1); // obtener la coordenada X del tile
    const int indexTileY = indexWarp / tilesPerDimension; // obtener la coordenada Y del tile

    // omitir tiles de borde por simplicidad por ahora
    if (indexTileX == 0 || indexTileX >= tilesPerDimension - 1 || indexTileY == 0 || indexTileY >= tilesPerDimension - 1)
        return;

    const int indexLane = (threadIdx.x & (warpSize - 1)); // obtener el índice del hilo dentro del warp

    // esquina superior izquierda del tile que tiene tamaño 6x6
    // la carga consiste en leer 6 filas comenzando con indexTile, cada una con 6 columnas    
    // cargar 36 píxeles usando 32 hilos ==> algunos hilos necesitan más iteraciones
    const int indexTile = (indexTileX*tileSize - 1) + (indexTileY * tileSize - 1) * imageSize;
    const int stepsRequired = 1 + (tileAccessPixels / warpSize);

    alignas(128)
    __shared__ half alignedAccess[256]; // memoria compartida alineada para acceso eficiente
    // 6x6 píxeles (para calcular el desenfoque gaussiano basado en el vecino más cercano para los píxeles interiores 4x4)

    // no requiere alineación porque ningún tensor accede a esto
    __shared__ half tileAccess[tileAccessPixels]; // memoria compartida para cargar los píxeles vecinos

    // cargar los píxeles vecinos en la memoria compartida
    for(int i=0;i<tileAccessSize;i++)
    {
        if (indexLane < tileAccessSize)
        {
            tileAccess[ i * tileAccessSize  + indexLane] =                 
                image[  i * imageSize       + indexLane + indexTile];
        }
    }
    __syncthreads();
    
    // mapear datos de píxeles vecinos (9 por píxel) a la fila de la matriz del núcleo tensor (cada fila = vecinos de un píxel, la mitad de los elementos son ceros)
    // mapear multiplicadores gaussianos a la fila de la matriz del núcleo tensor
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> neighborPixels;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> gaussian;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> result;
    // mapeo
    if (indexLane < tilePixels)
    {
        const int ix = indexLane & (tileSize - 1);
        const int iy = indexLane / tileSize;

        int ctr = 0;
        const int accessX = ix + 1;
        const int accessY = iy + 1;
        for (int jy = -1; jy <= 1; jy++)
            for (int jx = -1; jx <= 1; jx++)
            {
                alignedAccess[ctr + indexLane * 16] = tileAccess[accessX + jx + (accessY + jy) * tileAccessSize];
                ctr++;
            }

        for (int k = 0; k < 7; k++)
            alignedAccess[9 + k + indexLane * 16] = 0; // rellenar con ceros
    }
    __syncthreads();
    
    // ahora los datos de vecinos mapeados pueden ser cargados
    nvcuda::wmma::load_matrix_sync(neighborPixels, alignedAccess, 16);

    // también cargar 16 conjuntos diferentes de multiplicadores de desenfoque gaussiano (1 por fila, 9 elementos llenos, el resto son ceros) 
    nvcuda::wmma::load_matrix_sync(gaussian, gaussianMultipliers, 16);
    
    // inicializar resultados a cero
    nvcuda::wmma::fill_fragment(result, 0.0f);
    
    // se calculan 16 operaciones de desenfoque gaussiano por píxel a la vez
    // cada resultado de un píxel se da en una fila
    // 16 píxeles diferentes en 16 filas
    nvcuda::wmma::mma_sync(result, neighborPixels, gaussian, result);
    __syncthreads();
    
    // almacenar resultados de vuelta en la memoria compartida
    nvcuda::wmma::store_matrix_sync(alignedAccess, result, 16, nvcuda::wmma::mem_row_major);
   
    __syncthreads(); // sincronizar porque se escribió en la memoria compartida
    
    // distribuir el resultado a 16 imágenes cada una con una fuerza/patrón de desenfoque gaussiano diferente
    if (indexLane < tileSize)
    {
        const int resultSubTileX = indexLane & (tileSize - 1);
        const int resultSubTileY = indexLane / tileSize;

        // iterar imágenes de salida        
        for (int i = 0; i < 16; i++)
        {
            // iterar filas de píxeles
            for (int k = 0; k < tileSize; k++)
            {
                outputImages[i * imagePixels + (indexTile + 1 + imageSize) + indexLane + k*imageSize]  // i itera la versión gaussiana, tileX & tileY iteran píxeles 4x4 del tile
                    =
                    alignedAccess[i + (k * tileSize + indexLane) * 16]; // i itera la versión gaussiana, indexLane itera píxel
            }
        }
    }
}

#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>

void test()
{
    auto img = cv::imread("test.jpg",cv::ImreadModes::IMREAD_GRAYSCALE);
    
    cv::namedWindow("input");
    cv::resizeWindow("input", cv::Size2i(1024, 1024));
    cv::imshow("input", img);

    // elementos para 1M matrices de tamaño 16x16
    constexpr int inputN = imagePixels * sizeof(half);
    constexpr int gaussN = 16 * 16 * sizeof(half);
    constexpr int outputN = inputN * 16;
    half* dvcA, * dvcB, * dvcC;
    cudaMalloc(&dvcA, inputN);
    cudaMalloc(&dvcB, gaussN);
    cudaMalloc(&dvcC, outputN);

    half* hstA, * hstB, * hstC;
    cudaHostAlloc(&hstA, inputN, cudaHostAllocDefault);
    cudaHostAlloc(&hstB, gaussN, cudaHostAllocDefault);
    cudaHostAlloc(&hstC, outputN, cudaHostAllocDefault);

    // imagen de entrada, escala de grises normalizada
    for (int i = 0; i < imagePixels; i++)
        hstA[i] = img.at<uchar>(i) / 256.0f;

    // coeficientes de desenfoque gaussiano (suma=16)
    // todos iguales por ahora por simplicidad
    for (int i = 0; i < 16; i++)
    {
        hstB[i * 16    ] = 1+i/16.0f;
        hstB[i * 16 + 1] = 2 + i / 8.0f;
        hstB[i * 16 + 2] = 1 + i / 16.0f;
        hstB[i * 16 + 3] = 2 + i / 8.0f;
        hstB[i * 16 + 4] = (8); // centro de la curva gaussiana (3x3)
        hstB[i * 16 + 5] = 2 + i / 8.0f;
        hstB[i * 16 + 6] = 1 + i / 16.0f;
        hstB[i * 16 + 7] = 2 + i / 8.0f;
        hstB[i * 16 + 8] = 1 + i / 16.0f;
        hstB[i * 16 + 9] = 0; 
        hstB[i * 16 + 10] = 0;
        hstB[i * 16 + 11] = 0;
        hstB[i * 16 + 12] = 0;
        hstB[i * 16 + 13] = 0;
        hstB[i * 16 + 14] = 0;
        hstB[i * 16 + 15] = 0;
    }
    // leer imagen con opencv
    std::cout << "desenfoque tensorial" << std::endl;

    cudaStream_t stream0;
    cudaStreamCreate(&stream0);

    // calentamiento 
    int numWarpsToLaunch = imagePixels / tilePixels;
    for (int i = 0; i < 1000; i++)
    {
        superGaussPipeline <<<numWarpsToLaunch, 32, 0, stream0 >>> (dvcA, dvcB, dvcC);
    }
    cudaEvent_t evt, evt2;
    auto err = cudaEventCreate(&evt);
    if (err)
    {
        std::cout << "Código de error:" << err << std::endl;
    }
    cudaEventCreate(&evt2);
    cudaEventRecord(evt, stream0);

    cudaMemcpyAsync(dvcA, hstA, inputN, ::cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(dvcB, hstB, gaussN, ::cudaMemcpyHostToDevice, stream0);
    for (int i = 0; i < 1000; i++)
    {
        superGaussPipeline <<<numWarpsToLaunch, 32, 0, stream0 >>> (dvcA, dvcB, dvcC);
    }
    err=cudaMemcpyAsync(hstC, dvcC, outputN, ::cudaMemcpyDeviceToHost, stream0);
    if (err)
    {
        std::cout <<"Código de error:" << err << std::endl;
    }
    cudaEventRecord(evt2, stream0);
    cudaEventSynchronize(evt2);
    float tim;
    cudaEventElapsedTime(&tim, evt, evt2);
    std::cout << "generando 16 imágenes 1000 veces: " << tim << " ms" << std::endl;
    
    // mostrar imágenes con opencv
    // división por 16 para normalización
    cv::namedWindow(std::string("output"));
    cv::resizeWindow(std::string("output"), cv::Size2i(1024, 1024));
    int frame = 0;
    while (cv::waitKey(200) != 27)
    {
        frame++;
        if (frame == 16)
            frame = 0;
        for (int i = 0; i < imagePixels; i++)
            img.at<uchar>(i) = (((float)hstC[i + frame * imagePixels]) / 10.0f) * 256.0f;
        cv::imshow(std::string("output"), img);
    }
 
    cv::destroyAllWindows();
    cudaFreeHost(hstA);
    cudaFreeHost(hstB);
    cudaFreeHost(hstC);
    cudaFree(dvcA);
    cudaFree(dvcB);
    cudaFree(dvcC);
}

int main()
{
    test();
    return 0;
}
