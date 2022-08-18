#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_kernel.cuh"

__global__ void vectorAdditionKernel(double* A, double* B, double* C, int arraySize) {
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadId < arraySize) {
		C[threadId] = A[threadId] + B[threadId];
	}
}

void kernel(double* A, double* B, double* C, int arraySize) {
	double* d_A, * d_B, * d_C;

	cudaMalloc((void**)&d_A, arraySize * sizeof(double));
	cudaMalloc((void**)&d_B, arraySize * sizeof(double));
	cudaMalloc((void**)&d_C, arraySize * sizeof(double));

	cudaMemcpy(d_A, A, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	dim3 blockSize(512, 1, 1);
	dim3 gridSize(512 / arraySize + 1, 1);

	vectorAdditionKernel <<<gridSize, blockSize >>> (d_A, d_B, d_C, arraySize);
	cudaMemcpy(C, d_C, arraySize * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}