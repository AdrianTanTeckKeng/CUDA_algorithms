#include "header\naive.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

namespace StreamCompaction
{
	namespace Naive
	{
		
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		

		// Define kernel function for Scan
		__global__ void kernelScan(int n, int d, const int* idata, int* odata)
		{
			int index = threadIdx.x + blockIdx.x * blockDim.x;
			if (index >= n)
				return;
			int pow2d = 1 << (d - 1);
			if (index >= pow2d)
				odata[index] = idata[index] + idata[index - pow2d];
		}

		/*
		Perform scan on idata, storing the result into odata.
		*/
		void scan(int n, int* odata, const int* idata)
		{
			// Initialize device pointers and allocate memory. Need a double buffer
			int* deviceIn;
			int* deviceOut;
			cudaMalloc((void**)&deviceIn, n * sizeof(int));
			cudaMalloc((void**)&deviceOut, n * sizeof(int));
			cudaMemcpy(deviceIn, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(deviceOut, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			
			// Initialize grid
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			// Start timer
			timer().startGpuTimer();

			// Iteratively perform naive scan
			for (int d = 1; d <= ilog2ceil(n); d++)
			{
				kernelScan << <fullBlocksPerGrid, blockSize >> > (n, d, deviceIn, deviceOut);
				cudaMemcpy(deviceIn, deviceOut, n * sizeof(int), cudaMemcpyDeviceToDevice);
			}
			odata[0] = 0;
			timer().endGpuTimer();

			// copy in data
			cudaMemcpy(odata + 1, deviceIn, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);

			cudaFree(deviceIn);
			cudaFree(deviceOut);
		}
	}
}