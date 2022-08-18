#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include "..\header\workEfficient.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


__global__ void StreamCompaction::Common::kernMapToBoolean(int n, int* bools, const int* idata)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index >= n)
		return;

	if (idata[index] != 0)
		bools[index] = 1;
	else
		bools[index] = 0;
}

__global__ void StreamCompaction::Common::kernScatter(int n, int* odata, const int* idata, const int* bools, const int* indices)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= n)
		return;
	if (bools[index] != 0)
		odata[indices[index]] = idata[index];
}

namespace StreamCompaction
{
	namespace workEfficient
	{
		
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		

		__global__ void kernelUp(int size, int d, int* data)
		{
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			if (index >= size)
				return;
			int stride = 1 << d;
			if ((index + 1) % (1 << (d+1)) == 0)
				data[index] += data[index - stride];
		}

		__global__ void kernelDown(int size, int d, int* data)
		{
			int index = threadIdx.x + blockDim.x * blockIdx.x;
			if(index >= size)
				return;
			int stride = 1 << d;
			if ((index + 1) % (1 << (d + 1)) == 0)
			{
				int temp = data[index - stride];
				data[index - stride] = data[index];
				data[index] += temp;
			}
				
		}

		void scan(int n, int* odata, const int* idata)
		{
			int* device;
			int size = 1 << ilog2ceil(n);
			cudaMalloc((void**) &device, size * sizeof(int));
			cudaMemcpy(device, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(device + n, 0, (size - n)*sizeof(int));

			dim3 BlocksPerGrid((size + blockSize - 1) / blockSize);

			// Start timer
			timer().startGpuTimer();

			// Now perform upsweep
			for (int d = 0; d <= ilog2ceil(n) - 1; d++)
			{
				kernelUp << <BlocksPerGrid, blockSize >> > (size, d, device);
			}
			// Zero out one element
			cudaMemset(device + size - 1, 0, sizeof(int));

			// Then perform downsweep
			for (int d = ilog2ceil(n)-1; d >= 0; d--)
			{
				kernelDown << <BlocksPerGrid, blockSize >> > (size, d, device);
			}
			timer().endGpuTimer();

			cudaMemcpy(odata, device, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(device);
		}

		int compact(int n, int* odata, const int* idata)
		{
			int size = 1 << ilog2ceil(n);
			dim3 BlocksPerGrid((size + blockSize - 1) / blockSize);

			// Define all device pointers and allocate all memories on GPU
			int* d_input, * d_boolean, * d_indices, * d_output;
			
			// Allocate memory for boolean array
			cudaMalloc((void**)&d_boolean, size * sizeof(int));
			cudaMemset(d_boolean + n, 0, (size - n) * sizeof(int));

			// Allocate memory for input and transfer data from host to gpu
			cudaMalloc((void**)&d_input, n * sizeof(int));
			cudaMemcpy(d_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			// Allocate memory for output and indices
			cudaMalloc((void**)&d_output, size * sizeof(int));
			cudaMalloc((void**)&d_indices, size * sizeof(int));
			
			// Now start loop.
			// We first map input to boolean
			timer().startGpuTimer();
			StreamCompaction::Common::kernMapToBoolean<<<BlocksPerGrid, blockSize>>>(n, d_boolean ,d_input);

			// Now we perform scan on Boolean. We first map data from Boolean to indices(we will need the boolean still)
			cudaMemcpy(d_indices, d_boolean, size * sizeof(int), cudaMemcpyDeviceToDevice);
			
			// Then perform upsweep
			for (int d = 0; d <= ilog2ceil(n) - 1; d++)
			{
				kernelUp <<<BlocksPerGrid, blockSize >> > (size, d, d_indices);
			}
			// Zero out one element
			cudaMemset(d_indices + size - 1, 0, sizeof(int));

			// Then perform downsweep
			for (int d = ilog2ceil(n) - 1; d >= 0; d--)
			{
				kernelDown << <BlocksPerGrid, blockSize >> > (size, d, d_indices);
			}
			

			// Now perform scatter
			StreamCompaction::Common::kernScatter << <BlocksPerGrid, blockSize >> > (n, d_output, d_input, d_boolean, d_indices);
			timer().endGpuTimer();
			
			// Retrieve all required data
			int* finalCount = new int[1];
			cudaMemcpy(finalCount, d_indices + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
			int count = finalCount[0];
			cudaMemcpy(odata, d_output, count * sizeof(int), cudaMemcpyDeviceToHost);
			
			// Free data
			delete[] finalCount;
			
			cudaFree(d_output);
			cudaFree(d_input);
			cudaFree(d_indices);
			cudaFree(d_boolean);
			
			return count;
		}
	}
}
