#include "..\header\thrustInBuilt.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include <thrust/scan.h>

namespace StreamCompaction
{
	namespace thrustInBuilt
	{
		
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		
		void scan(int n, int* odata, const int* idata)
		{
			timer().startGpuTimer();
			thrust::exclusive_scan(idata, idata+n, odata);
			timer().endGpuTimer();
		}
	}
}