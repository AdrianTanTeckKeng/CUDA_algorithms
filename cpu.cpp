#include "common.h"

namespace StreamCompaction
{
	namespace CPU
	{
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer &timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		void scan(int n, int* odata, const int* idata)
		{
			timer().startCpuTimer();
			odata[0] = 0;
			for (int i = 1; i < n; i++)
			{
				odata[i] += odata[i - 1] + idata[i - 1];
			}
			timer().endCpuTimer();
		}

		int compactWithoutScan(int n, int* odata, const int* idata)
		{
			timer().startCpuTimer();
			int count = 0;
			for (int i = 0; i < n; i++)
			{
				if (idata[i] != 0)
				{
					odata[count] = idata[i];
					count += 1;
				}
			}
			timer().endCpuTimer();
			return count;
		}

		int compactWithScan(int n, int* odata, const int* idata)
		{
			int count = 0;

			int* map = new int[n];
			int* idx = new int[n];

			timer().startCpuTimer();

			for (int i = 0; i < n; i++)
			{
				if (idata[i] != 0)
					map[i] = 1;
				else
					map[i] = 0;
			}

			// exclusive scan algorithm
			int tmp = 0;
			for (int i = 0; i < n; i++)
			{
				idx[i] = tmp;
				tmp += map[i];
			}

			for (int i = 0; i < n; i++)
			{
				if (map[i] != 0)
				{
					odata[idx[i]] = idata[i];
					count += 1;
				}
			}

			timer().endCpuTimer();
			delete[] map;
			delete[] idx;
			return count;
		}
	}
}