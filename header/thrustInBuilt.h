#pragma once
#include "common.h"

namespace StreamCompaction
{
	namespace thrustInBuilt
	{
		StreamCompaction::Common::PerformanceTimer& timer();
		void scan(int n, int* odata, const int* idata);
	}
}