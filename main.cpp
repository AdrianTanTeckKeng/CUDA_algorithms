#include <iostream>
#include "cpu.h"
#include "naive.h"
#include "workEfficient.h"
#include "testing_helpers.h"
#include "thrustInBuilt.h"
#include "timer.h"

const int SIZE = 1 << 12;
const int NPOT = SIZE - 3;
int* a = new int[SIZE];
int* b = new int[SIZE];
int* c = new int[SIZE];

int main()
{
	printf("\n");
	printf("*****************************\n");
	printf("**        SCAN TESTS       **\n");
	printf("*****************************\n");
	std::cout << "Input array is: " << std::endl;
	genArray(SIZE - 1, a, 50);
	a[SIZE - 1] = 0;
	printArray(SIZE, a, true);
	
	// Testing CPU scan, first check power of two
	zeroArray(SIZE, b);
	printDesc("cpu scan, power-of-two");
	StreamCompaction::CPU::scan(SIZE, b, a);
	printArray(SIZE, b, true);
	printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
	
	// Testing CPU scan, without power of two
	zeroArray(SIZE, c);
	printDesc("cpu scan, non power-of-two");
	StreamCompaction::CPU::scan(NPOT, c, a);
	printArray(NPOT, c, true);
	std::cout << "Checking non power-of-two result \n";
	printCmpResult(NPOT, b, c);
	printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
	
	// Testing work-inefficient GPU scan, without power-of-two
	zeroArray(SIZE, c);
	printDesc("Naive scan implemented on GPU, non power-of-two");
	StreamCompaction::Naive::scan(NPOT, c, a);
	printArray(SIZE, c, true);
	std::cout << "Checking results for naive scan on GPU\n";
	printCmpResult(NPOT, b, c);
	printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");

	// Testing work-efficient GPU scan, with no power-of-two
	zeroArray(SIZE, c);
	printDesc("Work efficient scan implemented on GPU, non power-of-two");
	StreamCompaction::workEfficient::scan(NPOT, c, a);
	printArray(SIZE, c, true);
	std::cout << "Checking results for work efficient scan on GPU\n";
	printCmpResult(NPOT, b, c);
	printElapsedTime(StreamCompaction::workEfficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");


	// Testing exclusive scan implemented in thrust, with no power-of-two
	zeroArray(SIZE, c);
	printDesc("Exclusive scan implemented in CUDA thrust library, non power-of-two");
	StreamCompaction::thrustInBuilt::scan(NPOT, c, a);
	printArray(SIZE, c, true);
	std::cout << "Checking results for exclusive scan implemented in CUDA library, non power-of-two\n";
	printCmpResult(NPOT, b, c);
	printElapsedTime(StreamCompaction::thrustInBuilt::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");

	// Now we test stream compaction, which is an algorithm that uses scan to remove 0s from an array of integers;
	printf("\n");
	printf("*****************************\n");
	printf("** STREAM COMPACTION TESTS **\n");
	printf("*****************************\n");

	// Generate data to test stream compaction
	genArray(SIZE - 1, a, 4);
	a[SIZE - 1] = 0;
	printArray(SIZE, a, true);

	// Testing algorithms implemented on CPU side
	int count, expectedCount, expectedNPOT;
	zeroArray(SIZE, b);
	printDesc("cpu compact without scan, power-of-two");
	count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
	expectedCount = count;
	printArray(count, b, true);
	printCmpLenResult(count, expectedCount, b, b);

	zeroArray(SIZE, c);
	printDesc("cpu compact without scan, non power-of-two");
	count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
	expectedNPOT = count;
	printArray(count, c, true);
	printCmpLenResult(count, expectedNPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("cpu compact with scan, non power-of-two");
	count = StreamCompaction::CPU::compactWithScan(NPOT, c, a);
	printArray(count, c, true);
	printCmpLenResult(count, expectedNPOT, b, c);

	// Testing work efficient algorithm implemented on GPU
	zeroArray(SIZE, c);
	printDesc("work-efficient compact with scan on GPU, non power-of-two");
	count = StreamCompaction::workEfficient::compact(NPOT, c, a);
	printArray(count, c, true);
	printCmpLenResult(count, expectedNPOT, b, c);

	// Now free up the memory
	delete[] a;
	delete[] b;
	delete[] c;
	
	return 0;
}