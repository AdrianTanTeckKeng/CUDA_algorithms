#pragma once
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <ctime>


int cmpArrays(int n, int* a, int* b);


void printCmpResult(int n, int* a, int* b);

void printCmpLenResult(int n, int expN, int* a, int* b);

void genArray(int n, int* a, int maxval);

void zeroArray(int n, int* a);

void printArray(int n, int* a, bool abridged);

void printDesc(const char* desc);