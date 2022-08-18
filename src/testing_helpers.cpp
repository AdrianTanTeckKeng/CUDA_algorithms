#include "..\header\testing_helpers.h"


int cmpArrays(int n, int* a, int* b)
{
    for (int i = 0; i < n; i++)
    {
        if (a[i] != b[i])
        {
            printf("    a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
            return 1;
        }
    }
    return 0;
}

void printCmpResult(int n, int* a, int* b)
{
    printf("    %s \n", cmpArrays(n, a, b) ? "failed value" : "passed");
}

void printCmpLenResult(int n, int expN, int* a, int* b)
{
    if (n != expN)
        printf("    expected %d elements, got %d\n", expN, n);
    printf("    %s \n",
        (n == -1 || n != expN) ? "failed count" : cmpArrays(n, a, b) ? "failed value" : "passed");
}

void genArray(int n, int* a, int maxval)
{
    //srand(time(nullptr));
    srand(5);
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % maxval;
    }
}

void zeroArray(int n, int* a)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = 0;
    }
}

void printArray(int n, int* a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("%3d ", a[i]);
    }
    printf("]\n");
}

void printDesc(const char* desc)
{
    printf("==== %s ====\n", desc);
}
