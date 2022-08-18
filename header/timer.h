#pragma once
#include <string>
#include <iostream>

template<typename T>
void printElapsedTime(T time, std::string note = "")
{
    std::cout << "   elapsed time: " << time << "ms    " << note << std::endl;
}