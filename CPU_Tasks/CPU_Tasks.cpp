// CPU_Tasks.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <iostream>
#include <random>
#include <iomanip>
#include <limits>
#include <chrono>

void print_array(const float* array, const unsigned int top_k);

void generate_array(float* array, const unsigned int array_size, const float float_min = std::numeric_limits<float>::min(), const float float_max = std::numeric_limits<float>::max());

bool is_array_sorted(const float* array, const unsigned int array_size);

void bubble_sort(float array[], int size);

void bubble_sort(float array[], int size)
{
    // loop to access each array element
    for (int step = 0; step < size; ++step)
    {

        // loop to compare array elements
        for (int i = 0; i < size - step; ++i)
        {

            // compare two adjacent elements
            // change > to < to sort in descending order
            if (array[i] > array[i + 1])
            {

                // swapping elements if elements
                // are not in the intended order
                float temp = array[i];
                array[i] = array[i + 1];
                array[i + 1] = temp;
            }
        }
    }
}

void print_array(const float* array, const unsigned int top_k)
{
    // print top k elements from float array
    for (int i = 0; i < top_k; i++)
    {
        std::cout << std::setprecision(10) << array[i] << " | ";
    }
    std::cout << "\n";
}

void generate_array(float* array, const unsigned int array_size, const float float_min, const float float_max)
{
    std::random_device rd;
    std::default_random_engine eng(rd());
    eng.seed(0); // set seed to same value to generate repetitive results
    std::uniform_real_distribution<> distr(float_min, float_max);

    for (int i = 0; i < array_size; ++i)
    {
        array[i] = distr(eng);
    }
}

bool is_array_sorted(const float* array, const unsigned int array_size)
{
    for (int i = 0; i < array_size - 1; ++i)
    {
        if (array[i] > array[i + 1])
            return false;
    }
    return true;
}


int main()
{
    std::cout << "Start!\n";
    const unsigned int array_size = 1048576; // 65536; // 262144;
    float* a = new float[array_size];

    generate_array(a, array_size); //, -1000, 1000);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    bubble_sort(a, array_size);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Sort Duration = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;

    //print_array(a, 10);

    if (is_array_sorted(a, array_size))
    {
        printf("\n\nProgram finished without errros and Array is sorted\n");
    }
    else
    {
        printf("\n\nERROR: Array is not sorted!\n");
    }

    delete[] a;
}


