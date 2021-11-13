#include "utility.h"

void print_array(float* array, const unsigned int top_k)
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
