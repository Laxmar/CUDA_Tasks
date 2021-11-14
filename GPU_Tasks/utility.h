#pragma once

#include <iostream>
#include <random>
#include <iomanip>
#include <limits>

void print_array(const float* array, const unsigned int top_k);

void generate_array(float* array, const unsigned int array_size, const float float_min = std::numeric_limits<float>::min(), const float float_max = std::numeric_limits<float>::max());

bool is_array_sorted(const float* array, const unsigned int array_size);