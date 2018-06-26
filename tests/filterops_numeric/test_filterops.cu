#include "gtest/gtest.h"
#include <iostream>
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <tuple>
#include "helper/utils.cuh"

template <typename LeftValueType, typename RightValueType>
void test_filterops_using_templates(gdf_comparison_operator gdf_operator = GDF_EQUALS)
{
    for (int column_size = 0; column_size < 100; column_size += 3)
    {
        const int max_size = 8;
        for (int init_value = 0; init_value <= 0; init_value++)
        {
            gdf_column lhs = gen_gdb_column<LeftValueType>(column_size, init_value); // 4, 2, 0
            // lhs.null_count = 2;

            gdf_column rhs = gen_gdb_column<RightValueType>(column_size, 0.01 + max_size - init_value); // 0, 2, 4
            // rhs.null_count = 1;

            gdf_column output = gen_gdb_column<int8_t>(column_size, 0);

            gdf_error error = gpu_comparison(&lhs, &rhs, &output, gdf_operator);
            EXPECT_TRUE(error == GDF_SUCCESS);

            // std::cout << "Left" << std::endl;
            // print_column<LeftValueType>(&lhs);

            // std::cout << "Right" << std::endl;
            // print_column<RightValueType>(&rhs);

            // std::cout << "Output" << std::endl;
            // print_column<int8_t>(&output);

            check_column_for_comparison_operation<LeftValueType, RightValueType>(&lhs, &rhs, &output, gdf_operator);

            gpu_apply_stencil(&lhs, &output, &rhs);

            check_column_for_stencil_operation<LeftValueType, RightValueType>(&lhs, &output, &rhs);


            delete_gdf_column(&lhs);
            delete_gdf_column(&rhs);
            delete_gdf_column(&output);
        }
    }
}

TEST(FilterOperationsTest, WithInt8AndOthers)
{
    test_filterops_using_templates<int8_t, int8_t>();
    test_filterops_using_templates<int8_t, int16_t>();
    
    test_filterops_using_templates<int8_t, int32_t>();
    test_filterops_using_templates<int8_t, int64_t>();
    test_filterops_using_templates<int8_t, float>(); 
    test_filterops_using_templates<int8_t, double>();
}

TEST(FilterOperationsTest, WithInt16AndOthers)
{
    test_filterops_using_templates<int16_t, int8_t>();
    test_filterops_using_templates<int16_t, int16_t>();
    test_filterops_using_templates<int16_t, int32_t>();
    test_filterops_using_templates<int16_t, int64_t>();
    test_filterops_using_templates<int16_t, float>();
    test_filterops_using_templates<int16_t, double>();
   
}

TEST(FilterOperationsTest, WithInt32AndOthers)
{
    test_filterops_using_templates<int32_t, int8_t>();
    test_filterops_using_templates<int32_t, int16_t>();
    test_filterops_using_templates<int32_t, int32_t>();
    test_filterops_using_templates<int32_t, int64_t>();
    test_filterops_using_templates<int32_t, float>();
    test_filterops_using_templates<int32_t, double>();
   
}

TEST(FilterOperationsTest, WithInt64AndOthers)
{
    test_filterops_using_templates<int64_t, int8_t>();
    test_filterops_using_templates<int64_t, int16_t>();
    test_filterops_using_templates<int64_t, int32_t>();
    test_filterops_using_templates<int64_t, int64_t>();
    test_filterops_using_templates<int64_t, float>();
    test_filterops_using_templates<int64_t, double>();
   
}

TEST(FilterOperationsTest, WithFloat32AndOthers)
{
    test_filterops_using_templates<float, int8_t>();
    test_filterops_using_templates<float, int16_t>();
    test_filterops_using_templates<float, int32_t>();
    test_filterops_using_templates<float, int64_t>();
    test_filterops_using_templates<float, float>();
    test_filterops_using_templates<float, double>();
   
}

TEST(FilterOperationsTest, WithFloat64AndOthers)
{
    test_filterops_using_templates<double, int8_t>();
    test_filterops_using_templates<double, int16_t>();
    test_filterops_using_templates<double, int32_t>();
    test_filterops_using_templates<double, int64_t>();
    test_filterops_using_templates<double, float>();
    test_filterops_using_templates<double, double>();
   
}
