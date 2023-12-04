#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

extern bool ocl_setup;
extern cl::Context ocl_context;
extern cl::CommandQueue ocl_queue;
extern cl::Kernel matmul_kernel;
extern cl::Kernel multiple_multi_matmul_kernel;
extern cl::Kernel multiple_matmul_kernel;
extern cl::Kernel multiple_add_kernel;
extern cl::Kernel multiple_transpose_kernel;
extern cl::Kernel multiple_sum_kernel;
extern cl::Kernel transpose_kernel;
extern cl::Kernel div_float_kernel;
extern cl::Kernel mul_float_kernel;
extern cl::Kernel add_float_kernel;
extern cl::Kernel sub_float_kernel;
extern cl::Kernel div_float_eq_kernel;
extern cl::Kernel mul_float_eq_kernel;
extern cl::Kernel add_float_eq_kernel;
extern cl::Kernel sub_float_eq_kernel;
extern cl::Kernel add_mat_kernel;
extern cl::Kernel sub_mat_kernel;
extern cl::Kernel dot_mat_kernel;
extern cl::Kernel add_mat_eq_kernel;
extern cl::Kernel sub_mat_eq_kernel;
extern cl::Kernel dot_mat_eq_kernel;
extern cl::Kernel add_col_kernel;
extern cl::Kernel sub_col_kernel;
extern cl::Kernel dot_col_kernel;
extern cl::Kernel relu_kernel;
extern cl::Kernel relu_inv_kernel;

void ocl_init();