#include "oclData.hpp"
#include "errors.hpp"
#include <fstream>
#include <iostream>

bool ocl_setup = false;
cl::Context ocl_context;
cl::CommandQueue ocl_queue;
cl::Kernel matmul_kernel;
cl::Kernel multiple_multi_matmul_kernel;
cl::Kernel multiple_matmul_kernel;
cl::Kernel multiple_add_kernel;
cl::Kernel multiple_dot_kernel;
cl::Kernel multiple_transpose_kernel;
cl::Kernel multiple_sum_kernel;
cl::Kernel transpose_kernel;
cl::Kernel div_float_kernel;
cl::Kernel mul_float_kernel;
cl::Kernel add_float_kernel;
cl::Kernel sub_float_kernel;
cl::Kernel div_float_eq_kernel;
cl::Kernel mul_float_eq_kernel;
cl::Kernel add_float_eq_kernel;
cl::Kernel sub_float_eq_kernel;
cl::Kernel add_mat_kernel;
cl::Kernel sub_mat_kernel;
cl::Kernel dot_mat_kernel;
cl::Kernel add_mat_eq_kernel;
cl::Kernel sub_mat_eq_kernel;
cl::Kernel dot_mat_eq_kernel;
cl::Kernel relu_kernel;
cl::Kernel relu_inv_kernel;
cl::Kernel sigmoid_kernel;
cl::Kernel sigmoid_inv_kernel;
cl::Kernel binary_CEL_kernel;
cl::Kernel binary_CEL_derivative_kernel;
cl::Kernel log_kernel;
cl::Kernel exp_kernel;
cl::Kernel convolution_kernel;
cl::Kernel parallel_convolution_kernel;
cl::Kernel rotate_conv_kernel;
cl::Kernel pad_kernel;
cl::Kernel parallel_pad_kernel;


void ocl_init()
{
   try {
   unsigned int platform_id=0, device_id=0;
   std::vector<cl::Platform> platforms;
   cl::Platform::get(&platforms);
   std::vector<cl::Device> devices;
   platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_CPU, &devices);
   ocl_context = cl::Context(devices);
   ocl_queue = cl::CommandQueue( ocl_context, devices[device_id] );
   std::vector<std::string> sourcePaths = {
      "kernels/matmul.cl",
      "kernels/multiple_matmul.cl",
      "kernels/multiple_add.cl",
      "kernels/multiple_dot.cl",
      "kernels/multiple_sum.cl",
      "kernels/multiple_transpose.cl",
      "kernels/multiple_multi_matmul.cl",
      "kernels/transpose.cl",
      "kernels/div_float.cl",
      "kernels/mul_float.cl",
      "kernels/add_float.cl",
      "kernels/sub_float.cl",
      "kernels/div_float_eq.cl",
      "kernels/mul_float_eq.cl",
      "kernels/add_float_eq.cl",
      "kernels/sub_float_eq.cl",
      "kernels/add_mat.cl",
      "kernels/sub_mat.cl",
      "kernels/dot_mat.cl",
      "kernels/add_mat_eq.cl",
      "kernels/sub_mat_eq.cl",
      "kernels/dot_mat_eq.cl",
      "kernels/relu.cl",
      "kernels/relu_inv.cl",
      "kernels/sigmoid.cl",
      "kernels/sigmoid_inv.cl",
      "kernels/binary_CEL.cl",
      "kernels/binary_CEL_derivative.cl",
      "kernels/log.cl",
      "kernels/exp.cl",
      "kernels/convolution.cl",
      "kernels/parallel_convolution.cl",
      "kernels/rotate_conv.cl",
      "kernels/pad.cl",
      "kernels/parallel_pad.cl"
   };
   cl::Program::Sources sources;
   for (auto& path : sourcePaths) {
      std::ifstream sourceFile(path);
      sources.push_back(std::string(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>())));
   }
   cl::Program program=cl::Program(ocl_context, sources);
   program.build(devices);

   matmul_kernel                    = cl::Kernel(program, "matmul");
   multiple_multi_matmul_kernel     = cl::Kernel(program, "multiple_multi_matmul");
   multiple_matmul_kernel           = cl::Kernel(program, "multiple_matmul");
   multiple_add_kernel              = cl::Kernel(program, "multiple_add");
   multiple_dot_kernel              = cl::Kernel(program, "multiple_dot");
   multiple_transpose_kernel        = cl::Kernel(program, "multiple_transpose");
   multiple_sum_kernel              = cl::Kernel(program, "multiple_sum");
   transpose_kernel                 = cl::Kernel(program, "transpose");
   div_float_kernel                 = cl::Kernel(program, "div_float");
   mul_float_kernel                 = cl::Kernel(program, "mul_float");
   add_float_kernel                 = cl::Kernel(program, "add_float");
   sub_float_kernel                 = cl::Kernel(program, "sub_float");
   div_float_eq_kernel              = cl::Kernel(program, "div_float_eq");
   mul_float_eq_kernel              = cl::Kernel(program, "mul_float_eq");
   add_float_eq_kernel              = cl::Kernel(program, "add_float_eq");
   sub_float_eq_kernel              = cl::Kernel(program, "sub_float_eq");
   add_mat_kernel                   = cl::Kernel(program, "add_mat");
   sub_mat_kernel                   = cl::Kernel(program, "sub_mat");
   dot_mat_kernel                   = cl::Kernel(program, "dot_mat");
   add_mat_eq_kernel                = cl::Kernel(program, "add_mat_eq");
   sub_mat_eq_kernel                = cl::Kernel(program, "sub_mat_eq");
   dot_mat_eq_kernel                = cl::Kernel(program, "dot_mat_eq");
   relu_kernel                      = cl::Kernel(program, "relu");
   relu_inv_kernel                  = cl::Kernel(program, "relu_inv");
   sigmoid_kernel                   = cl::Kernel(program, "sigmoid");
   sigmoid_inv_kernel               = cl::Kernel(program, "sigmoid_inv");
   binary_CEL_kernel                = cl::Kernel(program, "binary_CEL");
   binary_CEL_derivative_kernel     = cl::Kernel(program, "binary_CEL_derivative");
   log_kernel                       = cl::Kernel(program, "log");
   exp_kernel                       = cl::Kernel(program, "exp");
   convolution_kernel               = cl::Kernel(program, "convolution");
   parallel_convolution_kernel      = cl::Kernel(program, "parallel_convolution");
   rotate_conv_kernel               = cl::Kernel(program, "rotate_conv");
   pad_kernel                       = cl::Kernel(program, "pad");
   parallel_pad_kernel              = cl::Kernel(program, "parallel_pad");

   ocl_queue.finish();

   ocl_setup = true;
   }
   catch(cl::Error& err) {
      std::cout << "Error in setup: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
}