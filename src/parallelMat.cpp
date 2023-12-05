#include "parallelMat.hpp"
#include "oclData.hpp"
#include <iostream>
#include "errors.hpp"

ParallelMat::ParallelMat(const std::vector<Mat>& mats)
{
   m_width = mats[0].getWidth();
   m_height = mats[0].getHeight();
   for (const auto &input : mats) {
      // ensure every vector is the same size
      assert(input.getWidth() == m_width);
      assert(input.getHeight() == m_height);
   }
   m_count = mats.size();
   const unsigned INPUT_SIZE = m_width*m_height;

   std::vector<float> inputs_data(m_count*INPUT_SIZE);
   for(unsigned i = 0; i < m_count; i++) {
      const std::vector<float> input_data = mats[i].getVals();
      std::copy(begin(input_data), end(input_data), begin(inputs_data) + i*INPUT_SIZE);
   }

   m_buffer = cl::Buffer(ocl_context, CL_MEM_READ_WRITE, (m_count*INPUT_SIZE)*sizeof(float));
   ocl_queue.enqueueWriteBuffer( m_buffer, CL_TRUE, 0, (m_count*INPUT_SIZE)*sizeof(float), inputs_data.data() );
}

std::vector<Mat> ParallelMat::toVector() const
{
   const unsigned MAT_SIZE = m_width*m_height;
   const unsigned N_ELEMENTS = MAT_SIZE*m_count;

   std::vector<float> all_mats_data(N_ELEMENTS);

   ocl_queue.enqueueReadBuffer(m_buffer, CL_TRUE, 0, N_ELEMENTS*sizeof(float), all_mats_data.data());

   std::vector<Mat> inputs;
   for (unsigned i = 0; i < m_count; i++) {
      std::vector<float> mat_data(begin(all_mats_data) + i*MAT_SIZE, begin(all_mats_data) + (i+1)*MAT_SIZE);
      inputs.push_back(Mat(m_height, m_width, mat_data));
   }  
   return inputs;
}

Mat ParallelMat::sum() const
{
   const cl_int arraySize = m_height*m_width;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, arraySize * sizeof(float));
   cl_int numArrays=m_count;

   try {
      multiple_sum_kernel.setArg( 0, m_buffer );
      multiple_sum_kernel.setArg( 1, out_buffer );
      multiple_sum_kernel.setArg( 2, sizeof(cl_int), &numArrays );
      multiple_sum_kernel.setArg( 3, sizeof(cl_int), &arraySize );

      cl::NDRange global( arraySize );
      ocl_queue.enqueueNDRangeKernel( multiple_sum_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in multipleSum: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return Mat(m_height, m_width, out_buffer);
}

ParallelMat ParallelMat::operator* (const ParallelMat &other) const
{
   assert(m_width == other.m_height);

   cl_int common = m_width;
   cl_int A_h    = m_height;
   cl_int B_w    = other.m_width;
   const int C_N_ELEMENTS = A_h*B_w*m_count;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, C_N_ELEMENTS * sizeof(float));
   try {
      multiple_multi_matmul_kernel.setArg( 0, m_buffer );
      multiple_multi_matmul_kernel.setArg( 1, other.m_buffer );
      multiple_multi_matmul_kernel.setArg( 2, out_buffer );
      multiple_multi_matmul_kernel.setArg( 3, sizeof(cl_int), &common );
      multiple_multi_matmul_kernel.setArg( 4, sizeof(cl_int), &B_w );
      multiple_multi_matmul_kernel.setArg( 5, sizeof(cl_int), &A_h );

      cl::NDRange global( C_N_ELEMENTS );
      ocl_queue.enqueueNDRangeKernel( multiple_multi_matmul_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in parallelMultiply: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return ParallelMat(out_buffer, A_h, other.m_width, m_count);
}

ParallelMat ParallelMat::transpose() const
{
   const int N_ELEMENTS = m_width * m_height * m_count;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   
   cl_int W = m_width;
   cl_int H = m_height;

   try {
   multiple_transpose_kernel.setArg( 0, m_buffer );
   multiple_transpose_kernel.setArg( 1, out_buffer );
   multiple_transpose_kernel.setArg( 2, sizeof(cl_int), &W );
   multiple_transpose_kernel.setArg( 3, sizeof(cl_int), &H );

   cl::NDRange global( N_ELEMENTS );
   ocl_queue.enqueueNDRangeKernel( multiple_transpose_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in parallelTranspose: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return ParallelMat(out_buffer, m_width, m_height, m_count); // W & H are swapped because of the transpose
}

ParallelMat ParallelMat::mat_add_sub_dot_op(char op, const ParallelMat &other) const
{
   assert(m_width == other.m_width && m_height == m_height && m_count == other.m_count);
   switch(op)
   {
      case '+': return mat_add_sub_dot(other, add_mat_kernel); break;
      case '-': return mat_add_sub_dot(other, sub_mat_kernel); break;
      case '^':
      case '.': return mat_add_sub_dot(other, dot_mat_kernel); break;
      default: throw std::exception();
   }
}

ParallelMat ParallelMat::mat_add_sub_dot(const ParallelMat &other, cl::Kernel& kernel) const
{
   const int N_ELEMENTS = m_width * m_height * m_count;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
   cl::NDRange global( N_ELEMENTS );
   try {
      kernel.setArg( 0, m_buffer );
      kernel.setArg( 1, other.m_buffer);
      kernel.setArg( 2, out_buffer );
      ocl_queue.enqueueNDRangeKernel( kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in mat_add_sub_dot: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(out_buffer, m_height, m_width, m_count);
}

ParallelMat ParallelMat::relu() const
{
   const int N_ELEMENTS = m_width*m_height*m_count;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   try {
      relu_kernel.setArg( 0, m_buffer );
      relu_kernel.setArg( 1, out_buffer);
      cl::NDRange global( N_ELEMENTS );
      ocl_queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in relu: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(out_buffer, m_height, m_width, m_count);
}

ParallelMat ParallelMat::relu_inv() const
{
   const int N_ELEMENTS = m_width*m_height*m_count;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   try {
      relu_inv_kernel.setArg( 0, m_buffer );
      relu_inv_kernel.setArg( 1, out_buffer);
      cl::NDRange global( N_ELEMENTS );
      ocl_queue.enqueueNDRangeKernel( relu_inv_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in relu_inv: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(out_buffer, m_height, m_width, m_count);
}

ParallelMat ParallelMat::sigmoid() const
{
   const int N_ELEMENTS = m_width*m_height*m_count;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   try {
      sigmoid_kernel.setArg( 0, m_buffer );
      sigmoid_kernel.setArg( 1, out_buffer);
      cl::NDRange global( N_ELEMENTS );
      ocl_queue.enqueueNDRangeKernel(sigmoid_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in sigmoid: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(out_buffer, m_height, m_width, m_count);
}

ParallelMat ParallelMat::sigmoid_inv() const
{
   const int N_ELEMENTS = m_width*m_height*m_count;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   try {
      sigmoid_inv_kernel.setArg( 0, m_buffer );
      sigmoid_inv_kernel.setArg( 1, out_buffer);
      cl::NDRange global( N_ELEMENTS );
      ocl_queue.enqueueNDRangeKernel( sigmoid_inv_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in sigmoid_inv: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(out_buffer, m_height, m_width, m_count);
}

ParallelMat ParallelMat::binary_crossentropy_loss(const ParallelMat& prediction) const
{
   const int N_ELEMENTS = m_width*m_height*m_count;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
   cl::NDRange global( N_ELEMENTS );
   try {
      binary_CEL_kernel.setArg( 0, m_buffer );
      binary_CEL_kernel.setArg( 1, prediction.m_buffer);
      binary_CEL_kernel.setArg( 2, out_buffer );
      ocl_queue.enqueueNDRangeKernel( binary_CEL_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in binary_crossentropy_loss: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(out_buffer, m_height, m_width, m_count);
}

ParallelMat ParallelMat::binary_crossentropy_loss_derivative(const ParallelMat& prediction)  const
{
   const int N_ELEMENTS = m_width*m_height*m_count;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
   cl::NDRange global( N_ELEMENTS );
   try {
      binary_CEL_derivative_kernel.setArg( 0, m_buffer );
      binary_CEL_derivative_kernel.setArg( 1, prediction.m_buffer);
      binary_CEL_derivative_kernel.setArg( 2, out_buffer );
      ocl_queue.enqueueNDRangeKernel( binary_CEL_derivative_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in binary_crossentropy_loss_derivative: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(out_buffer, m_height, m_width, m_count);
}
