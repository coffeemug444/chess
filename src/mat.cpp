#include "mat.hpp"
#include "parallelMat.hpp"
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <array>
#include <string>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <future>
#include <chrono>
#include <omp.h>
#include <fstream>
#include "errors.hpp"
#include <mutex>
#include "oclData.hpp"

using std::vector, std::unique_ptr, std::array, std::async, std::future;
using namespace std::chrono_literals;

// static variable setup
std::random_device Mat::rd;
std::mt19937 Mat::gen = std::mt19937(rd());

Mat operator* (float f, const Mat& mat) { return mat.float_op('*', f); }


std::ostream& operator<<(std::ostream& out, const Mat& mat)
{
   int h = mat.getHeight();
   int w = mat.getWidth();
   auto vals = mat.getVals();

   std::string row_delim = "[";
   for (int row = 0; row < h; row++)
   {
      out << row_delim << "[";
      row_delim = ",\n ";

      std::string delim = "";
      for (int col = 0; col < w; col++)
      {
         out << delim << vals[row*w + col];
         delim = ",";
      }

      out << "]";
   }
   out << "]";
   
   return out;
}

ParallelMat operator+(const ParallelMat& l, const Mat& r)
{
   return r + l;
}


void Mat::setup() {
   if (ocl_setup) return;
   ocl_init();
}

Mat::Mat()
{
   setup();

   m_width = m_height = 0;

   m_buffer = cl::Buffer(ocl_context, CL_MEM_READ_WRITE, 0);
}


Mat::Mat(unsigned int height, unsigned int width, const std::vector<float>& vals)
{
   setup();

   m_width = width;
   m_height = height;

   m_buffer = cl::Buffer(ocl_context, CL_MEM_READ_WRITE, (m_width*m_height)*sizeof(float));
   ocl_queue.enqueueWriteBuffer( m_buffer, CL_TRUE, 0, (m_width*m_height)*sizeof(float), vals.data() );
}

Mat::Mat(unsigned int height, unsigned int width, const cl::Buffer& new_buffer)
{
   setup();

   m_width = width;
   m_height = height;

   m_buffer = new_buffer;
}

Mat::Mat(const Mat &mat)
{
   setup();

   m_width = mat.m_width;
   m_height = mat.m_height;

   m_buffer = cl::Buffer(ocl_context, CL_MEM_READ_WRITE, (m_width*m_height)*sizeof(float));
   ocl_queue.enqueueCopyBuffer(mat.m_buffer, m_buffer, 0, 0, (m_width*m_height)*sizeof(float));
}

const Mat& Mat::operator=(const Mat &other)
{
   m_width = other.m_width;
   m_height = other.m_height;
   ocl_queue.enqueueCopyBuffer(other.m_buffer, m_buffer, 0, 0, (m_width*m_height)*sizeof(float));

   return *this;
}

const Mat& Mat::operator=(const Mat &&other)
{
   m_width = other.m_width;
   m_height = other.m_height;
   m_buffer = other.m_buffer;

   return *this;
}

Mat Mat::ones(unsigned height, unsigned width)
{
   return val(height, width, 1.f);
}

Mat Mat::zeros(unsigned height, unsigned width)
{
   return val(height, width, 0.f);
}

Mat Mat::val(unsigned height, unsigned width, float val)
{
   std::vector<float> vals(height*width);
   std::fill(begin(vals), end(vals), val);
   return Mat(height, width, vals);
}

Mat Mat::random(unsigned height, unsigned width)
{
   std::vector<float> vals(height*width);
   std::normal_distribution<float> d(0.f, 1.f);
   for (unsigned i = 0; i < height * width; i++)
   {
      vals[i] = d(gen);
   }

   return Mat(height, width, vals);
}

Mat Mat::he(unsigned height, unsigned width)
{
   std::vector<float> vals(height*width);
   float deviation = sqrt(2.0f / ((float)width));
   std::normal_distribution<float> d(0.f, deviation);
   for (unsigned i = 0; i < height * width; i++)
   {
      vals[i] = d(gen);
   }

   return Mat(height, width, vals);
}

Mat Mat::mat_add_sub_dot(const Mat &other, cl::Kernel &kernel) const {
   const int N_ELEMENTS = m_width * m_height;
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
   return Mat(m_height, m_width, out_buffer);
}

Mat& Mat::mat_add_sub_dot_eq(const Mat &other, cl::Kernel &kernel) {
   const int N_ELEMENTS = m_width * m_height;
   cl::NDRange global( N_ELEMENTS );
   try {
      kernel.setArg( 0, m_buffer );
      kernel.setArg( 1, other.m_buffer);
      ocl_queue.enqueueNDRangeKernel( kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in mat_add_sub_dot_eq: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   
   return *this;
}


const Mat& Mat::float_eq_op(char op, float val)
{
   const int N_ELEMENTS = m_width * m_height;
   cl::NDRange global( N_ELEMENTS );

   cl_float buffer_val = val;

   try {

      cl::Kernel& kernel = [&]() -> cl::Kernel& {
         switch (op)
         {
            case '*': return mul_float_eq_kernel;
            case '/': return div_float_eq_kernel;
            case '+': return add_float_eq_kernel;
            case '-': return sub_float_eq_kernel;
            default: throw new std::exception();
         }
      }();

      kernel.setArg( 0, m_buffer );
      kernel.setArg( 1, sizeof(cl_float), &buffer_val );

      ocl_queue.enqueueNDRangeKernel( kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in mat_add_sub_dot_eq: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   
   return *this;
}

Mat Mat::mat_add_sub_dot_op(char op, const Mat &other) const {
   assert(m_height == other.m_height);
   assert(m_width == other.m_width);

   switch (op)
   {
      case '+': return (mat_add_sub_dot(other, add_mat_kernel));
      case '-': return (mat_add_sub_dot(other, sub_mat_kernel));
      case '^':
      case '.': return (mat_add_sub_dot(other, dot_mat_kernel));
      default : throw std::exception();
   }
}

Mat& Mat::mat_add_sub_dot_eq_op(char op, const Mat &other) {
   assert(m_height == other.m_height);
   assert(m_width == other.m_width);

   switch (op)
   {
      case '+': return (mat_add_sub_dot_eq(other, add_mat_eq_kernel));
      case '-': return (mat_add_sub_dot_eq(other, sub_mat_eq_kernel));
      case '^':
      case '.': return (mat_add_sub_dot_eq(other, dot_mat_eq_kernel));
      default : throw std::exception();
   }
}

template <class T>
void shrinkActivePoolToSize(vector<future<T>> &active_pool, unsigned s)
{
   while (active_pool.size() >= s)
   {
      for (unsigned i = 0; i < active_pool.size(); i++)
      {
         if (active_pool[i].wait_for(0ms) == std::future_status::ready)
         {
            active_pool.erase(active_pool.begin() + i);
            i--;
         }
      }
   }
}


Mat Mat::relu() const
{
   const int N_ELEMENTS = m_width*m_height;
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
   return Mat(m_height, m_width, out_buffer);
}

Mat Mat::relu_inv() const
{
   const int N_ELEMENTS = m_width*m_height;
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
   return Mat(m_height, m_width, out_buffer);
}


Mat Mat::sigmoid() const
{
   const int N_ELEMENTS = m_width*m_height;
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
   return Mat(m_height, m_width, out_buffer);
}

Mat Mat::sigmoid_inv() const
{
   const int N_ELEMENTS = m_width*m_height;
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
   return Mat(m_height, m_width, out_buffer);
}

Mat Mat::log() const
{
   const int N_ELEMENTS = m_width*m_height;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   try {
      log_kernel.setArg( 0, m_buffer );
      log_kernel.setArg( 1, out_buffer);
      cl::NDRange global( N_ELEMENTS );
      ocl_queue.enqueueNDRangeKernel( log_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in log: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return Mat(m_height, m_width, out_buffer);
}

Mat Mat::exp() const
{
   const int N_ELEMENTS = m_width*m_height;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   try {
      exp_kernel.setArg( 0, m_buffer );
      exp_kernel.setArg( 1, out_buffer);
      cl::NDRange global( N_ELEMENTS );
      ocl_queue.enqueueNDRangeKernel( exp_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in exp: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return Mat(m_height, m_width, out_buffer);
}

Mat Mat::softmax() const
{
   auto e = exp();
   return e / (e.sum());
}

Mat Mat::binary_crossentropy_loss(const Mat& prediction) const
{
   const int N_ELEMENTS = m_width * m_height;
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
   return Mat(m_height, m_width, out_buffer);
}

Mat Mat::binary_crossentropy_loss_derivative(const Mat& prediction)  const
{
   const int N_ELEMENTS = m_width * m_height;
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
   return Mat(m_height, m_width, out_buffer);
}

// sum of all the elements in this matrix
float Mat::sum() const
{
   auto vals = getVals();
   return std::accumulate(vals.begin(), vals.end(), 0.f);
}

float Mat::sumOfSquares() const
{
   auto vals = getVals();
   float total = 0;
   for (float val : vals)
   {
      total += val * val;
   }

   return total;
}


Mat Mat::runFun(float function(float)) const
{
   std::vector<float> invals(m_width * m_height);
   std::vector<float> outvals(m_width * m_height);
   ocl_queue.enqueueReadBuffer( m_buffer, CL_TRUE, 0, (m_width * m_height)*sizeof(float), invals.data() );

   using fl_iter = std::vector<float>::iterator;
   vector<future<fl_iter>> active_pool{};

   for (unsigned row = 0; row < m_height; row++)
   {
      shrinkActivePoolToSize(active_pool, 16);
      active_pool.push_back(std::async(
         std::transform<fl_iter, fl_iter, decltype(function)>
         , begin(invals) + row*m_width, begin(invals) + row*(m_width + 1), begin(outvals), function
      ));
   }
   for (auto &ftr : active_pool)
   {
      ftr.wait();
   }

   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, (m_width * m_height)*sizeof(float));
   ocl_queue.enqueueWriteBuffer( out_buffer, CL_FALSE, 0, (m_width * m_height)*sizeof(float), outvals.data() );

   return Mat(m_height, m_width, out_buffer);
}

Mat Mat::rectify() const
{
   std::vector<float> vals(m_width * m_height);
   std::vector<float> out_vals(m_width * m_height);
   ocl_queue.enqueueReadBuffer(m_buffer, CL_TRUE, 0, m_width*m_height*sizeof(float),vals.data());
   float sum = 0;
   for (unsigned i = 0; i < m_height * m_width; i++)
   {
      out_vals[i] = vals[i] * vals[i];
      sum += out_vals[i];
   }
   if (sum == 0)
   {
      return Mat(m_height, m_width, m_buffer);
   }

   return Mat(m_height, m_width, out_vals)/sum;
}

Mat Mat::float_op(char op, float val) const 
{
   const int N_ELEMENTS = m_height*m_width;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   
   cl_float buffer_val = val;
   try {

   cl::Kernel& kernel = [&]() -> cl::Kernel& {
      switch (op)
      {
         case '*': return mul_float_kernel;
         case '/': return div_float_kernel;
         case '+': return add_float_kernel;
         case '-': return sub_float_kernel;
         default: throw new std::exception();
      }
   }();

   kernel.setArg( 0, m_buffer );
   kernel.setArg( 1, out_buffer );
   kernel.setArg( 2, sizeof(cl_float), &buffer_val );

   cl::NDRange global( N_ELEMENTS );
   ocl_queue.enqueueNDRangeKernel( kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return Mat(m_height, m_width, out_buffer);
}

float Mat::getVal(unsigned i, unsigned j)
{
   assert(m_height >= i && m_width >= j);
   float outval;
   int offset = i*m_width + j;
   ocl_queue.enqueueReadBuffer( m_buffer, CL_TRUE, offset*sizeof(float), sizeof(float), &outval );
   return outval;
}

Mat Mat::operator*(const Mat &other) const
{
   assert(m_width == other.m_height);

   const int C_N_ELEMENTS = m_height*other.m_width;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, C_N_ELEMENTS * sizeof(float));
   try {
      cl_int buffer_a_w = m_width;
      cl_int buffer_b_w = other.m_width;
      matmul_kernel.setArg( 0, m_buffer );
      matmul_kernel.setArg( 1, other.m_buffer );
      matmul_kernel.setArg( 2, out_buffer );
      matmul_kernel.setArg( 3, sizeof(cl_int), &buffer_a_w );
      matmul_kernel.setArg( 4, sizeof(cl_int), &buffer_b_w );

      cl::NDRange global( C_N_ELEMENTS );
      ocl_queue.enqueueNDRangeKernel( matmul_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in operator*: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return Mat(m_height, other.getWidth(), out_buffer);
}


const Mat& Mat::operator*=(const Mat &other)
{
   assert(m_width == other.m_height);

   const int C_N_ELEMENTS = m_height*other.m_width;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, C_N_ELEMENTS * sizeof(float));
   try {
      cl_int buffer_a_w = m_width;
      cl_int buffer_b_w = other.m_width;
      matmul_kernel.setArg( 0, m_buffer );
      matmul_kernel.setArg( 1, other.m_buffer );
      matmul_kernel.setArg( 2, out_buffer );
      matmul_kernel.setArg( 3, sizeof(cl_int), &buffer_a_w );
      matmul_kernel.setArg( 4, sizeof(cl_int), &buffer_b_w );

      cl::NDRange global( C_N_ELEMENTS );
      ocl_queue.enqueueNDRangeKernel( matmul_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in operator*: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   m_buffer = out_buffer;
   m_width = other.getWidth();

   return *this;
}

ParallelMat Mat::operator* (const ParallelMat &other) const
{
   assert(m_width == other.m_height);

   const int C_N_ELEMENTS = m_height*other.m_width*other.m_count;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, C_N_ELEMENTS * sizeof(float));
   try {
      cl_int buffer_common=m_width;
      cl_int buffer_a_h=m_height;
      cl_int buffer_b_w=other.m_width;

      multiple_matmul_kernel.setArg( 0, m_buffer );
      multiple_matmul_kernel.setArg( 1, other.m_buffer );
      multiple_matmul_kernel.setArg( 2, out_buffer );
      multiple_matmul_kernel.setArg( 3, sizeof(cl_int), &buffer_common );
      multiple_matmul_kernel.setArg( 4, sizeof(cl_int), &buffer_b_w );
      multiple_matmul_kernel.setArg( 5, sizeof(cl_int), &buffer_a_h );

      cl::NDRange global( C_N_ELEMENTS );
      ocl_queue.enqueueNDRangeKernel( multiple_matmul_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in multipleMultiply: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return ParallelMat(m_height, other.m_width, other.m_count, out_buffer);
}

std::vector<float> Mat::getVals() const {
   const int N_ELEMENTS = m_width * m_height;
   std::vector<float> out(N_ELEMENTS);

   ocl_queue.enqueueReadBuffer(m_buffer, CL_TRUE, 0, N_ELEMENTS*sizeof(float), out.data());
   return out;
}


ParallelMat Mat::operator+ (const ParallelMat &other) const
{
   assert(m_width == other.m_width && m_height == other.m_height);

   const int N_ELEMENTS = m_width * m_height * other.m_count;
   const int B_size = m_width * m_height;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
   cl_int bufferB_size=B_size;
   cl::NDRange global( N_ELEMENTS );
   try {
      multiple_add_kernel.setArg( 0, m_buffer );
      multiple_add_kernel.setArg( 1, other.m_buffer);
      multiple_add_kernel.setArg( 2, out_buffer );
      multiple_add_kernel.setArg( 3, sizeof(cl_int), &bufferB_size );
      ocl_queue.enqueueNDRangeKernel( multiple_add_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in multipleAdd: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(m_height, m_width, other.m_count, out_buffer);
}

ParallelMat Mat::operator^ (const ParallelMat &other) const
{
   assert(m_width == other.m_width && m_height == other.m_height);

   const int N_ELEMENTS = m_width * m_height * other.m_count;
   const int B_size = m_width * m_height;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
   cl_int bufferB_size=B_size;
   cl::NDRange global( N_ELEMENTS );
   try {
      multiple_dot_kernel.setArg( 0, m_buffer );
      multiple_dot_kernel.setArg( 1, other.m_buffer);
      multiple_dot_kernel.setArg( 2, out_buffer );
      multiple_dot_kernel.setArg( 3, sizeof(cl_int), &bufferB_size );
      ocl_queue.enqueueNDRangeKernel( multiple_dot_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in multipleAdd: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(m_height, m_width, other.m_count, out_buffer);
}

Mat Mat::transpose() const
{
   const int N_ELEMENTS = m_width * m_height;
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   
   cl_int W = m_width;
   cl_int H = m_height;

   try {
   transpose_kernel.setArg( 0, m_buffer );
   transpose_kernel.setArg( 1, out_buffer );
   transpose_kernel.setArg( 2, sizeof(cl_int), &W );
   transpose_kernel.setArg( 3, sizeof(cl_int), &H );

   cl::NDRange global( N_ELEMENTS );
   ocl_queue.enqueueNDRangeKernel( transpose_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in transpose: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return Mat(m_width, m_height, out_buffer);
}
