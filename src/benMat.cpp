#include "benMat.hpp"
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

using std::vector, std::unique_ptr, std::array, std::async, std::future;
using namespace std::chrono_literals;

// static variable setup
std::random_device Mat::rd;
std::mt19937 Mat::gen = std::mt19937(rd());
bool Mat::openClIsSetup = false;
cl::Context Mat::context;
cl::CommandQueue Mat::queue;



void Mat::setup() {
   if (openClIsSetup) {
      return;
   }      
   try {
   unsigned int platform_id=0, device_id=0;
   std::vector<cl::Platform> platforms;
   cl::Platform::get(&platforms);
   std::vector<cl::Device> devices;
   platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_CPU, &devices);
   context = cl::Context(devices);
   queue = cl::CommandQueue( context, devices[device_id] );
   std::vector<std::string> sourcePaths = {
      "kernels/matmul.cl",
      "kernels/multiple_matmul.cl",
      "kernels/multiple_add.cl",
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
      "kernels/add_col.cl",
      "kernels/sub_col.cl",
      "kernels/dot_col.cl",
      "kernels/relu.cl",
      "kernels/relu_inv.cl"
   };
   cl::Program::Sources sources;
   for (auto& path : sourcePaths) {
      std::ifstream sourceFile(path);
      sources.push_back(std::string(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>())));
   }
   cl::Program program=cl::Program(context, sources);
   program.build(devices);

   matmul_kernel                    = cl::Kernel(program, "matmul");
   multiple_multi_matmul_kernel     = cl::Kernel(program, "multiple_multi_matmul");
   multiple_matmul_kernel           = cl::Kernel(program, "multiple_matmul");
   multiple_add_kernel              = cl::Kernel(program, "multiple_add");
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
   add_col_kernel                   = cl::Kernel(program, "add_col");
   sub_col_kernel                   = cl::Kernel(program, "sub_col");
   dot_col_kernel                   = cl::Kernel(program, "dot_col");
   relu_kernel                      = cl::Kernel(program, "relu");
   relu_inv_kernel                  = cl::Kernel(program, "relu_inv");

   queue.finish();

   openClIsSetup = true;
   }
   catch(cl::Error& err) {
      std::cout << "Error in setup: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
}

Mat::Mat()
{
   setup();

   _w = _h = 0;

   buffer = cl::Buffer(context, CL_MEM_READ_WRITE, 0);
}


Mat::Mat(unsigned int height, unsigned int width, std::vector<float> vals)
{
   setup();

   _w = width;
   _h = height;

   buffer = cl::Buffer(context, CL_MEM_READ_WRITE, (_w*_h)*sizeof(float));
   queue.enqueueWriteBuffer( buffer, CL_TRUE, 0, (_w*_h)*sizeof(float), vals.data() );
}

Mat::Mat(unsigned int height, unsigned int width, const cl::Buffer& new_buffer)
{
   setup();

   _w = width;
   _h = height;

   buffer = new_buffer;
}

Mat::Mat(const Mat &mat)
{
   setup();

   _w = mat._w;
   _h = mat._h;

   buffer = cl::Buffer(context, CL_MEM_READ_WRITE, (_w*_h)*sizeof(float));
   queue.enqueueCopyBuffer(mat.buffer, buffer, 0, 0, (_w*_h)*sizeof(float));
}

const Mat& Mat::operator=(const Mat &other)
{
   _w = other._w;
   _h = other._h;
   queue.enqueueCopyBuffer(other.buffer, buffer, 0, 0, (_w*_h)*sizeof(float));

   return *this;
}

const Mat& Mat::operator=(const Mat &&other)
{
   _w = other._w;
   _h = other._h;
   buffer = other.buffer;

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
   const int N_ELEMENTS = _w * _h;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
   cl::NDRange global( N_ELEMENTS );
   try {
      kernel.setArg( 0, buffer );
      kernel.setArg( 1, other.buffer);
      kernel.setArg( 2, out_buffer );
      queue.enqueueNDRangeKernel( kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in mat_add_sub_dot: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return Mat(_h, _w, out_buffer);
}

Mat& Mat::mat_add_sub_dot_eq(const Mat &other, cl::Kernel &kernel) {
   const int N_ELEMENTS = _w * _h;
   cl::NDRange global( N_ELEMENTS );
   try {
      kernel.setArg( 0, buffer );
      kernel.setArg( 1, other.buffer);
      queue.enqueueNDRangeKernel( kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in mat_add_sub_dot_eq: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   
   return *this;
}

Mat Mat::mat_add_sub_dot_op(char op, const Mat &other) const {
   assert(_h == other._h);
   assert(_w == other._w);

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
   assert(_h == other._h);
   assert(_w == other._w);

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
   const int N_ELEMENTS = _w*_h;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   try {
      relu_kernel.setArg( 0, buffer );
      relu_kernel.setArg( 1, out_buffer);
      cl::NDRange global( N_ELEMENTS );
      queue.enqueueNDRangeKernel(relu_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in relu: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return Mat(_h, _w, out_buffer);
}

Mat Mat::relu_inv() const
{
   const int N_ELEMENTS = _w*_h;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   try {
      relu_inv_kernel.setArg( 0, buffer );
      relu_inv_kernel.setArg( 1, out_buffer);
      cl::NDRange global( N_ELEMENTS );
      queue.enqueueNDRangeKernel( relu_inv_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in relu_inv: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return Mat(_h, _w, out_buffer);
}


Mat Mat::runFun(float function(float)) const
{
   std::vector<float> invals(_w * _h);
   std::vector<float> outvals(_w * _h);
   queue.enqueueReadBuffer( buffer, CL_TRUE, 0, (_w * _h)*sizeof(float), invals.data() );

   using fl_iter = std::vector<float>::iterator;
   vector<future<fl_iter>> active_pool{};

   for (unsigned row = 0; row < _h; row++)
   {
      shrinkActivePoolToSize(active_pool, 16);
      active_pool.push_back(std::async(
         std::transform<fl_iter, fl_iter, decltype(function)>
         , begin(invals) + row*_w, begin(invals) + row*(_w + 1), begin(outvals), function
      ));
   }
   for (auto &ftr : active_pool)
   {
      ftr.wait();
   }

   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, (_w * _h)*sizeof(float));
   queue.enqueueWriteBuffer( out_buffer, CL_FALSE, 0, (_w * _h)*sizeof(float), outvals.data() );

   return Mat(_h, _w, out_buffer);
}

Mat Mat::rectify() const
{
   std::vector<float> vals(_w * _h);
   std::vector<float> out_vals(_w * _h);
   queue.enqueueReadBuffer(buffer, CL_TRUE, 0, _w*_h*sizeof(float),vals.data());
   float sum = 0;
   for (unsigned i = 0; i < _h * _w; i++)
   {
      out_vals[i] = vals[i] * vals[i];
      sum += out_vals[i];
   }
   if (sum == 0)
   {
      return Mat(_h, _w, buffer);
   }

   return Mat(_h, _w, out_vals)/sum;
}

Mat Mat::float_op(char op, float val) const 
{
   const int N_ELEMENTS = _h*_w;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   
   cl_float buffer_val = val;
   try {

   cl::Kernel kernel;
   switch (op)
   {
      case '+': kernel = add_float_kernel; break;
      case '-': kernel = sub_float_kernel; break;
      case '*': kernel = mul_float_kernel; break;
      case '/': kernel = div_float_kernel; break;
      default: throw new std::exception();
   }

   kernel.setArg( 0, buffer );
   kernel.setArg( 1, out_buffer );
   kernel.setArg( 2, sizeof(cl_float), &buffer_val );

   cl::NDRange global( N_ELEMENTS );
   queue.enqueueNDRangeKernel( kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return Mat(_h, _w, out_buffer);
}

float Mat::getVal(unsigned i, unsigned j)
{
   assert(_h >= i && _w >= j);
   float outval;
   int offset = i*_w + j;
   queue.enqueueReadBuffer( buffer, CL_TRUE, offset*sizeof(float), sizeof(float), &outval );
   return outval;
}

Mat Mat::joinVector(const std::vector<Mat>& inputs) 
{
   const unsigned w = inputs[0].getWidth();
   const unsigned h = inputs[0].getHeight();
   for (const auto &input : inputs) {
      assert(input.getWidth() == w);
      assert(input.getHeight() == h);
   }
   const unsigned NUM_INPUTS = inputs.size();
   const unsigned INPUT_SIZE = w*h;

   std::vector<float> inputs_dup_data(NUM_INPUTS*INPUT_SIZE);
   for(unsigned i = 0; i < NUM_INPUTS; i++) {
      const std::vector<float> inputData = inputs[i].getVals();
      std::copy(begin(inputData), end(inputData), begin(inputs_dup_data) + i*INPUT_SIZE);
   }
   return Mat(NUM_INPUTS*h, w, inputs_dup_data);
}

std::vector<Mat> Mat::extractVector(const Mat& input_dup, unsigned NUM_INPUTS) 
{
   const unsigned w = input_dup.getWidth();
   const unsigned h = input_dup.getHeight() / NUM_INPUTS;
   const unsigned INPUT_SIZE = w*h;

   auto input_dup_data = input_dup.getVals();
   std::vector<Mat> inputs;
   for (unsigned i = 0; i < NUM_INPUTS; i++) {
      std::vector<float> input_data(begin(input_dup_data) + i*INPUT_SIZE, begin(input_dup_data) + (i+1)*INPUT_SIZE);
      inputs.push_back(Mat(h, w, input_data));
   }  
   return inputs;
}

Mat Mat::operator*(const Mat &other) const
{
   assert(_w == other._h);

   const int C_N_ELEMENTS = _h*other._w;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, C_N_ELEMENTS * sizeof(float));
   try {
      cl_int buffer_a_w = _w;
      cl_int buffer_b_w = other._w;
      matmul_kernel.setArg( 0, buffer );
      matmul_kernel.setArg( 1, other.buffer );
      matmul_kernel.setArg( 2, out_buffer );
      matmul_kernel.setArg( 3, sizeof(cl_int), &buffer_a_w );
      matmul_kernel.setArg( 4, sizeof(cl_int), &buffer_b_w );

      cl::NDRange global( C_N_ELEMENTS );
      queue.enqueueNDRangeKernel( matmul_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in operator*: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return Mat(_h, other.getWidth(), out_buffer);
}


const Mat& Mat::operator*=(const Mat &other)
{
   assert(_w == other._h);

   const int C_N_ELEMENTS = _h*other._w;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, C_N_ELEMENTS * sizeof(float));
   try {
      cl_int buffer_a_w = _w;
      cl_int buffer_b_w = other._w;
      matmul_kernel.setArg( 0, buffer );
      matmul_kernel.setArg( 1, other.buffer );
      matmul_kernel.setArg( 2, out_buffer );
      matmul_kernel.setArg( 3, sizeof(cl_int), &buffer_a_w );
      matmul_kernel.setArg( 4, sizeof(cl_int), &buffer_b_w );

      cl::NDRange global( C_N_ELEMENTS );
      queue.enqueueNDRangeKernel( matmul_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in operator*: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   buffer = out_buffer;
   _w = other.getWidth();

   return *this;
}


Mat Mat::multipleMultiply(const Mat& other, unsigned num_dupes) const
{
   assert(_w == other._h/num_dupes);

   const int C_N_ELEMENTS = _h*other._w*num_dupes;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, C_N_ELEMENTS * sizeof(float));
   try {
      cl_int buffer_common=_w;
      cl_int buffer_a_h=_h;
      cl_int buffer_b_w=other._w;

      multiple_matmul_kernel.setArg( 0, buffer );
      multiple_matmul_kernel.setArg( 1, other.buffer );
      multiple_matmul_kernel.setArg( 2, out_buffer );
      multiple_matmul_kernel.setArg( 3, sizeof(cl_int), &buffer_common );
      multiple_matmul_kernel.setArg( 4, sizeof(cl_int), &buffer_b_w );
      multiple_matmul_kernel.setArg( 5, sizeof(cl_int), &buffer_a_h );

      cl::NDRange global( C_N_ELEMENTS );
      queue.enqueueNDRangeKernel( multiple_matmul_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in multipleMultiply: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return Mat(_h*num_dupes, other.getWidth(), out_buffer);
}

// sometimes you just need to multiply a whole bunch of matrices at the same time:
// A1  x  B1  ->  C1
// A2  x  B2  ->  C2
// A3  x  B3  ->  C3
// A4  x  B4  ->  C4
// A5  x  B5  ->  C5
// A6  x  B6  ->  C6
// A7  x  B7  ->  C7
// A8  x  B8  ->  C8
//
// that's what this method is for
// all of the A matrices must be the same shape, likewise with B's
Mat Mat::multipleMultiMultiply(const Mat& other, unsigned num_dupes) const
{
   assert(_w == other._h/num_dupes);

   cl_int common = _w;
   cl_int A_h    = _h/num_dupes;
   cl_int B_w    = other._w;
   const int C_N_ELEMENTS = A_h*B_w*num_dupes;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, C_N_ELEMENTS * sizeof(float));
   try {
      multiple_multi_matmul_kernel.setArg( 0, buffer );
      multiple_multi_matmul_kernel.setArg( 1, other.buffer );
      multiple_multi_matmul_kernel.setArg( 2, out_buffer );
      multiple_multi_matmul_kernel.setArg( 3, sizeof(cl_int), &common );
      multiple_multi_matmul_kernel.setArg( 4, sizeof(cl_int), &B_w );
      multiple_multi_matmul_kernel.setArg( 5, sizeof(cl_int), &A_h );

      cl::NDRange global( C_N_ELEMENTS );
      queue.enqueueNDRangeKernel( multiple_multi_matmul_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in multipleMultiMultiply: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return Mat(A_h*num_dupes, other.getWidth(), out_buffer);
}

std::vector<float> Mat::getVals() const {
   const int N_ELEMENTS = _w * _h;
   std::vector<float> out(N_ELEMENTS);

   queue.enqueueReadBuffer(buffer, CL_TRUE, 0, N_ELEMENTS*sizeof(float), out.data());
   return out;
}

Mat Mat::multipleAdd(const Mat& other, unsigned num_dupes) const
{
   const int N_ELEMENTS = other._w * other._h;
   const int B_size = N_ELEMENTS / num_dupes;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
   cl_int bufferB_size=B_size;
   cl::NDRange global( N_ELEMENTS );
   try {
      multiple_add_kernel.setArg( 0, buffer );
      multiple_add_kernel.setArg( 1, other.buffer);
      multiple_add_kernel.setArg( 2, out_buffer );
      multiple_add_kernel.setArg( 3, sizeof(cl_int), &bufferB_size );
      queue.enqueueNDRangeKernel( multiple_add_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in multipleAdd: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return Mat(_h*num_dupes, _w, out_buffer);
}

Mat Mat::transpose() const
{
   const int N_ELEMENTS = _w * _h;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   
   cl_int W = _w;
   cl_int H = _h;

   try {
   transpose_kernel.setArg( 0, buffer );
   transpose_kernel.setArg( 1, out_buffer );
   transpose_kernel.setArg( 2, sizeof(cl_int), &W );
   transpose_kernel.setArg( 3, sizeof(cl_int), &H );

   cl::NDRange global( N_ELEMENTS );
   queue.enqueueNDRangeKernel( transpose_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in transpose: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return Mat(_w, _h, out_buffer);
}

Mat Mat::multipleTranspose(unsigned num_dupes) const
{
   const int N_ELEMENTS = _w * _h;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
   
   cl_int W = _w;
   cl_int H = _h / num_dupes;

   try {
   multiple_transpose_kernel.setArg( 0, buffer );
   multiple_transpose_kernel.setArg( 1, out_buffer );
   multiple_transpose_kernel.setArg( 2, sizeof(cl_int), &W );
   multiple_transpose_kernel.setArg( 3, sizeof(cl_int), &H );

   cl::NDRange global( N_ELEMENTS );
   queue.enqueueNDRangeKernel( multiple_transpose_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in transpose: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return Mat(_w*num_dupes, _h/num_dupes, out_buffer);
}


Mat Mat::multipleSum(unsigned num_dupes) const
{
   const cl_int arraySize = _w * _h / num_dupes;
   cl::Buffer out_buffer(context, CL_MEM_READ_WRITE, arraySize * sizeof(float));
   cl_int numArrays=num_dupes;

   try {
   multiple_sum_kernel.setArg( 0, buffer );
   multiple_sum_kernel.setArg( 1, out_buffer );
   multiple_sum_kernel.setArg( 2, sizeof(cl_int), &numArrays );
   multiple_sum_kernel.setArg( 3, sizeof(cl_int), &arraySize );

   cl::NDRange global( arraySize );
   queue.enqueueNDRangeKernel( multiple_sum_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in transpose: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return Mat(_h/num_dupes, _w, out_buffer);
}
