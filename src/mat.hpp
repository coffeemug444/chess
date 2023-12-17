#pragma once

#include <vector>
#include <assert.h>
#include <ostream>
#include <random>
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

class ParallelMat;
class Mat;
class ConvKernel;

Mat operator* (float f, const Mat& mat);
std::ostream& operator<<(std::ostream& out, const Mat& mat);
ParallelMat operator+(const ParallelMat& l, const Mat& r);

class Mat {
private:
   static std::random_device rd; 
   static std::mt19937 gen;

   static void setup();

   cl::Buffer m_buffer;
   unsigned m_width = 0;
   unsigned m_height = 0;
   Mat float_op(char op, float val) const;
   const Mat& float_eq_op(char op, float val);
   Mat mat_add_sub_dot_op(char op, const Mat &other) const;
   Mat mat_add_sub_dot(const Mat &other, cl::Kernel& kernel) const;

   Mat& mat_add_sub_dot_eq_op(char op, const Mat &other);
   Mat& mat_add_sub_dot_eq(const Mat &other, cl::Kernel& kernel);

   Mat(unsigned height, unsigned width, const cl::Buffer& buffer);

public:

   Mat(unsigned height, unsigned width, const std::vector<float>& vals);
   Mat(const Mat& mat);
   Mat();

   ParallelMat operator* (const ParallelMat &other) const;
   ParallelMat operator+ (const ParallelMat &other) const;
   ParallelMat operator^ (const ParallelMat &other) const;

   Mat operator* (const Mat &other) const;
   Mat operator+ (const Mat &other) const { return mat_add_sub_dot_op('+', other); };
   Mat operator- (const Mat &other) const { return mat_add_sub_dot_op('-', other); };
   Mat operator^ (const Mat &other) const { return mat_add_sub_dot_op('^', other); };

   Mat operator/ (float other) const { return float_op('/', other); };
   Mat operator+ (float other) const { return float_op('+', other); };
   Mat operator- (float other) const { return float_op('-', other); };
   friend Mat operator* (float f, const Mat& mat);

   const Mat& operator*= (const Mat &other);
   const Mat& operator+= (const Mat &other) { return mat_add_sub_dot_eq_op('+', other); };
   const Mat& operator-= (const Mat &other) { return mat_add_sub_dot_eq_op('-', other); };
   const Mat& operator^= (const Mat &other) { return mat_add_sub_dot_eq_op('^', other); };
   const Mat& operator*= (float other) { return float_eq_op('*', other); };
   const Mat& operator/= (float other) { return float_eq_op('/', other); };
   const Mat& operator+= (float other) { return float_eq_op('+', other); };
   const Mat& operator-= (float other) { return float_eq_op('-', other); };

   const Mat& operator= (const Mat &other);
   const Mat& operator= (const Mat &&other);


   // assumes this matrix is your true values, `prediction` is your nn output
   Mat binary_crossentropy_loss(const Mat& prediction) const;
   Mat binary_crossentropy_loss_derivative(const Mat& prediction) const;

   // sum of all the elements in this matrix
   float sum() const;
   float sumOfSquares() const;

   Mat relu() const;
   Mat relu_inv() const;

   Mat sigmoid() const;
   Mat sigmoid_inv() const;

   Mat log() const;
   Mat exp() const;

   Mat softmax() const;

   Mat transpose() const;

   float getVal(unsigned row, unsigned col);
   std::vector<float> getVals() const;

   Mat runFun(float function(float)) const;
   Mat rectify() const;

   unsigned getWidth() const { return m_width; };
   unsigned getHeight() const { return m_height; };

   static Mat zeros(unsigned rows, unsigned cols);
   static Mat ones(unsigned rows, unsigned cols);
   static Mat val(unsigned rows, unsigned cols, float val);
   static Mat random(unsigned rows, unsigned cols);
   static Mat he(unsigned height, unsigned width);

   friend ParallelMat;
   friend ConvKernel;
};

