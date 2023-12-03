#ifndef MAT
#define MAT

#include <vector>
#include <assert.h>
#include <random>
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

class Mat {
   private:
      static std::random_device rd; 
      static std::mt19937 gen;
      static bool openClIsSetup;
      static cl::Context context;
      static cl::CommandQueue queue;
      
      inline static cl::Kernel matmul_kernel;
      inline static cl::Kernel multiple_multi_matmul_kernel;
      inline static cl::Kernel multiple_matmul_kernel;
      inline static cl::Kernel multiple_add_kernel;
      inline static cl::Kernel multiple_transpose_kernel;
      inline static cl::Kernel multiple_sum_kernel;
      inline static cl::Kernel transpose_kernel;
      inline static cl::Kernel div_float_kernel;
      inline static cl::Kernel mul_float_kernel;
      inline static cl::Kernel add_float_kernel;
      inline static cl::Kernel sub_float_kernel;
      inline static cl::Kernel div_float_eq_kernel;
      inline static cl::Kernel mul_float_eq_kernel;
      inline static cl::Kernel add_float_eq_kernel;
      inline static cl::Kernel sub_float_eq_kernel;
      inline static cl::Kernel add_mat_kernel;
      inline static cl::Kernel sub_mat_kernel;
      inline static cl::Kernel dot_mat_kernel;
      inline static cl::Kernel add_mat_eq_kernel;
      inline static cl::Kernel sub_mat_eq_kernel;
      inline static cl::Kernel dot_mat_eq_kernel;
      inline static cl::Kernel add_col_kernel;
      inline static cl::Kernel sub_col_kernel;
      inline static cl::Kernel dot_col_kernel;
      inline static cl::Kernel relu_kernel;
      inline static cl::Kernel relu_inv_kernel;

      static void setup();

      cl::Buffer buffer;
      unsigned _w = 0;
      unsigned _h = 0;
      Mat float_op(char op, float val) const;
      const Mat& float_eq_op(char op, float val);
      Mat mat_add_sub_dot_op(char op, const Mat &other) const;
      Mat mat_add_sub_dot(const Mat &other, cl::Kernel& kernel) const;

      Mat& mat_add_sub_dot_eq_op(char op, const Mat &other);
      Mat& mat_add_sub_dot_eq(const Mat &other, cl::Kernel& kernel);

      Mat(unsigned rows, unsigned cols, const cl::Buffer& buffer);

   public:
      Mat(unsigned rows, unsigned cols, std::vector<float> vals);
      Mat(const Mat& mat);
      Mat();


      Mat multipleMultiMultiply(const Mat& other, unsigned num_dupes) const;
      Mat multipleMultiply(const Mat& other, unsigned num_dupes) const;
      Mat multipleAdd(const Mat& other, unsigned num_dupes) const;

      Mat operator* (const Mat &other) const;
      Mat operator+ (const Mat &other) const { return mat_add_sub_dot_op('+', other); };
      Mat operator- (const Mat &other) const { return mat_add_sub_dot_op('-', other); };
      Mat operator^ (const Mat &other) const { return mat_add_sub_dot_op('^', other); };
      Mat operator+ (float other) const { return float_op('+', other); };
      Mat operator- (float other) const { return float_op('-', other); };
      Mat operator* (float other) const { return float_op('*', other); };
      Mat operator/ (float other) const { return float_op('/', other); };

      const Mat& operator*= (const Mat &other);
      const Mat& operator+= (const Mat &other) { return mat_add_sub_dot_eq_op('+', other); };
      const Mat& operator-= (const Mat &other) { return mat_add_sub_dot_eq_op('-', other); };
      const Mat& operator^= (const Mat &other) { return mat_add_sub_dot_eq_op('^', other); };
      const Mat& operator+= (float other) { return float_eq_op('+', other); };
      const Mat& operator-= (float other) { return float_eq_op('-', other); };
      const Mat& operator*= (float other) { return float_eq_op('*', other); };
      const Mat& operator/= (float other) { return float_eq_op('/', other); };

      const Mat& operator= (const Mat &other);
      const Mat& operator= (const Mat &&other);

      Mat relu() const;
      Mat relu_inv() const;
      
      Mat transpose() const;
      Mat multipleTranspose(unsigned num_dupes) const;

      Mat multipleSum(unsigned num_dupes) const;

      float getVal(unsigned row, unsigned col);
      std::vector<float> getVals() const;

      static Mat joinVector(const std::vector<Mat>& inputs);
      static std::vector<Mat> extractVector(const Mat& input_dup, unsigned num_inputs);

      Mat runFun(float function(float)) const;
      Mat rectify() const;

      unsigned getWidth() const { return _w; };
      unsigned getHeight() const { return _h; };

      static Mat zeros(unsigned rows, unsigned cols);
      static Mat ones(unsigned rows, unsigned cols);
      static Mat val(unsigned rows, unsigned cols, float val);
      static Mat random(unsigned rows, unsigned cols);
      static Mat he(unsigned height, unsigned width);
};



#endif