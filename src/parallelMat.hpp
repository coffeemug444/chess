#pragma once
#include "mat.hpp"

class ConvKernel;

class ParallelMat
{
public:
   ParallelMat ();
   ParallelMat (const std::vector<Mat>& mats);

   std::vector<Mat> toVector() const;
   Mat sum() const;

   ParallelMat operator* (const ParallelMat &other) const;
   ParallelMat operator+ (const ParallelMat &other) const { return mat_add_sub_dot_op('+', other); };
   ParallelMat operator- (const ParallelMat &other) const { return mat_add_sub_dot_op('-', other); };
   ParallelMat operator^ (const ParallelMat &other) const { return mat_add_sub_dot_op('^', other); };

   ParallelMat transpose() const;

   ParallelMat relu() const;
   ParallelMat relu_inv() const;
   ParallelMat sigmoid() const;
   ParallelMat sigmoid_inv() const;

   ParallelMat log() const;
   ParallelMat exp() const;
   ParallelMat softmax() const;

   // assumes this matrix is your true values, `prediction` is your nn output
   ParallelMat binary_crossentropy_loss(const ParallelMat& prediction) const;
   ParallelMat binary_crossentropy_loss_derivative(const ParallelMat& prediction) const;

   unsigned getWidth() const { return m_width; }
   unsigned getHeight() const { return m_height; }
   unsigned getCount() const { return m_count; }

private:
   ParallelMat( unsigned height, unsigned width, unsigned count, const cl::Buffer& buffer )
      :m_buffer{buffer}
      ,m_height{height}
      ,m_width{width}
      ,m_count{count}
   {
   }

   cl::Buffer m_buffer;
   unsigned m_height = 0;
   unsigned m_width = 0;
   unsigned m_count = 0;


   ParallelMat mat_add_sub_dot_op(char op, const ParallelMat &other) const;
   ParallelMat mat_add_sub_dot(const ParallelMat &other, cl::Kernel& kernel) const;

   friend Mat;
   friend ConvKernel;
};