#pragma once
#include "mat.hpp"

enum Padding
{
   VALID = 0,  // no padding
   SAME = 1    // output has same dimensions as input
};

class ConvKernel
{
public:
   ConvKernel (int channels,
               int kernel_height,
               int kernel_width,
               Padding padding,
               int filters,
               int input_height,
               int input_width,
               const Mat& vals);

   ParallelMat operator* (const ParallelMat &other) const;
   Mat operator* (const Mat &other) const;

private:
   cl::Buffer m_buffer;
   int m_channels;
   int m_height;
   int m_width;
   Padding m_padding;
   int m_filters;
   int m_input_height;
   int m_input_width;

   friend Mat;
};
