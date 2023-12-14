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
   ConvKernel (unsigned channels,
               unsigned kernel_height,
               unsigned kernel_width,
               unsigned filters,
               Padding padding,
               unsigned input_height,
               unsigned input_width,
               const Mat& vals);

   static std::pair<unsigned,unsigned> getOutputHeightWidth(
               unsigned kernel_height,
               unsigned kernel_width,
               Padding padding,
               unsigned input_height,
               unsigned input_width);

   ParallelMat operator* (const ParallelMat &other) const;
   Mat operator* (const Mat &other) const;

private:
   cl::Buffer m_buffer;
   unsigned m_channels;
   unsigned m_height;
   unsigned m_width;
   Padding m_padding;
   unsigned m_filters;
   unsigned m_input_height;
   unsigned m_input_width;

   friend Mat;
};
