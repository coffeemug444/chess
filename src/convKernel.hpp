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
               Padding padding);

   ParallelMat operator* (const ParallelMat &other) const;
   Mat operator* (const Mat &other) const;

private:
   cl::Buffer m_buffer;
   int m_channels;
   int m_height;
   int m_width;
   Padding m_padding;

   friend Mat;
};
