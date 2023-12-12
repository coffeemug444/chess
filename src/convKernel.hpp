#pragma once
#include "mat.hpp"

enum Padding
{
   SAME,
   NONE
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

   friend Mat;
};