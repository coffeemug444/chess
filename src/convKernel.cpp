#include <iostream>

#include "convKernel.hpp"
#include "parallelMat.hpp"
#include "oclData.hpp"
#include "errors.hpp"


ConvKernel::ConvKernel (int channels,
                        int kernel_height,
                        int kernel_width,
                        Padding padding)
   :m_channels{channels}
   ,m_height{kernel_height}
   ,m_width{kernel_width}
   ,m_padding{padding}
{
   
}

ParallelMat ConvKernel::operator* (const ParallelMat &other) const
{
   cl_int convkernel_w = m_width;
   cl_int convkernel_h = m_height;
   cl_int input_w = other.getWidth();
   cl_int input_h = other.getHeight();
   cl_int output_h;
   cl_int output_w;
   cl_int u_padding;
   cl_int l_padding;

   if (m_padding == SAME)
   {
      output_h = input_h;
      output_w = input_w;
      l_padding = convkernel_w / 2; // r_padding = (convkernel_w - 1) - l_padding
      u_padding = convkernel_h / 2; // d_padding = (convkernel_h - 1) - u_padding
   }
   else
   {
      output_h = input_h - convkernel_h + 1;
      output_w = input_w - convkernel_w + 1;
      u_padding = 0;
      l_padding = 0;
   }

   int N_ELEMENTS = output_h*output_w*other.m_count;
   cl::NDRange global( N_ELEMENTS );
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
       
   try {
      convolution_kernel.setArg( 0,  m_buffer );
      convolution_kernel.setArg( 1,  other.m_buffer);
      convolution_kernel.setArg( 2,  out_buffer );
      convolution_kernel.setArg( 3,  convkernel_w );
      convolution_kernel.setArg( 4,  convkernel_h );
      convolution_kernel.setArg( 5,  input_w );
      convolution_kernel.setArg( 6,  input_h );
      convolution_kernel.setArg( 7,  output_w );
      convolution_kernel.setArg( 8,  output_h );
      convolution_kernel.setArg( 9,  u_padding );
      convolution_kernel.setArg( 10, l_padding );

      ocl_queue.enqueueNDRangeKernel( convolution_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in binary_crossentropy_loss_derivative: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(out_buffer, output_w, output_h, other.m_count);
}

Mat ConvKernel::operator* (const Mat &other) const
{
   cl_int convkernel_w = m_width;
   cl_int convkernel_h = m_height;
   cl_int input_w = other.getWidth();
   cl_int input_h = other.getHeight();
   cl_int output_h;
   cl_int output_w;
   cl_int u_padding;
   cl_int l_padding;

   if (m_padding == SAME)
   {
      output_h = input_h;
      output_w = input_w;
      l_padding = convkernel_w / 2; // r_padding = (convkernel_w - 1) - l_padding
      u_padding = convkernel_h / 2; // d_padding = (convkernel_h - 1) - u_padding
   }
   else
   {
      output_h = input_h - convkernel_h + 1;
      output_w = input_w - convkernel_w + 1;
      u_padding = 0;
      l_padding = 0;
   }

   int N_ELEMENTS = output_h*output_w;
   cl::NDRange global( N_ELEMENTS );
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
       
   try {
      convolution_kernel.setArg( 0, m_buffer );
      convolution_kernel.setArg( 1, other.m_buffer);
      convolution_kernel.setArg( 2, out_buffer );
      convolution_kernel.setArg( 3, convkernel_w );
      convolution_kernel.setArg( 4, convkernel_h );
      convolution_kernel.setArg( 5, input_w );
      convolution_kernel.setArg( 6, input_h );
      convolution_kernel.setArg( 7, output_w );
      convolution_kernel.setArg( 8, u_padding );
      convolution_kernel.setArg( 9, l_padding );

      ocl_queue.enqueueNDRangeKernel( convolution_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in binary_crossentropy_loss_derivative: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return Mat(output_h, output_w, out_buffer);
}

