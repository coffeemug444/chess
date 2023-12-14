#include <iostream>

#include "convKernel.hpp"
#include "parallelMat.hpp"
#include "oclData.hpp"
#include "errors.hpp"


ConvKernel::ConvKernel (unsigned channels,
                        unsigned kernel_height,
                        unsigned kernel_width,
                        unsigned filters,
                        Padding padding,
                        unsigned input_height,
                        unsigned input_width,
                        const Mat& vals)
   :m_channels{channels}
   ,m_height{kernel_height}
   ,m_width{kernel_width}
   ,m_padding{padding}
   ,m_filters{filters}
   ,m_input_height{input_height}
   ,m_input_width{input_width}
{
   assert(vals.getWidth() == 1);
   assert(vals.getHeight() == kernel_height*kernel_width*filters);

   m_buffer = cl::Buffer(ocl_context, CL_MEM_READ_WRITE, (kernel_height*kernel_width*filters)*sizeof(float));
   ocl_queue.enqueueCopyBuffer(vals.m_buffer, m_buffer, 0, 0, (kernel_height*kernel_width*filters)*sizeof(float));
}

std::pair<unsigned,unsigned> ConvKernel::getOutputHeightWidth(
            unsigned kernel_height,
            unsigned kernel_width,
            Padding padding,
            unsigned input_height,
            unsigned input_width)
{
   if (padding == SAME) return {input_height, input_width};

   unsigned output_h = input_height - kernel_height + 1;
   unsigned output_w = input_width - kernel_width + 1;
   return {output_h, output_w};
}

ParallelMat ConvKernel::operator* (const ParallelMat &other) const
{
   cl_int convkernel_w = m_width;
   cl_int convkernel_h = m_height;
   cl_int input_w = m_input_width;
   cl_int input_h = m_input_height;
   cl_int channels = m_channels;
   cl_int filters = m_filters;
   auto [output_h, output_w] = getOutputHeightWidth(convkernel_h, convkernel_w, m_padding, input_h, input_w);
   cl_int u_padding = 0;
   cl_int l_padding = 0;

   if (m_padding == SAME)
   {
      l_padding = convkernel_w / 2; // r_padding = (convkernel_w - 1) - l_padding
      u_padding = convkernel_h / 2; // d_padding = (convkernel_h - 1) - u_padding
   }

   int N_ELEMENTS = output_h*output_w*filters*other.m_count;
   cl::NDRange global( N_ELEMENTS );
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
       
   try {
      parallel_convolution_kernel.setArg( 0,  m_buffer );
      parallel_convolution_kernel.setArg( 1,  other.m_buffer);
      parallel_convolution_kernel.setArg( 2,  out_buffer );
      parallel_convolution_kernel.setArg( 3,  convkernel_w );
      parallel_convolution_kernel.setArg( 4,  convkernel_h );
      parallel_convolution_kernel.setArg( 5,  input_w );
      parallel_convolution_kernel.setArg( 6,  input_h );
      parallel_convolution_kernel.setArg( 7,  channels );
      parallel_convolution_kernel.setArg( 8,  filters );
      parallel_convolution_kernel.setArg( 9,  static_cast<cl_int>(output_w));
      parallel_convolution_kernel.setArg( 10, static_cast<cl_int>(output_h));
      parallel_convolution_kernel.setArg( 11, u_padding );
      parallel_convolution_kernel.setArg( 12, l_padding );

      ocl_queue.enqueueNDRangeKernel( parallel_convolution_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in binary_crossentropy_loss_derivative: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(output_w*output_h*filters, 1, other.m_count, out_buffer);
}

Mat ConvKernel::operator* (const Mat &other) const
{
   cl_int convkernel_w = m_width;
   cl_int convkernel_h = m_height;
   cl_int input_w = m_input_width;     // can't get input_w, input_h from input `other`!
   cl_int input_h = m_input_height;    // input is a Nx1 column
   cl_int channels = m_channels;
   cl_int filters = m_filters;
   auto [output_h, output_w] = getOutputHeightWidth(convkernel_h, convkernel_w, m_padding, input_h, input_w);
   cl_int u_padding = 0;
   cl_int l_padding = 0;

   if (m_padding == SAME)
   {
      l_padding = convkernel_w / 2; // r_padding = (convkernel_w - 1) - l_padding
      u_padding = convkernel_h / 2; // d_padding = (convkernel_h - 1) - u_padding
   }

   int N_ELEMENTS = output_h*output_w*m_filters;
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
      convolution_kernel.setArg( 7,  channels );
      convolution_kernel.setArg( 8,  filters );
      convolution_kernel.setArg( 9,  static_cast<cl_int>(output_w));
      convolution_kernel.setArg( 10, static_cast<cl_int>(output_h));
      convolution_kernel.setArg( 11, u_padding );
      convolution_kernel.setArg( 12, l_padding );

      ocl_queue.enqueueNDRangeKernel( convolution_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in binary_crossentropy_loss_derivative: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return Mat(output_w*output_h*filters, 1, out_buffer);
}
