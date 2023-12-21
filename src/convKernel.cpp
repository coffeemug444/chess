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

ConvKernel::ConvKernel (unsigned channels,
                        unsigned kernel_height,
                        unsigned kernel_width,
                        unsigned filters,
                        Padding padding,
                        unsigned input_height,
                        unsigned input_width,
                        const cl::Buffer& vals)
   :m_channels{channels}
   ,m_height{kernel_height}
   ,m_width{kernel_width}
   ,m_padding{padding}
   ,m_filters{filters}
   ,m_input_height{input_height}
   ,m_input_width{input_width}
{
   m_buffer = cl::Buffer(ocl_context, CL_MEM_READ_WRITE, (kernel_height*kernel_width*filters)*sizeof(float));
   ocl_queue.enqueueCopyBuffer(vals, m_buffer, 0, 0, (kernel_height*kernel_width*filters)*sizeof(float));
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

std::pair<unsigned,unsigned> ConvKernel::getPaddedHeightWidth(
            unsigned kernel_height,
            unsigned kernel_width,
            Padding padding,
            unsigned input_height,
            unsigned input_width)
{
   if (padding == VALID) return {input_height, input_width};

   unsigned l_padding = kernel_width / 2;
   unsigned r_padding = (kernel_width - 1) - l_padding;
   unsigned u_padding = kernel_height / 2;
   unsigned d_padding = (kernel_height - 1) - u_padding;

   unsigned padded_w = l_padding + r_padding + input_width;
   unsigned padded_h = u_padding + d_padding + input_height;
   return {padded_h, padded_w};
}

ParallelMat ConvKernel::operator* (const ParallelMat &other) const
{
   cl_int convkernel_w = m_width;
   cl_int convkernel_h = m_height;
   cl_int channels = m_channels;
   cl_int filters = m_filters;
   auto [output_h, output_w] = getOutputHeightWidth();

   int N_ELEMENTS = output_h*output_w*filters*other.m_count;
   cl::NDRange global( N_ELEMENTS );
   cl::Buffer in_buffer = parallelPad(other.m_buffer, other.getCount());
   auto [padded_h, padded_w] = getPaddedHeightWidth();
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
       
   try {
      parallel_convolution_kernel.setArg( 0,  m_buffer );
      parallel_convolution_kernel.setArg( 1,  in_buffer);
      parallel_convolution_kernel.setArg( 2,  out_buffer );
      parallel_convolution_kernel.setArg( 3,  convkernel_w );
      parallel_convolution_kernel.setArg( 4,  convkernel_h );
      parallel_convolution_kernel.setArg( 5,  static_cast<cl_int>(padded_w));
      parallel_convolution_kernel.setArg( 6,  static_cast<cl_int>(padded_h));
      parallel_convolution_kernel.setArg( 7,  channels );
      parallel_convolution_kernel.setArg( 8,  filters );
      parallel_convolution_kernel.setArg( 9,  static_cast<cl_int>(output_w));
      parallel_convolution_kernel.setArg( 10, static_cast<cl_int>(output_h));

      ocl_queue.enqueueNDRangeKernel( parallel_convolution_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in convKernel parallel convolution: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(output_w*output_h*filters, 1, other.m_count, out_buffer);
}

Mat ConvKernel::operator* (const Mat &other) const
{
   cl_int convkernel_w = m_width;
   cl_int convkernel_h = m_height;
   cl_int channels = m_channels;
   cl_int filters = m_filters;
   auto [output_h, output_w] = getOutputHeightWidth();

   int N_ELEMENTS = output_h*output_w*m_filters;
   cl::NDRange global( N_ELEMENTS );
   cl::Buffer in_buffer = pad(other.m_buffer);
   auto [padded_h, padded_w] = getPaddedHeightWidth();

   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
       
   try {
      convolution_kernel.setArg( 0,  m_buffer );
      convolution_kernel.setArg( 1,  in_buffer);
      convolution_kernel.setArg( 2,  out_buffer );
      convolution_kernel.setArg( 3,  convkernel_w );
      convolution_kernel.setArg( 4,  convkernel_h );
      convolution_kernel.setArg( 5,  static_cast<cl_int>(padded_w));
      convolution_kernel.setArg( 6,  static_cast<cl_int>(padded_h));
      convolution_kernel.setArg( 7,  channels );
      convolution_kernel.setArg( 8,  filters );
      convolution_kernel.setArg( 9,  static_cast<cl_int>(output_w));
      convolution_kernel.setArg( 10, static_cast<cl_int>(output_h));

      ocl_queue.enqueueNDRangeKernel( convolution_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in convKernel convolution: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return Mat(output_w*output_h*filters, 1, out_buffer);
}


Mat ConvKernel::operator^(const Mat& other) const
{
   auto [output_h, output_w] = getOutputHeightWidth();
   cl::Buffer out_buffer = [this, other, &output_h, &output_w](){
      if (m_padding == SAME) 
      {
         output_w += m_width - 1;
         output_h += m_height - 1;
         return pad(other.m_buffer);
      }
      else
      {
         int w_padding = m_width - 1;
         int h_padding = m_height - 1;

         int l = (w_padding+1) / 2;
         int r = w_padding / 2;
         int u = (h_padding+1) / 2;
         int d = h_padding / 2;

         output_w += l + r;
         output_h += u + d;

         return pad(other.m_buffer, l, r, u, d);
      }
   }();

   cl_int convkernel_w = m_width;
   cl_int convkernel_h = m_height;
   cl_int channels = m_channels;
   cl_int filters = m_filters;

   int N_ELEMENTS = m_input_height*m_input_width*m_channels;
   cl::NDRange global( N_ELEMENTS );

   cl::Buffer in_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
       
   try {
      transpose_conv_kernel.setArg( 0,  m_buffer );
      transpose_conv_kernel.setArg( 1,  in_buffer);
      transpose_conv_kernel.setArg( 2,  out_buffer );
      transpose_conv_kernel.setArg( 3,  convkernel_w );
      transpose_conv_kernel.setArg( 4,  convkernel_h );
      transpose_conv_kernel.setArg( 5,  static_cast<cl_int>(m_input_width));
      transpose_conv_kernel.setArg( 6,  static_cast<cl_int>(m_input_height));
      transpose_conv_kernel.setArg( 7,  channels );
      transpose_conv_kernel.setArg( 8,  filters );
      transpose_conv_kernel.setArg( 9,  static_cast<cl_int>(output_w));
      transpose_conv_kernel.setArg( 10, static_cast<cl_int>(output_h));

      ocl_queue.enqueueNDRangeKernel( transpose_conv_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in convKernel convolution: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return Mat(N_ELEMENTS, 1, in_buffer);
}


ParallelMat ConvKernel::operator^(const ParallelMat& other) const
{
   auto [output_h, output_w] = getOutputHeightWidth();
   cl::Buffer out_buffer = [this, other, &output_h, &output_w](){
      if (m_padding == SAME) 
      {
         output_w += m_width - 1;
         output_h += m_height - 1;
         return pad(other.m_buffer);
      }
      else
      {
         int w_padding = m_width - 1;
         int h_padding = m_height - 1;

         int l = (w_padding+1) / 2;
         int r = w_padding / 2;
         int u = (h_padding+1) / 2;
         int d = h_padding / 2;

         output_w += l + r;
         output_h += u + d;

         return pad(other.m_buffer, l, r, u, d);
      }
   }();

   cl_int convkernel_w = m_width;
   cl_int convkernel_h = m_height;
   cl_int channels = m_channels;
   cl_int filters = m_filters;

   int N_ELEMENTS = m_input_height*m_input_width*m_channels*other.getCount();
   cl::NDRange global( N_ELEMENTS );

   cl::Buffer in_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));
       
   try {
      parallel_transpose_conv_kernel.setArg( 0,  m_buffer );
      parallel_transpose_conv_kernel.setArg( 1,  in_buffer);
      parallel_transpose_conv_kernel.setArg( 2,  out_buffer );
      parallel_transpose_conv_kernel.setArg( 3,  convkernel_w );
      parallel_transpose_conv_kernel.setArg( 4,  convkernel_h );
      parallel_transpose_conv_kernel.setArg( 5,  static_cast<cl_int>(m_input_width));
      parallel_transpose_conv_kernel.setArg( 6,  static_cast<cl_int>(m_input_height));
      parallel_transpose_conv_kernel.setArg( 7,  channels );
      parallel_transpose_conv_kernel.setArg( 8,  filters );
      parallel_transpose_conv_kernel.setArg( 9,  static_cast<cl_int>(output_w));
      parallel_transpose_conv_kernel.setArg( 10, static_cast<cl_int>(output_h));

      ocl_queue.enqueueNDRangeKernel( parallel_transpose_conv_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in convKernel convolution: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }
   return ParallelMat(N_ELEMENTS/other.getCount(), 1, other.getCount(), in_buffer);
}

cl::Buffer ConvKernel::pad(const cl::Buffer& input) const
{  
   if (m_padding == VALID) return input;

   cl_int l = m_width / 2;
   cl_int r = (m_width - 1) - l;
   cl_int u = m_height / 2;
   cl_int d = (m_height - 1) - u;

   return pad(input, l, r, u, d);
}

cl::Buffer ConvKernel::parallelPad(const cl::Buffer& input, int num) const
{  
   if (m_padding == VALID) return input;

   int l = m_width / 2;
   int r = (m_width - 1) - l;
   int u = m_height / 2;
   int d = (m_height - 1) - u;

   return parallelPad(input, num, l, r, u, d);
}


cl::Buffer ConvKernel::pad(const cl::Buffer& input, int l, int r, int u, int d) const
{
   cl_int l_padding = l;
   cl_int r_padding = r;
   cl_int u_padding = u;
   cl_int d_padding = d;

   cl_int padded_w = l_padding + r_padding + m_input_width;
   cl_int padded_h = u_padding + d_padding + m_input_height;

   cl_int input_width = m_input_width;
   cl_int input_height = m_input_height;

   cl_int channels = m_channels;

   int N_ELEMENTS = padded_w*padded_h*m_channels;
   cl::NDRange global( N_ELEMENTS );
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));

   try {
      pad_kernel.setArg( 0,  input );
      pad_kernel.setArg( 1,  out_buffer);
      pad_kernel.setArg( 2,  input_width );
      pad_kernel.setArg( 3,  input_height );
      pad_kernel.setArg( 4,  channels );
      pad_kernel.setArg( 5,  l_padding );
      pad_kernel.setArg( 6,  r_padding );
      pad_kernel.setArg( 7,  u_padding );
      pad_kernel.setArg( 8,  d_padding );

      ocl_queue.enqueueNDRangeKernel( pad_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in convKernel pad: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return out_buffer;
}

cl::Buffer ConvKernel::parallelPad(const cl::Buffer& input, int num, int l, int r, int u, int d) const
{
   cl_int l_padding = l;
   cl_int r_padding = r;
   cl_int u_padding = u;
   cl_int d_padding = d;

   cl_int padded_w = l_padding + r_padding + m_input_width;
   cl_int padded_h = u_padding + d_padding + m_input_height;

   cl_int input_width = m_input_width;
   cl_int input_height = m_input_height;

   cl_int channels = m_channels;

   int N_ELEMENTS = padded_w*padded_h*m_channels*num;
   cl::NDRange global( N_ELEMENTS );
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));

   try {
      parallel_pad_kernel.setArg( 0,  input );
      parallel_pad_kernel.setArg( 1,  out_buffer);
      parallel_pad_kernel.setArg( 2,  input_width );
      parallel_pad_kernel.setArg( 3,  input_height );
      parallel_pad_kernel.setArg( 4,  channels );
      parallel_pad_kernel.setArg( 5,  l_padding );
      parallel_pad_kernel.setArg( 6,  r_padding );
      parallel_pad_kernel.setArg( 7,  u_padding );
      parallel_pad_kernel.setArg( 8,  d_padding );

      ocl_queue.enqueueNDRangeKernel( parallel_pad_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in convKernel parallel pad: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return out_buffer;
}

ConvKernel ConvKernel::rotated() const
{
   int N_ELEMENTS = m_width*m_height*m_filters;
   cl::NDRange global( N_ELEMENTS );
   cl::Buffer out_buffer(ocl_context, CL_MEM_READ_WRITE, N_ELEMENTS*sizeof(float));

   cl_int width = m_width;
   cl_int height = m_height;

   try {
      rotate_conv_kernel.setArg( 0,  m_buffer );
      rotate_conv_kernel.setArg( 1,  out_buffer);
      rotate_conv_kernel.setArg( 2,  width );
      rotate_conv_kernel.setArg( 3,  height );

      ocl_queue.enqueueNDRangeKernel( rotate_conv_kernel, cl::NullRange, global );
   }
   catch(cl::Error& err) {
      std::cout << "Error in convKernel rotated: " << err.what() << "(" << getErrorString(err.err()) << ")" << std::endl;
   }

   return ConvKernel(m_channels,
                     m_height,
                     m_width,
                     m_filters,
                     m_padding,
                     m_input_height,
                     m_input_width,
                     out_buffer);
}
