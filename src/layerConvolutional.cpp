#include "layerConvolutional.hpp"

LayerConvolutional::LayerConvolutional
(int input_height, 
 int input_width,
 int channels,
 int kernel_height,
 int kernel_width,
 int kernel_filters,
 Padding padding,
 InitializationMode initialization_mode)
:UpdatableLayer(
   input_height*input_width*channels, 
   [kernel_height, kernel_width, padding, input_height, input_width, kernel_filters]()->int{ 
      auto [h,w] = ConvKernel::getOutputHeightWidth(kernel_height, kernel_width, padding, input_height, input_width);
      return h*w*kernel_filters;
   }()
)
,m_weights(
   channels, 
   kernel_height, 
   kernel_width, 
   kernel_filters, 
   padding, 
   input_height, 
   input_width, 
   [initialization_mode,kernel_width,kernel_height,kernel_filters]()->Mat{
      switch(initialization_mode) {
         case HE:
            return Mat::he(kernel_width*kernel_height*kernel_filters,1);
         case NORMAL:
            return Mat::random(kernel_width*kernel_height*kernel_filters,1);
         default: throw std::exception();
      }
   }())
,m_biases{Mat::zeros(output_size,1)}
{

}
