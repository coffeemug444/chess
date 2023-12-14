#pragma once
#include "layer.hpp"
#include "convKernel.hpp"


class LayerConvolutional : public UpdatableLayer
{
public:
   LayerConvolutional(int input_height, 
                      int input_width,
                      int channels,
                      int kernel_height,
                      int kernel_width,
                      int kernel_filters,
                      Padding padding,
                      InitializationMode initialization_mode);

private:
   ConvKernel m_weights;
   Mat m_biases;
};