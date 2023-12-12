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
                      Padding padding);


};