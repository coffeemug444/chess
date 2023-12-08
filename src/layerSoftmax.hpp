#pragma once

#include "layer.hpp"

class LayerSoftmax : public Layer
{
public:
   LayerSoftmax(int size)
   :Layer(size, size)
   {}

   Mat compute(const Mat& input) const override {
      return input.softmax();
   }
};