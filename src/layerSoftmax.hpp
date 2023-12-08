#pragma once

#include "layer.hpp"

class LayerSoftmax : public Layer
{
public:
   LayerSoftmax(int size);

   Mat compute(const Mat& input) const override;

   ParallelMat compute(const ParallelMat& input) const override;
};