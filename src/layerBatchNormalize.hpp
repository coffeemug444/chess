#pragma once

#include "layer.hpp"

class LayerBatchNormalize : public Layer
{
public:
   LayerBatchNormalize(int size);

   Mat compute(const Mat& input) const override;

   ParallelMat compute(const ParallelMat& input) const override;
};