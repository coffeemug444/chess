#include "layerBatchNormalize.hpp"

LayerBatchNormalize::LayerBatchNormalize(int size)
:Layer(size, size)
{}

Mat LayerBatchNormalize::compute(const Mat& input) const {
   float mean = input.sum() / (input.getHeight() * input.getWidth());
   return input - mean;
}

ParallelMat LayerBatchNormalize::compute(const ParallelMat& input) const {
   auto sub_mats = input.toVector();
   for (Mat& mat : sub_mats)
   {
      mat = compute(mat);
   }
   return ParallelMat{sub_mats};
}