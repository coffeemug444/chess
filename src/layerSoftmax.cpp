#include "layerSoftmax.hpp"

LayerSoftmax::LayerSoftmax(int size)
:Layer(size, size)
{}

Mat LayerSoftmax::compute(const Mat& input) const {
   return input.softmax();
}

ParallelMat LayerSoftmax::compute(const ParallelMat& input) const {
   return input.softmax();
}