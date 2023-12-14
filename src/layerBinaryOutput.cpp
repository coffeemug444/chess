#include "layerBinaryOutput.hpp"

ParallelMat LayerBinaryOutput::createCostDerivative(const ParallelMat& final_activation, const ParallelMat& desired_output)
{
   return desired_output.binary_crossentropy_loss_derivative(final_activation);
}
