#pragma once

#include "layerFullyConnected.hpp"

class LayerBinaryOutput : public LayerFullyConnected
{
public:
   LayerBinaryOutput(int input_size, InitializationMode initialization_mode)
   :LayerFullyConnected(input_size, 1, initialization_mode, SIGMOID)
   {}
   ParallelMat createCostDerivative(const ParallelMat& final_activation, const ParallelMat& desired_output) override;
};