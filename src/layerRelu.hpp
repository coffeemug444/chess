#pragma once

#include "layer.hpp"

class LayerRelu : public FullyConnected
{
public:
   LayerRelu(int input_size, int output_size, InitializationMode initialization_mode)
   :FullyConnected(input_size, output_size, initialization_mode)
   {}
   Mat compute(const Mat& input) const override;
   ParallelMat compute(const ParallelMat& input) const override;
   std::pair<ParallelMat, ParallelMat> feedForward(const ParallelMat& input) const override;
   ParallelMat createCostDerivative(const ParallelMat& final_activation, const ParallelMat& desired_output) override;
   ParallelMat updateWeightsAndBiasesGradients(const ParallelMat& output, const ParallelMat& activation, const ParallelMat& delta) override;
   void applyWeightsAndBiasesGradients(float learning_rate) override;
};