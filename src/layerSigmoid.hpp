#pragma once

#include "layer.hpp"

class LayerSigmoid : public FullyConnected
{
public:
   Mat compute(const Mat& input) const override;
   std::pair<ParallelMat, ParallelMat> feedForward(const ParallelMat& input) const override;
   ParallelMat createCostDerivative(const ParallelMat& final_activation, const ParallelMat& desired_output) override;
   ParallelMat updateWeightsAndBiasesGradients(const ParallelMat& output, const ParallelMat& activation, const ParallelMat& delta) override;
   void applyWeightsAndBiasesGradients(float learning_rate) override;
};
