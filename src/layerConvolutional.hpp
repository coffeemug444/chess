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
                      int kernel_filters,
                      Padding padding,
                      InitializationMode initialization_mode,
                      ActivationFunction activation_function);


   Mat compute(const Mat& input) const override;
   ParallelMat compute(const ParallelMat& input) const override;
   std::pair<ParallelMat, ParallelMat> feedForward(const ParallelMat& input) const override;
   ParallelMat createCostDerivative(const ParallelMat& final_activation, const ParallelMat& desired_output) override;
   ParallelMat updateWeightsAndBiasesGradients(const ParallelMat& output, const ParallelMat& activation, const ParallelMat& delta) override;
   void applyWeightsAndBiasesGradients(float learning_rate) override;

private:
   ConvKernel m_weights;
   Mat m_biases;
};