#pragma once

#include "layer.hpp"

class LayerFullyConnected : public UpdatableLayer
{
public:
   LayerFullyConnected(int input_size, int output_size, InitializationMode initialization_mode, ActivationFunction activation_function)
   :UpdatableLayer(input_size, output_size, activation_function)
   ,m_weights([initialization_mode, input_size, output_size](){
      switch (initialization_mode)
      {
      case HE:
         return Mat::he(output_size, input_size);
      case NORMAL:
         return Mat::random(output_size, input_size);
      default:
         return Mat::zeros(output_size, input_size);
      }
   }())
   ,m_biases(Mat::zeros(output_size, 1))
   ,m_weight_grads(Mat::zeros(output_size, input_size))
   ,m_bias_grads(Mat::zeros(output_size, 1))
   ,m_batch_size{0}
   {
   }

   Mat compute(const Mat& input) const override;
   ParallelMat compute(const ParallelMat& input) const override;
   std::pair<ParallelMat, ParallelMat> feedForward(const ParallelMat& input) const override;
   ParallelMat createCostDerivative(const ParallelMat& final_activation, const ParallelMat& desired_output) override;
   ParallelMat updateWeightsAndBiasesGradients(const ParallelMat& output, const ParallelMat& activation, const ParallelMat& delta) override;
   void applyWeightsAndBiasesGradients(float learning_rate) override;

protected:
   Mat m_weights;
   Mat m_biases;

   Mat m_weight_grads;
   Mat m_bias_grads;
   int m_batch_size;
};
