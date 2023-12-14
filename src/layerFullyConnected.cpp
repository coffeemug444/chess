#include "layerFullyConnected.hpp"

Mat LayerFullyConnected::compute(const Mat& input) const
{
   switch (m_activation_function)
   {
   case RELU: return (m_weights * input + m_biases).relu();
   case SIGMOID: return (m_weights * input + m_biases).sigmoid();
   default: throw std::exception();
   }
}

ParallelMat LayerFullyConnected::compute(const ParallelMat& input) const
{
   switch (m_activation_function)
   {
   case RELU: return (m_weights * input + m_biases).relu();
   case SIGMOID: return (m_weights * input + m_biases).sigmoid();
   default: throw std::exception();
   }
}

std::pair<ParallelMat, ParallelMat> LayerFullyConnected::feedForward(const ParallelMat& input) const
{
   ParallelMat pre_activation = m_weights * input + m_biases;
   switch (m_activation_function)
   {
   case RELU: return {pre_activation, pre_activation.relu()};
   case SIGMOID: return {pre_activation, pre_activation.sigmoid()};
   default: throw std::exception();
   }
}

ParallelMat LayerFullyConnected::createCostDerivative(const ParallelMat& final_activation, const ParallelMat& desired_output)
{
   return final_activation - desired_output;
}

ParallelMat LayerFullyConnected::updateWeightsAndBiasesGradients(const ParallelMat& preactivation, const ParallelMat& activation, const ParallelMat& delta)
{
   ParallelMat this_delta = delta ^ [this, preactivation](){
      switch (m_activation_function)
      {
      case RELU: return preactivation.relu_inv();
      case SIGMOID: return preactivation.sigmoid_inv();
      default: throw std::exception();
      }
   }();

   m_weight_grads += (this_delta * activation.transpose()).sum();
   m_bias_grads += delta.sum();
   m_batch_size += delta.getCount();

   return m_weights.transpose() * this_delta;
}

void LayerFullyConnected::applyWeightsAndBiasesGradients(float learning_rate)
{
   float d = learning_rate / m_batch_size;
   m_weights -= d * m_weight_grads;
   m_biases -= d * m_bias_grads;

   m_weight_grads = Mat::zeros(output_size, input_size);
   m_bias_grads = Mat::zeros(output_size, 1);
   m_batch_size = 0;
}