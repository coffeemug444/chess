#include "layerSigmoid.hpp"

Mat LayerSigmoid::compute(const Mat& input) const
{
   return (m_weights * input + m_biases).sigmoid();
}

ParallelMat LayerSigmoid::compute(const ParallelMat& input) const
{
   return (m_weights * input + m_biases).relu();
}

std::pair<ParallelMat, ParallelMat> LayerSigmoid::feedForward(const ParallelMat& input) const
{
   ParallelMat pre_activation = m_weights * input + m_biases;
   ParallelMat activation = pre_activation.sigmoid();
   return {pre_activation, activation};
}

ParallelMat LayerSigmoid::createCostDerivative(const ParallelMat& final_activation, const ParallelMat& desired_output)
{
   return final_activation - desired_output;
}

ParallelMat LayerSigmoid::updateWeightsAndBiasesGradients(const ParallelMat& preactivation, const ParallelMat& activation, const ParallelMat& delta)
{
   ParallelMat this_delta = delta ^ preactivation.sigmoid_inv();

   auto weight_grad = (this_delta * activation.transpose()).sum();
   m_weight_grads += weight_grad;
   m_bias_grads += delta.sum();
   m_batch_size += delta.getCount();

   return m_weights.transpose() * this_delta;
}

void LayerSigmoid::applyWeightsAndBiasesGradients(float learning_rate)
{
   float d = learning_rate / m_batch_size;
   m_weights += d * m_weight_grads;
   m_biases += d * m_bias_grads;

   m_weight_grads = Mat::zeros(output_size, input_size);
   m_bias_grads = Mat::zeros(output_size, 1);
   m_batch_size = 0;
}