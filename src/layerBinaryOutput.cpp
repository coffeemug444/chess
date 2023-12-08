#include "layerBinaryOutput.hpp"


Mat LayerBinaryOutput::compute(const Mat& input) const
{
   return (m_weights * input + m_biases).sigmoid();
}

ParallelMat LayerBinaryOutput::compute(const ParallelMat& input) const
{
   return (m_weights * input + m_biases).sigmoid();
}

std::pair<ParallelMat, ParallelMat> LayerBinaryOutput::feedForward(const ParallelMat& input) const
{
   ParallelMat pre_activation = m_weights * input + m_biases;
   ParallelMat activation = pre_activation.sigmoid();
   return {pre_activation, activation};
}

ParallelMat LayerBinaryOutput::createCostDerivative(const ParallelMat& final_activation, const ParallelMat& desired_output)
{
   return desired_output.binary_crossentropy_loss_derivative(final_activation);
}

ParallelMat LayerBinaryOutput::updateWeightsAndBiasesGradients(const ParallelMat& preactivation, const ParallelMat& activation, const ParallelMat& delta)
{
   ParallelMat this_delta = delta ^ preactivation.relu_inv();

   m_weight_grads += (this_delta * activation.transpose()).sum();
   m_bias_grads += delta.sum();
   m_batch_size += delta.getCount();

   return m_weights.transpose() * this_delta;
}

void LayerBinaryOutput::applyWeightsAndBiasesGradients(float learning_rate)
{
   float d = learning_rate / m_batch_size;
   m_weights -= d * m_weight_grads;
   m_biases -= d * m_bias_grads;

   m_weight_grads = Mat::zeros(output_size, input_size);
   m_bias_grads = Mat::zeros(output_size, 1);
   m_batch_size = 0;
}