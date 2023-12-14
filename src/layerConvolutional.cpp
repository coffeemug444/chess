#include "layerConvolutional.hpp"

LayerConvolutional::LayerConvolutional
(int input_height, 
 int input_width,
 int channels,
 int kernel_height,
 int kernel_width,
 int kernel_filters,
 Padding padding,
 InitializationMode initialization_mode,
 ActivationFunction activation_function)
:UpdatableLayer(
   input_height*input_width*channels, 
   [kernel_height, kernel_width, padding, input_height, input_width, kernel_filters]()->int{ 
      auto [h,w] = ConvKernel::getOutputHeightWidth(kernel_height, kernel_width, padding, input_height, input_width);
      return h*w*kernel_filters;
   }(),
   activation_function
)
,m_weights(
   channels, 
   kernel_height, 
   kernel_width, 
   kernel_filters, 
   padding, 
   input_height, 
   input_width, 
   [initialization_mode,kernel_width,kernel_height,kernel_filters]()->Mat{
      switch(initialization_mode) {
         case HE:
            return Mat::he(kernel_width*kernel_height*kernel_filters,1);
         case NORMAL:
            return Mat::random(kernel_width*kernel_height*kernel_filters,1);
         default: throw std::exception();
      }
   }())
,m_biases{Mat::zeros(output_size,1)}
{

}

Mat LayerConvolutional::compute(const Mat& input) const
{
   Mat preactivation = m_weights * input + m_biases;
   switch (m_activation_function)
   {
   case RELU: return preactivation.relu();
   case SIGMOID: return preactivation.sigmoid();
   default: throw std::exception();
   }
}

ParallelMat LayerConvolutional::compute(const ParallelMat& input) const
{
   ParallelMat preactivation = m_weights * input + m_biases;
   switch (m_activation_function)
   {
   case RELU: return preactivation.relu();
   case SIGMOID: return preactivation.sigmoid();
   default: throw std::exception();
   }
}

std::pair<ParallelMat, ParallelMat> LayerConvolutional::feedForward(const ParallelMat& input) const
{
   ParallelMat preactivation = m_weights * input + m_biases;
   switch (m_activation_function)
   {
   case RELU: return {preactivation, preactivation.relu()};
   case SIGMOID: return {preactivation, preactivation.sigmoid()};
   default: throw std::exception();
   }
}

ParallelMat LayerConvolutional::createCostDerivative(const ParallelMat& final_activation, const ParallelMat& desired_output)
{
   return final_activation - desired_output;
}

ParallelMat LayerConvolutional::updateWeightsAndBiasesGradients(const ParallelMat& output, const ParallelMat& activation, const ParallelMat& delta)
{

}

void LayerConvolutional::applyWeightsAndBiasesGradients(float learning_rate)
{

}
