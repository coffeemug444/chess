#pragma once

#include <utility>
#include "mat.hpp"
#include "parallelMat.hpp"


class Layer
{
public:
   Layer(int input_size, int output_size)
   :input_size{input_size}
   ,output_size{output_size}
   {}

   const int input_size;
   const int output_size;

   // returns the final output of this layer after any activation functions
   virtual Mat compute(const Mat& input) const = 0;

   virtual bool isUpdatable() { return false; }

   virtual ~Layer() = default;
};


class UpdatableLayer : public Layer
{
public:
   enum InitializationMode
   {
      HE,
      NORMAL
   };

   UpdatableLayer(int input_size, int output_size)
   :Layer(input_size, output_size)
   {}


   // returns a pair of
   // 1. pre-activation, the output of the weights and biases 
   //    before the activation function
   // 2. activation, final output of this layer
   virtual std::pair<ParallelMat, ParallelMat> feedForward(const ParallelMat& input) const = 0;

   // Only valid for the final UpdatableLayer - returns the delta used to update the weights and biases
   virtual ParallelMat createCostDerivative(const ParallelMat& final_activation, const ParallelMat& desired_output) = 0;

   // Updates the weights and biases gradients using delta. Returns the delta to be used for
   // the previous layer. Will stay saved until `applyWeightsAndBiasesGradients` is called
   virtual ParallelMat updateWeightsAndBiasesGradients(const ParallelMat& output, const ParallelMat& activation, const ParallelMat& delta) = 0;

   virtual void applyWeightsAndBiasesGradients(float learning_rate) = 0;

   bool isUpdatable() override { return true; }
};


class FullyConnected : public UpdatableLayer
{
   FullyConnected(int input_size, int output_size, InitializationMode initialization_mode)
   :UpdatableLayer(input_size, output_size)
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

protected:
   Mat m_weights;
   Mat m_biases;

   Mat m_weight_grads;
   Mat m_bias_grads;
   int m_batch_size;
};
