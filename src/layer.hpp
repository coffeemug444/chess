#pragma once

#include <utility>
#include "mat.hpp"
#include "parallelMat.hpp"

enum InitializationMode
{
   HE,
   NORMAL
};

enum ActivationFunction
{
   RELU,
   SIGMOID
};

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
   virtual ParallelMat compute(const ParallelMat& input) const = 0;

   virtual bool isUpdatable() { return false; }

   virtual ~Layer() = default;
};


class UpdatableLayer : public Layer
{
public:

   UpdatableLayer(int input_size, int output_size, ActivationFunction activation_function)
   :Layer(input_size, output_size)
   ,m_activation_function{activation_function}
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
   virtual ParallelMat updateWeightsAndBiasesGradients(const ParallelMat& output_preactivation, const ParallelMat& input_activation, const ParallelMat& delta) = 0;

   virtual void applyWeightsAndBiasesGradients(float learning_rate) = 0;

   bool isUpdatable() override { return true; }

protected:
   ActivationFunction m_activation_function;
};

