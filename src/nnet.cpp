#include "nnet.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <assert.h>
#include <iostream>

using std::shared_ptr, std::vector, std::make_shared, std::make_unique, std::setw, std::setprecision, std::ofstream, std::ifstream;


NNet::NNet(const std::vector<std::reference_wrapper<Layer>>& layers)
:m_layers(layers)
{
   for (Layer& layer : layers)
   {
      if (layer.isUpdatable())
      {
         m_updatable_layers.push_back(dynamic_cast<UpdatableLayer&>(layer));
      }
   }
}

std::vector<Mat> NNet::compute(const std::vector<Mat>& inputs) const {
   const unsigned INPUT_SIZE = m_layers.front().get().input_size;
   for(auto &input : inputs) {
      assert(input.getWidth() == 1);
      assert(input.getHeight() == INPUT_SIZE);
   }

   auto a_l = ParallelMat{inputs};

   for (Layer& layer : m_layers)
   {
      a_l = layer.compute(a_l);
   }
   return a_l.toVector();
}

Mat NNet::compute(const Mat &input) const
{
   auto a_l = input;

   for (Layer& layer : m_layers)
   {
      a_l = layer.compute(a_l);
   }
   return a_l;
}

void NNet::applyWeightsAndBiasesGradients(float learning_rate)
{
   for (UpdatableLayer& layer : m_updatable_layers)
   {
      layer.applyWeightsAndBiasesGradients(learning_rate);
   }
}

void NNet::backPropagate(
   const std::vector<Mat>& inputs_vec, 
   const std::vector<Mat>& desired_outputs_vec) const
{
   assert(inputs_vec.size() == desired_outputs_vec.size());

   std::vector<Mat> weight_grads;
   std::vector<Mat> bias_grads;

   vector<ParallelMat> activations;
   activations.reserve(m_layers.size() + 1);
   vector<ParallelMat> preactivations;
   auto inputs = ParallelMat{inputs_vec};
   auto desired_outputs = ParallelMat{desired_outputs_vec};

   activations.push_back(inputs);

   for (Layer& layer : m_layers)
   {
      if (layer.isUpdatable())
      {
         UpdatableLayer& updatable_layer = dynamic_cast<UpdatableLayer&>(layer);
         auto[preactivation, activation] = updatable_layer.feedForward(activations.back());
         activations.push_back(activation);
         preactivations.push_back(preactivation);
      }
      else
      {
         activations.back() = layer.compute(activations.back());
      }
   }

   ParallelMat delta = m_updatable_layers.back().get().createCostDerivative(activations.back(), desired_outputs);

   for (int i = m_updatable_layers.size() - 1; i >= 0; i--)
   {
      delta = m_updatable_layers.at(i).get().updateWeightsAndBiasesGradients(preactivations[i], activations[i], delta);
   }
   
}
