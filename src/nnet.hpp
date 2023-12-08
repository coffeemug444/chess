#pragma once

#include "mat.hpp"
#include "parallelMat.hpp"
#include <vector>
#include <functional>
#include "layer.hpp"

class NNet {
public:
   enum Mode {
      BINARY_CLASSIFICATION,
      MULTICLASS_CLASSIFICATION,
      REGRESSION
   };

private:
   std::vector<std::reference_wrapper<Layer>> m_layers;
   std::vector<std::reference_wrapper<UpdatableLayer>> m_updatable_layers;
   
public:
   NNet(const std::vector<std::reference_wrapper<Layer>>& layers);

   std::vector<Mat> compute(const std::vector<Mat>& inputs) const;
   Mat compute(const Mat& input) const;

   // adds to the weightgrad and biasgrad update terms in each layer. A call to
   // `applyWeightsAndBiasesGradients` is required in order to apply these gradients
   void backPropagate(const std::vector<Mat>& inputs, const std::vector<Mat>& desired_outputs) const;

   void applyWeightsAndBiasesGradients(float learning_rate);
};


