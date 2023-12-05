#pragma once

#include "mat.hpp"
#include "parallelMat.hpp"
#include <vector>

class NNet {
   public:
      enum Mode {
         BINARY_CLASSIFICATION,
         REGRESSION
      };

   private:
      std::vector<int> m_layer_sizes;
      std::vector<Mat> m_weights;
      std::vector<Mat> m_biases;
      Mode m_mode;

      ParallelMat multipleCompute(const ParallelMat& batch) const;
      
   public:

      NNet();
      NNet(std::vector<int> layer_sizes, char initialisation, Mode mode);
      NNet(std::string file_path);
      void save(std::string file_path) const;

      // returns [layer_outputs, activations]
      std::pair<std::vector<Mat>, std::vector<Mat>> forwardPass(const Mat& input) const;

      ParallelMat multipleCompute(const std::vector<Mat>& inputs) const;
      Mat compute(const Mat& input) const;
      void adjustWeightsAndBiases(const std::vector<Mat> &weight_grad, const std::vector<Mat> &bias_grad, float learning_rate);

      // returns [weight_grads, bias_grads]
      std::pair<std::vector<Mat>,std::vector<Mat>> backPropagate(const std::vector<Mat>& inputs, const std::vector<Mat>& desired_outputs) const;

      std::vector<Mat> weightsZero();
      std::vector<Mat> biasesZero();
};


