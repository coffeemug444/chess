#pragma once

#include "mat.hpp"
#include "parallelMat.hpp"
#include <vector>

class NNet {
   private:
      std::vector<int> m_layer_sizes;
      std::vector<Mat> m_weights;
      std::vector<Mat> m_biases;
      ParallelMat multipleCompute(const ParallelMat& batch) const;
      std::vector<Mat> computeWeights(std::vector<Mat>& diffs, std::vector<Mat>& as) const;
      std::vector<Mat> computeBiases(std::vector<Mat>& diffs) const;
      
   public:
      static float sigmoid_act (float x);
      static float sigmoid_inv (float x);
      static float reLU_act(float x);
      static float reLU_inv(float x);
      static float leaky_reLU_act(float x);
      static float leaky_reLU_inv(float x);

      NNet();
      NNet(std::vector<int> layer_sizes, char mode);
      NNet(std::string file_path);
      void save(std::string file_path) const;
      void forwardPass(const Mat& input, std::vector<Mat>& zs, std::vector<Mat>& as) const;
      ParallelMat multipleCompute(const std::vector<Mat>& inputs) const;
      Mat compute(const Mat& input) const;
      void adjustWeightsAndBiases(std::vector<Mat>& weight_grad, std::vector<Mat>& bias_grad, float learning_rate, unsigned iterations);
      void backPropagate(const Mat& input, const Mat& desired_output, std::vector<Mat>& weight_grads_out, std::vector<Mat>& bias_grads_out) const;
      void multipleBackPropagate(const std::vector<Mat>& inputs, const std::vector<Mat>& desired_outputs, std::vector<Mat>& weight_grads_out, std::vector<Mat>& bias_grads_out) const;

      std::vector<Mat> weightsZero();
      std::vector<Mat> biasesZero();
};


