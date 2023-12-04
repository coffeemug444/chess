#include "nnet.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <assert.h>

using std::shared_ptr, std::vector, std::make_shared, std::make_unique, std::setw, std::setprecision, std::ofstream, std::ifstream;

NNet::NNet(){};

// layer_sizes defines the number of outputs for each layer.
// An extra layer is added to the beginning notating the number
// of inputs into the net (layer_sizes' length is 1 greater than
// the number of layers in the net)
// mode is '1' for ones, '0' for zeros, or 'r' for random floats between 0 and 1
NNet::NNet(vector<int> layer_sizes, char mode)
{
   assert(mode == '0' || mode == '1' || mode == 'r' || mode == 'h');

   m_layer_sizes.push_back(layer_sizes[0]);

   for (unsigned i = 1; i < layer_sizes.size(); i++)
   {
      m_layer_sizes.push_back(layer_sizes[i]);
      m_biases.push_back(Mat::zeros(layer_sizes[i], 1));
      switch (mode)
      {
      case '0':
         m_weights.push_back(Mat::zeros(layer_sizes[i], layer_sizes[i - 1]));
         break;
      case '1':
         m_weights.push_back(Mat::ones(layer_sizes[i], layer_sizes[i - 1]));
         break;
      case 'r':
         m_weights.push_back(Mat::random(layer_sizes[i], layer_sizes[i - 1]));
         break;
      case 'h':
         m_weights.push_back(Mat::he(layer_sizes[i], layer_sizes[i - 1]));
      }
   }
}

vector<Mat> NNet::weightsZero()
{
   auto vec = vector<Mat>();
   auto prev_it = m_layer_sizes.begin();
   for (auto it = m_layer_sizes.begin() + 1; it != m_layer_sizes.end(); ++it)
   {
      auto w1 = *it.base();
      auto w2 = *prev_it.base();
      auto zero_mat = Mat::zeros(w1, w2);
      vec.push_back(zero_mat);
      prev_it = it;
   }
   return vec;
}

vector<Mat> NNet::biasesZero()
{
   auto vec = vector<Mat>();
   for (auto it = m_layer_sizes.begin() + 1; it != m_layer_sizes.end(); ++it)
   {
      auto b = *it.base();
      auto zero_mat = Mat::zeros(b, 1);
      vec.push_back(zero_mat);
   }
   return vec;
}

void printMatDim(Mat mat)
{
   std::cout << "{" << mat.getHeight() << ", " << mat.getWidth() << "}" << std::endl;
}

void printVecDims(vector<Mat> vec)
{
   for (auto mat : vec)
   {
      printMatDim(mat);
   }
}

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream &str)
{
   std::vector<std::string> result;
   std::string line;
   std::getline(str, line);

   std::stringstream line_stream(line);
   std::string cell;

   while (std::getline(line_stream, cell, ','))
   {
      result.push_back(cell);
   }
   return result;
}

NNet::NNet(std::string file_path)
{
   ifstream file(file_path);
   std::vector<std::string> line_tokens;
   line_tokens = getNextLineAndSplitIntoTokens(file);
   for (auto layer_size : line_tokens)
   {
      m_layer_sizes.push_back(std::stoi(layer_size));
   }
   
   for (unsigned i = 1; i < m_layer_sizes.size(); i++)
   {
      int previous_size = m_layer_sizes[i-1];
      int layer_size = m_layer_sizes[i];
      // weights
      line_tokens = getNextLineAndSplitIntoTokens(file);
      vector<float> weights(previous_size * (layer_size));
      std::transform(begin(line_tokens), end(line_tokens), begin(weights), [](std::string in){ return std::stof(in); });

      // biases
      line_tokens = getNextLineAndSplitIntoTokens(file);
      vector<float> biases(layer_size);
      std::transform(begin(line_tokens), end(line_tokens), begin(biases), [](std::string in){ return std::stof(in); });

      m_weights.push_back(Mat(layer_size, previous_size, weights));
      m_biases.push_back(Mat(layer_size, 1, biases));
   }
}

void NNet::save(std::string file_path) const
{
   ofstream file(file_path);
   // csv, format:
   // [LAYER_SIZES]\n,
   // [
   //   WEIGHT_n\n
   //   BIAS_n\n
   // ]
   for (auto size : m_layer_sizes)
   {
      file << size << ",";
   }
   file << "\n";
   for (unsigned layer_size = 0; layer_size < m_layer_sizes.size() - 1; layer_size++)
   {
      auto layer_weights = m_weights[layer_size].getVals();
      for (unsigned i = 0; i < m_weights[layer_size].getHeight() * m_weights[layer_size].getWidth(); i++) {
         file << layer_weights[i] << ",";
      }
      file << "\n";

      auto layer_biases = m_biases[layer_size].getVals();
      for (unsigned i = 0; i < m_biases[layer_size].getHeight() * m_biases[layer_size].getWidth(); i++) {
         file << layer_biases[i] << ",";
      }
      file << "\n";
   }
}

ParallelMat NNet::multipleCompute(const std::vector<Mat>& inputs) const {
   const unsigned INPUT_SIZE = m_layer_sizes[0];
   for(auto &input : inputs) {
      assert(input.getWidth() == 1);
      assert(input.getHeight() == INPUT_SIZE);
   }

   return multipleCompute(inputs);
}

ParallelMat NNet::multipleCompute(const ParallelMat& batch) const {
   auto a_l = batch;

   for (unsigned i = 0; i < m_weights.size(); i++)
   {
      auto wa_l = m_weights[i] * a_l;
      auto zl = m_biases[i] + wa_l;
      a_l = zl.relu();
   }
   return a_l;
}

Mat NNet::compute(const Mat &input) const
{
   auto a_l = input;

   for (unsigned i = 0; i < m_weights.size(); i++)
   {
      auto zl = m_weights[i] * a_l + m_biases[i];
      a_l = zl.relu();
   }
   return a_l;
}

// returns a vector of gradients for each weight based on
// computed diffs from backpropagate
vector<Mat> NNet::computeWeights(vector<Mat> &diffs, vector<Mat>& as) const
{
   auto new_weights = vector<Mat>();
   for (unsigned i = 0; i < m_weights.size(); i++)
   {
      Mat new_weight = diffs[i] * as[i].transpose();
      new_weights.push_back(new_weight);
   }
   return new_weights;
}

// returns a vector of gradients for each bias based on
// computed diffs from backpropagate
vector<Mat> NNet::computeBiases(vector<Mat> &diffs) const
{
   auto new_biases = vector<Mat>();
   for (unsigned i = 0; i < m_biases.size(); i++)
   {
      new_biases.push_back(diffs[i]);
   }
   return new_biases;
}

void NNet::adjustWeightsAndBiases(vector<Mat> &weight_grad, vector<Mat> &bias_grad, float learning_rate, unsigned iterations)
{
   float d = learning_rate / iterations;
   for (unsigned i = 0; i < weight_grad.size(); i++)
   {
      m_weights[i] = m_weights[i] - d * weight_grad[i];
      m_biases[i] = m_biases[i] - d * bias_grad[i];
   }
}

void NNet::forwardPass(const Mat& input, vector<Mat>& zs, vector<Mat>& as) const
{
   as.push_back(input);

   for (unsigned i = 0; i < m_weights.size(); i++)
   {
      auto zl = m_weights[i] * as[i] + m_biases[i];
      zs.push_back(zl);
      as.push_back(zl.relu());
   }
}


std::pair<std::vector<Mat>,std::vector<Mat>> NNet::multipleBackPropagate(
   const std::vector<Mat>& inputs, 
   const std::vector<Mat>& desired_outputs) const
{
   assert(inputs.size() == desired_outputs.size());

   std::vector<Mat> weight_grads;
   weight_grads.reserve(m_weights.size());
   std::vector<Mat> bias_grads;
   bias_grads.reserve(m_biases.size());

   vector<ParallelMat> as;
   vector<ParallelMat> zs;
   vector<ParallelMat> diffs;
   auto inputs_dup = ParallelMat{inputs};
   auto desired_outputs_dup = ParallelMat{desired_outputs};

   as.push_back(inputs_dup);
   for (unsigned i = 0; i < m_weights.size(); i++)
   {
      auto wa_l = m_weights[i] * as[i];
      auto zl = m_biases[i] + wa_l;
      zs.push_back(zl);
      as.push_back(zl.relu());
   }

   auto DaC = as.back() - desired_outputs_dup;

   ParallelMat final_z_relu_inv = zs.back().relu_inv();
   ParallelMat diff_end = DaC ^ final_z_relu_inv;
   diffs.push_back(diff_end);

   unsigned num_layers = m_layer_sizes.size() - 1;

   for (unsigned i = num_layers - 1; i >= 1; i--)
   {
      ParallelMat t_weights_x_diff_next = m_weights[i].transpose() * diffs[0];
      ParallelMat diff_0 = t_weights_x_diff_next ^ zs[i - 1].relu_inv();
      diffs.insert(diffs.begin(), diff_0);
   }


   for (unsigned i = 0; i < m_weights.size(); i++)
   {
      ParallelMat asit = as[i].transpose();
      ParallelMat new_weight_dup = diffs[i] * asit;
      Mat new_weight = new_weight_dup.sum();
      weight_grads.push_back(new_weight);
      bias_grads.push_back(diffs[i].sum());
   }
}

// returns a vector of diffs that can be passed into
// computeWeights or computeBiases
std::pair<std::vector<Mat>,std::vector<Mat>> NNet::backPropagate(
   const Mat &input,
   const Mat &desired_output) const
{
   std::vector<Mat> weight_grads;
   weight_grads.reserve(m_weights.size());
   std::vector<Mat> bias_grads;
   bias_grads.reserve(m_biases.size());

   vector<Mat> zs;
   vector<Mat> as;
   vector<Mat> diffs;
   forwardPass(input, zs, as);
   auto DaC = as.back() - desired_output;

   Mat final_z_relu_inv = zs.back().relu_inv();
   Mat diff_end = DaC ^ final_z_relu_inv;
   diffs.push_back(diff_end);

   int num_layers = m_layer_sizes.size() - 1; // has extra input layer we don't care about

   for (unsigned i = num_layers - 1; i >= 1; i--)
   {
      Mat t_weights_x_diff_next = m_weights[i].transpose() * diffs[0];
      Mat diff_0 = t_weights_x_diff_next ^ zs[i - 1].relu_inv();
      diffs.insert(diffs.begin(), diff_0);
   }

   for (unsigned i = 0; i < m_weights.size(); i++)
   {
      Mat asit = as[i].transpose();
      Mat new_weight = diffs[i] * asit;
      weight_grads.push_back(new_weight);
      bias_grads.push_back(diffs[i]);
   }
}

float NNet::sigmoid_act(float x)
{
   return 1 / (1 + std::exp(-x));
}

float NNet::sigmoid_inv(float x)
{
   float a_x = sigmoid_act(x);
   return a_x * (1 - a_x);
}

float NNet::reLU_act(float x)
{
   if (x < 0)
      return 0;
   return x;
}

float NNet::reLU_inv(float x)
{
   if (x < 0)
      return 0;
   return 1;
}

float NNet::leaky_reLU_act(float x)
{
   if (x < 0)
      return 0.1 * x;
   return x;
}

float NNet::leaky_reLU_inv(float x)
{
   if (x < 0)
      return 0.1;
   return 1;
}
