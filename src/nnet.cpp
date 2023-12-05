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

NNet::NNet(){};

// layer_sizes defines the number of outputs for each layer.
// An extra layer is added to the beginning notating the number
// of inputs into the net (layer_sizes' length is 1 greater than
// the number of layers in the net)
// initialisation is '1' for ones, '0' for zeros, or 'r' for random floats between 0 and 1
NNet::NNet(vector<int> layer_sizes, char initialisation, Mode mode)
:m_mode{mode}
{
   assert(initialisation == '0' || initialisation == '1' || initialisation == 'r' || initialisation == 'h');

   m_layer_sizes.push_back(layer_sizes[0]);

   for (unsigned i = 1; i < layer_sizes.size(); i++)
   {
      m_layer_sizes.push_back(layer_sizes[i]);
      m_biases.push_back(Mat::zeros(layer_sizes[i], 1));
      switch (initialisation)
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

   return multipleCompute(ParallelMat{inputs});
}

ParallelMat NNet::multipleCompute(const ParallelMat& batch) const {
   auto a_l = batch;

   for (unsigned i = 0; i < m_weights.size(); i++)
   {
      auto weight_x_activation = m_weights[i] * a_l;
      auto zl = m_biases[i] + weight_x_activation;
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

      if (i < m_weights.size() - 1 || m_mode == REGRESSION)
      {
         a_l = zl.relu();
      }
      else
      {
         a_l = zl.sigmoid();
      }
   }
   return a_l;
}

void NNet::adjustWeightsAndBiases(const vector<Mat> &weight_grad, const vector<Mat> &bias_grad, float learning_rate)
{
   for (unsigned i = 0; i < weight_grad.size(); i++)
   {
      m_weights[i] = m_weights[i] - learning_rate * weight_grad[i];
      m_biases[i] = m_biases[i] - learning_rate * bias_grad[i];
   }
}

std::pair<vector<Mat>, vector<Mat>> NNet::forwardPass(const Mat& input) const
{
   vector<Mat> layer_outputs, activations;
   layer_outputs.reserve(m_weights.size());
   activations.reserve(m_weights.size());

   activations.push_back(input);

   for (unsigned i = 0; i < m_weights.size(); i++)
   {
      auto zl = m_weights[i] * activations[i] + m_biases[i];
      layer_outputs.push_back(zl);
      if (i < m_weights.size() - 1 || m_mode == REGRESSION)
      {
         activations.push_back(zl.relu());
      }
      else
      {
         activations.push_back(zl.sigmoid());
      }
   }

   return {layer_outputs, activations};
}


std::pair<std::vector<Mat>,std::vector<Mat>> NNet::backPropagate(
   const std::vector<Mat>& inputs, 
   const std::vector<Mat>& desired_outputs) const
{
   assert(inputs.size() == desired_outputs.size());

   float input_size_inv = 1.f / inputs.size();

   std::vector<Mat> weight_grads;
   std::vector<Mat> bias_grads;

   vector<ParallelMat> activations;    
   vector<ParallelMat> layer_outputs;  // 'Zs'
   auto inputs_dup = ParallelMat{inputs};
   auto desired_outputs_dup = ParallelMat{desired_outputs};

   activations.push_back(inputs_dup);
   for (unsigned i = 0; i < m_weights.size(); i++)
   {
      layer_outputs.push_back(m_biases[i] + m_weights[i] * activations[i]);
      activations.push_back(layer_outputs.back().relu());
   }

   ParallelMat cost_derivative = [&]() -> ParallelMat
   {
      if (m_mode == CLASSIFICATION)
      {
         return activations.back().binary_crossentropy_loss_derivative(desired_outputs_dup);
      } 
      else return activations.back() - desired_outputs_dup;
   }();
   

   ParallelMat final_z_relu_inv = layer_outputs.back().relu_inv();
   ParallelMat delta = cost_derivative ^ final_z_relu_inv;  

   weight_grads.push_back(input_size_inv * (delta * activations[m_weights.size() - 1].transpose()).sum());
   bias_grads.push_back(input_size_inv * delta.sum());


   int num_layers = m_weights.size();

   for (int i = 2 ; i <= num_layers; i++)
   {
      int l = num_layers - i;

      const ParallelMat& layer_output_i = layer_outputs[l];
      const ParallelMat sp = layer_output_i.relu_inv();
      delta = (m_weights[l + 1].transpose() * delta) ^ sp;

      weight_grads.insert(weight_grads.begin(), input_size_inv * (delta * activations[l].transpose()).sum());
      bias_grads.insert(bias_grads.begin(), input_size_inv * delta.sum());
   }

   return {weight_grads, bias_grads};
}
