#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <ranges>
#include <thread>
#include <random>

#include "piece.hpp"
#include "board.hpp"
#include "nnet.hpp"

#include <iostream>

using namespace std::chrono_literals;

int main() 
{
   std::random_device rd;
   std::mt19937 gen(rd());

   std::uniform_real_distribution<float> dist(-5.f, 5.f);
   
   NNet nn({2, 8, 4, 2}, 'r', NNet::REGRESSION);

   int batch_size = 20;

   for (int epoch = 0; epoch < 100; epoch++)
   {

      std::cout << "------ EPOCH " << epoch << " ------\n";
      std::vector<Mat> input_batch;
      std::vector<Mat> desired_batch;
      for (int batch = 0; batch < batch_size; batch++)
      {
         std::vector<float> input_vec {dist(gen), dist(gen)};
         float desired_result = (input_vec[0] < 0) ^ (input_vec[1] < 0) ? 0 : 1;

         input_batch.push_back({2,1,input_vec});
         desired_batch.push_back({2,1,std::vector<float>{1.f - desired_result, desired_result}});
      }

      auto [weight_diffs, bias_diffs] = nn.backPropagate(input_batch, desired_batch);

      float learning_rate = 0.03f;

      nn.adjustWeightsAndBiases(weight_diffs, bias_diffs, learning_rate);
   }

   int total_correct = 0;
   for (int i = 0; i < 10; i++)
   {
      std::vector<float> input_vec {dist(gen), dist(gen)};
      float desired_result = (input_vec[0] < 0) ^ (input_vec[1] < 0) ? 0 : 1;
      Mat true_v{2,1,std::vector<float>{1.f - desired_result, desired_result}};
      Mat result = nn.compute({2,1,input_vec});
      
      auto vals = result.getVals();
      int output = vals[1] > vals[0] ? 0 : 1;

      bool correct = static_cast<float>(output) == desired_result;

      int confidence = 100 * vals[1-output];

      float a,b;
      a = input_vec[0] < 0 ? 0 : 1;
      b = input_vec[1] < 0 ? 0 : 1;
      
      std::cout << (correct ? "✔" : "✘") << " " << a << b << " desired: " << desired_result << ", actual " << output << " (" << confidence << "\% confident" << ")\n";

      total_correct += correct;
   }

   std::cout << total_correct << "/10 correct\n";


}




