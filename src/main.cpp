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
   
   NNet nn({2, 8, 4, 1}, 'h', NNet::BINARY_CLASSIFICATION);

   int batch_size = 50;

   for (int epoch = 0; epoch < 1000; epoch++)
   {
      std::vector<Mat> input_batch;
      std::vector<Mat> desired_batch;
      for (int batch = 0; batch < batch_size; batch++)
      {
         float a, b;
         a = static_cast<float>(gen() % 2);
         b = static_cast<float>(gen() % 2);
         std::vector<float> input_vec {a,b};
         float desired_result = (a == b) ? 0 : 1;

         input_batch.push_back({2,1,input_vec});
         desired_batch.push_back({1,1,std::vector<float>{desired_result}});
      }

      auto [weight_diffs, bias_diffs] = nn.backPropagate(input_batch, desired_batch);

      float learning_rate = 0.03f;

      nn.adjustWeightsAndBiases(weight_diffs, bias_diffs, learning_rate);
   }

   int total_correct = 0;
   for (int i = 0; i < 10; i++)
   {
      float a, b;
      a = static_cast<float>(gen() % 2);
      b = static_cast<float>(gen() % 2);
      std::vector<float> input_vec {a,b};
      float desired_result = (a == b) ? 0 : 1;
      Mat true_v{1,1,std::vector<float>{desired_result}};
      Mat result = nn.compute({2,1,input_vec});
      
      float output = result.getVals()[0];

      int rounded = static_cast<int>(output + 0.5f);

      bool correct = std::abs(desired_result-output) < 0.5f;

      int confidence = 100*2*std::abs(output - 0.5f);
      
      std::cout << (correct ? "✔" : "✘") << " " << a << b << " desired: " << desired_result << ", actual " << rounded << " (" << confidence << "\% confident" << ")\n";

      total_correct += correct;
   }

   std::cout << total_correct << "/10 correct\n";


}




