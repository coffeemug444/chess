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
   
   NNet nn({2, 8, 4, 1}, 'r');

   int batch_size = 10;

   for (int epoch = 0; epoch < 100; epoch++)
   {
      std::vector<Mat> input_batch;
      std::vector<Mat> desired_batch;
      for (int batch = 0; batch < batch_size; batch++)
      {
         std::vector<float> input_vec {dist(gen), dist(gen)};
         float result = (input_vec[0] < 0) ^ (input_vec[1] < 0) ? -1.f : 1.f;
         input_batch.push_back({2,1,input_vec});
         desired_batch.push_back({1,1,{result}});
      }

      std::cout << "epoch " << epoch << ' ';
      auto [weight_diffs, bias_diffs] = nn.multipleBackPropagate(input_batch, desired_batch);
      // std::cout << "[0,0] of weight diffs on epoch " << epoch << ": {";
      // std::string delim = "";
      // for (const auto& w : weight_diffs)
      // {
      //    std::cout << delim << w.getVals().at(0);
      //    delim = ", ";
      // }
      // std::cout << "}\n";
      // std::cout << "[0,0] of bias diffs on epoch " << epoch << ": {";
      // delim = "";
      // for (const auto& b : bias_diffs)
      // {
      //    std::cout << delim << b.getVals().at(0);
      //    delim = ", ";
      // }
      // std::cout << "}\n";


      float learning_rate = 0.03f;

      nn.adjustWeightsAndBiases(weight_diffs, bias_diffs, learning_rate);
   }

   int total_correct = 0;
   for (int i = 0; i < 10; i++)
   {
      std::vector<float> input_vec {dist(gen), dist(gen)};
      float desired_result = (input_vec[0] < 0) ^ (input_vec[1] < 0) ? -1.f : 1.f;
      Mat result = nn.compute({2,1,input_vec});

      float actual = result.getVal(0,0);

      bool correct = (desired_result > 0 && actual > 0.5) || (desired_result < 0 && actual < -0.5);

      std::cout << (correct ? "✔" : "✘") << ": Wanted " << desired_result << ", got " << actual << '\n';
      total_correct += correct;
   }

   std::cout << total_correct << "/" << 10 << '\n';




}




