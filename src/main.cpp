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
#include "layerFullyConnected.hpp"
#include "layerSoftmax.hpp"

#include <iostream>

using namespace std::chrono_literals;

std::vector<float> get_input_vec(int num)
{
   return {
      static_cast<float>((num>>0) & 1),
      static_cast<float>((num>>1) & 1),
      static_cast<float>((num>>2) & 1),
      static_cast<float>((num>>3) & 1)
   };
}

std::vector<float> get_desired_vec(int num)
{
   return {
      static_cast<float>(num == 0),
      static_cast<float>(num == 1),
      static_cast<float>(num == 2),
      static_cast<float>(num == 3),
      static_cast<float>(num == 4),
      static_cast<float>(num == 5),
      static_cast<float>(num == 6),
      static_cast<float>(num == 7),
      static_cast<float>(num == 8),
      static_cast<float>(num == 9)
   };
}


int main() 
{
   std::random_device rd;
   std::mt19937 gen(rd());

   LayerFullyConnected layer0{4,  30, HE, RELU};
   LayerFullyConnected layer1{30, 20, HE, RELU};
   LayerFullyConnected layer2{20, 14, HE, RELU};
   LayerFullyConnected layer3{14, 10, HE, RELU};
   LayerSoftmax        layer4{10};
   
   NNet nn({
      layer0,
      layer1,
      layer2,
      layer3,
      layer4
   });

   int batch_size = 20;

   for (int epoch = 0; epoch < 3000; epoch++)
   {
      std::vector<Mat> input_batch;
      std::vector<Mat> desired_batch;
      for (int batch = 0; batch < batch_size; batch++)
      {
         int num = gen() % 10;

         input_batch.push_back({4,1,get_input_vec(num)});
         desired_batch.push_back({10,1,get_desired_vec(num)});
      }

      nn.backPropagate(input_batch, desired_batch);

      float learning_rate = 0.03f;

      nn.applyWeightsAndBiasesGradients(learning_rate);
   }

   int total_correct = 0;
   for (int i = 0; i < 10; i++)
   {
      int num = i;
      Mat true_v{10,1,get_desired_vec(num)};
      Mat result = nn.compute({4,1,get_input_vec(num)});
      
      auto results = result.getVals();

      int guessed_num = 0;
      float max_confidence = 0;
      for (int i = 0; i < 10; i++)
      {
         if (results[i] > max_confidence)
         {
            guessed_num = i;
            max_confidence = results[i];
         }
      }

      bool correct = num == guessed_num;

      std::cout << (correct ? "✔" : "✘") <<  " desired: " << num << ", actual " << guessed_num << " (" << static_cast<int>(100*max_confidence) << "\% confident" << ")\n";

      total_correct += correct;
   }

   std::cout << total_correct << "/10 correct\n";


}




