/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <tuple>
#include <map>
#include <algorithm>
#include <random>

using namespace std;

#include "tiny_dnn/tiny_dnn.h"
#include "gray_train_images.cpp"
#include "gray_train_labels.cpp"

extern std::vector<tiny_dnn::vec_t> train_images_data;
extern std::vector<tiny_dnn::label_t> train_labels_data;


static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
                          tiny_dnn::core::backend_t backend_type) {
  using namespace tiny_dnn::layers;

  using tiny_dnn::core::connection_table;
  using padding = tiny_dnn::padding;

  using conv = tiny_dnn::convolutional_layer;
  using fc = tiny_dnn::fully_connected_layer;
  using max_pool = tiny_dnn::max_pooling_layer;
  using batch_norm = tiny_dnn::batch_normalization_layer;
  using dropout = tiny_dnn::dropout_layer;
  using relu = tiny_dnn::relu_layer;
  using softmax = tiny_dnn::softmax_layer;
  // DeepThin
  const int n_fmaps1 = 32; // number of feature maps for upper layer
  const int n_fmaps2 = 48; // number of feature maps for lower layer
  const int n_fc = 512;  //number of hidden units in fully-connected layer

  const int input_w = 45;
  const int input_h = 45;
  const int input_c = 1;

  const int num_classes = 43;

  nn << batch_norm(input_w * input_h, input_c)
    << conv(input_w, input_h, 3, 3, input_c, n_fmaps1, tiny_dnn::padding::same, true, 2, 2, 0, 0)  // 3x3 kernel, 2 stride

    << batch_norm(23 * 23, n_fmaps1)
    << relu()
    << conv(23, 23, 3, 3, n_fmaps1, n_fmaps1, tiny_dnn::padding::same)  // 3x3 kernel, 1 stride

    << batch_norm(23 * 23, n_fmaps1)
    << relu()
    << max_pool(23, 23, n_fmaps1, 2, 1, false)
    << conv(22, 22, 3, 3, n_fmaps1, n_fmaps2, tiny_dnn::padding::same, true, 2, 2)  // 3x3 kernel, 2 stride

    << batch_norm(11 * 11, n_fmaps2)
    << relu()
    << conv(11, 11, 3, 3, n_fmaps2, n_fmaps2, tiny_dnn::padding::same)  // 3x3 kernel, 1 stride

    << batch_norm(11 * 11, n_fmaps2)
    << relu()
    << max_pool(11, 11, n_fmaps2, 2, 1, false)
    << fc(10 * 10 * n_fmaps2, n_fc, true, backend_type)

    << batch_norm(1 * 1, n_fc)
    << relu()
    << dropout(n_fc, 0.5)
    << fc(n_fc, num_classes, true, backend_type)
    << softmax();

  // // generate graph model in dot language
  // std::ofstream ofs("graph_deepthin.txt");
  // tiny_dnn::graph_visualizer viz(nn, "graph");
  // viz.generate(ofs);
}

static void train_lenet(const std::string &data_dir_path,
                        double learning_rate,
                        const int n_train_epochs,
                        const int n_minibatch,
                        tiny_dnn::core::backend_t backend_type) {


  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;

  // optimizer
  // tiny_dnn::adagrad optimizer;
  tiny_dnn::adam optimizer;
  // optimizer.alpha = learning_rate;
   // optimizer.alpha *=
  //   std::min(tiny_dnn::float_t(4),
  //            static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

  // optimizer.alpha *=
  //   static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate);


  std::cout << "load models..." << std::endl;
  construct_net(nn, backend_type);
  // nn.weight_init(tiny_dnn::weight_init::xavier(2.0));
  // nn.bias_init(tiny_dnn::weight_init::xavier(2.0));

  // std::vector<tiny_dnn::vec_t> mnist_all_train_images, mnist_all_test_images;
  // tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
  //     &mnist_all_train_images, -1.0, 1.0, 2, 2);
  // std::cout << "mnist_all_train_images.size():" << mnist_all_train_images.size() << std::endl; 
  // for (int i = 0; i < mnist_all_train_images.size(); ++i){
  //   std::cout << "mnist_all_train_images[i].size()" << mnist_all_train_images[i].size() << std::endl;
  //   for (int j = 0; j < mnist_all_train_images[i].size(); ++j){
  //     std::cout << mnist_all_train_images[i][j] << " ";
  //   }
  //   break;
  // }


  // シャッフル
  std::vector<tiny_dnn::vec_t> train_images, test_images;
  std::vector<tiny_dnn::label_t> train_labels, test_labels;
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::vector<int> num_list;
  for (int i=0; i<train_images_data.size(); i++) num_list.push_back(i);
  std::shuffle(num_list.begin(), num_list.end(), engine);

  for(int i = 0;i < num_list.size();i++){
    // train
    train_images.push_back(train_images_data[num_list[i]]);
    train_labels.push_back(train_labels_data[num_list[i]]);

    // test
    // if (i == 0){
      test_images.push_back(train_images_data[num_list[i]]);
      test_labels.push_back(train_labels_data[num_list[i]]);
    // }
  }

  // std::vector<tiny_dnn::vec_t> train_images = train_images_data;
  // std::vector<tiny_dnn::vec_t> test_images = train_images_data;
  // std::vector<tiny_dnn::label_t> train_labels = train_labels_data;
  // std::vector<tiny_dnn::label_t> test_labels = train_labels_data;
  // std::vector<tiny_dnn::label_t> test_labels = {1};

  // std::random_device rd;
  // std::mt19937 g(123);
  // std::shuffle(num_list.begin(), num_list.end(), g);

  // for(int i = 0;i < num_list.size();i++){
  //   if(i % 5 == 0) {
  //     test_images.push_back(train_images_data[num_list[i]]);
  //     test_labels.push_back(train_labels_data[num_list[i]]);
  //   } else {
  //     train_images.push_back(train_images_data[num_list[i]]);
  //     train_labels.push_back(train_labels_data[num_list[i]]);
  //   }
  // }


  std::cout << "start training" << std::endl;

  std::cout << "train_images size : " << train_images.size() << std::endl;
  std::cout << "train_labels size : " << train_labels.size() << std::endl;
  std::cout << "test_images size : " << test_images.size() << std::endl;
  std::cout << "test_labels size : " << test_labels.size() << std::endl;


  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;

  // std::vector<tiny_dnn::tensor_t *> prev_input_vector;
  // // tiny_dnn::tensor_t tensor_image(1, tiny_dnn::vec_t(45*45,1));
  // tiny_dnn::tensor_t tensor_image = {{77,76,77,83,90,98,116,125,126,136,144,146,193,234,248,247,240,236,204,159,142,142,156,181,205,229,240,241,227,205,179,145,113,89,97,103,107,97,87,78,75,72,68,67,67,81,79,79,85,94,107,127,130,117,122,135,160,206,243,250,249,242,222,187,148,147,154,166,184,204,225,236,241,234,218,195,157,122,94,98,102,107,98,89,81,76,74,73,73,74,83,82,82,87,100,118,137,132,105,113,138,185,225,252,251,250,242,205,171,143,163,175,179,182,194,216,231,239,240,233,215,174,135,103,103,104,107,100,92,83,78,76,76,78,79,79,81,85,89,106,130,133,121,94,137,183,229,245,252,251,246,234,195,175,170,208,214,193,165,165,199,222,236,238,240,233,193,152,115,117,115,111,102,93,86,78,73,74,76,78,77,80,86,92,111,134,126,115,102,158,205,237,248,253,249,243,229,193,185,194,223,231,221,194,183,191,211,229,239,241,237,216,180,141,119,108,105,109,106,95,81,73,73,75,78,75,79,86,96,115,136,119,114,118,173,215,235,246,251,246,237,223,193,194,211,225,234,238,222,205,186,198,214,227,225,220,219,194,156,117,98,94,113,118,108,87,73,73,76,79,73,78,86,101,119,137,118,118,136,175,210,240,243,242,239,227,212,194,194,205,215,215,208,198,189,181,179,177,176,165,157,166,151,126,111,97,84,101,113,120,94,76,74,81,87,73,83,98,117,126,128,112,114,131,164,188,199,193,186,182,173,162,152,150,152,155,153,147,141,136,132,128,126,123,115,110,114,108,99,95,93,91,98,106,118,95,78,74,87,98,78,91,111,133,131,114,105,107,118,145,156,147,138,129,124,119,114,109,105,101,100,98,97,95,92,91,91,91,89,88,86,84,85,87,88,95,104,98,99,111,92,78,75,91,105,101,107,117,139,127,96,99,102,105,111,114,113,112,111,111,110,109,107,106,107,108,110,113,114,115,118,122,122,114,118,122,119,118,117,115,116,119,100,95,104,86,74,74,88,99,126,128,134,145,135,114,115,116,117,119,121,123,126,128,132,135,137,138,140,142,142,143,145,147,151,155,152,145,134,129,123,113,108,105,102,101,102,91,87,89,78,71,73,81,88,149,151,154,152,148,143,133,129,131,132,133,135,139,144,150,155,160,163,165,166,163,161,161,164,168,174,167,157,144,129,114,98,90,86,83,82,82,82,79,75,71,70,71,74,76,170,173,173,165,167,171,138,123,125,119,114,111,114,118,126,130,134,137,135,128,118,115,117,117,121,131,139,140,127,119,110,99,92,87,86,85,85,81,77,72,71,71,71,73,74,181,181,181,177,171,163,133,117,114,105,101,102,104,107,114,115,113,112,108,102,99,98,102,99,99,103,110,113,110,108,108,112,106,96,90,86,84,80,76,73,73,72,71,73,76,184,183,182,181,170,153,129,112,102,95,93,100,101,103,107,106,103,97,97,100,108,113,115,112,107,99,97,96,99,100,105,122,119,106,94,86,81,78,76,75,75,73,71,74,76,180,178,175,167,165,165,130,107,94,93,96,106,107,105,98,108,119,113,129,156,182,192,189,189,178,155,134,118,112,101,96,111,114,111,96,86,79,78,78,77,75,73,71,73,74,155,156,155,137,143,160,129,107,94,96,100,107,107,105,102,116,134,148,174,205,222,230,230,226,215,197,175,156,141,121,104,103,104,104,98,89,80,80,80,81,77,75,74,77,81,137,136,131,108,119,149,126,108,96,101,106,108,106,107,111,127,148,179,206,229,237,243,248,237,226,216,197,179,163,141,119,100,96,97,100,94,84,84,84,85,80,77,78,83,87,163,140,111,95,110,140,124,109,96,107,114,116,111,111,124,141,159,179,185,185,208,222,230,208,191,179,155,140,146,147,140,114,102,99,105,100,88,91,92,89,83,79,82,84,87,184,151,109,87,102,135,124,113,101,114,118,110,108,117,146,156,159,161,155,149,174,196,215,192,172,157,140,129,131,149,157,123,104,96,112,110,96,100,100,93,85,80,83,86,89,193,159,114,85,97,132,126,118,107,120,120,99,104,125,171,173,161,150,138,129,147,167,189,171,157,149,147,139,121,151,173,133,107,92,114,116,105,108,105,96,87,82,83,88,92,175,152,121,93,101,131,133,127,114,125,120,91,109,144,197,195,179,175,165,152,139,134,136,130,139,169,189,182,128,158,188,152,116,88,104,113,115,107,100,93,87,83,83,90,96,159,144,123,99,105,129,140,137,122,129,121,88,116,157,210,214,204,198,188,174,145,141,156,131,138,184,206,196,133,158,190,168,127,87,102,113,118,106,96,91,88,85,84,89,94,143,135,123,105,110,127,143,144,130,134,122,88,122,169,219,229,225,215,201,183,152,157,190,143,140,190,211,200,135,156,186,178,135,86,103,114,120,103,93,90,88,86,85,88,91,131,129,125,114,117,127,132,135,135,137,124,87,127,178,226,232,225,215,188,155,149,164,195,144,134,175,208,205,137,153,179,171,130,86,104,115,121,101,90,89,88,87,86,89,93,120,118,115,109,110,115,117,125,139,141,127,90,122,167,220,230,222,193,162,135,150,173,204,152,136,163,188,186,132,153,181,168,128,86,106,116,119,98,88,89,88,87,88,91,94,117,114,111,108,108,110,110,122,144,143,128,95,114,151,208,219,206,161,135,123,152,180,206,160,140,152,163,160,131,158,185,164,124,86,106,115,115,96,87,89,88,89,91,93,95,135,135,136,135,136,137,133,138,152,139,122,99,110,139,197,188,155,118,115,129,152,171,188,160,143,139,135,136,146,173,188,148,113,86,100,109,113,97,88,89,90,91,93,94,95,160,160,160,159,158,157,153,149,146,139,129,111,107,119,162,169,159,131,123,126,143,157,170,169,167,163,164,170,185,184,169,126,103,94,111,113,105,96,91,90,90,90,92,93,94,180,179,178,176,174,171,167,154,135,137,134,124,109,104,124,152,173,162,149,138,150,160,170,187,197,198,205,212,215,184,145,110,100,104,120,116,97,94,92,92,89,88,89,90,91,181,179,178,176,175,173,166,150,124,121,124,134,123,110,101,128,164,192,202,204,214,219,220,224,226,225,229,223,195,156,124,120,117,114,110,101,88,90,93,96,90,86,86,86,86,183,179,175,172,170,169,166,150,122,112,112,125,124,118,104,111,129,168,195,214,225,231,235,236,236,237,227,207,172,140,117,118,114,109,100,92,85,90,93,92,88,85,83,84,85,175,170,164,159,155,152,160,150,123,107,100,107,118,123,115,104,100,131,162,192,202,213,224,221,222,228,211,187,155,131,115,112,106,99,90,85,83,91,92,87,85,83,80,81,83,137,134,130,124,113,102,139,146,124,107,94,86,99,115,128,123,112,105,114,131,137,153,177,155,152,171,182,180,156,135,117,110,101,93,86,82,81,89,90,82,80,78,77,78,79,169,151,127,101,94,100,135,142,125,108,94,84,99,115,128,131,127,109,105,108,108,116,128,116,115,128,144,153,151,137,121,113,102,90,82,80,81,89,89,81,81,80,77,79,81,205,177,138,90,84,104,133,140,127,110,96,88,101,115,123,131,136,121,110,103,103,101,96,99,101,104,116,130,143,137,125,116,103,89,80,78,82,89,88,79,81,82,78,80,83,190,180,160,105,82,82,124,139,128,112,100,93,98,104,110,115,120,125,122,116,132,131,116,122,126,125,130,133,135,132,126,114,101,89,80,77,79,85,85,79,79,79,79,80,82,160,154,142,109,90,84,125,142,136,118,107,105,106,108,111,114,119,126,128,128,138,138,130,131,131,129,133,137,140,133,123,112,101,90,80,77,77,81,83,80,80,81,81,82,83,133,127,119,110,100,94,128,144,145,128,119,121,120,119,120,121,123,127,132,136,136,137,139,134,129,124,128,135,141,131,119,111,102,94,83,77,74,78,80,81,81,81,81,83,84,133,129,122,113,107,104,126,142,155,147,141,140,138,136,136,135,134,133,132,131,131,133,138,131,123,114,113,115,118,118,115,110,107,102,89,78,71,74,76,79,78,78,78,84,90,134,130,124,117,114,112,121,128,135,131,127,125,123,122,122,121,119,119,113,106,113,126,143,134,122,105,106,112,122,126,125,114,108,102,92,82,72,73,76,81,83,85,87,92,96,127,124,120,115,111,109,109,108,106,105,104,103,102,101,101,100,99,99,91,81,94,116,146,138,123,99,103,114,133,141,139,118,107,100,94,85,74,74,77,85,91,95,99,100,101,101,101,99,91,84,79,81,83,84,87,90,92,93,94,96,93,88,82,80,80,84,106,142,138,124,98,103,116,137,151,153,122,107,99,92,84,76,74,78,89,96,102,108,105,101,90,93,95,82,74,70,71,74,79,86,89,89,88,88,91,93,91,82,79,79,81,102,139,138,124,97,101,115,138,152,153,121,105,98,94,88,80,75,76,85,95,102,105,101,98,85,92,97,78,71,70,68,72,80,89,92,89,86,84,89,94,97,86,81,79,81,101,137,137,125,96,100,113,138,149,149,119,104,96,97,92,84,76,74,80,92,101,100,97,94}};
  // prev_input_vector.push_back(&tensor_image);


  int epoch = 1;
  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;

    // for (int i = 0; i < nn.depth(); i++) {
    //   std::cout << "############ layer " << i << " ############\n";
    //   std::cout << "layer type:" << nn[i]->layer_type() << " ";
    //   std::cout << "input:" << nn[i]->in_data_size() << "(" << nn[i]->in_data_shape() << ") ";
    //   std::cout << "output:" << nn[i]->out_data_size() << "(" << nn[i]->out_data_shape() << ")\n";

    //   std::cout << "prev_input_vector.size(): " << prev_input_vector.size() << ", " << prev_input_vector[0]->size()
    //     << ", " << prev_input_vector[0][0].size() << ", " << prev_input_vector[0][0][0].size() << std::endl;

    //   // 今回の入力を一つ前の出力にする
    //   std::vector<tiny_dnn::tensor_t *> cur_input_vector;
    //   tiny_dnn::tensor_t cur_input_tensor(1, tiny_dnn::vec_t(prev_input_vector[0][0][0].size(),1));
    //   cur_input_vector.push_back(&cur_input_tensor);
    //   // copy(prev_input_vector.begin(), prev_input_vector.end(), cur_input_vector.begin());
    //   cur_input_vector[0][0].pop_back();
    //   cur_input_vector[0][0].push_back(prev_input_vector[0][0][0]);

    //   // 今回の出力のsizeを決める
    //   std::vector<tiny_dnn::tensor_t *> cur_output_vector;
    //   tiny_dnn::tensor_t cur_output_tensor(1, tiny_dnn::vec_t(nn[i]->out_data_size(),1));
    //   cur_output_vector.push_back(&cur_output_tensor);

    //   // forward始まる前のsizeを確認
    //   std::cout << "fwd in\n";
    //   std::cout << "prev_input_vector.size(): " << prev_input_vector.size() << ", " << prev_input_vector[0]->size()
    //     << ", " << prev_input_vector[0][0].size() << ", " << prev_input_vector[0][0][0].size() << std::endl;
    //   std::cout << "cur_input_vector.size(): " << cur_input_vector.size() << ", " << cur_input_vector[0]->size()
    //     << ", " << cur_input_vector[0][0].size() << ", " << cur_input_vector[0][0][0].size() << std::endl;
    //   std::cout << "cur_output_vector.size(): " << cur_output_vector.size() << ", " << cur_output_vector[0]->size()
    //     << ", " << cur_output_vector[0][0].size() << ", " << cur_output_vector[0][0][0].size() << std::endl;

    //   nn[i]->forward_propagation(cur_input_vector, cur_output_vector);
    //   std::cout << "fwd end\n";

    //   // outputの値をprevに代入
    //   prev_input_vector[0][0].pop_back();
    //   prev_input_vector[0][0].push_back(cur_output_vector[0][0][0]);
      
    //   // forward終わった後のsizeを確認
    //   std::cout << "prev_input_vector.size(): " << prev_input_vector.size() << ", " << prev_input_vector[0]->size()
    //     << ", " << prev_input_vector[0][0].size() << ", " << prev_input_vector[0][0][0].size() << std::endl;
    //   std::cout << "cur_input_vector.size(): " << cur_input_vector.size() << ", " << cur_input_vector[0]->size()
    //     << ", " << cur_input_vector[0][0].size() << ", " << cur_input_vector[0][0][0].size() << std::endl;
    //   std::cout << "cur_output_vector.size(): " << cur_output_vector.size() << ", " << cur_output_vector[0]->size()
    //     << ", " << cur_output_vector[0][0].size() << ", " << cur_output_vector[0][0][0].size() << std::endl;

    //   // 各レイヤーのfeature mapを出力
    //   ofstream fout;
    //   string file_name = "feature_map/" + std::to_string(i) + "_" + nn[i]->layer_type() + ".txt";
    //   fout.open(file_name, std::ofstream::out);
    //   if(fout.fail()){
    //       std::cout << file_name << "fout.open() failed." << std::endl;
    //       return -1;
    //    }
    //   for (int r=0; r<cur_output_vector[0][0][0].size(); ++r){
    //     fout << cur_output_vector[0][0][0][r] << " ";
    //   }
    //   fout << "\n";
    //   fout.close();
    //   std::cout << "fout sccess." << std::endl;
      
    //   // input_vectorにはもう用はないので消去
    //   cur_input_vector.clear();
    //   cur_output_vector.clear();
    // }


    ++epoch;
    tiny_dnn::result res = nn.test(test_images, test_labels);
    std::cout << res.num_success << "/" << res.num_total << std::endl;

    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

  // training
  nn.train<tiny_dnn::cross_entropy_multiclass>(optimizer, train_images, train_labels, n_minibatch,
                          n_train_epochs, on_enumerate_minibatch,
                          on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  // test and show results
  nn.test(test_images, test_labels).print_detail(std::cout);
  // save network model & trained weights
  // nn.save("LeNet-model");
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
  const std::array<const std::string, 5> names = {{
    "internal", "nnpack", "libdnn", "avx", "opencl",
  }};
  for (size_t i = 0; i < names.size(); ++i) {
    if (name.compare(names[i]) == 0) {
      return static_cast<tiny_dnn::core::backend_t>(i);
    }
  }
  return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
            << " --learning_rate 1"
            << " --epochs 30"
            << " --minibatch_size 16"
            << " --backend_type internal" << std::endl;
}

int main(int argc, char **argv) {
  double learning_rate                   = 0.01;
  int epochs                             = 30;
  std::string data_path                  = "";
  int minibatch_size                     = 32;
  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  if (argc == 2) {
    std::string argname(argv[1]);
    if (argname == "--help" || argname == "-h") {
      usage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--minibatch_size") {
      minibatch_size = atoi(argv[count + 1]);
    } else if (argname == "--backend_type") {
      backend_type = parse_backend_name(argv[count + 1]);
    } else if (argname == "--data_path") {
      data_path = std::string(argv[count + 1]);
    } else {
      std::cerr << "Invalid parameter specified - \"" << argname << "\""
                << std::endl;
      usage(argv[0]);
      return -1;
    }
  }
  if (data_path == "") {
    std::cerr << "Data path not specified." << std::endl;
    usage(argv[0]);
    return -1;
  }
  if (learning_rate <= 0) {
    std::cerr
      << "Invalid learning rate. The learning rate must be greater than 0."
      << std::endl;
    return -1;
  }
  if (epochs <= 0) {
    std::cerr << "Invalid number of epochs. The number of epochs must be "
                 "greater than 0."
              << std::endl;
    return -1;
  }
  if (minibatch_size <= 0 || minibatch_size > 60000) {
    std::cerr
      << "Invalid minibatch size. The minibatch size must be greater than 0"
         " and less than dataset size (60000)."
      << std::endl;
    return -1;
  }
  std::cout << "Running with the following parameters:" << std::endl
            << "Data path: " << data_path << std::endl
            << "Learning rate: " << learning_rate << std::endl
            << "Minibatch size: " << minibatch_size << std::endl
            << "Number of epochs: " << epochs << std::endl
            << "Backend type: " << backend_type << std::endl
            << std::endl;
  try {
    train_lenet(data_path, learning_rate, epochs, minibatch_size, backend_type);
  } catch (tiny_dnn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
  return 0;
}
