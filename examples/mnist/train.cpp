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

#include <sys/socket.h> //アドレスドメイン
#include <sys/types.h>  //ソケットタイプ
#include <arpa/inet.h>  //バイトオーダの変換に利用
#include <unistd.h>     //close()に利用
#include <string>       //string型

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


  std::cout << "load models..." << std::endl;
  construct_net(nn, backend_type);

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

  std::cout << "start training" << std::endl;

  std::cout << "train_images size : " << train_images.size() << std::endl;
  std::cout << "train_labels size : " << train_labels.size() << std::endl;
  std::cout << "test_images size : " << test_images.size() << std::endl;
  std::cout << "test_labels size : " << test_labels.size() << std::endl;

  /************************結果受信用ソケット************************************/
  /* IP アドレス、ポート番号、ソケット */
  char destination[] = "127.0.0.1";
  unsigned short port = 8080;
  int dstSocket;

  /* sockaddr_in 構造体 */
  struct sockaddr_in dstAddr;

  /* sockaddr_in 構造体のセット */
  memset(&dstAddr, 0, sizeof(dstAddr));
  dstAddr.sin_port = htons(port);
  dstAddr.sin_family = AF_INET;
  dstAddr.sin_addr.s_addr = inet_addr(destination);

  /* ソケット生成 */
  dstSocket = socket(AF_INET, SOCK_STREAM, 0);

  /* 接続 */
  printf("Trying to connect to %s: \n", destination);
  connect(dstSocket, (struct sockaddr *) &dstAddr, sizeof(dstAddr));

  /************************結果受信用ソケット************************************/

  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;

  int epoch = 1;
  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;

    // lossの計算
    std::cout << "calculate loss..." << std::endl;
    auto train_loss = nn.get_loss<tiny_dnn::mse>(train_images, train_labels);
    auto test_loss = nn.get_loss<tiny_dnn::mse>(test_images, test_labels);

    // accuracyの計算
    std::cout << "calculate accuracy..." << std::endl;
    tiny_dnn::result train_results = nn.test(train_images, train_labels);
    tiny_dnn::result test_results = nn.test(test_images, test_labels);
    float_t train_accuracy = (float_t)train_results.num_success * 100 / train_results.num_total;
    float_t test_accuracy = (float_t)test_results.num_success * 100 / test_results.num_total;

    std::cout << "train loss: " << train_loss << " test loss: " << test_loss << std::endl;
    std::cout << "train accuracy: " << train_accuracy << "% test accuracy: " << test_accuracy << "%" << std::endl;

    // データ送信 (train_loss,test_loss,train_accuracy,test_accuracy を送る)
    auto send_str = std::to_string(train_loss) + "," + std::to_string(test_loss);  // loss
    send_str += "," + std::to_string(train_accuracy) + "," + std::to_string(test_accuracy);  // accuracy
    char* send_char = const_cast<char*>(send_str.c_str());
    send(dstSocket, send_char, 40, 0);   // 送信
    std::cout << send_char << std::endl;

    // きちんと送れたか応答確認
    char reccieved_message[128];
    recv(dstSocket, reccieved_message, sizeof(reccieved_message), 0);
    printf("reccieved_message: %s\n", reccieved_message);


    ++epoch;
    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

  // training
  nn.train<tiny_dnn::cross_entropy_multiclass>(optimizer, train_images, train_labels, n_minibatch,
                          n_train_epochs, on_enumerate_minibatch,
                          on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  // close socket
  close(dstSocket);

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
