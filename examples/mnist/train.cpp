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

// #include "gray_train_images.cpp"
// #include "gray_train_labels.cpp"
// extern std::vector<tiny_dnn::vec_t> train_images_data;
// extern std::vector<tiny_dnn::label_t> train_labels_data;

#define BUFFER_SIZE 256
#define MAX_VALUE 256

/* 画像データ */
typedef struct {
    unsigned int width; /* 画像の横サイズ */
    unsigned int height; /* 画像の縦サイズ */
    unsigned int num_bit; /* 1ピクセルあたりのビット数 */
    unsigned int max_value; /* 最大輝度値 */
    unsigned char *data; /* 画像データの先頭アドレス */
} IMAGE;

/* PNMのタイプ */
typedef enum {
    PNM_TYPE_PBM_ASCII,
    PNM_TYPE_PGM_ASCII,
    PNM_TYPE_PPM_ASCII,
    PNM_TYPE_PBM_BINARY, 
    PNM_TYPE_PGM_BINARY,
    PNM_TYPE_PPM_BINARY,
    PNM_TYPE_ERROR
} PNM_TYPE;


/* プロトタイプ宣言 */
int allocImage(IMAGE *);
unsigned int getImageInfo(IMAGE *, char *, unsigned int, PNM_TYPE);
unsigned int getNextValue(unsigned int *, char *, unsigned int, unsigned int);
int readP5(IMAGE *, char *, unsigned int);

/**
 * 画像データ格納用のバッファを確保する
 * 
 * 引数
 * image: 画像データ格納用の構造体
 * 
 * 返却値
 * 成功: 0
 * 失敗: 0以外
 */
int allocImage(IMAGE *image) {
    unsigned int size;
    unsigned char *data;
    unsigned int line_byte;

    if (image == NULL) {
        return -1;
    }

    /* 1行あたりのバイト数を計算（切り上げ） */
    line_byte = (image->width * image->num_bit + 7) / 8;

    /* サイズを決定してメモリ取得 */
    size = line_byte * image->height;
    data = (unsigned char *)malloc(sizeof(unsigned char) * size);
    if (data == NULL) {
        printf("mallocに失敗しました\n");
        return -1;
    }

    /* 取得したメモリのアドレスをimage構造体にセット */
    image->data = data;

    return 0;

}

/**
 * ヘッダーを読み込み結果をIMAGE構造体に格納する
 * 
 * 引数
 * image: 画像データ格納用の構造体
 * file_data: ファイルデータの先頭アドレス
 * file_size: ファイルデータのサイズ
 * 
 * 返却値
 * 成功: 画像データの先頭位置
 * 失敗: 0
 */
unsigned int getImageInfo(IMAGE *image, char *file_data, unsigned int file_size, PNM_TYPE type) {

    unsigned int read_pos;
    unsigned int value;
    unsigned int read_size;

    /* データ読み込み位置を先頭にセット */
    read_pos = 0;

    /* マジックナンバー分を読み飛ばす */
    read_pos += 2;

    /* 画像の横サイズを取得する */
    read_size = getNextValue(&value, file_data, read_pos, file_size);
    image->width = value;

    read_pos += read_size;

    /* 画像の縦サイズを取得する */
    read_size = getNextValue(&value, file_data, read_pos, file_size);
    image->height = value;

    read_pos += read_size;

    /* 画像の最大輝度値を取得する */
    switch (type) {
        case PNM_TYPE_PGM_ASCII:
        case PNM_TYPE_PPM_ASCII:
        case PNM_TYPE_PGM_BINARY:
        case PNM_TYPE_PPM_BINARY:
            /* 取得するのはPGMとPBMのみ */
            read_size = getNextValue(&value, file_data, read_pos, file_size);

            /* 最大輝度値の値チェック */
            if (value > MAX_VALUE) {
                printf("最大輝度値が不正です\n");
                return 0;
            }

            image->max_value = value;
            read_pos += read_size;
            break;
        default:
            break;
    }

    /* PNMタイプに応じてピクセルあたりのバイト数を設定 */
    switch (type) {
        case PNM_TYPE_PBM_ASCII:
        case PNM_TYPE_PBM_BINARY:
            image->num_bit = 1;
            break;
        case PNM_TYPE_PGM_ASCII:
        case PNM_TYPE_PGM_BINARY:
            image->num_bit = 8;
            break;
        case PNM_TYPE_PPM_ASCII:
        case PNM_TYPE_PPM_BINARY:
            image->num_bit = 24;
            break;
        default:
            break;
    }

    return read_pos;
}


/**
 * ファイルデータの次の文字列を数値化して取得する
 * 
 * 引数
 * value: 数値化した結果
 * file: ファイルデータの先頭アドレス
 * read_pos: 読み込み位置
 * file_size: ファイルデータのサイズ
 * 
 * 返却値
 * 成功: ファイルデータから読み込んだサイズ
 * 失敗: 0
 */
unsigned int getNextValue(unsigned int *value, char *data, unsigned int read_pos, unsigned int file_size) {
    char str[256];

    /* 空白系の文字やコメントを除いた次の文字列を取得する */
    unsigned int i, j, k;
    
    i = 0;
    while (read_pos + i < file_size) {
        /* 空白系の文字の場合は次の文字へスキップ */
        if (isspace(data[read_pos + i])) {
            i++;
            continue;
        }

        /* #ならそれ以降はコメントなので次の行へ */
        if (data[read_pos + i] == '#') {
            do {
                i++;
            } while (read_pos + i < file_size && data[read_pos + i] != '\n');

            /* \nの１文字文進める */
            i++;
        }

        break;
    }

    /* 文字列を取得 */
    j = 0;
    while (read_pos + i + j < file_size && !isspace(data[read_pos + i + j])) {
        /* 読み込んだバイト数をカウントアップ */
        j++;
    }

    /* 文字列を数字に変換 */
    for (k = 0; k < j; k++) {
        str[k] = data[read_pos + i + k];
    }
    str[k] = '\0';

    /* int化 */
    *value = (unsigned int)atoi(str);

    /* 読み込んだ文字数を返却 */
    return (i + j);
}


/**
 * P5ファイルのヘッダーと画像データをIMAGE構造体に格納する
 * 
 * 引数
 * image: 読み込んだ画像データ構造体
 * file_data: ファイルデータの先頭アドレス
 * file_size; ファイルデータのサイズ
 * 
 * 返却値
 * 成功: 0
 * 失敗: 0以外
 * 
 */
int readP5(IMAGE *image, char *file_data, unsigned int file_size) {
    unsigned int read_pos;
    unsigned int num_byte;
    unsigned int i, j, c;
    unsigned char byte_data;
    unsigned int color;
 
    /* ヘッダーを読み込んでImage構造体にデータをつめる */
    read_pos = getImageInfo(image, file_data, file_size, PNM_TYPE_PPM_BINARY);

    if (read_pos == 0) {
        /* ヘッダー読み込みに失敗していたら終了 */
        printf("ヘッダーがおかしいです\n");
        return -1;
    }

    /* ヘッダーの情報に基づいてメモリ確保 */
    if (allocImage(image) != 0) {
        printf("メモリ取得に失敗しました\n");
        return -1;
    }

    /* 最大輝度値の次にある空白系文字の分、読み込み位置を加算 */
    read_pos += 1;

    /* １ピクセルあたりの色の数を設定 */
    color = image->num_bit / 8;

    /* ファイル全体を読み終わるか必要な数分の輝度数をセットするまでループ */
    num_byte = 0;
    for (j = 0; j < image->height; j++) {
        for (i = 0; i < image->width; i++) {
            for (c = 0; c < color; c++) {

                /* 輝度値をIMAGE構造体に格納 */
                byte_data = (unsigned char)file_data[read_pos];
                image->data[num_byte] = byte_data;

                /* 格納したデータ数をインクリメント */
                num_byte += 1;

                /* データ読み込み位置と読み込んだデータ数を計算 */
                read_pos += 1;
                
                /* ファイルサイズ分読み込んでいたら終了 */
                if (read_pos >= file_size) {
                    return 0;
                }
            }
        }
    }

    return 0;
}

/**
 * バイトデータをencodeしてuint_image_dataに格納する
 * 
 * 今は引数がIMAGEになっているが、
 * 実際は unsigned char *image->data でもいい。
 * 
 */
// void encodePnm(IMAGE *image, unsigned int data_size, std::vector<unsigned int> &uint_image_data) {
void encodePnm(IMAGE *image, unsigned int data_size, tiny_dnn::vec_t &uint_image_data) {
    // std::cout << "--- encode start --- " << std::endl;
    // std::cout << "data_size: " << data_size << "\n";
    
    for (int i = 1; i < data_size + 1; i++)
    {
        unsigned int value = (unsigned int)image->data[i - 1];
        // std::cout << value << " ";
        uint_image_data[i-1] = value;
    }

    // std::cout << "\n";
    // std::cout << "--- encode end --- " << "\n";
}

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

void recieve_train_images(
  std::vector<tiny_dnn::vec_t> &recieved_images,
  std::vector<tiny_dnn::label_t> &recieved_labels
){
  /* IP アドレス、ポート番号、ソケット */
  char destination[] = "127.0.0.1";
  unsigned short port = 8080;
  int dstSocket;

  /* sockaddr_in 構造体 */
  struct sockaddr_in dstAddr;

  /* 各種パラメータ */
  int status;
  int numsnt;
  char toSendText[] = "Hello! This is a client.";

  /************************************************************/

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

  /* 受信内容 */
  int file_size;
  int n_images;
  char *file_data;
  IMAGE image;
  tiny_dnn::label_t label;

  // 接続確認
  printf("sending...\n");
  send(dstSocket, toSendText, strlen(toSendText)+1, 0);

  // この後送られてくる画像の数を受け取る
  recv(dstSocket, (char*)&n_images, sizeof(int), 0);
  printf("n_images: %d\n", n_images);

  /* パケット送受信 */
  for(int i=0; i<n_images; i++) {
    // 今回の画像のクラスラベル
    recv(dstSocket, (char*)&label, sizeof(tiny_dnn::label_t), 0);
    printf("label: %d\n", label);

    // この後送られるファイルのサイズを受け取る
    recv(dstSocket, (char*)&file_size, sizeof(int), 0);
    printf("received: %d\n", file_size);

    // ファイルデータ読み込み用のメモリ確保してから受け取る
    file_data = (char*)malloc(sizeof(char) * file_size);
    recv(dstSocket, file_data, file_size, 0);
    // printf("%s\n", file_data);

    // 画像の読み込み
    readP5(&image, file_data, file_size);

    // 画像のエンコード
    std::cout << "w:" << image.width << ", h:" << image.height << std::endl;
    unsigned int data_size = image.width * image.height * 1;
    tiny_dnn::vec_t uint_image_data(data_size);
    // std::vector<unsigned int> uint_image_data(data_size);
    encodePnm(&image, data_size, uint_image_data);

    // std::cout << "uint_image_data: " << uint_image_data.size() << "\n";
    // for(int i = 0; i < uint_image_data.size(); i++){
    //     std::cout <<  uint_image_data[i] << " ";
    // }
    // std::cout << "\n";

    recieved_labels.push_back(label);
    recieved_images.push_back(uint_image_data);

    free(file_data);
  }

  /* ソケット終了 */
  close(dstSocket);
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


  // 画像の受け取り
  std::vector<tiny_dnn::vec_t> recieved_images;
  std::vector<tiny_dnn::label_t> recieved_labels;
  recieve_train_images(recieved_images, recieved_labels);


  // シャッフル
  std::vector<tiny_dnn::vec_t> train_images, test_images;
  std::vector<tiny_dnn::label_t> train_labels, test_labels;
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::vector<int> num_list;
  for (int i=0; i<recieved_images.size(); i++) num_list.push_back(i);
  std::shuffle(num_list.begin(), num_list.end(), engine);

  for(int i = 0;i < num_list.size();i++){
    // train
    train_images.push_back(recieved_images[num_list[i]]);
    train_labels.push_back(recieved_labels[num_list[i]]);

    // test
    // if (i == 0){
      test_images.push_back(recieved_images[num_list[i]]);
      test_labels.push_back(recieved_labels[num_list[i]]);
    // }
  }

  std::cout << "start training" << std::endl;

  std::cout << "train_images size : " << train_images.size() << std::endl;
  std::cout << "train_labels size : " << train_labels.size() << std::endl;
  std::cout << "test_images size : " << test_images.size() << std::endl;
  std::cout << "test_labels size : " << test_labels.size() << std::endl;


  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;

  int epoch = 1;
  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;

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
