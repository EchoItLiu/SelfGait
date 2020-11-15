# SelfGait
### PyTorch implementation of [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733).

<div align="center">
  <img src="https://github.com/EchoItLiu/SelfGait/blob/main/exp/SelfGait_framework.png">
</div>
<p align="center">
  Figure 1: The framework of SelfGait
</p>

### Installation

Clone the repository and run
```
$ conda env create --name byol --file env.yml
$ conda activate byol
$ python main.py
```

## Config

Before running PyTorch SelfGait, make sure you choose the correct running configurations on the config.yaml file.

```yaml
#network:
#  name: resnet18

  # Specify a folder containing a pre-trained model to fine-tune. If training from scratch, pass None.
#  fine_tune_from: 'resnet-18_40-epochs'

#  projection_head:
#    mlp_hidden_size: 512
#    projection_size: 128

network:
  hidden_dim: 128
  lr: 1e-5
  hard_or_full_trip: 'full'
  batch_size: (8, 16)
  restore_path:
  restore_iter: 0
  total_iter: 80000
  margin: 0.2
  num_workers: 0
  frame_num: 30
  model_name: 'GaitSet'
  fine_tune_from: 'None'
  projection_head:
    mlp_hidden_size: 256
    projection_size: 128

data:
  dataset_path: "/home/yqliu/Dataset/CASIA-B/silhouettes"
  #dataset_path: "/home/yqliu/Dataset/oumvlp_pre/"
  resolution: '64'
  dataset: 'CASIA-B'
  #dataset: 'OUMVLP'
  f_length: 30
  partition_rate: 0.8
  #label_rate: 0.008
  label_rate: 0.0
  pid_num: 73
  pid_shuffle: False

data_transforms:
  s: 1
  input_shape: (96,96,3)

trainer:
  batch_size: (8, 16)
  m: 0.996 # momentum update
  checkpoint_interval: 5000
  max_epochs: 40
  num_workers: 4

optimizer:
  params:
    lr: 0.03
    momentum: 0.9
    weight_decay: 0.0004
    
CUDA_VISIBLE_DEVICES: '1,0'

```

## Evaluation

We measure the performance of SelfGait on two benchmark datasets that are CASIA64 and OUMVLP.

During training, BYOL learns features using the STL10 ```train+unsupervised``` set and evaluates in the held-out ```test``` set.

|       Linear Classifier      | Feature  Extractor | Architecture | Feature dim | Projection Head  dim | Epochs | Batch  Size | STL10 Top 1 |
|:----------------------------:|:------------------:|:------------:|:-----------:|:--------------------:|:------:|:-----------:|:-----------:|
|      Logistic Regression     |    PCA Features    |       -      |     256     |           -          |    -   |             |    36.0%    |
|              KNN             |    PCA Features    |       -      |     256     |           -          |    -   |             |    31.8%    |
| Logistic Regression  (Adam) |     BYOL (SGD)     |   [ResNet-18](https://drive.google.com/file/d/1Qj01H8cox8067cpCwhHZSQ0nfQl2RHbQ/view?usp=sharing)  |     512     |          128         |   40   | 64          |    70.1%    |
| Logistic Regression  (Adam) |     BYOL (SGD)     |   [ResNet-18](https://drive.google.com/file/d/1CFQZWKfBzAZp56EADYfMgq0HHua3XCQW/view?usp=sharing)  |     512     |          128         |   80   | 64          |    75.2%    |
