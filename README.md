# SelfGait
### PyTorch implementation of [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733).

<div align="center">
  <img src="https://github.com/EchoItLiu/SelfGait/blob/main/exp/SelfGait_framework.png">
</div>
<p align="center">
  Figure 1: The framework of SelfGait.
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

### Quantitative Experiment
We measure the performance of SelfGait on two benchmark datasets that are CASIA64 and OUMVLP.

<div align="center">
  <img src="https://github.com/EchoItLiu/SelfGait/blob/main/exp/CASIA64_Tab.jpg">
</div>
<p align="center">
  Table 1: Averaged rank-1 accuracies on Gallery All 14 views CASIA-B, excluding identical-view cases.
</p>


<div align="center">
  <img src="https://github.com/EchoItLiu/SelfGait/blob/main/exp/OUMVLP_Tab.png">
</div>
<p align="center">
  Table 2: Averaged rank-1 accuracies on OU-MVLP, excluding identical-view cases.
</p>

### Qualitative Experiment

<div align="center">
  <img src="https://github.com/EchoItLiu/SelfGait/blob/main/exp/EX_CL.svg">
</div>
<p align="center">
  Table 1: The CL rank-1 accuracy.
</p>


<div align="center">
  <img src="https://github.com/EchoItLiu/SelfGait/blob/main/exp/EX_BG.svg">
</div>
<p align="center">
  Table 2: The BG rank-1 accuracy.
</p>


