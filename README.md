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


\begin{table*}[]
\centering
\begin{tabular}{l|l|cccccccccccc}
\hline
\multicolumn{2}{l|}{Gallery NM\#1-4} & \multicolumn{11}{c|}{0 - 180}                                                                  & \multicolumn{1}{l}{\multirow{2}{*}{Mean}} \\ \cline{1-13}
\multicolumn{2}{l|}{Probe}           & 0    & 18   & 36   & 54   & 72   & 90   & 108  & 126  & 144  & 162  & \multicolumn{1}{c|}{180} & \multicolumn{1}{l}{}                      \\ \hline
\multirow{6}{*}{NM \#5-6} & GaitSet \cite{chao2019gaitset}  & 90.8 & 97.9 & 99.4 & 96.9 & 93.6 & 91.7 & 95.0 & 97.8 & 98.9 & 96.8 & 85.8                     & 95.0                                      \\
                          & GaitNet \cite{song2019gaitnet}  & 91.2 & 92.0 & 90.5 & 95.6 & 86.9 & 92.6 & 93.5 & 96.0 & 90.9 & 88.8 & 89.0                     & 91.6                                      \\
                          & GaitPart \cite{fan2020gaitpart} & 94.1 & 98.6 & 99.3 & 98.5 & 94.0 & 92.3 & 95.9 & 98.4 & 99.2 & 97.8 & 90.4                     & 96.2                                      \\
                          & SG-2-$40\%$     & 84.8  & 88.4  & 93.8  & 96.1  & 88.8  & 85.0  & 86.2  & 89.4  & 93.0  & 90.8 & 84.0                      & 89.1                                       \\
                          & SG-4-$40\%$     & 88.4  & 93.2  & 95.2  & 96.0  & 91.4  & 90.8  & 92.2  & 93.6  & 95.2  & 92.2  & 90.2                     & 92.6                                       \\
                          & SG-6-$40\%$     & 90.4  & 93.8  & 96.8  & 96.7  & 92.0  & 92.2  & 92.2  & 94.6  & 96.8  & 94.8  & 88.6                      & 93.5                                      \\ \hline

\multirow{6}{*}{BG \#1-2} & GaitSet \cite{chao2019gaitset} & 83.8 & 91.2 &  91.8 & 88.8 & 83.3 & 81.0 & 84.1 & 90.0 & 92.2 & 94.4 & 79.0                     & 87.2                                      \\
                          & GaitNet \cite{song2019gaitnet} & 83.0 & 87.8 & 88.3 & 93.3 & 82.6 & 74.8 & 89.5 & 91.0 & 86.1 & 81.2 & 85.6                     & 85.7                                      \\
                          & GaitPart \cite{fan2020gaitpart} & 89.1 & 94.8 & 96.7 & 95.1 & 88.3 & 94.9 & 89.0 & 93.5 & 96.1 & 93.8 & 85.8                     & 91.5                                      \\
                          & SG-2     & 76.9  & 83.1  & 83.1  & 86.9  & 76.7  & 71.5  & 75.0  & 83.3  & 86.7  & 86.3  & 80.8                      & 81.3                                       \\
                          & SG-4     & 85.6  & 89.2  & 91.9  & 89.0  & 82.9  & 81.3  & 83.5  & 87.9  & 86.5  & 90.2  & 82.9                      & 86.4                                       \\
                          & SG-6    & 90.2  & 93.9  & 96.8  & 92.9  & 87.9  & 93.8  & 91.2  & 93.8  & 94.9  & 95.9  & 87.6                      & {\bf 92.6}                                       \\ \hline
\multirow{6}{*}{CL \#1-2} & GaitSet \cite{chao2019gaitset} & 61.4 & 75.4 & 80.7 & 77.3 & 72.1 & 70.1 & 71.5 & 73.5 & 73.5 & 68.4 & 50.0                     & 70.4                                      \\
                          & GaitNet \cite{song2019gaitnet} & 42.1 & 58.2 & 65.1 & 70.7 & 68.0 & 70.6 & 65.3 & 69.4 & 51.5 & 50.1 & 36.6                     & 58.9                                      \\
                          & GaitPart \cite{fan2020gaitpart} & 70.7 & 85.5 & 86.9 & 83.3 & 77.1 & 72.5 & 76.9 & 82.2 & 83.8 & 80.2 & 66.5                     & 78.7                                      \\
                          & SG-2     & 69.4  & 78.8 & 76.9 & 75.4 & 69.2 & 65.0 & 71.3 & 72.5 & 75.2 & 72.7 & 68.1                      & 72.2                                       \\
                          & SG-4     & 71.2  & 84.6  & 82.3  & 80.8  & 76.3  & 74.4  & 80.0  & 85.2  & 79.0  & 79.6  & 75.8                      & 79.0                                       \\
                          & SG-6    & 72.1  & 85.7  & 86.0  & 84.1  & 77.4  & 75.8  & 81.5  & 85.9  & 81.8  & 80.1  & 76.8                      & {\bf 80.6}                                       \\ \hline
\end{tabular}
\caption{Averaged rank-1 accuracies on CASIA-B, excluding identical-view cases.
}
\label{casia}
\end{table*}
