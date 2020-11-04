import os
import os.path as osp
import numpy as np
from .data_set import DataSet
import math
from data.transforms import get_simclr_data_transforms, get_test_transforms


## load数据
def load_data(config):
    # path
    seq_dir = list()
    # 视角
    view = list()
    # 状态
    seq_type = list()
    # 身份标签
    label = list()
    ##
    seqs_l = list()

    # → (s = 1, input_shape = xxx)
    # 增广
    Transformer = get_simclr_data_transforms(**config['data_transforms'])
    Transformer_test=get_test_transforms(**config['data_transforms'])
    dataset_path=config['data']['dataset_path']
    dataset=config['data']['dataset']
    pid_num=config['data']['pid_num']
    pid_shuffle=config['data']['pid_shuffle']
    resolution=config['data']['resolution']
    f_length=config['data']['f_length']
    partition_rate=config['data']['partition_rate']
    label_rate=config['data']['label_rate']

    ## 按身份文件进行迭代： 如从001--120
    for _label in sorted(list(os.listdir(dataset_path))):
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if dataset == 'CASIA-B' and _label == '005':
                continue
        ## ID标签文件夹路径
        label_path = osp.join(dataset_path, _label)
        ## sorted缺省升序排列: 状态文件夹路径迭代
        for _seq_type in sorted(list(os.listdir(label_path))):

            ## 类别路径
            seq_type_path = osp.join(label_path, _seq_type)
            ## sorted缺省升序排列: 各类视角文件夹路径迭代
            for _view in sorted(list(os.listdir(seq_type_path))):
                
                _seq_dir = osp.join(seq_type_path, _view)
                ## png list:身份ID(label)-状态(seq_type)-状态ID(seq_dir)-视角(view)
                ## -时间戳帧(i).png
                seqs = os.listdir(_seq_dir)
                ## 
                ## 从N张.png里面随机截取连续的T帧 
                if len(seqs) >= f_length:
                    # 时间戳顺序  
                    seqs = sorted(seqs)
                    # 随机数
                    rand_st = np.random.randint(0,len(seqs) - f_length + 1) 
                    rand_ed = rand_st + f_length
                    # slice seq
                    curr_seq_imgs = seqs[rand_st:rand_ed]
                    ## [(..,..), ..., (..,..)]
                    seqs_l.append((curr_seq_imgs, _seq_dir, _label, _seq_type, _view))

                    # info路径 ['','','']
                    # seq_dir.append(_seq_dir)
                    # 身份标签 00x
                    # label.append(_label)
                    # 附属物标签 bg-0x
                    # seq_type.append(_seq_type)
                    # 视角标签 108
                    # view.append(_view)
                        
                        
    ## pid_num为partition点           
    partition_info = osp.join('partition_rate', '{}_{}_{}.npy'.format(
        dataset, partition_rate, pid_shuffle))

    ## 分割点
    partition_num = math.floor(len(seqs_l) * partition_rate) 
    label_num = math.floor(len(seqs_l) * label_rate)

    ## 保存原始文件配置信息
    seqs_partition = [seqs_l[label_num:partition_num], seqs_l[0:label_num], seqs_l[partition_num:]]

    ## 根据划分点pid_fname --- 获取训练测试的身份ID号列表
    train_list = seqs_partition[0]
    ft_list = seqs_partition[1]
    test_list = seqs_partition[2]
    
    

    ##  xarray list | 帧身份标签 | 视角 | 状态 | 路径
    train_source = DataSet(
        [train_list[i][0] for i in range(len(train_list))],
        [train_list[i][1] for i in range(len(train_list))],
        [train_list[i][2] for i in range(len(train_list))],
        [train_list[i][3] for i in range(len(train_list))],
        [train_list[i][4] for i in range(len(train_list))],
        Transformer,
        resolution)
        
    ##  xarray list | 帧身份标签 | 视角 | 状态 | 路径
    ft_source = DataSet(
        [train_list[i][0] for i in range(len(ft_list))],
        [train_list[i][1] for i in range(len(ft_list))],
        [train_list[i][2] for i in range(len(ft_list))],
        [train_list[i][3] for i in range(len(ft_list))],
        [train_list[i][4] for i in range(len(ft_list))],
        Transformer,
        resolution)
        
    ##  xarray list | 帧身份标签 | 视角 | 状态 | 路径
    test_source = DataSet(
        [test_list[i][0] for i in range(len(test_list))],
        [test_list[i][1] for i in range(len(test_list))],
        [test_list[i][2] for i in range(len(test_list))],
        [test_list[i][3] for i in range(len(test_list))],
        [test_list[i][4] for i in range(len(test_list))],
        Transformer_test,
        resolution)

    return train_source,ft_source, test_source
