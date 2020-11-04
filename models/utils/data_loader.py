import os
import os.path as osp
import numpy as np
from .data_set import DataSet
import math
from data.transforms import get_simclr_data_transforms, get_test_transforms


## load����
def load_data(config):
    # path
    seq_dir = list()
    # �ӽ�
    view = list()
    # ״̬
    seq_type = list()
    # ��ݱ�ǩ
    label = list()
    ##
    seqs_l = list()

    # �� (s = 1, input_shape = xxx)
    # ����
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

    ## ������ļ����е����� ���001--120
    for _label in sorted(list(os.listdir(dataset_path))):
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if dataset == 'CASIA-B' and _label == '005':
                continue
        ## ID��ǩ�ļ���·��
        label_path = osp.join(dataset_path, _label)
        ## sortedȱʡ��������: ״̬�ļ���·������
        for _seq_type in sorted(list(os.listdir(label_path))):

            ## ���·��
            seq_type_path = osp.join(label_path, _seq_type)
            ## sortedȱʡ��������: �����ӽ��ļ���·������
            for _view in sorted(list(os.listdir(seq_type_path))):
                
                _seq_dir = osp.join(seq_type_path, _view)
                ## png list:���ID(label)-״̬(seq_type)-״̬ID(seq_dir)-�ӽ�(view)
                ## -ʱ���֡(i).png
                seqs = os.listdir(_seq_dir)
                ## 
                ## ��N��.png���������ȡ������T֡ 
                if len(seqs) >= f_length:
                    # ʱ���˳��  
                    seqs = sorted(seqs)
                    # �����
                    rand_st = np.random.randint(0,len(seqs) - f_length + 1) 
                    rand_ed = rand_st + f_length
                    # slice seq
                    curr_seq_imgs = seqs[rand_st:rand_ed]
                    ## [(..,..), ..., (..,..)]
                    seqs_l.append((curr_seq_imgs, _seq_dir, _label, _seq_type, _view))

                    # info·�� ['','','']
                    # seq_dir.append(_seq_dir)
                    # ��ݱ�ǩ 00x
                    # label.append(_label)
                    # �������ǩ bg-0x
                    # seq_type.append(_seq_type)
                    # �ӽǱ�ǩ 108
                    # view.append(_view)
                        
                        
    ## pid_numΪpartition��           
    partition_info = osp.join('partition_rate', '{}_{}_{}.npy'.format(
        dataset, partition_rate, pid_shuffle))

    ## �ָ��
    partition_num = math.floor(len(seqs_l) * partition_rate) 
    label_num = math.floor(len(seqs_l) * label_rate)

    ## ����ԭʼ�ļ�������Ϣ
    seqs_partition = [seqs_l[label_num:partition_num], seqs_l[0:label_num], seqs_l[partition_num:]]

    ## ���ݻ��ֵ�pid_fname --- ��ȡѵ�����Ե����ID���б�
    train_list = seqs_partition[0]
    ft_list = seqs_partition[1]
    test_list = seqs_partition[2]
    
    

    ##  xarray list | ֡��ݱ�ǩ | �ӽ� | ״̬ | ·��
    train_source = DataSet(
        [train_list[i][0] for i in range(len(train_list))],
        [train_list[i][1] for i in range(len(train_list))],
        [train_list[i][2] for i in range(len(train_list))],
        [train_list[i][3] for i in range(len(train_list))],
        [train_list[i][4] for i in range(len(train_list))],
        Transformer,
        resolution)
        
    ##  xarray list | ֡��ݱ�ǩ | �ӽ� | ״̬ | ·��
    ft_source = DataSet(
        [train_list[i][0] for i in range(len(ft_list))],
        [train_list[i][1] for i in range(len(ft_list))],
        [train_list[i][2] for i in range(len(ft_list))],
        [train_list[i][3] for i in range(len(ft_list))],
        [train_list[i][4] for i in range(len(ft_list))],
        Transformer,
        resolution)
        
    ##  xarray list | ֡��ݱ�ǩ | �ӽ� | ״̬ | ·��
    test_source = DataSet(
        [test_list[i][0] for i in range(len(test_list))],
        [test_list[i][1] for i in range(len(test_list))],
        [test_list[i][2] for i in range(len(test_list))],
        [test_list[i][3] for i in range(len(test_list))],
        [test_list[i][4] for i in range(len(test_list))],
        Transformer_test,
        resolution)

    return train_source,ft_source, test_source
