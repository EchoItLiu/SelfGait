import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import xarray as xr
from PIL import Image

class DataSet(tordata.Dataset):
    def __init__(self, seq_imgs, seq_dir, label, seq_type, view, transformer, resolution,cache=False):
        ## 
        self.seq_imgs = seq_imgs
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.transformer = transformer
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution)/64*10)
        self.data_size = len(self.label)
        ## 
        self.data_1 = [None] * self.data_size
        self.frame_set_1 = [None] * self.data_size
        self.data_2 = [None] * self.data_size
        self.frame_set_2 = [None] * self.data_size
        ## 
        self.label_set = set(self.label)
        self.seq_type_set = set(self.seq_type)
        self.view_set = set(self.view)
        self.cache=cache

        ## Label | Status | View 的 Shape
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1

        ## xarray: Label * Seq * View
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])

        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i

    def load_all_data(self):
        ## 按label进行loader封装
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, seq_img, path):
        ## 去除填充部分，10 -10 共20
        ## norm 255: 0-1
        return self.img2xarray(seq_img,
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32')

    def __getitem__(self, index):
        # pose sequence sampling
        ## Data Array类型: f_length * 64 (y) * 44 (x)
        if not self.cache:
            data_1 = [self.__loader__(self.seq_imgs[index], self.seq_dir[index])]
            data_2 = [self.__loader__(self.seq_imgs[index], self.seq_dir[index])]
            frame_set_1 = [set(feature.coords['frame'].values.tolist()) for feature in data_1]
            frame_set_1 = list(set.intersection(*frame_set_1))
            frame_set_2 = [set(feature.coords['frame'].values.tolist()) for feature in data_2]
            frame_set_2 = list(set.intersection(*frame_set_2))
        
        elif(self.data_1[index] is None):
            data_1 = [self.__loader__(self.seq_imgs[index], self.seq_dir[index])]
            data_2 = [self.__loader__(self.seq_imgs[index], self.seq_dir[index])]
            frame_set_1 = [set(feature.coords['frame'].values.tolist()) for feature in data_1]
            frame_set_1 = list(set.intersection(*frame_set_1))
            frame_set_2 = [set(feature.coords['frame'].values.tolist()) for feature in data_2]
            frame_set_2 = list(set.intersection(*frame_set_2))
            self.data_1[index] = data_1
            self.frame_set_1[index] = frame_set_1
            self.data_2[index] = data_2
            self.frame_set_2[index] = frame_set_2
        else:
            data_1 = self.data_1[index]
            frame_set_1 = self.frame_set_1[index]
            data_2 = self.data_2[index]
            frame_set_2 = self.frame_set_2[index]
        labels = self.label[index] 
        viewer = self.view[index]
        status = self.seq_type[index]
        seq_dir = self.seq_dir[index]
        #print(len(data),len(labels),len(viewer),len(status),len(seq_dir))
        #print(data.shape)
        #print(data[1,1,1])
        #print(type(data[0]),type(data[1]),type(data[2]))
        #print(data[0].shape,data[1].shape,data[2].shape)
        ##  xarray list | 帧身份标签 | 视角 | 状态 | 路径
        
        #return (data_1,data_2, labels, viewer, status, seq_dir)
        #print("get item first")
        #print(type(data_1),type(data_2))
        #print(len(data_1))#1
        #print(data_1[0].shape)#[30,64,44]
        return data_1,data_2,frame_set_1,frame_set_2, viewer, status, labels
        #data, frame_set, self.view[index], self.seq_type[index], self.label[index]

    def convertPIL(self, seq_imgs, view_path):
        # BGR
        #arr=np.array(cv2.resize(cv2.imread(osp.join(view_path, seq_imgs[0])),(64,64)))
        #print(arr.shape)
        #arr=np.array(Image.open(osp.join(view_path, seq_imgs[0])).convert("RGB"))
        #print(arr.shape)
        seq_PILs = [
        Image.fromarray(np.uint8(np.reshape(Image.open(osp.join(view_path, _img_file_name)).convert("RGB"),
        #Image.fromarray(np.uint8(np.reshape(cv2.resize(cv2.imread(osp.join(view_path, _img_file_name)),(64,64)),
            [self.resolution, self.resolution, -1])))
            for _img_file_name in seq_imgs
            if osp.isfile(osp.join(view_path, _img_file_name))]
        # BGR →  RGB     
        # seq_RGB_PILs = [cv2.cvtColor(pil, cv2.COLOR_BGR2RGB) for pil in seq_PILs]
        return seq_PILs

    def augPipeline(self, target_seq):
        # → [0,1] RGB
        # aug_imgs_seq = [self.transformer(tar) for tar in target_seq]
        # 单通道
        aug_imgs_seq = [self.transformer(tar)[0,:,:] for tar in target_seq]
        #
        return aug_imgs_seq

    ## 
    def img2xarray(self, seq_imgs, view_path):
        # 根据当前文件info路径序列下得到所有的.png文件进行img→array
        #print(len(seq_imgs))
        PILs_seq = self.convertPIL(seq_imgs, view_path)
        #print(len(PILs_seq))
        #print(np.array(PILs_seq[0]).shape)
        #PILs_seq=[8,64,64,3]
        #print(PILs_seq[0].dtype)
        #print(np.max(PILs_seq[0]),np.min(PILs_seq[0]))
        aug_seq = self.augPipeline(PILs_seq)
        #print(len(aug_seq))
        #aug_seq=[8,Tensor(3,64)]
        #print(type(aug_seq[0]))
        #print(aug_seq[0].size())
        ## 
        num_list = list(range(len(aug_seq)))
        #print(type(num_list))
        #print(num_list[1].size(),num_list[2].size())
        aug_seq=[aug.numpy() for aug in aug_seq]
            
        
        ## 第一维是帧数 第二维是图y，第三维是图x
        data_dict = xr.DataArray(
            aug_seq,
            ## coords shape大小等于frame_list
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict
    
    def __len__(self):
        return len(self.label)
