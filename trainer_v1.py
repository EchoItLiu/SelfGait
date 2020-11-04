import math
import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import _create_model_training_folder

from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata

from models.utils import TripletSampler
from models.network import TripletLoss


class BYOLTrainer:
    def __init__(self, online_network, target_network,conv_trans,TP_1,TP_2,projection_1, projection_2, predictor, optimizer, device,config,train_source,test_source, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.conv_trans=conv_trans
        self.TP_1=TP_1
        self.TP_2=TP_2
        self.projection_1=projection_1
        self.projection_2=projection_2
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.config=config
        self.m = params['m']
        #self.batch_size = params['batch_size']
        self.batch_size =(4, 16)
        self.P, self.M = self.batch_size
        #self.num_workers = params['num_workers']
        self.num_workers = 0
        self.train_source=train_source
        self.test_source=test_source
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])
        
        self.sample_type = 'all'
        self.hard_or_full_trip=config['network']['hard_or_full_trip']
        self.margin=config['network']['margin']
        
        self.Gait_list=[self.online_network,self.conv_trans,self.TP_1]
        self.Encode_list=[self.online_network,self.conv_trans,self.TP_1,self.projection_1,self.predictor]
        self.Target_list=[self.target_network,self.TP_2,self.projection_2]
        
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.triplet_loss.cuda()
        
        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01
        
        self.niter=config['network']['restore_iter']
        #self.regression_loss=nn.CosineSimilarity(dim=2, eps=1e-6)
        #self.regression_loss=nn.MSELoss()
        
        #print(type(self.projection_1),type(projection_1))

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projection_1.parameters(), self.projection_2.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.TP_1.parameters(), self.TP_2.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        #x=[b(1),n(31),d(256)]
        cos=nn.CosineSimilarity(dim=1, eps=1e-6)
        #x = F.normalize(x, dim=2)
        #y = F.normalize(y, dim=2)
        return 1 - 1 * cos(x,y)

    def initializes_target_network(self):
        # init momentum network as encoder net
        #print(type(self.online_network),type(self.target_network))
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        #print(type(self.projection_1),type(self.projection_2))    
        for param_q, param_k in zip(self.projection_1.parameters(), self.projection_2.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        for param_q, param_k in zip(self.TP_1.parameters(), self.TP_2.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs_1 = [batch[i][0] for i in range(batch_size)]
        seqs_2 = [batch[i][1] for i in range(batch_size)]
        frame_sets_1 = [batch[i][2] for i in range(batch_size)]
        frame_sets_2 = [batch[i][3] for i in range(batch_size)]
        view = [batch[i][4] for i in range(batch_size)]
        seq_type = [batch[i][5] for i in range(batch_size)]
        label = [batch[i][6] for i in range(batch_size)]
        batch = [seqs_1,seqs_2, view, seq_type, label, None, None]
        #print(batch_size,len(batch[0]),feature_num,len(seqs))

        def select_frame_1(index):
            sample = seqs_1[index]
            frame_set = frame_sets_1[index]
            if self.sample_type == 'random':
                if len(frame_set) >= 30:
                    if len(frame_set) > 40:
                        x = random.randint(0, (len(frame_set) - 40))
                        frame_set = frame_set[x:x+40]
                        frame_id_list = np.random.choice(frame_set, size=self.frame_num, replace=False)
                        frame_id_list.sort()
                        _ = [feature.loc[frame_id_list].values for feature in sample]
                    else:
                        frame_id_list = np.random.choice(frame_set, size=self.frame_num, replace=False)
                        frame_id_list.sort()
                        _ = [feature.loc[frame_id_list].values for feature in sample]
                else:
                    s_frame_id_list = np.random.choice(frame_set, size=(self.frame_num - len(frame_set)), replace=False).tolist()
                    frame_id_list = s_frame_id_list + frame_set
                    frame_id_list.sort()
                    _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.loc[frame_set].values for feature in sample]
            return _
        
        def select_frame_2(index):
            sample = seqs_2[index]
            frame_set = frame_sets_2[index]
            if self.sample_type == 'random':
                if len(frame_set) >= 30:
                    if len(frame_set) > 40:
                        x = random.randint(0, (len(frame_set) - 40))
                        frame_set = frame_set[x:x+40]
                        frame_id_list = np.random.choice(frame_set, size=self.frame_num, replace=False)
                        frame_id_list.sort()
                        _ = [feature.loc[frame_id_list].values for feature in sample]
                    else:
                        frame_id_list = np.random.choice(frame_set, size=self.frame_num, replace=False)
                        frame_id_list.sort()
                        _ = [feature.loc[frame_id_list].values for feature in sample]
                else:
                    s_frame_id_list = np.random.choice(frame_set, size=(self.frame_num - len(frame_set)), replace=False).tolist()
                    frame_id_list = s_frame_id_list + frame_set
                    frame_id_list.sort()
                    _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.loc[frame_set].values for feature in sample]
            return _

        seqs_1 = list(map(select_frame_1, range(len(seqs_1))))
        seqs_2 = list(map(select_frame_2, range(len(seqs_2))))

        if self.sample_type == 'random':
            seqs_1 = [np.asarray([seqs_1[i][j] for i in range(batch_size)]) for j in range(feature_num)]
            seqs_2 = [np.asarray([seqs_2[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            
            batch_frames_1 = [[
                                len(frame_sets_1[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            
            batch_frames_2 = [[
                                len(frame_sets_2[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            
            if len(batch_frames_1[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames_1[-1])):
                    batch_frames_1[-1].append(0)
                    
            if len(batch_frames_2[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames_2[-1])):
                    batch_frames_2[-1].append(0)
                    
            max_sum_frame_1 = np.max([np.sum(batch_frames_1[_]) for _ in range(gpu_num)])
            
            max_sum_frame_2 = np.max([np.sum(batch_frames_2[_]) for _ in range(gpu_num)])
            
            seqs_1 = [[
                        np.concatenate([
                                           seqs_1[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            
            seqs_2 = [[
                        np.concatenate([
                                           seqs_2[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]        
            
            seqs_1 = [np.asarray([
                                   np.pad(seqs_1[j][_],
                                          ((0, max_sum_frame_1 - seqs_1[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]
            
            seqs_2 = [np.asarray([
                                   np.pad(seqs_2[j][_],
                                          ((0, max_sum_frame_2 - seqs_2[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]        
            
            batch[5] = np.asarray(batch_frames_1)
            batch[6] = np.asarray(batch_frames_2)

        batch[0] = seqs_1
        batch[1] = seqs_2
        return batch

    def train(self, train_dataset,maxiter=10000):
        train_source=train_dataset
        #train_loader = DataLoader(train_dataset, batch_size=self.batch_size,num_workers=self.num_workers, drop_last=False, shuffle=True)
        for subnet in self.Encode_list:
            subnet.train()
        triplet_sampler = TripletSampler(train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        #self.niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        print("model save: ",model_checkpoints_folder)

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):

            for batch_view in train_loader:

                #batch_view_1 = batch_view_1.to(self.device)
                #batch_view_2 = batch_view_2.to(self.device)

                #if self.niter == 0:
                #    grid = torchvision.utils.make_grid(batch_view_1[:32])
                #    self.writer.add_image('views_1', grid, global_step=self.niter)

                #    grid = torchvision.utils.make_grid(batch_view_2[:32])
                #    self.writer.add_image('views_2', grid, global_step=self.niter)

                loss = self.update(batch_view)
                self.writer.add_scalar('loss', loss, global_step=self.niter)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if(self.niter%100==0):
                    print("iter=",str(self.niter)," loss=",loss.cpu().data)
                if(self.niter%1000==0):
                    self.test(test_dataset=self.test_source,niter=-1)
                    self.save_model(os.path.join(model_checkpoints_folder, 'model_'+str(self.niter)+'.pth'))
                self._update_target_network_parameters()  # update the key encoder
                self.niter += 1

            print("End of epoch {}".format(epoch_counter))
            if(self.niter==maxiter):
                return

        # save checkpoints
        

    def update(self, batch_view_1):
        
        seq_1,seq_2, view, seq_type, label, batch_frame_1,batch_frame_2=batch_view_1
        
        seq_1_x=seq_1
        seq_1_y=seq_1
        seq_2_x=seq_2
        seq_2_y=seq_2
        
        for i in range(len(seq_1)):
            _,_,w,h=seq_1[i].shape
            seq_1[i] = self.np2var(seq_1[i].reshape([-1,30,w,h])).float()
            seq_1_x[i] = seq_1[i][:,0:-1,:,:]
            seq_1_y[i] = seq_1[i][:,-1:,:,:]
        if batch_frame_1 is not None:
            batch_frame_1 = self.np2var(batch_frame_1).int()
                
        for i in range(len(seq_2)):
            _,_,w,h=seq_2[i].shape
            seq_2[i] = self.np2var(seq_2[i].reshape([-1,30,w,h])).float()
            seq_2_x[i] = seq_2[i][:,0:-1,:,:]
            seq_2_y[i] = seq_2[i][:,-1:,:,:]
        if batch_frame_2 is not None:
            batch_frame_2 = self.np2var(batch_frame_2).int()
        
        #print(len(seq_1))#1
        #print(seq_1[0].size())#[2,1920,64,44]
        
        # compute query feature
        predictions_from_view_1= self.predictor(self.projection_1(self.TP_1(self.conv_trans(self.online_network(*seq_1_x,batch_frame_1)))))
        predictions_from_view_2= self.predictor(self.projection_1(self.TP_1(self.conv_trans(self.online_network(*seq_2_x,batch_frame_2)))))
        
        # compute key features
        with torch.no_grad():
            targets_to_view_2= self.projection_2(self.TP_2(self.target_network(*seq_1_y,batch_frame_1)))
            targets_to_view_1= self.projection_2(self.TP_2(self.target_network(*seq_2_y,batch_frame_2)))
        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()
        
    
    def finetune(self, finetune_source,maxiter=50000):
        #train_loader = DataLoader(train_dataset, batch_size=self.batch_size,num_workers=self.num_workers, drop_last=False, shuffle=True)
        self.finetune_source=finetune_source
        for subnet in self.Gait_list:
            subnet.train()
        
        triplet_sampler = TripletSampler(finetune_source, self.batch_size)
        finetune_loader = tordata.DataLoader(
            dataset=finetune_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        self.niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        
        train_label_set = list(self.finetune_source.label_set)
        train_label_set.sort()

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):

            for batch_view in finetune_loader:
            
                seq_1,seq_2, view, seq_type, label, batch_frame_1,batch_frame_2=batch_view
        
                for i in range(len(seq_1)):
                    _,_,w,h=seq_1[i].shape
                    seq_1[i] = self.np2var(seq_1[i].reshape([-1,30,w,h])).float()
                if batch_frame_1 is not None:
                    batch_frame_1 = self.np2var(batch_frame_1).int()

                #print(len(seq_1))#1
                #print(seq_1[0].size())#[2,1920,64,44]
                
                feature= self.TP_1(self.conv_trans(self.online_network(*seq_1,batch_frame_1)))

                target_label = [train_label_set.index(l) for l in label]
                target_label = self.np2var(np.array(target_label)).long()
    
                triplet_feature = feature.permute(1, 0, 2).contiguous()
                triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
                (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num) = self.triplet_loss(triplet_feature, triplet_label)
                if self.hard_or_full_trip == 'hard':
                    loss = hard_loss_metric.mean()
                elif self.hard_or_full_trip == 'full':
                    loss = full_loss_metric.mean()
    
                self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
                self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
                self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
                self.dist_list.append(mean_dist.mean().data.cpu().numpy())
                
                self.writer.add_scalar('loss_ft', loss, global_step=self.niter)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if(self.niter%100==0):
                    print("iter=",str(self.niter)," loss=",loss.cpu().data)
                if(self.niter%500==0):
                    self.test(test_dataset=self.test_source,niter=-1)
                    self.save_model(os.path.join(model_checkpoints_folder, 'model_'+str(self.niter)+'.pth'))

                self._update_target_network_parameters()  # update the key encoder
                self.niter += 1
                if(self.niter==maxiter):
                    return

            print("End of epoch {}".format(epoch_counter))

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def cuda_dist(self,x, y):
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        dist = torch.sum(x ** 2, 2).unsqueeze(2) + torch.sum(y ** 2, 2).unsqueeze(
            2).transpose(1, 2) - 2 * torch.matmul(x, y.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist)).mean(0)
        return dist
    
    def evaluation(self,data, config):
        dataset = config['dataset'].split('-')[0]
        feature, view, seq_type, label = data
        label = np.array(label)
        view_list = list(set(view))
        view_list.sort()
        view_num = len(view_list)
        sample_num = len(feature)
    
        probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                          'OUMVLP': [['00']]}
        gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                            'OUMVLP': [['01']]}
    
        num_rank = 5
        acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
        for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
            for gallery_seq in gallery_seq_dict[dataset]:
                for (v1, probe_view) in enumerate(view_list):
                    for (v2, gallery_view) in enumerate(view_list):
                        gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                        gallery_x = feature[:, gseq_mask, :]
                        gallery_y = label[gseq_mask]
    
                        pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                        probe_x = feature[:, pseq_mask, :]
                        probe_y = label[pseq_mask]
    
                        dist = self.cuda_dist(probe_x, gallery_x)
                        idx = dist.sort(1)[1].cpu().numpy()
                        acc[p, v1, v2, :] = np.round(
                            np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                   0) * 100 / dist.shape[0], 2)
    
        return acc
        
    def de_diag(self,acc, each_angle=False):
        result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
        if not each_angle:
            result = np.mean(result)
        return result
    
    def transform(self, test_dataset,batch_size=1):
        self.online_network.eval()
        self.conv_trans.eval()
        self.TP_1.eval()
        
        for subnet in self.Gait_list:
            subnet.eval()
        
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(test_dataset),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            seq_1,seq_2, view, seq_type, label, batch_frame_1,batch_frame_2 = x
            for j in range(len(seq_1)):
                _,_,w,h=seq_1[j].shape
                seq_1[j] = self.np2var(seq_1[j]).float()
            if batch_frame_1 is not None:
                batch_frame_1 = self.np2var(batch_frame_1).int()
            # print(batch_frame, np.sum(batch_frame))

            feature= self.TP_1(self.conv_trans(self.online_network(*seq_1,batch_frame_1)))
            feature = feature.permute(1, 0, 2).contiguous()
            feature_list.append(feature.data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 1), view_list, seq_type_list, label_list\
        
    def test(self,test_dataset,niter=-1):
        if(niter>=0):
            self.load_model(niter)
        print('Transforming...')
        time = datetime.now()
        test = self.transform(test_dataset, batch_size=128)
        print('Evaluating...')
        acc = self.evaluation(test, self.config['data'])
        print('Evaluation complete. Cost:', datetime.now() - time)
        
        # Print rank-1 accuracy of the best model
        # e.g.
        # ===Rank-1 (Include identical-view cases)===
        # NM: 95.405,     BG: 88.284,     CL: 72.041
        for i in range(1):
            print('===Rank-%d (Include identical-view cases)===' % (i + 1))
            print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))
            self.writer.add_scalar('IN_NM', np.mean(acc[0, :, :, i]), global_step=self.niter)
            self.writer.add_scalar('IN_BG', np.mean(acc[1, :, :, i]), global_step=self.niter)
            self.writer.add_scalar('IN_CL', np.mean(acc[2, :, :, i]), global_step=self.niter)
                
        
        # Print rank-1 accuracy of the best model，excluding identical-view cases
        # e.g.
        # ===Rank-1 (Exclude identical-view cases)===
        # NM: 94.964,     BG: 87.239,     CL: 70.355
        for i in range(1):
            print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
            print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                self.de_diag(acc[0, :, :, i]),
                self.de_diag(acc[1, :, :, i]),
                self.de_diag(acc[2, :, :, i])))
            self.writer.add_scalar('EX_NM', self.de_diag(acc[0, :, :, i]), global_step=self.niter)
            self.writer.add_scalar('EX_BG', self.de_diag(acc[1, :, :, i]), global_step=self.niter)
            self.writer.add_scalar('EX_CL', self.de_diag(acc[2, :, :, i]), global_step=self.niter)
        
        # Print rank-1 accuracy of the best model (Each Angle)
        # e.g.
        # ===Rank-1 of each angle (Exclude identical-view cases)===
        # NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
        # BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
        # CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]
        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            print('NM:', self.de_diag(acc[0, :, :, i], True))
            print('BG:', self.de_diag(acc[1, :, :, i], True))
            print('CL:', self.de_diag(acc[2, :, :, i], True))
    '''
    self.online_network = online_network
    self.target_network = target_network
    self.conv_trans=conv_trans
    self.TP_1=TP_1
    self.TP_2=TP_2
    self.projection_1=projection_1
    self.projection_2=projection_2
    self.optimizer = optimizer
    self.device = device
    self.predictor = predictor
    '''

    def save_model(self, PATH):
        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'conv_trans_state_dict':self.conv_trans.state_dict(),
            'TP_1_state_dict':self.TP_1.state_dict(),
            'TP_2_state_dict':self.TP_2.state_dict(),
            'projection_1_state_dict':self.projection_1.state_dict(),
            'projection_2_state_dict':self.projection_2.state_dict(),
            'predictor_state_dict':self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
        
    def load_model(self, PATH):
        #print("not finish load")
        if(not os.path.exists(PATH)):
            print("model path not exist:",PATH)
            return
        stat_dict=torch.load(PATH)
        self.online_network.load_state_dict(stat_dict['online_network_state_dict'])
        self.target_network.load_state_dict(stat_dict['target_network_state_dict'])
        self.conv_trans.load_state_dict(stat_dict['conv_trans_state_dict'])
        self.TP_1.load_state_dict(stat_dict['TP_1_state_dict'])
        self.TP_2.load_state_dict(stat_dict['TP_2_state_dict'])
        self.projection_1.load_state_dict(stat_dict['projection_1_state_dict'])
        self.projection_2.load_state_dict(stat_dict['projection_2_state_dict'])
        self.predictor.load_state_dict(stat_dict['predictor_state_dict'])
        self.optimizer.load_state_dict(stat_dict['optimizer_state_dict'])
        
    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))
