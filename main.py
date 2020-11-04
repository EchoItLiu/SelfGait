import os

import torch
import yaml
from torchvision import datasets
import torch.nn as nn
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
#from trainer_v2 import BYOLTrainer
from trainer import BYOLTrainer
from models.network import SetNet
from models.network.basic_blocks import MCM_NOTP,TP,TP_FULL,predictor
from models.utils.data_set import DataSet
from models.utils.data_loader import load_data
print(torch.__version__)
torch.manual_seed(0)


def main():
    config = yaml.load(open("./config/config.yaml", "r"))
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    #data_transform = get_simclr_data_transforms(**config['data_transforms'])

    train_source,ft_source,test_source = load_data(config)
    '''
    train_source,_,_ = load_data(config)
    config_test=config
    config_test['dataset_path']="/home/yqliu/Dataset/CASIA-B/silhouettes"
    config_test['dataset']='CASIA-B'
    ft_source,_,test_source = load_data(config_test)
    print(config_test['dataset'])
    '''
    #print(type(train_source),type(test_source))
    print(train_source.__len__(),ft_source.__len__(),test_source.__len__())
    train=True
    test=False
    Train_Set=Test_Set=None
    if train:
        print("Loading training data...")
        #Train_Set = train_source.load_all_data()
    if test:
        print("Loading test data...")
        #Test_Set = test_source.load_all_data()
    #print(type(Train_Set),type(Test_Set))

    # online network
    #online_network = ResNet18(**config['network']).to(device)
    online_network = SetNet(config['network']['hidden_dim'])
    online_network=nn.DataParallel(online_network).cuda()
    pretrained_folder = config['network']['fine_tune_from']
    
    
    conv_trans=MCM_NOTP(in_channels=128, out_channels=128, p=31, div=4)
    conv_trans=nn.DataParallel(conv_trans).cuda()
    #conv_trans=MCM_NOTP(in_channels=config['network']['hidden_dim'], out_channels=config['network']['hidden_dim'], p=16, div=4).to(device)
    TP_1=TP_FULL(hidden_dim=config['network']['hidden_dim'])
    TP_1=nn.DataParallel(TP_1).cuda()
    
    # load pre-trained model if defined
    if pretrained_folder!='None':
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    projection_1 = MLPHead(in_channels=config['network']['hidden_dim'],**config['network']['projection_head'])
    projection_1=nn.DataParallel(projection_1).cuda()
    #projection = MLPHead(in_channels=128,**config['network']['projection_head']).to(device)
    #predictor_1=predictor(hidden_dim=config['network']['hidden_dim']).cuda()
    predictor = MLPHead(in_channels=config['network']['hidden_dim'],**config['network']['projection_head'])
    predictor=nn.DataParallel(predictor).cuda()
    
    # target encoder
    #target_network = ResNet18(**config['network']).to(device)
    target_network = SetNet(config['network']['hidden_dim'])
    target_network=nn.DataParallel(target_network).cuda()
    TP_2=TP_FULL(hidden_dim=config['network']['hidden_dim'])
    TP_2=nn.DataParallel(TP_2).cuda()
    #predictor_2=predictor(hidden_dim=config['network']['hidden_dim']).cuda()
    projection_2 = MLPHead(in_channels=config['network']['hidden_dim'],**config['network']['projection_head'])
    projection_2=nn.DataParallel(projection_2).cuda()
    
    
    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(conv_trans.parameters()) + list(TP_1.parameters()) + list(projection_1.parameters()) + list(predictor.parameters()),**config['optimizer']['params'])
    #print(type(projection_1),type(projection_2))
    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          conv_trans=conv_trans,
                          TP_1=TP_1,
                          TP_2=TP_2,
                          projection_1=projection_1,
                          projection_2=projection_2,
                          predictor=predictor,
                          optimizer=optimizer,
                          device=device,
                          config=config,
                          train_source=train_source,
                          test_source=test_source,
                          **config['trainer'])

    #trainer.train_ft(train_source,maxiter=10000)
    #trainer.finetune(ft_source,maxiter=80000)
    trainer.load_model("/home/yqliu/PJs/BYOL_MCM/PyTorch-BYOL/runs/Oct20_11-37-30_123.pami.group/checkpoints/model_9000.pth")
    trainer.finetune(train_source,maxiter=80000)
    #trainer.test(test_source)


if __name__ == '__main__':
    main()
