import arff2pandas as a2p
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from CONFIG import *

class DF2Torchloader(torch.utils.data.Dataset):
    """
    Task가 train이라면 데이터만 출력되고, val이라면 데이터와 라벨이 출력된다
    라벨은 0 이 정상, 1이 비정상이다
    * 입력파라미터
    data = pandas dataframe
    devoce = 'cpu' or 'cuda'
    task = 'train' or 'val'
    """

    def __init__(self, data, device, task):
        super(DF2Torchloader, self).__init__()
        
        assert task in ['train', 'val'], f'Task must be train or val :: Input = {task}'
        
        self.data = data
        self.device = device
        self.task = task

    def __getitem__(self, index):
        if self.task == 'train':
            return torch.FloatTensor(self.data.iloc[index].drop('target')).to(self.device)
        elif self.task == 'val':
            # 라벨 생성
            if self.data.iloc[index]['target'] == '1': # 정상이라면
                label = torch.IntTensor([0]).to(self.device)
            else: # 비정상이라면
                label = torch.IntTensor([1]).to(self.device)
            return torch.FloatTensor(self.data.iloc[index].drop('target')).to(self.device), label

    def __len__(self):
        return len(self.data)

def load_dataloader(device):
    """
    return train, validation, test, abnormal
    """
    df_train = a2p.load('ECG5000/ECG5000_TEST.arff')
    df_test = a2p.load('ECG5000/ECG5000_TRAIN.arff')

    normal_df = df_train[df_train['target'] == '1']
    abnormal_df = df_train[df_train['target'] != '1']

    train_normal, validation_normal = train_test_split(normal_df, test_size= RATIO, random_state=SEED)

    train_generator = DF2Torchloader(train_normal, device, 'train')
    validation_generator = DF2Torchloader(validation_normal, device, 'train')
    test_generator = DF2Torchloader(df_test, device, 'val')
    abnormal_generator = DF2Torchloader(abnormal_df, device, 'train')

    train_loader = torch.utils.data.DataLoader(train_generator, batch_size=BATCH, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_generator, batch_size=BATCH, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_generator, batch_size=BATCH, shuffle=True)
    abnormal_loader = torch.utils.data.DataLoader(abnormal_generator, batch_size = BATCH, shuffle=True)

    return train_loader, validation_loader, test_loader, abnormal_loader