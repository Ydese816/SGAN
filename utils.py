import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
from torch.utils.data import Dataset
from query_probability import load_ucr
import warnings

warnings.filterwarnings('ignore')


class UcrDataset(Dataset):
    def __init__(self, txt_file, channel_last=True, normalize=True):
        '''
        :param txt_file: path of file
        :param channel_last
        '''
        # self.data = np.loadtxt(txt_file)
        self.data = load_ucr(txt_file, normalize)
        self.channel_last = channel_last
        self.box_max = self.data.max()
        self.box_min = self.data.min()
        if self.channel_last:
            self.data = np.reshape(self.data, [self.data.shape[0], self.data.shape[1], 1])
        else:
            self.data = np.reshape(self.data, [self.data.shape[0], 1, self.data.shape[1]])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        if not self.channel_last:
            return self.data[idx, :, 1:], self.data[idx, :, 0]
        else:
            return self.data[idx, 1:, :], self.data[idx, 0, :]

    def get_seq_len(self):
        if self.channel_last:
            return self.data.shape[1] - 1
        else:
            return self.data.shape[2] - 1
    def get_max_min(self):
        return self.box_max,self.box_min
    def getclass(self):
        test_data = torch.from_numpy(self.data)
        Y = test_data[:, 0]
        return torch.unique(Y).shape[0]


class AdvDataset(Dataset):
    def __init__(self, txt_file):
        self.data = np.loadtxt(txt_file)
        self.data = np.reshape(self.data, [self.data.shape[0], self.data.shape[1], 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, 2:, :], self.data[idx, 1, :],self.data[idx, 0, :]

    def get_seq_len(self):
        return self.data.shape[1] - 2


def UCR_dataloader(dataset, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return data_loader


def pearson_correlation(x, y):
    x = torch.tensor(x)
    y = torch.tensor(y)
    x = x.clone().detach()
    y = y.clone().detach()
    if len(x) != len(y):
        raise ValueError("not equal")

    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    covariance = torch.sum((x - mean_x) * (y - mean_y))

    std_x = torch.sqrt(torch.sum((x - mean_x)**2))
    std_y = torch.sqrt(torch.sum((y - mean_y)**2))

    if std_x == 0 or std_y == 0:
        return 0  
    else:
        pearson_coefficient = covariance / (std_x * std_y)
        return pearson_coefficient


def smoothness_v2(sequence):
    sequence = torch.tensor(sequence)
    sequence = sequence.clone().detach()
    second_order_diff = sequence[2:] - 2 * sequence[1:-1] + sequence[:-2]

    smoothness_value = torch.norm(second_order_diff, p=2)

    smoothness_value = torch.sigmoid(smoothness_value)
    return smoothness_value.item()
