import glob
import numpy as np
import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class SceneflowDataset(Dataset):
    def __init__(self, npoints=8192, root='datasets/data_processed_maxcut_35_20k_2k_8192', train=True, cache=None):
        self.npoints = npoints
        self.train = train
        self.root = root
        if self.train:
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))

        if cache is None:
            self.cache = {}
        else:
            self.cache = cache

        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['points1'].astype('float32')
                pos2 = data['points2'].astype('float32')
                color1 = data['color1'].astype('float32') / 255
                color2 = data['color2'].astype('float32') / 255
                flow = data['flow'].astype('float32')
                mask1 = data['valid_mask1']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

        if self.train:
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            color1 = color1[sample_idx1, :]
            color2 = color2[sample_idx2, :]
            flow = flow[sample_idx1, :]
            mask1 = mask1[sample_idx1]
        else:
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            color1 = color1[:self.npoints, :]
            color2 = color2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            mask1 = mask1[:self.npoints]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        pos1 = torch.from_numpy(pos1)
        pos2 = torch.from_numpy(pos2)
        color1 = torch.from_numpy(color1)
        color2 = torch.from_numpy(color2)
        flow = torch.from_numpy(flow)
        mask1 = torch.from_numpy(mask1).unsqueeze(-1).type(torch.float32)

        return pos1, pos2, color1, color2, flow, mask1

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    train_set = SceneflowDataset(train=True)
    points1, points2, color1, color2, flow, mask1 = train_set[5]

    print(points1.shape, points1.dtype)
    print(points2.shape, points2.dtype)
    print(color1.shape, color1.dtype)
    print(color2.shape, color2.dtype)
    print(flow.shape, flow.dtype)
    print(mask1.shape, mask1.dtype)

