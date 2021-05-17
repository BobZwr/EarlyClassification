import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import numpy as np
from tqdm import tqdm

from model import MyDataset, MLSTM
from utils import *

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

def pair_rank(i, j):
    y_i = train_label[i][0]
    y_j = train_label[j][0]
    paired_label = [[y_i], [y_j]]
    paired_data = [train_data[i], train_data[j]]
    rank = 0.5
    if y_i > y_j or train_label[i][1] < train_label[j][1]:
        rank = 1.0
    if y_i < y_j or train_label[i][1] > train_label[j][1]:
        rank = 0.0

    return paired_data, paired_label, rank



def train(model, x, y, lr, epochs, batch_size):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-8)
    for epoch in tqdm(range(epochs)):
        x, y = shuffle(x, y)
        for i in tqdm(range(len(x))):
            current_data = []
            current_label = []
            current_rank = []
            for j in range(len(x)):
                if i == j:
                    continue
                paired_data, paired_label, rank = pair_rank(i, j)
                current_data.append(paired_data)
                current_label.append(paired_label)
                current_rank.append(rank)

            dataset = MyDataset(current_data, current_label, current_rank)
            dataloader = DataLoader(dataset, batch_size)
            for batch_idx, batch in tqdm(enumerate(dataloader), leave= False, desc='dataloader'):
                input_x, input_y, rank = tuple(t.to(device) for t in batch)
                input_x = input_x[0]
                loss, _, _, _ = model(input_x, input_y, rank)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        scheduler.step(epoch)
    torch.save(model, 'testmodel.pkl')
    return

if __name__ == '__main__':
    # load data
    train_data = np.load('data/train_data.npy', allow_pickle=True)
    train_data = np.expand_dims(train_data, -1)
    train_label = np.load('data/train_label.npy', allow_pickle=True)
    test_data = np.load('data/test_data.npy', allow_pickle=True)
    test_label = np.load('data/test_label.npy',allow_pickle=True)

    #set parameters
    lr = 1e-4
    epochs = 10
    batch_size = 1

    # build model
    input_dim = 1
    output_dim = 1
    model =MLSTM(input_dim=input_dim, output_dim = output_dim,hidden_dim=256, fc_dim=128, keep_prob=0.25).to(dtype=float)
    train(model=model, x = train_data, y = train_label, lr=lr, epochs=epochs, batch_size = batch_size)