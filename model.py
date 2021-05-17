import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor

from torch.utils.data import Dataset
import  numpy as np

class MyDataset(Dataset):
    def __init__(self, data, label, rank):
        self.data = data
        self.label = label
        self.rank = rank

    def __getitem__(self, index):
        return (torch.tensor(self.data[index]), torch.tensor(self.label[index]), torch.tensor(self.rank[index]))

    def __len__(self):
        return len(self.data)

class MLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, keep_prob):
        super(MLSTM, self).__init__()
        # [batchsize, seq length, input dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim
        self.keep_prob = keep_prob
        self.LSTM = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim)
        self.W_decomp = Parameter(Tensor(self.hidden_dim, self.hidden_dim))
        self.b_decomp = Parameter(Tensor(self.hidden_dim))
        self.W_fc1 = Parameter(Tensor(self.hidden_dim, self.fc_dim))
        self.b_fc1 = Parameter(Tensor(self.fc_dim))
        self.W_fc2 = Parameter(Tensor(self.hidden_dim, self.fc_dim))
        self.b_fc2 = Parameter(Tensor(self.fc_dim))
        self.W_score1 = Parameter(Tensor(self.fc_dim, self.output_dim))
        self.b_score1 = Parameter(Tensor(self.output_dim))
        self.W_score2 = Parameter(Tensor(self.fc_dim, self.output_dim))
        self.b_score2 = Parameter(Tensor(self.output_dim))

    def get_output1(self, state):
        output = F.relu(torch.matmul(state, self.W_fc1) + self.b_fc1)
        output = F.dropout(output, self.keep_prob)
        output = torch.matmul(output, self.W_score1) + self.b_score1
        return output

    def get_output2(self, state):
        output = F.relu(torch.matmul(state, self.W_fc2) + self.b_fc2)
        output = F.dropout(output, self.keep_prob)
        output = torch.matmul(output, self.W_score2) + self.b_score2
        return output

    def get_outputs1(self, x):
        all_states, _ = self.LSTM(x)
        tmp = list(map(self.get_output1, all_states))
        all_outputs = tmp[0]
        return all_outputs[-1, :, :]

    def get_outputs2(self, x):
        all_states, _, _ = self.LSTM(x)
        all_outputs = torch.tensor(list(map(self.get_output2, all_states)))
        return all_outputs[-1, :, :]

    def get_outputs(self, x):
        output1 = self.get_output1(x)
        output2 = self.get_output2(x)
        output = torch.stack((output1, output2), 1)
        return output

    def loss_func(self, x, label, rank):
        # [seq_length x batch_size x input_dim)
        output1 = self.get_outputs1(x)
        output2 = self.get_outputs2(x)
        pairwise_ranking1 = F.cross_entropy(output1, label, reduction='mean') + F.cross_entropy(output1[0] - output1[1], rank)
        pairwise_ranking2 = F.cross_entropy(output2, label, reduction='mean') + F.cross_entropy(output2[0] - output2[1], rank)

        pairwise_ranking = pairwise_ranking2 + pairwise_ranking1

        output = self.get_outputs(x)
        RCE = F.cross_entropy(output, label, reduction='mean')

        loss = RCE + pairwise_ranking

        y_pred = F.softmax(output)
        y = label

        return loss, y_pred, y, output

    def forward(self, x, label, rank):
        loss, y_pred, y, output = self.loss_func(x, label, rank)

        return loss, y_pred, y, output