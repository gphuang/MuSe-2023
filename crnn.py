# https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from config import device

class CRNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2, n_to_1=False):
        super(CRNN, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_layers = n_layers
        self.d_out = d_out
        self.n_directions = 2 if bi else 1
        self.n_to_1 = n_to_1
        
        self.c1 = nn.Conv1d(d_in, d_out, 1)
        self.c2 = nn.Conv1d(d_out, d_out,1)
        self.relu = nn.ReLU(inplace=True)
        self.gru = nn.GRU(d_out, d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)

    def forward(self, x, x_len):
        # Turn ( batch_size x seq_len x input_size) into (batch_size x input_size x seq_len) for CNN
        x = x.transpose(1, 2)
        
        # Run through Conv1d layers
        x = self.c1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.relu(x)
        # Turn (batch_size x hidden_size x seq_len) back into (batch_size x seq_len x hidden_size) for RNN
        x = x.transpose(1, 2)
        x = F.tanh(x)
        
        # x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        rnn_enc = self.gru(x)

        if self.n_to_1:
            # hiddenstates, h_n, only last layer
            return last_item_from_packed(rnn_enc[0], x_len)
            # batch_size = x.shape[0]
            # h_n = h_n.view(self.n_layers, self.n_directions, batch_size, self.d_out) # (NL, ND, BS, dim)
            # last_layer = h_n[-1].permute(1,0,2) # (BS, ND, dim)
            # x_out = last_layer.reshape(batch_size, self.n_directions * self.d_out) # (BS, ND*dim)

        else:
            x_out = rnn_enc[0]
            #? x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        return x_out
