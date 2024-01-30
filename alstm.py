# https://github.com/0aqz0/pytorch-attention-mechanism/blob/master/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from attention import TemporalAttn
from arnn import last_item_from_packed

"""
LSTM with attention
"""
class AttnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(AttnLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)
        self.attn = TemporalAttn(hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def X_forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        x, weights = self.attn(x)
        x = self.fc(x)
        return x, weights

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        x, (h_n, c_n) = self.lstm(x_packed)
        x, weights = self.attn(x)
        x = self.fc(x)

        if self.n_to_1:
            return last_item_from_packed(x[0], x_len)
        else:
            x_out = x[0]
            x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        return x_out

# Test
if __name__ == '__main__':
    model = AttnLSTM(input_size=1, hidden_size=128, num_layers=1)
    x = torch.randn(16, 20, 1)
    print(model(x))