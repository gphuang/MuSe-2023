import torch
import torch.nn as nn
from config import ACTIVATION_FUNCTIONS
from rnn import RNN

class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y

class Model(nn.Module):
    """MUSE 2023 default RNN model."""
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params

        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)

        self.rnn = RNN(params.model_dim, params.model_dim, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=params.rnn_dropout, n_to_1=params.n_to_1)


        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        x = self.inp(x)
        x = self.rnn(x, x_len) # encoder
        y = self.out(x)
        activation = self.final_activation(y)
        return activation, x

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1

class CRNNModel(nn.Module):
    def __init__(self, params):
        super(CRNNModel, self).__init__()
        self.params = params

        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)
        self.rnn = RNN(params.model_dim, 
                                params.model_dim, 
                                n_layers=params.rnn_n_layers, 
                                bi=params.rnn_bi,
                                dropout=params.rnn_dropout, 
                                n_to_1=params.n_to_1)

        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()
        
        self.conv = nn.Conv1d(params.d_in, params.model_dim, 1)
        self.maxpool = nn.MaxPool1d(2)
        self.avgpool = nn.AvgPool1d(2)
        self.relu = nn.ReLU()

    def forward(self, x, x_len):
        """
        x: ([batch_size, win_len (time_step), 1])
        """        
        # eval does not pass self.params?
        x = x.transpose(1, 2)
        x = self.conv(x)                  # ([batch_size, hidden_size, win_len])
        x1 = self.maxpool(x)
        x2 = self.avgpool(x)
        x =  torch.cat((x1, x2), 2)       # ([batch_size, hidden_size, win_len])
        x = x.transpose(1, 2)             # ([batch_size, win_len, hidden_size])
        x = self.rnn(x, x_len)                # ([batch_size, win_len, hidden_size])
        x = self.relu(x)
        y = self.out(x)                       # ([batch_size, win_len, n_targets])
        activation = self.final_activation(y) # ([batch_size, win_len, n_targets]) # preds
        return activation, x

    def set_n_to_1(self, n_to_1):
        self.rnn.n_to_1 = n_to_1


