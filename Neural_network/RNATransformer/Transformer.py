import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :] * 0.1
        x = self.dropout(x)
        return x

class FC(nn.Module):
        def __init__(self, input_size):
          super(FC, self).__init__()
          self.activation= nn.ReLU()
          self.fc1 = nn.Linear(input_size, input_size // 2)
          self.bn1 = nn.BatchNorm1d(input_size // 2)
          self.fc2 = nn.Linear(input_size // 2, input_size // 4) # Use floor division to ensure integer output
          self.bn2 = nn.BatchNorm1d(input_size // 4)
          self.fc3 = nn.Linear(input_size // 4, 1)

          # self.optim = torch.optim.Adam(self.parameters(), lr=0.01)
          # self.criterion = nn.CrossEntropyLoss()

        def forward(self, x):
          x = self.bn1(self.fc1(x))
          x = self.activation(x)
          x = self.bn2(self.fc2(x))
          x = self.activation(x)
          return self.fc3(x)



class TransformerRegressor(nn.Module):
    def __init__(self, seq_len, n_tokens, d_model, n_heads, n_layers, dim_feedforward,
                 dropout=0.2, layer_norm_eps=1e-5):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True,
                                                   dropout=dropout, layer_norm_eps=layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.bn = nn.BatchNorm1d(seq_len * d_model)
        self.regressor = FC(input_size = seq_len * d_model)

    def forward(self, x):
        x = self.embedding(x)

        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.flatten(start_dim=1)
        x = self.bn(x)
        x = self.regressor(x)
        return x

    def encode(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.flatten(start_dim=1)
        return x