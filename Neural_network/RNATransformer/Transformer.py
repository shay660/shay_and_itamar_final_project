import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional Encoding class to provide positional information to the input embeddings.

    Args:
        d_model (int): The dimension of the model.
        max_len (int): The maximum length of the input sequences.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerRegressor(nn.Module):
    """
    Transformer-based Regressor model for predicting mRNA degradation rate
    according to it sequence.

    Args:
        seq_len (int): The length of the input sequences.
        n_tokens (int): The number of input tokens.
        d_model (int): The dimension of the model.
        n_heads (int): The number of attention heads in the transformer.
        n_layers (int): The number of transformer encoder layers.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float, optional): The dropout value. Default is 0.2.
        layer_norm_eps (float, optional): The epsilon value for layer normalization. Default is 1e-5.
    """
    def __init__(self, seq_len, d_model, n_heads, n_layers,
                 dim_feedforward, n_tokens=4, dropout=0.2, layer_norm_eps=1e-5):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Linear(n_tokens, d_model) # maybe need to increase
        # self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        # nn.init.xavier_uniform_(self.positional_encoding)
        self.positional_encoding = PositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   batch_first=True,
                                                   dropout=dropout,
                                                   layer_norm_eps=layer_norm_eps, )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=n_layers)
        # self.batch_norm = nn.BatchNorm1d(d_model * seq_len)
        # TODO add activation function
        self.regressor = nn.Linear(seq_len * d_model, 1)

    def forward(self, x):
        """
        Forward pass for the transformer regressor.

        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_len, n_tokens).

        Returns:
            Tensor: The output tensor of shape (batch_size, 1).
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.flatten(start_dim=1)
        # x = self.batch_norm(x)
        x = self.regressor(x)
        return x


class TransformerWithBatchNorm(TransformerRegressor):
    def __init__(self, seq_len=110, n_tokens=4, d_model=128, n_heads=4,
                 n_layers=8, dim_feedforward=512, dropout=0.2, layer_norm_eps=1e-5):
        super().__init__(seq_len, d_model, n_heads, n_layers,
                         dim_feedforward,n_tokens,  dropout, layer_norm_eps)
        self.batch_norm = nn.BatchNorm1d(seq_len*d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.flatten(start_dim=1)
        x = self.batch_norm(x)
        x = self.regressor(x)
        return x


