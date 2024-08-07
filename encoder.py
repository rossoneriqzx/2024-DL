import torch
from torch import nn
from .attention import Attention_block
from .embedding import EBD

num_hiddens = 256

# Add(residual) & Norm 
class AddNorm(nn.Module):
    def __init__(self):
        super(AddNorm, self).__init__()
        self.add_norm = nn.LayerNorm(num_hiddens)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, X, X1):
        X1 = self.add_norm(X1)
        X = X + X1
        X = self.dropout(X)
        return X
    
# Feedforward Networks: 
# Enhancing the Nonlinear Expressive Power of Models, 
# Providing Local Processing and Feature Extraction, and Increasing the Depth and Capacity of Models.
class Pos_FFN(nn.Module):
    def __init__(self):
        super(Pos_FFN, self).__init__()
        self.lin1 = nn.Linear(num_hiddens, 512, bias=True)
        self.norm1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(512, num_hiddens, bias=True)
        self.norm2 = nn.BatchNorm1d(num_hiddens)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, X):
        X = self.lin1(X)
        X = self.norm1(X)
        X = self.relu1(X)
        X = self.dropout(out)
        X = self.lin2(X)
        X = self.norm2(X)
        X = self.relu2(X)
        X = self.dropout(out)
        return X

class Encoder_block(nn.Module):
    def __init__(self):
        super(Encoder_block, self).__init__()
        self.attention = Attention_block()
        self.addnorm1 = AddNorm()
        self.FFN = Pos_FFN()
        self.addnorm2 = AddNorm()
    
    def forward(self, X, I_m):
        I_m = I_m.unsqueeze(-2)
        X_1 = self.attention(X, I_m)
        X = self.addnorm1(X, X_1)
        X_1 = self.FFN(X)
        X = self.addnorm2(X, X_1)
        return X

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.ebd = EBD()
        self.encoder_blks = nn.Sequential()
        for _ in range(4):
            self.encoder_blks.append(Encoder_block())
    
    def forward(self, X, I_m):
        X = self.ebd(X)
        for encoder_blk in self.encoder_blks:
            X = encoder_blk(X, I_m)
        return X
