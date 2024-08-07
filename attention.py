import torch
from torch import nn

# Embedding dimensions
num_hiddens = 256

def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, M: torch.Tensor):
    A = Q @ K.transpose(-1, -2) / (Q.shape[-1] ** 0.5)
    M = M.unsqueeze(1)
    A.masked_fill_(M == 0, -torch.tensor(float('inf')))
    A = torch.softmax(A, dim=-1)
    O = A @ V
    return O

def transpose_qkv(QKV: torch.Tensor):
    QKV = QKV.reshape(QKV.shape[0], QKV.shape[1], 4, QKV.shape[-1] // 4)
    QKV = QKV.transpose(-2, -3)
    return QKV

def transpose_o(O: torch.Tensor):
    O = O.transpose(-2, -3)
    O = O.reshape(O.shape[0], O.shape[1], -1)
    return O

# Multi-head attention
class Attention_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Attention_block, self).__init__(*args, **kwargs)
        self.Wq = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wk = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wv = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wo = nn.Linear(num_hiddens, num_hiddens, bias=False)
    
    def forward(self, X, M: torch.Tensor):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        Q, K, V = transpose_qkv(Q), transpose_qkv(K), transpose_qkv(V)
        O = attention(Q, K, V, M)
        O = transpose_o(O)
        O = self.Wo(O)
        return O
# Cross attention
class CrossAttention_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CrossAttention_block, self).__init__(*args, **kwargs)
        self.Wq = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wk = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wv = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wo = nn.Linear(num_hiddens, num_hiddens, bias=False)
    
    def forward(self, X, X_en, I_M):
        Q, K, V = self.Wq(X), self.Wk(X_en), self.Wv(X_en)
        Q, K, V = transpose_qkv(Q), transpose_qkv(K), transpose_qkv(V)
        O = attention(Q, K, V, I_M)
        O = transpose_o(O)
        O = self.Wo(O)
        return O
