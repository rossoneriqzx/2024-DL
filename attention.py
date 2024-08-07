import torch
from torch import nn

# Embedding dimensions
num_hiddens = 256

def attention(Q, K, V, M): # all are tensor
    A = Q @ K.transpose(-1, -2) / (Q.shape[-1] ** 0.5)
    M = M.unsqueeze(1)
    A.masked_fill_(M == 0, -torch.tensor(float('inf')))
    A = torch.softmax(A, dim=-1)
    O = A @ V
    return O

def transpose_1(A):
    A = A.reshape(A.shape[0], A.shape[1], 4, A.shape[-1] // 4)
    A = A.transpose(-2, -3)
    return A

def transpose_2(O):
    O = O.transpose(-2, -3)
    O = O.reshape(O.shape[0], O.shape[1], -1)
    return O

# Multi-head attention
class Attention_block(nn.Module):
    def __init__(self):
        super(Attention_block, self).__init__()
        self.Wq = nn.Linear(num_hiddens, num_hiddens, bias=True)
        self.Wk = nn.Linear(num_hiddens, num_hiddens, bias=True)
        self.Wv = nn.Linear(num_hiddens, num_hiddens, bias=True)
        self.Wo = nn.Linear(num_hiddens, num_hiddens, bias=True) # Bias may be False based on 
    
    def forward(self, X, M: torch.Tensor):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        Q, K, V = transpose_1(Q), transpose_1(K), transpose_1(V)
        O = attention(Q, K, V, M)
        O = transpose_2(O)
        O = self.Wo(O)
        return O
# Cross attention
class CrossAttention_block(nn.Module):
    def __init__(self):
        super(CrossAttention_block, self).__init__()
        self.Wq = nn.Linear(num_hiddens, num_hiddens, bias=True)
        self.Wk = nn.Linear(num_hiddens, num_hiddens, bias=True)
        self.Wv = nn.Linear(num_hiddens, num_hiddens, bias=True)
        self.Wo = nn.Linear(num_hiddens, num_hiddens, bias=True)
    
    def forward(self, X, X_en, I_M):
        Q, K, V = self.Wq(X), self.Wk(X_en), self.Wv(X_en)
        Q, K, V = transpose_1(Q), transpose_1(K), transpose_1(V)
        O = attention(Q, K, V, I_M)
        O = transpose_2(O)
        O = self.Wo(O)
        return O
