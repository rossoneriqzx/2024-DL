import torch
from torch import nn
from .attention import Attention_block, CrossAttention_block
from .embedding import EBD
from .encoder import AddNorm, Pos_FFN

num_hiddens = 256

class Decoder_blk(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Decoder_blk, self).__init__(*args, **kwargs)
        self.attention = Attention_block()
        self.add_norm_1 = AddNorm()
        self.cross_attention = CrossAttention_block()
        self.add_norm_2 = AddNorm()
        self.FFN = Pos_FFN()
        self.add_norm_3 = AddNorm()
        mask_matrix = torch.ones(256, 256)
        self.tril_mask = torch.tril(mask_matrix).unsqueeze(0)
        
    def forward(self, X_t, O_m, X_en, I_m):
        O_m = O_m.unsqueeze(-2)
        I_m = I_m.unsqueeze(-2)
        X_1 = self.attention(X_t, O_m * self.tril_mask[:, :O_m.shape[-1], :O_m.shape[-1]].to(X_t.device))
        X_t = self.add_norm_1(X_t, X_1)
        X_1 = self.cross_attention(X_t, X_en, I_m)
        X_t = self.add_norm_2(X_t, X_1)
        X_1 = self.FFN(X_t)
        X_t = self.add_norm_3(X_t, X_1)
        return X_t

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Decoder, self).__init__(*args, **kwargs)
        self.ebd = EBD()
        self.decoder_blks = nn.Sequential()
        for _ in range(4):
            self.decoder_blks.append(Decoder_blk())
        self.dense = nn.Linear(num_hiddens, 1000, bias=False)
    
    def forward(self, X_t, O_m, X_en, I_m):
        X_t = self.ebd(X_t)
        for layer in self.decoder_blks:
            X_t = layer(X_t, O_m, X_en, I_m)
        X_t = self.dense(X_t)
        return X_t
