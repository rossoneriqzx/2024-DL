import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Transformer, self).__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, X_s, I_m, X_t, O_m):
        X_en = self.encoder(X_s, I_m)
        X = self.decoder(X_t, O_m, X_en, I_m)
        return X
