import torch
from torch import nn

# Embedding&Position
num_hiddens = 256

class EBD(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(EBD, self).__init__(*args, **kwargs)
        # 1000 means vocabulary size
        self.word_ebd = nn.Embedding(1000, num_hiddens)
        # 256 is the Position length
        self.pos_ebd = nn.Embedding(256, num_hiddens)
        self.pos_t = torch.arange(0, 256).reshape(1, 256)
    
    # X: (batch_size, length)
    def forward(self, X: torch.Tensor):
        return self.word_ebd(X) + self.pos_ebd(self.pos_t[:, :X.shape[-1]].to(X.device))
