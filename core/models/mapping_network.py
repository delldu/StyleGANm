import torch
import torch.nn as nn
from .modules.legacy import PixelNorm, EqualLinear
import pdb

class MappingNetwork(nn.Module):
    def __init__(
            self,
            style_dim,
            n_layers,
            lr_mlp=0.01
    ):
        super().__init__()
        self.style_dim = style_dim
        layers = [PixelNorm()]
        for i in range(n_layers):
            layers.append(
                EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu")
            )
        self.layers = nn.Sequential(*layers)
        # pdb.set_trace()
        # (Pdb) a
        # self = MappingNetwork(
        #   (layers): Sequential(
        #     (0): PixelNorm()
        #     (1): EqualLinear(512, 512)
        #     (2): EqualLinear(512, 512)
        #     (3): EqualLinear(512, 512)
        #     (4): EqualLinear(512, 512)
        #     (5): EqualLinear(512, 512)
        #     (6): EqualLinear(512, 512)
        #     (7): EqualLinear(512, 512)
        #     (8): EqualLinear(512, 512)
        #   )
        # )
        # style_dim = 512
        # n_layers = 8
        # lr_mlp = 0.01


    def forward(self, x):
        return self.layers(x)
