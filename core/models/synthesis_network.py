import torch
import torch.nn as nn
from .modules.legacy import *

import pdb

class SynthesisBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            style_dim,
            kernel_size=3,
            blur_kernel=[1, 3, 3, 1]
    ):
        super().__init__()
        self.conv1 = StyledConv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=True,
            blur_kernel=blur_kernel
        )
        self.conv2 = StyledConv(
            out_channel,
            out_channel,
            kernel_size,
            style_dim,
            blur_kernel=blur_kernel
        )
        self.to_rgb = ToRGB(out_channel, style_dim)
        # self = SynthesisBlock(
        #   (conv1): StyledConv(
        #     (conv): ModulatedConv2d(512, 512, 3, upsample=True, downsample=False)
        #     (noise): NoiseInjection()
        #     (activate): FusedLeakyReLU()
        #   )
        #   (conv2): StyledConv(
        #     (conv): ModulatedConv2d(512, 512, 3, upsample=False, downsample=False)
        #     (noise): NoiseInjection()
        #     (activate): FusedLeakyReLU()
        #   )
        #   (to_rgb): ToRGB(
        #     (upsample): Upsample()
        #     (conv): ModulatedConv2d(512, 3, 1, upsample=False, downsample=False)
        #   )
        # )
        # in_channel = 512
        # out_channel = 512
        # style_dim = 512
        # kernel_size = 3
        # blur_kernel = [1, 3, 3, 1]



    def forward(self, hidden, style, noise=[None, None]):
        hidden = self.conv1(hidden, style, noise=noise[0])
        hidden = self.conv2(hidden, style, noise=noise[1])
        rgb = self.to_rgb(hidden, style)
        return hidden, rgb


class SynthesisNetwork(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            blur_kernel=[1, 3, 3, 1],
            channels = [512, 512, 512, 512, 512, 256, 128, 64, 32]
    ):
        super().__init__()
        # self = SynthesisNetwork(
        #   (input): ConstantInput()
        #   (conv1): StyledConv(
        #     (conv): ModulatedConv2d(512, 512, 3, upsample=False, downsample=False)
        #     (noise): NoiseInjection()
        #     (activate): FusedLeakyReLU()
        #   )
        #   (to_rgb1): ToRGB(
        #     (conv): ModulatedConv2d(512, 3, 1, upsample=False, downsample=False)
        #   )
        #   (layers): ModuleList()
        # )
        # size = 1024
        # style_dim = 512
        # blur_kernel = [1, 3, 3, 1]
        # channels = [512, 512, 512, 512, 512, 256, 128, 64, 32]

        self.size = size
        self.style_dim = style_dim

        self.input = ConstantInput(channels[0])
        self.conv1 = StyledConv(
            channels[0], channels[0], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(channels[0], style_dim, upsample=False)

        self.layers = nn.ModuleList()
        in_channel = channels[0]

        for out_channel in channels[1:]:
            self.layers.append(
                SynthesisBlock(
                    in_channel,
                    out_channel,
                    style_dim,
                    3,
                    blur_kernel=blur_kernel
                )
            )
            in_channel = out_channel

        self.upsample = Upsample(blur_kernel)
        # pdb.set_trace()

    def forward(self, style, noise=None):
        out = {"noise": [], "rgb": [], "img": None}
        pdb.set_trace()

        hidden = self.input(style)
        if noise is None:
            _noise = torch.randn(1, 1, hidden.size(-1), hidden.size(-1)).to(style.device)
        else:
            _noise = noise[0]
        out["noise"].append(_noise)
        hidden = self.conv1(hidden, style, noise=_noise)
        img = self.to_rgb1(hidden, style)
        out["rgb"].append(img)
        pdb.set_trace()

        for i, m in enumerate(self.layers):
            shape = [2, 1, 1, 2 ** (i + 3), 2 ** (i + 3)]
            if noise is None:
                _noise = torch.randn(*shape).to(style.device)
            else:
                _noise = noise[i + 1]
            out["noise"].append(_noise)
            hidden, rgb = m(hidden, style, _noise)
            out["rgb"].append(rgb)
            img = self.upsample(img) + rgb

        out["img"] = img
        pdb.set_trace()

        return out
