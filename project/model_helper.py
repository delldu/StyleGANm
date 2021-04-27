"""Create model."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 04月 27日 星期二 16:03:15 CST
# ***
# ************************************************************************************/
#
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .modules import StyledConv2d,
#     ConstantInput, 
#     MultichannelImage, 
#     ModulatedDWConv2d, 
#     MobileSynthesisBlock, 
#     DWTInverse

import pdb

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5, onnx_trace=False):
    return F.leaky_relu(input + bias.view(1, input.size(1)), negative_slope=0.2) * scale

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, onnx_trace=False
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        # xxxx8888
        # if bias:
        #     self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        # else:
        #     self.bias = None

        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        # self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.onnx_trace = onnx_trace

    def forward(self, input):
        # xxxx8888
        # if self.activation:
        #     out = F.linear(input, self.weight * self.scale)
        #     out = fused_leaky_relu(out, self.bias * self.lr_mul, onnx_trace=self.onnx_trace)

        # else:
        #     out = F.linear(
        #         input, self.weight * self.scale, bias=self.bias * self.lr_mul
        #     )

        out = F.linear(input, self.weight * self.scale)
        out = fused_leaky_relu(out, self.bias * self.lr_mul, onnx_trace=self.onnx_trace)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

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

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        # create conv
        self.weight = nn.Parameter(
            torch.randn(channels_out, channels_in, kernel_size, kernel_size)
        )
        # create modulation network
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        # create demodulation parameters
        self.demodulate = demodulate
        if self.demodulate:
            self.register_buffer("style_inv", torch.randn(1, 1, channels_in, 1, 1))
        # some service staff
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2
        # self = ModulatedConv2d(
        #   (modulation): Linear(in_features=512, out_features=512, bias=True)
        # )
        # channels_in = 512
        # channels_out = 12
        # style_dim = 512
        # kernel_size = 1
        # demodulate = False

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight, padding=self.padding)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        w = self.weight.unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv * w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)

class StyledConv2d(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        style_dim,
        kernel_size,
        demodulate=True,
        conv_module=ModulatedConv2d
    ):
        super().__init__()

        self.conv = conv_module(
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=demodulate
        )

        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.act(out + self.bias)
        return out

class MultichannelImage(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=1
    ):
        super().__init__()
        self.conv = ModulatedConv2d(channels_in, channels_out, style_dim, kernel_size, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))

    def forward(self, hidden, style):
        out = self.conv(hidden, style)
        out = out + self.bias
        return out

class IDWTUpsaplme(nn.Module):
    def __init__(
            self,
            channels_in,
            style_dim,
    ):
        super().__init__()
        self.channels = channels_in // 4
        assert self.channels * 4 == channels_in
        # upsample
        self.idwt = DWTInverse(mode='zero', wave='db1')
        # modulation
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)

    def forward(self, x, style):
        b, _, h, w = x.size()
        x = self.modulation(style).view(b, -1, 1, 1) * x
        low = x[:, :self.channels]
        high = x[:, self.channels:]
        high = high.view(b, self.channels, 3, h, w)
        x = self.idwt((low, [high]))
        return x

class MobileSynthesisBlock(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=3,
            conv_module=ModulatedConv2d
    ):
        super().__init__()
        self.up = IDWTUpsaplme(channels_in, style_dim)
        self.conv1 = StyledConv2d(
            channels_in // 4,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.conv2 = StyledConv2d(
            channels_out,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.to_img = MultichannelImage(
            channels_in=channels_out,
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )

    def forward(self, hidden, style, noise=[None, None]):
        hidden = self.up(hidden, style)
        hidden = self.conv1(hidden, style, noise=noise[0])
        hidden = self.conv2(hidden, style, noise=noise[1])
        img = self.to_img(hidden, style)
        return hidden, img

class ModulatedDWConv2d(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        # create conv
        self.weight_dw = nn.Parameter(
            torch.randn(channels_in, 1, kernel_size, kernel_size)
        )
        self.weight_permute = nn.Parameter(
            torch.randn(channels_out, channels_in, 1, 1)
        )
        # create modulation network
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        # create demodulation parameters
        self.demodulate = demodulate
        if self.demodulate:
            self.register_buffer("style_inv", torch.randn(1, 1, channels_in, 1, 1))
        # some service staff
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight_dw, padding=self.padding, groups=x.size(1))
        x = F.conv2d(x, self.weight_permute)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        w = (self.weight_dw.transpose(0, 1) * self.weight_permute).unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv * w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)

class MobileSynthesisNetwork(nn.Module):
    def __init__(
            self,
            style_dim,
            channels = [512, 512, 512, 512, 512, 256, 128, 64]
    ):
        super().__init__()
        self.style_dim = style_dim

        self.input = ConstantInput(channels[0])
        self.conv1 = StyledConv2d(
            channels[0],
            channels[0],
            style_dim,
            kernel_size=3
        )
        self.to_img1 = MultichannelImage(
            channels_in=channels[0],
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )

        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock(
                    channels_in,
                    channels_out,
                    style_dim,
                    3,
                    conv_module=ModulatedDWConv2d
                )
            )
            channels_in = channels_out

        self.idwt = DWTInverse(mode="zero", wave="db1")
        # self = MobileSynthesisNetwork(
        #   (input): ConstantInput()
        #   (conv1): StyledConv2d(
        #     (conv): ModulatedConv2d(
        #       (modulation): Linear(in_features=512, out_features=512, bias=True)
        #     )
        #     (noise): NoiseInjection()
        #     (act): LeakyReLU(negative_slope=0.2)
        #   )
        #   (to_img1): MultichannelImage(
        #     (conv): ModulatedConv2d(
        #       (modulation): Linear(in_features=512, out_features=512, bias=True)
        #     )
        #   )
        #   (layers): ModuleList(
        #     (0): MobileSynthesisBlock(
        #       (up): IDWTUpsaplme(
        #         (idwt): DWTInverse()
        #         (modulation): Linear(in_features=512, out_features=512, bias=True)
        #       )
        #       (conv1): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=128, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (conv2): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=512, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (to_img): MultichannelImage(
        #         (conv): ModulatedConv2d(
        #           (modulation): Linear(in_features=512, out_features=512, bias=True)
        #         )
        #       )
        #     )
        #     (1): MobileSynthesisBlock(
        #       (up): IDWTUpsaplme(
        #         (idwt): DWTInverse()
        #         (modulation): Linear(in_features=512, out_features=512, bias=True)
        #       )
        #       (conv1): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=128, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (conv2): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=512, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (to_img): MultichannelImage(
        #         (conv): ModulatedConv2d(
        #           (modulation): Linear(in_features=512, out_features=512, bias=True)
        #         )
        #       )
        #     )
        #     (2): MobileSynthesisBlock(
        #       (up): IDWTUpsaplme(
        #         (idwt): DWTInverse()
        #         (modulation): Linear(in_features=512, out_features=512, bias=True)
        #       )
        #       (conv1): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=128, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (conv2): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=512, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (to_img): MultichannelImage(
        #         (conv): ModulatedConv2d(
        #           (modulation): Linear(in_features=512, out_features=512, bias=True)
        #         )
        #       )
        #     )
        #     (3): MobileSynthesisBlock(
        #       (up): IDWTUpsaplme(
        #         (idwt): DWTInverse()
        #         (modulation): Linear(in_features=512, out_features=512, bias=True)
        #       )
        #       (conv1): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=128, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (conv2): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=512, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (to_img): MultichannelImage(
        #         (conv): ModulatedConv2d(
        #           (modulation): Linear(in_features=512, out_features=512, bias=True)
        #         )
        #       )
        #     )
        #     (4): MobileSynthesisBlock(
        #       (up): IDWTUpsaplme(
        #         (idwt): DWTInverse()
        #         (modulation): Linear(in_features=512, out_features=512, bias=True)
        #       )
        #       (conv1): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=128, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (conv2): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=256, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (to_img): MultichannelImage(
        #         (conv): ModulatedConv2d(
        #           (modulation): Linear(in_features=512, out_features=256, bias=True)
        #         )
        #       )
        #     )
        #     (5): MobileSynthesisBlock(
        #       (up): IDWTUpsaplme(
        #         (idwt): DWTInverse()
        #         (modulation): Linear(in_features=512, out_features=256, bias=True)
        #       )
        #       (conv1): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=64, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (conv2): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=128, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (to_img): MultichannelImage(
        #         (conv): ModulatedConv2d(
        #           (modulation): Linear(in_features=512, out_features=128, bias=True)
        #         )
        #       )
        #     )
        #     (6): MobileSynthesisBlock(
        #       (up): IDWTUpsaplme(
        #         (idwt): DWTInverse()
        #         (modulation): Linear(in_features=512, out_features=128, bias=True)
        #       )
        #       (conv1): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=32, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (conv2): StyledConv2d(
        #         (conv): ModulatedDWConv2d(
        #           (modulation): Linear(in_features=512, out_features=64, bias=True)
        #         )
        #         (noise): NoiseInjection()
        #         (act): LeakyReLU(negative_slope=0.2)
        #       )
        #       (to_img): MultichannelImage(
        #         (conv): ModulatedConv2d(
        #           (modulation): Linear(in_features=512, out_features=64, bias=True)
        #         )
        #       )
        #     )
        #   )
        #   (idwt): DWTInverse()
        # )
        # style_dim = 512
        # channels = [512, 512, 512, 512, 512, 256, 128, 64]
        # (Pdb) 



    def forward(self, style, noise=None):
        out = {"noise": [], "freq": [], "img": None}
        # pdb.set_trace()
        # (Pdb) style.size()
        # torch.Size([1, 512]
        # (Pdb) pp noise -- None
        hidden = self.input(style)
        # (Pdb) hidden.size()
        # torch.Size([1, 512, 4, 4])

        if noise is None:
            _noise = torch.randn(1, 1, hidden.size(-1), hidden.size(-1)).to(style.device)
        else:
            _noise = noise[0]
        out["noise"].append(_noise)
        hidden = self.conv1(hidden, style, noise=_noise)
        img = self.to_img1(hidden, style)
        out["freq"].append(img)
        # pdb.set_trace()
        # (Pdb) img.size()
        # torch.Size([1, 12, 4, 4])
        for i, m in enumerate(self.layers):
            shape = [2, 1, 1, 2 ** (i + 3), 2 ** (i + 3)]
            if noise is None:
                _noise = torch.randn(*shape).to(style.device)
            else:
                _noise = noise[i + 1]
            out["noise"].append(_noise)
            hidden, freq = m(hidden, style, _noise)
            out["freq"].append(freq)

        out["img"] = self.dwt_to_img(out["freq"][-1])

        return out

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

