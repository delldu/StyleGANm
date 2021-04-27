import torch
import torch.nn as nn
from .modules import StyledConv2d, \
    ConstantInput, \
    MultichannelImage, \
    ModulatedDWConv2d, \
    MobileSynthesisBlock, \
    DWTInverse

import pdb

# xxxx8888
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
