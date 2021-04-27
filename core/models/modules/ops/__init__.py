import torch
from .fused_act import FusedLeakyReLU, fused_leaky_relu
from .upfirdn2d import upfirdn2d
# xxxx8888
# if torch.cuda.is_available():
#     from .fused_act_cuda import FusedLeakyReLUFunction
#     from .upfirdn2d_cuda import UpFirDn2d
