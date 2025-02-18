import torch
import torch.nn as nn
from pytorch_wavelets.dwt.lowlevel import *
import pdb

def _SFB2D(low, highs, g0_row, g1_row, g0_col, g1_col):
    # low size torch.Size([1, 32, 256, 256]) highs size: torch.Size([1, 32, 3, 256, 256]

    lh, hl, hh = torch.unbind(highs, dim=2)
    lo = sfb1d(low, lh, g0_col, g1_col, mode='zero', dim=2)
    hi = sfb1d(hl, hh, g0_col, g1_col, mode='zero', dim=2)
    y = sfb1d(lo, hi, g0_row, g1_row, mode='zero', dim=3)

    return y


class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """
    def __init__(self, wave='db1', mode='zero', onnx_trace=False):
        super().__init__()

        if isinstance(wave, str):
            print("CheckPoint 1 ----------------------------")

            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            # ==>
            print("CheckPoint 2 ----------------------------")

            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            # (Pdb) g0_col, g1_col
            # ([0.7071067811865476, 0.7071067811865476], 
            #     [0.7071067811865476, -0.7071067811865476])
        else:
            print("CheckPoint 3 ----------------------------")

            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        filts = prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        # (Pdb) filts
        # (tensor([[[[0.7071],
        #           [0.7071]]]]), tensor([[[[ 0.7071],
        #           [-0.7071]]]]), tensor([[[[0.7071, 0.7071]]]]), tensor([[[[ 0.7071, -0.7071]]]]))
        
        self.register_buffer('g0_col', filts[0])
        self.register_buffer('g1_col', filts[1])
        self.register_buffer('g0_row', filts[2])
        self.register_buffer('g1_row', filts[3])
        self.mode = mode
        self.onnx_trace = onnx_trace

        # self = DWTInverse()
        # wave = pywt._extensions._pywt.Wavelet(name='db1', 
        # filter_bank=([0.7071067811865476, 0.7071067811865476], 
        #     [-0.7071067811865476, 0.7071067811865476], 
        #     [0.7071067811865476, 0.7071067811865476], 
        #     [0.7071067811865476, -0.7071067811865476]))
        # mode = 'zero'
        # onnx_trace = False

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        # (Pdb) len(coeffs), coeffs[0].size(), type(coeffs[1]), len(coeffs[1]), coeffs[1][0].size()
        # (2, torch.Size([1, 128, 4, 4]), <class 'list'>, 1, torch.Size([1, 128, 3, 4, 4]))

        yl, yh = coeffs
        ll = yl
        mode = mode_to_int(self.mode)
        # self.mode -- 'zero'
        # pdb.set_trace()

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            # if not self.onnx_trace:
            #     print("CheckPoint 5 ----------------------------")
            #     ll = SFB2D.apply(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
            # else:
            #     print("CheckPoint 6 ----------------------------")
            #     ll = _SFB2D(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
            # xxxx8888
            ll = _SFB2D(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row)

        # (Pdb) pp ll.size() -- torch.Size([1, 128, 8, 8])

        return ll
