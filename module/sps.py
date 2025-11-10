import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode


class SepConvMix(nn.Module):
    r"""
    Spatial Temporal fusion spiking separable convolution
    """

    def __init__(
        self,
        in_channels=1,
        embed_dims=128,
        img_size_h=32,
        img_size_w=512,
        bias=False,
        patch_size=4
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch = img_size_h // self.patch_size

        self.pe_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )

        self.pe_bn = nn.BatchNorm2d(embed_dims)
        self.pe_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')
        self.dwconv1 = nn.Conv2d(
            self.patch,
            self.patch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.patch,
            bias=bias,
        )  # depthwise conv
        self.bn1 = nn.BatchNorm2d(self.patch)
        self.lif1 = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.pwconv1 = nn.Conv2d(self.patch, self.patch, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(self.patch)
        self.lif2 = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.dwconv2 = nn.Conv2d(
            self.patch,
            self.patch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.patch,
            bias=bias,
        )  # depthwise conv
        self.bn3 = nn.BatchNorm2d(self.patch)
        self.lif3 = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.pwconv2 = nn.Conv2d(self.patch, self.patch, kernel_size=1, stride=1, bias=bias)
        self.bn4 = nn.BatchNorm2d(self.patch)
        self.lif4 = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        # RPE
        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x, hook=None):

        T, B, C, H, W = x.shape
        # PE
        H = H //self.patch_size
        W = W // self.patch_size
        x = x.flatten(0, 1)
        x = self.pe_conv(x)
        x = self.pe_bn(x).reshape(T, B, -1, H, W).contiguous()

        x_res = x.flatten(0, 1).permute(0, 2, 1, 3)
        x = self.pe_lif(x)
        if hook is not None:
            hook[self._get_name() + "_pe_lif"] = x.detach()
        x = self.bn1(self.dwconv1(x.flatten(0, 1).permute(0, 2, 1, 3)))
        x = x + x_res
        x = x.reshape(T, B, *x.shape[1:])
        x = self.lif1(x)
        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()
        x = self.bn2(self.pwconv1(x.flatten(0, 1)))
        x = x.reshape(T, B, *x.shape[1:])

        x_res = x.flatten(0, 1)
        x = self.lif2(x)
        if hook is not None:
            hook[self._get_name() + "_lif2"] = x.detach()
        x = self.bn3(self.dwconv2(x.flatten(0, 1)))
        x = x + x_res
        x = x.reshape(T, B, *x.shape[1:])

        x = self.lif3(x)
        if hook is not None:
            hook[self._get_name() + "_lif3"] = x.detach()
        x = self.bn4(self.pwconv2(x.flatten(0, 1)))
        x = x.reshape(T, B, *x.shape[1:])

        x_res = x.permute(0, 1, 3, 2, 4)
        x = x_res
        x = self.lif4(x)
        if hook is not None:
            hook[self._get_name() + "_lif4"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = x.reshape(T, B, *x.shape[1:])
        x = x + x_res

        x = x.reshape(T, B, -1, H, W)

        return x, hook

class SepConv_pps(nn.Module):
    r"""
    Temporal spiking separable convolution
    FOR PPS
    """

    def __init__(
        self,
        in_channels=1,
        embed_dims=128,
        img_size_h=8,
        img_size_w=512,
        bias=False,
    ):
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.embed_dims = embed_dims
        self.pe_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.pe_bn = nn.BatchNorm2d(embed_dims)
        self.pe_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.dwconv1 = nn.Conv1d(
            self.img_size_h,
            self.img_size_h,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=self.img_size_h,
            bias=bias,
        )  # depthwise conv
        self.bn1 = nn.BatchNorm1d(self.img_size_h)
        self.lif1 = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')


        self.dwconv2 = nn.Conv1d(
            self.img_size_h,
            self.img_size_h,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.img_size_h,
            bias=bias,
        )  # depthwise conv
        self.bn2 = nn.BatchNorm1d(self.img_size_h)


    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        x = x.flatten(0, 1)
        x = self.pe_conv(x)
        x = self.pe_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.pe_lif(x)
        if hook is not None:
            hook[self._get_name() + "_pe_lif"] = x.detach()

        x = self.bn1(self.dwconv1(x.flatten(0, 2)))
        x = x.reshape(T, B, -1, H, W)

        x = self.lif1(x)
        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()
        x = self.bn2(self.dwconv2(x.flatten(0, 2)))
        x = x.reshape(T, B, -1, H, W)

        return x, hook

class SepConv_pps_hci(nn.Module):
    r"""
    Temporal spiking separable convolution
    FOR PPS
    """

    def __init__(
        self,
        in_channels=1,
        embed_dims=128,
        img_size_h=8,
        img_size_w=512,
        bias=False,
    ):
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.embed_dims = embed_dims
        self.pe_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.pe_bn = nn.BatchNorm2d(embed_dims)
        self.pe_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.dwconv1 = nn.Conv1d(
            3,
            3,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=3,
            bias=bias,
        )  # depthwise conv
        self.bn1 = nn.BatchNorm1d(3)
        self.lif1 = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.dwconv1_ecg = nn.Conv1d(
            3,
            5,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=bias,
        )  # depthwise conv
        self.bn1_ecg = nn.BatchNorm1d(5)


        self.dwconv2 = nn.Conv1d(
            8,
            8,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=8,
            bias=bias,
        )  # depthwise conv
        self.bn2 = nn.BatchNorm1d(8)


    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        x = x.flatten(0, 1)
        x = self.pe_conv(x)
        x = self.pe_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.pe_lif(x)
        if hook is not None:
            hook[self._get_name() + "_pe_lif"] = x.detach()

        x_ecg = x[:,:,:,:3,:]
        x = x[:,:,:,3:6,:]
        x = self.bn1(self.dwconv1(x.flatten(0, 2)))
        x_ecg = self.bn1_ecg(self.dwconv1_ecg(x_ecg.flatten(0, 2)))
        x=torch.cat([x, x_ecg], dim=1)
        H = 8
        x = x.reshape(T, B, -1, H, W)

        x = self.lif1(x)
        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()
        x = self.bn2(self.dwconv2(x.flatten(0, 2)))
        x = x.reshape(T, B, -1, H, W)

        return x, hook