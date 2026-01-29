import torch
import torch.nn as nn
import torchvision
import numbers
from einops import rearrange


from ultralytics.nn.modules.conv import LightConv


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)

        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out
'''
class FS_SCWM_1(nn.Module):
    def __init__(self, dim, bias=False):
        super(FSAS_1, self).__init__()

        self.to_q = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_k = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_v = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.to_hidden_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.fsda_1 = FSDA_1(dim * 2)

    def forward(self, x):
        q = self.to_hidden_dw(self.to_q(x))
        k = self.to_hidden_dw(self.to_k(x))
        v = self.to_hidden_dw(self.to_v(x))

        q_fft = self.fsda_1(q)
        k_fft = self.fsda_1(k)
        v_fft = self.fsda_1(v)
        out = q_fft * k_fft

        out = self.norm(out)
        output = v_fft * out
        output = self.project_out(output)

        return output

class FS_SCWM_2(nn.Module):
    def __init__(self, dim, bias=False):
        super(FSAS_1, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)

        #self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.to_q0_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.to_q1_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=5, stride=1, padding=2, groups=dim * 2, bias=bias)
        self.to_k0_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.to_k1_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=5, stride=1, padding=2, groups=dim * 2, bias=bias)
        self.to_v0_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.to_v1_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=5, stride=1, padding=2, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim * 2 * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2 * 2, LayerNorm_type='WithBias')

        self.fsda_1 = FSDA_1(dim * 2 * 2)

    def forward(self, x):
        hidden = self.to_hidden(x)
        q0, k0, v0 = hidden.chunk(3, dim=1)

        q = torch.cat((self.to_q0_dw(q0),self.to_q1_dw(q0)),dim=1)
        k = torch.cat((self.to_k0_dw(k0), self.to_k1_dw(k0)), dim=1)
        v = torch.cat((self.to_v0_dw(v0), self.to_v1_dw(v0)), dim=1)

        q_fft = self.fsda_1(q)
        k_fft = self.fsda_1(k)
        v_fft = self.fsda_1(v)
        out = q_fft * k_fft

        out = self.norm(out)
        output = v_fft * out
        output = self.project_out(output)

        return output
'''
class FFIEM(nn.Module):
    def __init__(self, dim, bias=False):
        super(FSAS_1, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)

        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.ff = FSD_SCWM_1(dim * 2)

        #self.patch_size = 8
        #self.dcn = DeformConv(dim * 6, kernel_size=(5, 5), padding=2, groups=1)

    def forward(self, x):
        hidden = self.to_hidden(x)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        #q, k, v = self.dcn(hidden).chunk(3, dim=1)
        #q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                    patch2=self.patch_size)
        #k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                    patch2=self.patch_size)
        #q_fft = torch.fft.rfft2(q_patch.float())
        #k_fft = torch.fft.rfft2(k_patch.float())
        q_fft = self.ff(q)
        k_fft = self.ff(k)
        v_fft = self.ff(v)
        out = q_fft * k_fft
        #out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        #out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
        #                patch2=self.patch_size)

        out = self.norm(out)
        #output = v * out
        output = v_fft * out
        output = self.project_out(output)
        #x = x + output

        return output

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FF_SCWM(nn.Module):
    def __init__(self, nc):
        super(FS_SCWM, self).__init__()

        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0))

        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')        
        ori_mag = torch.abs(x_freq)  
        ori_pha = torch.angle(x_freq)  
        mag = self.processmag(ori_mag)  
        mag = ori_mag + mag  
        pha = self.processpha(ori_pha)  
        pha = ori_pha + pha  
        real = mag * torch.cos(pha)  
        imag = mag * torch.sin(pha)  
        x_out = torch.complex(real, imag)  
        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_freq_spatial 

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.att = FFIEM(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.att(self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))


class C2f_SCWM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))





