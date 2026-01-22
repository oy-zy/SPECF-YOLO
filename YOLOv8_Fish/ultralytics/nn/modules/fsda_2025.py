import torch
import torch.nn as nn
#from .conv import Conv

# Squeeze-and-Excitation (SE) 层，用于通过通道加权来重新校准特征图
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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        # 自适应平均池化，将每个通道缩小到单个值（1x1 的空间大小）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # SE块的全连接层，包含一个用于控制复杂度的降维率
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 减少通道维度
            nn.ReLU(inplace=True),  # ReLU激活函数引入非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 恢复原始通道维度
            nn.Sigmoid()  # Sigmoid激活，将每个通道的权重限制在0到1之间
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入张量的批量大小和通道数量
        y = self.avg_pool(x).view(b, c)  # 对每个通道进行全局平均池化并展平
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层生成每个通道的权重
        return x * y.expand_as(x)  # 对输入特征图进行通道加权


# 频谱动态聚合层
class FSDA(nn.Module):
    def __init__(self, nc):
        super(FSDA, self).__init__()

        # 用于处理幅度部分的卷积和激活操作
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  # 1x1卷积保持特征图大小不变
            nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活函数
            SELayer(channel=nc),  # 加入SE层进行通道加权
            nn.Conv2d(nc, nc, 1, 1, 0))  # 另一个1x1卷积

        # 用于处理相位部分的卷积和激活操作
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),  # 1x1卷积保持特征图大小不变
            nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活函数
            SELayer(channel=nc),  # 加入SE层进行通道加权
            nn.Conv2d(nc, nc, 1, 1, 0))  # 另一个1x1卷积

    def forward(self, x):
        _, _, H, W = x.shape
        # 使用了PyTorch的快速傅里叶变换(FFT)函数rfft2来对输入张量x执行二维离散傅里叶变换。以下是每个参数的详细说明：
        # torch.fft.rfft2(x): 计算输入张量x的二维实数快速傅里叶变换。
        # 这个函数只计算频谱中的正频率部分，因为输入数据是实数，频谱在正负频率上具有对称性。这种变换在处理图像、信号的频域分析时非常有用。
        # norm = 'backward': 设定了傅里叶变换的归一化方式。norm
        # 参数可以取以下值：
        # 'backward'(默认值): 将输入值不做归一化变换。
        # 'forward': 将结果除以总数，适合与逆傅里叶变换对称配合使用。
        # 'ortho': 提供单位能量的正交归一化。
        x_freq = torch.fft.rfft2(x, norm='backward')

        # 获取输入张量的幅度和相位信息
        ori_mag = torch.abs(x_freq)  # 计算复数张量的幅度
        ori_pha = torch.angle(x_freq)  # 计算复数张量的相位

        # 处理幅度信息
        mag = self.processmag(ori_mag)  # 使用处理幅度的网络
        mag = ori_mag + mag  # 将处理后的结果与原始幅度相加

        # 处理相位信息
        pha = self.processpha(ori_pha)  # 使用处理相位的网络
        pha = ori_pha + pha  # 将处理后的结果与原始相位相加

        # 重建复数形式的输出
        real = mag * torch.cos(pha)  # 实部：幅度 * cos(相位)
        imag = mag * torch.sin(pha)  # 虚部：幅度 * sin(相位)
        x_out = torch.complex(real, imag)  # 组合成复数输出

        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_freq_spatial  # 返回处理后的复数张量


class Bottleneck_FSDA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = FSDA(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_FSDA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_FSDA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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


if __name__ == '__main__':
    FSDA = FSDA(256)
    #创建一个输入张量
    batch_size = 8
    #input_tensor=torch.randn(batch_size, 256, 64, 64 )
    input_tensor = torch.randn(1, 256, 640, 640)
    #运行模型并打印输入和输出的形状
    output_tensor =FSDA(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)
