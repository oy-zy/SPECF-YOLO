import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CBFuse', 'CBLinear', 'Silence']


class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()

    def forward(self, x):
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class CBLinear(nn.Module):
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]

        '''
        h=xs[-1:]
        for i, x in enumerate(xs[:-1]):
            #print(i, x.shape)
            hh=x[self.idx[i]]
            h=F.interpolate(x[self.idx[i]], size = target_size, mode = 'nearest')
            jh=1
        '''
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out

if __name__ == '__main__':
    image_size = (1, 3, 640, 640)
    x = torch.rand(*image_size)
    model = CBLinear(c1=3,c2s=(64, 128, 256))
    model1 = CBFuse(idx=(0, 0))
    outs = model(x)
    outs1 = model1(outs)
    print(model(x).shape)
    h=1