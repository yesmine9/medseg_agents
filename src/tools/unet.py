import torch
import torch.nn as nn


def block(i, o, p=0.0):
    layers = [
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    ]
    if p > 0:
        layers.append(nn.Dropout2d(p))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, in_ch=3, base=16, depth=4, p=0.2):
        super().__init__()
        chs = [base * 2 ** k for k in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.enc.append(block(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bott = block(chs[-2], chs[-1], p)
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        prev = chs[-1]
        for ch in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(prev, ch, 2, stride=2))
            self.dec.append(block(ch * 2, ch, p))
            prev = ch
        self.out = nn.Conv2d(prev, 1, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bott(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([s, x], dim=1))
        return self.out(x)
