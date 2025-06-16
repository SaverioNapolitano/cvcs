import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(inplanes, planes, (3, 3), stride=stride, bias=False, padding=1),
                               nn.BatchNorm2d(planes),
                               nn.ReLU(), nn.Conv2d(planes, planes, (3, 3), padding=1, bias=False),
                               nn.BatchNorm2d(planes))
        if inplanes != planes or stride > 1:
            self.g = nn.Sequential(nn.Conv2d(inplanes, planes, (1, 1), stride=stride, bias=False),
                                   nn.BatchNorm2d(planes))
        else:
            self.g = nn.Identity()
        self.out = nn.ReLU()

    def forward(self, x):
        return self.out(self.f(x) + self.g(x))