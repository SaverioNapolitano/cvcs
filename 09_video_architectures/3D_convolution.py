import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 6)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, T, H, W)
kernel = torch.rand(oC, iC, kT, kH, kW)
bias = torch.rand(oC)

oT = T - (kT - 1)
oH = H - (kH - 1)
oW = W - (kW - 1)

out = torch.zeros((n, oC, oT, oH, oW))

for t in range(oT):
    for h in range(oH):
        for w in range(oW):
        # (n, oC) = (n, 1, iC, kT, kH, kW) * (1, oC, iC, kT, kH, kW) -> (n, oC, iC, kT, kH, kW) -> (n, oC) + (1, oC) -> (n, oC)
            out[:, :, t, h, w] = torch.sum(input[:, :, t:(t+kT), h:(h+kH), w:(w+kW)].unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4, 5)) + bias.unsqueeze(0)
