import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)
s = random.randint(2, 6)

input = torch.rand(n, iC, H, W)
kernel = torch.rand(iC, oC, kH, kW)

oH = (H - 1) * s + kH
oW = (W - 1) * s + kW

out = torch.zeros((n, oC, oH, oW))

h_start = 0
w_start = 0
for h in range(H):
    for w in range(W):
        #  (n, oC, kH, kW) =  (n, iC, 1, 1, 1) * (1, iC, oC, kH, kW) -> (n, iC, oC, kH, kW) -> (n, oC, kH, kW)
        out[:, :, h_start:(h_start+kH), w_start:(w_start+kW)] += \
            (input[:, :, h, w].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * kernel.unsqueeze(0)).sum(dim=1)
        w_start += s
    h_start += s
    w_start = 0

print(out)