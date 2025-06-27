import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
T = random.randint(10, 20)
H = random.randint(10, 20)
W = random.randint(10, 20)
kT = random.randint(2, 5)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(2, 3)
input = torch.rand(n, iC, T, H, W)

oT = (T - kT) // s + 1
oH = (H - kH) // s + 1
oW = (W - kW) // s + 1

out = torch.zeros((n, iC, oT, oH, oW))

t_start = 0
h_start = 0
w_start = 0
for t in range(oT):
    for h in range(oH):
        for w in range(oW):
            out[:, :, t, h, w] = torch.amax(input[:, :, t_start:(t_start + kT), h_start:(h_start + kH), w_start:(w_start + kW)], dim=(2, 3, 4))
            w_start += s
        w_start = 0
        h_start += s
    h_start = 0
    t_start += s