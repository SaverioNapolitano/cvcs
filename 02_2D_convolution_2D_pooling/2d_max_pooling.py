import random
import torch
import math

n = random.randint(2, 6)
iC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(2, 3)
input = torch.rand((n, iC, H, W), dtype=torch.float32)

oH = (H - kH) // s + 1
oW = (W - kW) // s + 1

out = torch.zeros((n, iC, oH, oW))

x = 0
for h in range(oH):
    y = 0
    for w in range(oW):
        out[:, :, h, w] = input[:, :, x:(x + kH), y:(y + kW)].amax(dim=(2, 3))
        y = y + s if y > 0 else w + s
    x = x + s if x > 0 else h + s
