import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, H, W, dtype=torch.float32)
kernel = torch.rand(oC, iC, kH, kW, dtype=torch.float32)

H = input.shape[2]
W = input.shape[3]
kH = kernel.shape[2]
kW = kernel.shape[3]
oC = kernel.shape[0]
iC = kernel.shape[1]
n = input.shape[0]
oH = H - (kH - 1)
oW = W - (kW - 1)
out = torch.zeros((n, oC, oH, oW))
old_h = 0
old_w = 0
for w in range(oW):
    for h in range(oH):
        for o in range(oC):
            out[:, o, h, w] = torch.sum(kernel[o, :, :, :] * input[:, :, h:(h + kH), w:(w + kW)], axis=(1, 2, 3))
