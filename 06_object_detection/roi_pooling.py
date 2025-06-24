import random
import torch

n = random.randint(1, 3)
C = random.randint(10, 20)
H = random.randint(5, 10)
W = random.randint(5, 10)
oH = random.randint(2, 4)
oW = random.randint(2, 4)
L = random.randint(2, 6)
input = torch.rand(n, C, H, W)
boxes = [torch.zeros(L, 4) for _ in range(n)]
for i in range(n):
    boxes[i][:, 0] = torch.rand(L) * (H - oH)  # y
    boxes[i][:, 1] = torch.rand(L) * (W - oW)  # x
    boxes[i][:, 2] = oH + torch.rand(L) * (H - oH)  # w
    boxes[i][:, 3] = oW + torch.rand(L) * (W - oW)  # h

    boxes[i][:, 2:] += boxes[i][:, :2]
    boxes[i][:, 2] = torch.clamp(boxes[i][:, 2], max=H - 1)
    boxes[i][:, 3] = torch.clamp(boxes[i][:, 3], max=W - 1)
output_size = (oH, oW)

out = torch.zeros((n, L, C, output_size[0], output_size[1]))

for k in range(n):
    for l in range(L):
        boxes[k][l, :] = torch.round(boxes[k][l, :])
        for i in range(oH):
            for j in range(oW):
                y1 = boxes[k][l, 0]
                y2 = boxes[k][l, 2]
                x1 = boxes[k][l, 1]
                x2 = boxes[k][l, 3]
                y_min = torch.floor(y1 + i * (y2 - y1 + 1) / oH).to(dtype=torch.int64)
                y_max = torch.ceil(y1 + (i + 1) * (y2 - y1 + 1) / oH).to(dtype=torch.int64)
                x_min = torch.floor(x1 + j * (x2 - x1 + 1) / oW).to(dtype=torch.int64)
                x_max = torch.ceil(x1 + (j + 1) * (x2 - x1 + 1) / oW).to(dtype=torch.int64)
                out[k, l, :, i, j] = torch.amax(input[k, :, y_min:y_max, x_min:x_max], dim=(1, 2))
