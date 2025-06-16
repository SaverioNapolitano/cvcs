import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(
    np.uint8)
im = torch.from_numpy(im)

hist, bin_edges = torch.histogram(im.to(dtype=torch.float64), 256, range=(0., 255.), density=True)

x = torch.arange(256)
w1 = torch.zeros(256)
w2 = torch.zeros(256)
mu1 = torch.zeros(256)
mu2 = torch.zeros(256)
variances = torch.zeros(256)

for t in range(256):
    w1[t] = torch.sum(hist[:(t + 1)])
    w2[t] = torch.sum(hist[(t + 1):])
    if w1[t] == 0 or w2[t] == 0:
        continue
    mu1[t] = torch.sum((hist[:(t + 1)] * x[:(t + 1)])) / w1[t]
    mu2[t] = torch.sum((hist[(t + 1):] * x[(t + 1):])) / w2[t]
    variances[t] = w1[t] * w2[t] * ((mu1[t] - mu2[t]) ** 2)

out = torch.argmax(variances).item()
