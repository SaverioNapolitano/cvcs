import random
import numpy as np
import torch
from skimage import data

im = data.astronaut()
im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
im = torch.from_numpy(im)
nbin = random.randint(32, 128)

im = im.reshape((3, im.shape[1] * im.shape[2])).to(dtype=torch.float64)

color1 = torch.bincount((im[0, :] * nbin // 256).to(dtype=torch.uint8))
color2 = torch.bincount((im[1, :] * nbin // 256).to(dtype=torch.uint8))
color3 = torch.bincount((im[2, :] * nbin // 256).to(dtype=torch.uint8))

out = torch.cat((color1, color2, color3))

out = out / torch.sum(out)
