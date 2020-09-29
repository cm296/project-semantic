import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from .curvature import banana_filter


class EdgeModel(nn.Module):

    def __init__(self, n_ories=320, gau_sizes=(5,), filt_size=9, fre=1.2, gamma=1, sigx=1, sigy=1):
        super().__init__()

        self.n_ories = n_ories
        self.gau_sizes = gau_sizes
        self.filt_size = filt_size
        self.fre = fre
        self.gamma = gamma
        self.sigx = sigx
        self.sigy = sigy

        # Construct filters
        i = 0
        ories = np.arange(0, np.pi, np.pi / n_ories)
        w = torch.zeros(size=(len(ories) * len(gau_sizes), 1, filt_size, filt_size))
        for gau_size in gau_sizes:
            for orie in ories:
                w[i, 0, :, :] = banana_filter(gau_size, fre, orie, 0, gamma, sigx, sigy, filt_size)
                i += 1
        self.weight = nn.Parameter(w)

    def forward(self, image):
        feats = F.conv2d(image, weight=self.weight, padding=math.floor(self.filt_size / 2))
        feats = feats.abs()
        return feats
