import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math


class CurvatureModel(nn.Module):

    def __init__(self,
                 n_ories=16,
                 curves=np.logspace(-2, -0.1, 20),
                 gau_sizes=(5,), filt_size=9, fre=1.2, gamma=1, sigx=1, sigy=1):
        super().__init__()

        self.n_ories = n_ories
        self.curves = curves
        self.gau_sizes = gau_sizes
        self.filt_size = filt_size
        self.fre = fre
        self.gamma = gamma
        self.sigx = sigx
        self.sigy = sigy

        # Construct filters
        i = 0
        ories = np.arange(0, 2 * np.pi, 2 * np.pi / n_ories)
        w = torch.zeros(size=(len(ories) * len(curves) * len(gau_sizes), 1, filt_size, filt_size))
        for curve in curves:
            for gau_size in gau_sizes:
                for orie in ories:
                    w[i, 0, :, :] = banana_filter(gau_size, fre, orie, curve, gamma, sigx, sigy, filt_size)
                    i += 1
        self.weight = nn.Parameter(w)

    def forward(self, image):
        feats = F.conv2d(image, weight=self.weight, padding=math.floor(self.filt_size / 2))
        feats = feats.abs()
        return feats


def banana_filter(s, fre, theta, cur, gamma, sigx, sigy, sz):
    # Define a matrix that used as a filter
    xv, yv = np.meshgrid(np.arange(np.fix(-sz/2).item(), np.fix(sz/2).item() + sz % 2),
                         np.arange(np.fix(sz/2).item(), np.fix(-sz/2).item() - sz % 2, -1))
    xv = xv.T
    yv = yv.T

    # Define orientation of the filter
    xc = xv * np.cos(theta) + yv * np.sin(theta)
    xs = -xv * np.sin(theta) + yv * np.cos(theta)

    # Define the bias term
    bias = np.exp(-sigx / 2)
    k = xc + cur * (xs ** 2)

    # Define the rotated Guassian rotated and curved function
    k2 = (k / sigx) ** 2 + (xs / (sigy * s)) ** 2
    G = np.exp(-k2 * fre ** 2 / 2)

    # Define the rotated and curved complex wave function
    F = np.exp(fre * k * 1j)

    # Multiply the complex wave function with the Gaussian function with a constant and bias
    filt = gamma * G * (F - bias)
    filt = np.real(filt)
    filt -= filt.mean()

    filt = torch.from_numpy(filt).float()
    return filt
