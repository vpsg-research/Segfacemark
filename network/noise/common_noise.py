import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint
import random as ra
import kornia
import math
from kornia.geometry.transform.imgwarp import get_perspective_transform
from kornia.geometry.transform.imgwarp import warp_perspective
from utils import Jpeg_compression
from config import training_config as cfg


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input, type=""):
        return input[0].clamp(-1, 1) - input[0]

def jpeg_compression_train(input, type=""):
    forward_watermarked_images = input[1]
    jp = Jpeg_compression.JpegCompression(cfg.device)
    rgb_jp = jp(forward_watermarked_images)
    return rgb_jp[:, [2], :, :].clamp(-1,1) - input[0]

class Resize(nn.Module):
    def __init__(self, down_scale=0.5):
        super(Resize, self).__init__()
        self.down_scale = down_scale

    def forward(self, input, type=""):
        forward_u_embedded = input[0]
        noised_down = F.interpolate(
            forward_u_embedded,
            size=(
                int(self.down_scale * forward_u_embedded.shape[2]),
                int(self.down_scale * forward_u_embedded.shape[3]),
            ),
            mode="nearest",
        )
        noised_up = F.interpolate(
            noised_down, size=(forward_u_embedded.shape[2], forward_u_embedded.shape[3]), mode="nearest"
        )
        return noised_up.clamp(-1,1) - forward_u_embedded

class MedianBlur(nn.Module):
    def __init__(self, kernel_size=(3, 3)):
        super(MedianBlur, self).__init__()
        self.transform = kornia.filters.MedianBlur(kernel_size=kernel_size)

    def forward(self, input, type=""):
        forward_u_embedded = input[0]
        return self.transform(forward_u_embedded).clamp(-1,1) - forward_u_embedded

class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.01, p=1):
        super(GaussianNoise, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianNoise(mean=mean, std=std, p=p)

    def forward(self, input, type=""):
        image = input[0]
        return self.transform(image).clamp(-1, 1) - image


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=(3,3), sigma=(2,2), p=1):
        super(GaussianBlur, self).__init__()
        self.transform = kornia.augmentation.RandomGaussianBlur(kernel_size=kernel_size, sigma=sigma, p=p)

    def forward(self, input, type=""):
        image = input[0]
        return self.transform(image).clamp(-1, 1) - image

class Dropout(nn.Module):
    def __init__(self, prob=0.3):
        super(Dropout, self).__init__()
        self.prob = prob
    
    def forward(self, input, type=""):
        forward_u_embedded, forward_cover_images = input[0], input[2]
        mask = torch.Tensor(np.random.choice([0.0, 1.0], forward_u_embedded.shape[2:], p=[self.prob, 1 - self.prob])).to(forward_u_embedded.device)
        mask = mask.expand_as(forward_u_embedded)
        output = forward_u_embedded * mask + forward_cover_images[:, [1], :, :] * (1 - mask)
        return output.clamp(-1,1) - forward_u_embedded

class SaltPepper(nn.Module):
	def __init__(self, prob=0.05):
		super(SaltPepper, self).__init__()
		self.prob = prob

	def sp_noise(self, image, prob):
		mask = torch.Tensor(np.random.choice((0, 1, 2), image.shape[2:], p=[1 - prob, prob / 2., prob / 2.])).to(image.device)
		mask = mask.expand_as(image)
		image[mask == 1] = 1  # salt
		image[mask == 2] = -1  # pepper
		return image

	def forward(self, input, type=""):
		forward_u_embedded = input[0]
		return self.sp_noise(forward_u_embedded, self.prob).clamp(-1,1) - forward_u_embedded
