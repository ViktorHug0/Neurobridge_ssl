import torch,os
import pandas as pd
import numpy as np
from PIL import Image
import logging
import open_clip
import pickle

import cv2
from PIL import Image
import random
import numpy as np
import torch
import logging
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from torchvision import transforms


class ColorJitter:
    def __init__(self, s=0.5, p=0.8):
        self.transform = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.p = p

    def __call__(self, img:Image)-> Image:
        transform = transforms.RandomApply([self.transform], p=self.p)
        img = transform(img)
        return img


class RandomCrop:
    def __init__(self, size=(224, 224)):
        self.size = size
        self.transform = transforms.RandomCrop(size=self.size)

    def __call__(self, img:Image)-> Image:
        img = img.resize((336, 336), Image.BILINEAR)
        img = self.transform(img)
        return img


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
        self.transform = transforms.RandomHorizontalFlip(p=self.p)
    def __call__(self, img:Image)-> Image:
        img = self.transform(img)
        return img

# Reduce resolution without changing the size
class LowResolution:
    def __init__(self, scale=0.5):
        """
        scale: The factor by which to reduce the resolution. For example, 0.5 means reducing the resolution to half of the original.
        """
        self.scale = scale

    def __call__(self, img:Image) -> Image:
        w, h = img.size
        new_size = (int(w * self.scale), int(h * self.scale))
        img = img.resize(new_size, Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img


class Mosaic:
    def __init__(self, mosaic_level=16):
        """
        mosaic_level: Controls the level of mosaic (block size). The larger the value, the coarser the mosaic.
        Typical values: 8, 16, 32, 64...
        """
        self.mosaic_level = mosaic_level

    def __call__(self, img:Image) -> Image:
        img = np.array(img)
        h, w, _ = img.shape

        # Resize the image to a smaller size and then back to the original size
        small = cv2.resize(img, (w // self.mosaic_level, h // self.mosaic_level), interpolation=cv2.INTER_LINEAR)
        mosaic_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return Image.fromarray(mosaic_img)


class GrayScale:
    def __init__(self, p=0.5):
        self.p = p
        self.transform = transforms.RandomGrayscale(p=self.p)

    def __call__(self, img)-> Image:
        img = self.transform(img)
        return img


class GaussianBlur:
    def __init__(self, blur_kernel_size, fluctuation_range=0):
        self.blur_kernel_size = blur_kernel_size
        self.fluctuation_range = fluctuation_range

    def __call__(self, img):
        img_np = np.array(img)
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        if self.fluctuation_range > 0:
            # Randomly fluctuate the blur kernel size
            blur_kernel_size = random.randint(max(1, self.blur_kernel_size - self.fluctuation_range), 
                                              self.blur_kernel_size + self.fluctuation_range)
            # Ensure the kernel size is within a valid range
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
        else:
            blur_kernel_size = self.blur_kernel_size
        
        img_blur = cv2.GaussianBlur(img_np, (blur_kernel_size, blur_kernel_size), 0)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_blur)
    
    
class GaussianNoise:
    def __init__(self, mean=0.0, std=10.0, fluctuation_range=0):
        """
        Adds a Gaussian noise augmentation class.

        Args:
            mean (float): The mean of the noise.
            std (float): The standard deviation (magnitude) of the noise.
        """
        self.mean = mean
        self.std = std
        self.fluctuation_range = fluctuation_range

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)
        
        if self.fluctuation_range > 0:
            # Randomly fluctuate the standard deviation
            # Ensure the standard deviation is within a valid range
            std = random.randint(max(1, self.std - self.fluctuation_range), self.std + self.fluctuation_range)
        else:
            std = self.std

        # Generate Gaussian noise with the same shape as the image
        noise = np.random.normal(self.mean, std, img_np.shape).astype(np.float32)

        # Add the noise to the image and clip the values to ensure they remain valid pixel values
        img_noisy = img_np + noise
        img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

        return Image.fromarray(img_noisy)
