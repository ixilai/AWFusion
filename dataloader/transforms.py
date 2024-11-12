import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, image_gt, target_gt):
        for t in self.transforms:
            image, target, image_gt, target_gt = t(image, target, image_gt, target_gt)
        return image, target, image_gt, target_gt


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, image_gt, target_gt):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        image_gt = F.resize(image_gt, self.size, interpolation=T.InterpolationMode.NEAREST)
        target_gt = F.resize(target_gt, self.size, interpolation=T.InterpolationMode.NEAREST)

        return image, target, image_gt, target_gt


class Resize_16(object):
    def __init__(self):
        pass

    def __call__(self, image, target, image_gt, target_gt):
        width, height = image.size

        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        image = F.resize(image, (new_height, new_width))
        target = F.resize(target, (new_height, new_width))
        image_gt = F.resize(image_gt, (new_height, new_width))
        target_gt = F.resize(target_gt, (new_height, new_width))

        return image, target, image_gt, target_gt


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, image_gt, target_gt):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
            image_gt = F.hflip(image_gt)
            target_gt = F.hflip(target_gt)
        return image, target, image_gt, target_gt


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, image_gt, target_gt):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
            image_gt = F.vflip(image_gt)
            target_gt = F.vflip(target_gt)
        return image, target, image_gt, target_gt


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, image_gt, target_gt):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size)
        image_gt = pad_if_smaller(image_gt, self.size)
        target_gt = pad_if_smaller(target_gt, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        image_gt = F.crop(image_gt, *crop_params)
        target_gt = F.crop(target_gt, *crop_params)
        return image, target, image_gt, target_gt

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, image_gt, target_gt):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        image_gt = F.center_crop(image_gt, self.size)
        target_gt = F.center_crop(target_gt, self.size)
        return image, target, image_gt, target_gt


class ToTensor(object):
    def __call__(self, image, target, image_gt, target_gt):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        image_gt = F.to_tensor(image_gt)
        target_gt = F.to_tensor(target_gt)
        return image, target, image_gt, target_gt