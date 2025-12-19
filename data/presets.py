import torch
from . import transforms as T


class DetectionPresetTrain:
    def __init__(self, data_augmentation, hflip_prob=0.5, mean=(123.0, 117.0, 104.0)):
        transforms = []

        # Chuyển PIL sang Tensor (giữ nguyên uint8 0-255)
        transforms.append(T.PILToTensor())

        if data_augmentation == "hflip":
            transforms.append(T.RandomHorizontalFlip(p=hflip_prob))
        transforms.append(T.ConvertImageDtype(torch.float))


        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)