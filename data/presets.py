import torch
from . import transforms as T


class DetectionPresetTrain:
    def __init__(self, data_augmentation, hflip_prob=0.5, mean=(123.0, 117.0, 104.0)):
        transforms = []

        # Chuyển PIL sang Tensor (giữ nguyên uint8 0-255)
        transforms.append(T.PILToTensor())

        # Data Augmentation: Horizontal Flip
        if data_augmentation == "hflip":
            transforms.append(T.RandomHorizontalFlip(p=hflip_prob))

        # Chuyển đổi sang Float (0.0 - 1.0) để đưa vào mạng
        transforms.append(T.ConvertImageDtype(torch.float))

        # Normalize (Optional: Faster R-CNN thường tự normalize bên trong,
        # nhưng nếu backbone yêu cầu thì thêm vào đây.
        # Ở đây ta giữ raw float 0-1 để model tự xử lý như mặc định torchvision)

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