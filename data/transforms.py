import torch
from torchvision.transforms import functional as F
from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple, Union
import torchvision.transforms as T

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target

class ConvertImageDtype(nn.Module):
    def __init__(self, dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                if "boxes" in target:
                    boxes = target["boxes"]
                    # Lật box: x1_new = width - x2_old
                    boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
                        [-1, 1, -1, 1], device=boxes.device
                    ) + torch.as_tensor([width, 0, width, 0], device=boxes.device)
                    target["boxes"] = boxes
        return image, target

class ToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target