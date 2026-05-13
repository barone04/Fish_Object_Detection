import torch
import numpy as np


class UnstructuredPruner:
    def __init__(self, model):
        self.model = model

    def prune(self, sensitivity=1.0):
        """
        Tính toán threshold và update mask cho từng layer ConvBNReLU.
        sensitivity: Hệ số nhân với độ lệch chuẩn (std).
        """
        print(f"Executing Song Han Pruning (Sensitivity={sensitivity})...")

        convs = self.model.get_prunable_layers(pruning_type="unstructured")

        for layer in convs:
            weight = layer.conv.weight.data

            # Tính threshold = sensitivity * std
            std = weight.std().item()
            threshold = sensitivity * std

            # Tạo mask mới: 1 nếu |w| > threshold, ngược lại 0
            new_mask = (weight.abs() > threshold).float()
            layer.mask_handler.update(new_mask)

            # Apply ngay lập tức (Zeroing)
            layer.mask_handler.apply(layer.conv)

        print(f"Unstructured pruning applied to {len(convs)} layers.")

    def apply_masks(self):
        """
        Gọi hàm này sau mỗi optimizer.step() để ép weight về 0.
        """
        convs = self.model.get_prunable_layers(pruning_type="unstructured")
        for layer in convs:
            if layer.mask_handler is not None:
                layer.mask_handler.apply(layer.conv)