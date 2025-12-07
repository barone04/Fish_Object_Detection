import torch
import torch.nn as nn

def l1inftyinfty(filter_tensor: torch.Tensor) -> float:
    """
    Tính chuẩn L1-inf-inf của một filter 3D (C_in, H, W).
    """
    abs_w = filter_tensor.abs()
    # Max over W (dim=2) -> (C_in, H)
    max_w = abs_w.max(dim=2).values
    # Max over H (dim=1) -> (C_in)
    max_hw = max_w.max(dim=1).values
    # Sum over C_in -> Scalar
    return max_hw.sum().item()

def l1inftyinfty_distance(filter1: torch.Tensor, filter2: torch.Tensor) -> float:
    diff = torch.abs(filter1 - filter2)
    return l1inftyinfty(diff)

def get_weight(layer: nn.Module):
    # Hỗ trợ lấy weight từ ConvBNReLU hoặc nn.Conv2d thường
    if hasattr(layer, 'conv'): # ConvBNReLU wrapper
        return layer.conv.weight
    elif hasattr(layer, 'weight'): # Standard Layer
        return layer.weight
    else:
        raise ValueError(f"Cannot extract weight from layer: {type(layer)}")