# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class UnstructuredMask(nn.Module):
#     """
#     Mask dùng cho Stage 1 (Song Han).
#     Kích thước mask = Kích thước weight (Ví dụ: 64x3x7x7).
#     """
#
#     def __init__(self, weight_shape):
#         super().__init__()
#         # Buffer: Được lưu vào state_dict nhưng không phải là tham số train (không có gradient)
#         self.register_buffer('mask', torch.ones(weight_shape))
#
#     def update(self, new_mask):
#         self.mask.data.copy_(new_mask)
#
#     def apply(self, conv_layer):
#         # Nhân trọng số với mask (ép về 0)
#         conv_layer.weight.data.mul_(self.mask)
#
#
# class StructuredMask(nn.Module):
#     """
#     Mask dùng cho Stage 2 (Filter Pruning).
#     Kích thước mask = Số kênh output (Ví dụ: 64).
#     """
#
#     def __init__(self, planes):
#         super().__init__()
#         self.register_buffer('mask', torch.ones(planes))
#
#     def apply(self, conv_layer):
#         # Dùng einsum để nhân vector mask vào toàn bộ filter
#         # cijk: channel_out, channel_in, h, w
#         # c: channel_out
#         conv_layer.weight.data = torch.einsum("cijk,c->cijk", conv_layer.weight.data, self.mask)
#
#
# class ConvBNReLU(nn.Module):
#     """
#     Wrapper 3-trong-1: Conv + BN + ReLU.
#     Hỗ trợ Pruning Masking bên trong.
#     """
#
#     def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False, relu=True):
#         super(ConvBNReLU, self).__init__()
#
#         self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size,
#                               stride=stride, padding=padding, bias=bias)
#         self.bn = nn.BatchNorm2d(planes)
#
#         if relu:
#             self.relu = nn.ReLU(inplace=True)
#         else:
#             self.relu = nn.Identity()
#
#         # Placeholder cho Mask (Sẽ được khởi tạo bởi Pruner)
#         self.mask_handler = None
#
#     @property
#     def out_channels(self):
#         # Property này CỰC KỲ QUAN TRỌNG để FPN của Faster R-CNN đọc được số kênh
#         return self.conv.out_channels
#
#     @property
#     def weight(self):
#         # Proxy để truy cập nhanh weight của conv bên trong
#         return self.conv.weight
#
#     def forward(self, x):
#         # Lưu ý: Mask thường được apply vào weight TRƯỚC KHI forward
#         # (thông qua hooks hoặc gọi hàm apply bên ngoài training loop)
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         return out
#
#     def get_prunable_layers(self, pruning_type="unstructured"):
#         """
#         Hàm để Pruner tìm thấy layer này và gắn mask vào.
#         """
#         # Nếu chưa có mask handler thì khởi tạo
#         if self.mask_handler is None:
#             if pruning_type == "unstructured":
#                 self.mask_handler = UnstructuredMask(self.conv.weight.shape)
#             elif pruning_type == "structured":
#                 self.mask_handler = StructuredMask(self.conv.out_channels)
#
#             # Move mask về cùng device với conv
#             self.mask_handler.to(self.conv.weight.device)
#
#         return [self]

import torch
import torch.nn as nn


class UnstructuredMask(nn.Module):
    """
    Mask 4D cho Song Han Pruning.
    """

    def __init__(self, weight_shape):
        super().__init__()
        self.register_buffer('mask', torch.ones(weight_shape))

    def update(self, new_mask):
        self.mask.data.copy_(new_mask)

    def apply(self, conv_layer):
        # Nhân element-wise
        conv_layer.weight.data.mul_(self.mask)


class StructuredMask(nn.Module):
    """
    Mask 1D cho Filter Pruning.
    """

    def __init__(self, planes):
        super().__init__()
        self.register_buffer('mask', torch.ones(planes))

    def update(self, new_mask):
        # new_mask là vector 1D
        self.mask.data.copy_(new_mask)

    def apply(self, conv_layer):
        # Broadcast mask (C_out) vào weight (C_out, C_in, H, W)
        # Reshape mask thành (C_out, 1, 1, 1) để nhân
        conv_layer.weight.data.mul_(self.mask.view(-1, 1, 1, 1))


class MaskProxy:
    """
    Lớp trung gian (Proxy) để đánh lừa Pruner.
    Giúp Pruner nghĩ rằng nó đang thao tác trực tiếp với layer,
    nhưng thực ra nó đang thao tác với đúng loại mask của nó.
    """

    def __init__(self, layer, mask_type):
        self.layer = layer
        self.mask_type = mask_type

    @property
    def mask_handler(self):
        # Trả về đúng loại mask mà Pruner cần
        if self.mask_type == 'unstructured':
            # Lazy init cho Unstructured Mask
            if self.layer.u_mask is None:
                self.layer.u_mask = UnstructuredMask(self.layer.conv.weight.shape).to(self.layer.conv.weight.device)
            return self.layer.u_mask

        elif self.mask_type == 'structured':
            # Lazy init cho Structured Mask
            if self.layer.s_mask is None:
                self.layer.s_mask = StructuredMask(self.layer.conv.out_channels).to(self.layer.conv.weight.device)
            return self.layer.s_mask
        return None

    @property
    def conv(self):
        return self.layer.conv

    def __getattr__(self, name):
        # Chuyển tiếp các truy cập attribute khác vào layer gốc
        return getattr(self.layer, name)


class ConvBNReLU(nn.Module):
    """
    Wrapper hỗ trợ Dual Masking (Cả Unstructured và Structured).
    """

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False, relu=True):
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(planes)

        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.Identity()

        # Lưu trữ riêng biệt 2 loại mask
        self.u_mask = None  # Unstructured
        self.s_mask = None  # Structured

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def weight(self):
        return self.conv.weight

    def forward(self, x):
        # Apply lần lượt các mask nếu tồn tại
        if self.u_mask is not None:
            self.u_mask.apply(self.conv)

        if self.s_mask is not None:
            self.s_mask.apply(self.conv)

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def get_prunable_layers(self, pruning_type="unstructured"):
        """
        Thay vì trả về self, ta trả về Proxy.
        Proxy sẽ tự động điều hướng request vào u_mask hoặc s_mask.
        """
        return [MaskProxy(self, pruning_type)]