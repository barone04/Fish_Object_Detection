import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .conv_bn_relu import ConvBNReLU


class BottleneckFPNBlock(nn.Module):
    """
    Cấu trúc: Compress (ConvBNReLU) -> Depthwise (ConvBN) -> Expand (ConvBN)
    Dùng ConvBNReLU cho Compress để code surgery cũ tự nhận diện được mask.
    """

    def __init__(self, channels=256, compress_rate=0.0):
        super().__init__()

        inner_channels = int(channels * (1.0 - compress_rate))

        self.compress_layer = ConvBNReLU(channels, inner_channels, kernel_size=1, padding=0, bias=False)
        self.dw_conv = nn.Conv2d(inner_channels, inner_channels, kernel_size=3,
                                 padding=1, groups=inner_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(inner_channels)
        self.relu = nn.ReLU(inplace=True)

        self.expand_conv = nn.Conv2d(inner_channels, channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # x: [N, 256, H, W]
        out = self.compress_layer(x)  # -> [N, inner, H, W]

        out = self.dw_conv(out)
        out = self.dw_bn(out)
        out = self.relu(out)

        out = self.expand_conv(out)  # -> [N, 256, H, W]
        out = self.expand_bn(out)
        return out

    def get_prunable_layers(self):
        # Trả về lớp compress để Pruner quản lý mask
        return [self.compress_layer]


class PrunableFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, compress_rate=None):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        if compress_rate is None:
            compress_rate = [0.0] * len(in_channels_list)

        for i, in_c in enumerate(in_channels_list):
            if in_c == 0: continue

            self.inner_blocks.append(nn.Conv2d(in_c, out_channels, 1))

            rate = compress_rate[i] if i < len(compress_rate) else 0.0
            self.layer_blocks.append(BottleneckFPNBlock(out_channels, compress_rate=rate))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x là Dict hoặc List features từ backbone
        if isinstance(x, dict):
            x = list(x.values())

        results = []
        last_inner = self.inner_blocks[-1](x[-1])
        results.append(self.layer_blocks[-1](last_inner))

        # Top-down Path (P4, P3, P2...)
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])

            # Upsample
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")

            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        out_map = OrderedDict()
        for i, res in enumerate(results):
            out_map[f"{i}"] = res

        out_map["pool"] = F.max_pool2d(results[-1], 1, 2, 0)
        return out_map