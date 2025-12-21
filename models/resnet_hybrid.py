import torch
import torch.nn as nn
from torchvision.models import resnet50 as torchvision_resnet50, ResNet50_Weights
from .conv_bn_relu import ConvBNReLU


stage_repeat = [3, 4, 6, 3]
stage_out_channel = [64] + [256] * 3 + [512] * 4 + [1024] * 6 + [2048] * 3


def adapt_channel(compress_rate):
    if compress_rate is None:
        compress_rate = [0.0] * 53

    stage_oup_cprate = []
    stage_oup_cprate += [compress_rate[0]]
    for i in range(len(stage_repeat) - 1):
        stage_oup_cprate += [compress_rate[i + 1]] * stage_repeat[i]
    stage_oup_cprate += [0.0] * stage_repeat[-1]

    mid_scale_cprate = compress_rate[len(stage_repeat):]

    overall_channel = []
    mid_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0:
            overall_channel += [int(stage_out_channel[i] * (1 - stage_oup_cprate[i]))]
        else:
            overall_channel += [int(stage_out_channel[i] * (1 - stage_oup_cprate[i]))]
            mid_channel += [
                int(stage_out_channel[i] // 4 * (1 - mid_scale_cprate[i - 1]))
            ]

    return overall_channel, mid_channel


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, midplanes, inplanes, planes, stride=1, is_downsample=False):
        super(Bottleneck, self).__init__()

        self.conv1 = ConvBNReLU(inplanes, midplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = ConvBNReLU(midplanes, midplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = ConvBNReLU(midplanes, planes, kernel_size=1, stride=1, padding=0, bias=False, relu=False)

        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()

        if is_downsample:
            self.shortcut = ConvBNReLU(
                inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False, relu=False
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

    # def get_prunable_layers(self, pruning_type="unstructured"):
    #     layers = [self.conv1, self.conv2, self.conv3]
    #     if isinstance(self.shortcut, ConvBNReLU):
    #         layers.append(self.shortcut)
    #     return layers

    def get_prunable_layers(self, pruning_type="unstructured"):
        layers = []
        # GỌI HÀM CỦA ConvBNReLU ĐỂ NÓ TỰ INIT MASK
        layers.extend(self.conv1.get_prunable_layers(pruning_type))
        layers.extend(self.conv2.get_prunable_layers(pruning_type))
        layers.extend(self.conv3.get_prunable_layers(pruning_type))

        if isinstance(self.shortcut, ConvBNReLU):
            layers.extend(self.shortcut.get_prunable_layers(pruning_type))
        return layers


class ResNet50(nn.Module):
    def __init__(self, compress_rate, num_classes=1000):
        super(ResNet50, self).__init__()

        overall_channel, mid_channel = adapt_channel(compress_rate)
        layer_num = 0

        # Stem
        self.conv1 = ConvBNReLU(3, overall_channel[layer_num], kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential()
        self.layer2 = nn.Sequential()
        self.layer3 = nn.Sequential()
        self.layer4 = nn.Sequential()

        layer_num += 1
        stages = ['layer1', 'layer2', 'layer3', 'layer4']

        for i, stage_name in enumerate(stages):
            stage = getattr(self, stage_name)
            stride = 1 if i == 0 else 2

            # First block (with downsample)
            stage.append(Bottleneck(
                midplanes=mid_channel[layer_num - 1],
                inplanes=overall_channel[layer_num - 1],
                planes=overall_channel[layer_num],
                stride=stride,
                is_downsample=True
            ))
            layer_num += 1

            # Other blocks
            for j in range(1, stage_repeat[i]):
                stage.append(Bottleneck(
                    midplanes=mid_channel[layer_num - 1],
                    inplanes=overall_channel[layer_num - 1],
                    planes=overall_channel[layer_num],
                    stride=1,
                    is_downsample=False
                ))
                layer_num += 1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(overall_channel[layer_num - 1], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # def get_prunable_layers(self, pruning_type="unstructured"):
    #     convs = [self.conv1]
    #     for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
    #         for block in stage:
    #             convs.extend(block.get_prunable_layers(pruning_type))
    #     return convs

    def get_prunable_layers(self, pruning_type="unstructured"):
        convs = []
        convs.extend(self.conv1.get_prunable_layers(pruning_type))

        for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in stage:
                convs.extend(block.get_prunable_layers(pruning_type))
        return convs

def load_pretrained_weights(model, weights="DEFAULT"):
    """
    Hàm ánh xạ trọng số từ ResNet50 chuẩn (Torchvision) sang Hybrid ResNet (Custom Wrapper).
    """
    if weights is None:
        return model

    print(f"Loading pretrained weights: {weights}")
    if weights == "DEFAULT":
        weights = ResNet50_Weights.DEFAULT

    # Lấy state_dict chuẩn
    std_state_dict = torchvision_resnet50(weights=weights).state_dict()
    custom_state_dict = model.state_dict()
    new_state_dict = {}

    # Mapping Loop
    for k, v in std_state_dict.items():
        new_k = k
        # Mapping Stem: conv1.weight -> conv1.conv.weight, bn1.weight -> conv1.bn.weight
        if k.startswith('conv1.'):
            new_k = k.replace('conv1.', 'conv1.conv.')
        elif k.startswith('bn1.'):
            new_k = k.replace('bn1.', 'conv1.bn.')

        # Mapping Downsample: downsample.0 -> shortcut.conv, downsample.1 -> shortcut.bn
        elif 'downsample.0' in k:
            new_k = k.replace('downsample.0', 'shortcut.conv')
        elif 'downsample.1' in k:
            new_k = k.replace('downsample.1', 'shortcut.bn')

        # Mapping Layers: conv1 -> conv1.conv, bn1 -> conv1.bn
        # Ví dụ: layer1.0.conv1.weight -> layer1.0.conv1.conv.weight
        # Ví dụ: layer1.0.bn1.weight   -> layer1.0.conv1.bn.weight
        elif 'layer' in k:
            if '.conv1.' in k:
                new_k = k.replace('.conv1.', '.conv1.conv.')
            elif '.bn1.' in k:
                new_k = k.replace('.bn1.', '.conv1.bn.')
            elif '.conv2.' in k:
                new_k = k.replace('.conv2.', '.conv2.conv.')
            elif '.bn2.' in k:
                new_k = k.replace('.bn2.', '.conv2.bn.')
            elif '.conv3.' in k:
                new_k = k.replace('.conv3.', '.conv3.conv.')
            elif '.bn3.' in k:
                new_k = k.replace('.bn3.', '.conv3.bn.')

        # 3. Load vào model nếu shape khớp (Bỏ qua FC layer nếu số class khác nhau)
        if new_k in custom_state_dict:
            if custom_state_dict[new_k].shape == v.shape:
                new_state_dict[new_k] = v
            else:
                print(f"Skipping {new_k}: Shape mismatch {custom_state_dict[new_k].shape} vs {v.shape}")
        else:
            # print(f"Key {new_k} not found in custom model") # Debug only
            pass

    model.load_state_dict(new_state_dict, strict=False)
    print("Pretrained weights loaded successfully (Compatible keys only).")
    return model


def resnet_50(compress_rate=None, num_classes=1000, weights=None):
    """
    Factory function hỗ trợ load weights.
    """
    model = ResNet50(compress_rate=compress_rate, num_classes=num_classes)
    if weights is not None:
        load_pretrained_weights(model, weights)
    return model