from typing import Any, Optional
import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn



def _resnet_fpn_extractor(
        backbone,
        trainable_layers: int,
        returned_layers: Optional[list] = None,
):
    # Logic đóng băng layer
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    # DYNAMIC IN_CHANNELS: Lấy số kênh thực tế từ backbone (đã bị prune)
    # Vì ta dùng ConvBNReLU, nó có property .out_channels
    in_channels_list = [
        backbone.layer1[-1].conv3.out_channels,
        backbone.layer2[-1].conv3.out_channels,
        backbone.layer3[-1].conv3.out_channels,
        backbone.layer4[-1].conv3.out_channels,
    ]

    out_channels = 256
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=LastLevelMaxPool(),
    )


def _efficientnet_b0_fpn_extractor(trainable_layers: int):
    # Load EfficientNet-B0 pretrained
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    effnet = efficientnet_b0(weights=weights)

    # Freeze layers according to trainable_layers (coarse-grained on features)
    layers_to_train = list(range(len(effnet.features)))[-trainable_layers:]
    for idx, module in enumerate(effnet.features):
        requires = idx in layers_to_train
        for p in module.parameters():
            p.requires_grad = requires

    # Identify indices for strides 4, 8, 16, 32 empirically for EfficientNet-B0
    # Torchvision EfficientNet-B0 strides accumulate as: 2 (stem), 4, 8, 16, 32 near the end.
    # We collect feature maps after blocks that approximately correspond to these strides.

    class EfficientNetB0Features(nn.Module):
        def __init__(self, eff):
            super().__init__()
            self.stem = nn.Sequential(eff.features[0])  # stride 2
            self.stage1 = nn.Sequential(*eff.features[1:3])  # up to stride 4
            self.stage2 = nn.Sequential(*eff.features[3:5])  # up to stride 8
            self.stage3 = nn.Sequential(*eff.features[5:7])  # up to stride 16
            self.stage4 = nn.Sequential(*eff.features[7:])  # up to stride 32

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            c2 = x  # stride ~4
            x = self.stage2(x)
            c3 = x  # stride ~8
            x = self.stage3(x)
            c4 = x  # stride ~16
            x = self.stage4(x)
            c5 = x  # stride ~32
            return {"0": c2, "1": c3, "2": c4, "3": c5}

    backbone = EfficientNetB0Features(effnet)

    # Infer in_channels for each returned feature by a dummy pass
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        feats = backbone(dummy)
        in_channels_list = [feats[str(i)].shape[1] for i in range(4)]

    out_channels = 256
    return BackboneWithFPN(
        backbone,
        return_layers={"0": "0", "1": "1", "2": "2", "3": "3"},
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )


# LOAD WEIGHT PRUNED BACKBONE
# def fasterrcnn_resnet50_fpn(
#         weights_backbone=None,
#         num_classes=91,
#         compress_rate=None,
#         trainable_backbone_layers=3,
#         **kwargs
# ) -> FasterRCNN:
#     """
#     Hàm dựng Faster R-CNN hỗ trợ Backbone Nén.
#     """
#
#     # 1. Khởi tạo Backbone Hybrid (Có thể nén hoặc không)
#     backbone = resnet_50(compress_rate=compress_rate)
#
#     # 2. Load trọng số Backbone (Nếu có)
#     if weights_backbone is not None:
#         print(f"Loading backbone weights from: {weights_backbone}")
#         state_dict = torch.load(weights_backbone)
#
#         # Xử lý trường hợp key state_dict có prefix 'module.' hoặc 'backbone.'
#         if 'state_dict' in state_dict:
#             state_dict = state_dict['state_dict']
#         elif 'model' in state_dict:
#             state_dict = state_dict['model']
#
#         # Mapping key (nếu cần thiết, do ConvBNReLU làm thay đổi tên biến)
#         # Tuy nhiên, nếu train từ đầu bằng resnet_hybrid thì key sẽ khớp.
#         # Ở đây giả sử key khớp hoặc người dùng tự xử lý mapping bên ngoài.
#         try:
#             backbone.load_state_dict(state_dict, strict=False)
#             # strict=False để bỏ qua fc layer nếu load từ classification
#         except Exception as e:
#             print(f"Warning loading backbone: {e}")
#
#     # 3. Gắn FPN
#     backbone_fpn = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
#
#     # 4. Tạo Faster R-CNN
#     model = FasterRCNN(backbone_fpn, num_classes=num_classes, **kwargs)
#
#     return model


# MODEL DEFAULT: EfficientNet-B0 backbone with FPN

def fasterrcnn_efficientnet_b0_fpn(
        weights_backbone=None,
        num_classes=91,
        compress_rate=None,
        trainable_backbone_layers=3,
        **kwargs
) -> FasterRCNN:
    # Ignoring compress_rate because EfficientNet backbone is used by default now

    # Build EfficientNet-B0 FPN backbone (pretrained)
    backbone_fpn = _efficientnet_b0_fpn_extractor(trainable_backbone_layers)

    model = FasterRCNN(backbone_fpn, num_classes=num_classes, **kwargs)
    return model