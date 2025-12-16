from typing import Any, Optional
import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .resnet_hybrid import resnet_50, ResNet50


def _resnet_fpn_extractor(
        backbone: ResNet50,
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


# MODEL BASELINE, DENSE BACKBONE
def fasterrcnn_resnet50_fpn(
        weights_backbone=None,
        num_classes=91,
        compress_rate=None,
        trainable_backbone_layers=3,
        **kwargs
) -> FasterRCNN:

    init_weights = "DEFAULT" if weights_backbone is None else None

    # 2. Khởi tạo Backbone (Thêm tham số weights=init_weights)
    backbone = resnet_50(compress_rate=compress_rate, weights=init_weights)

    # 3. Nếu có file weight backbone (ví dụ file .pth nén), thì load đè lên
    if weights_backbone is not None and str(weights_backbone) != "DEFAULT":
        print(f"Loading backbone weights from file: {weights_backbone}")
        try:
            state_dict = torch.load(weights_backbone)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            backbone.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning loading backbone file: {e}")

    # 4. Gắn FPN và tạo Model
    backbone_fpn = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone_fpn, num_classes=num_classes, **kwargs)

    return model