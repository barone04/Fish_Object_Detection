import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models._utils import IntermediateLayerGetter
from .custom_fpn import PrunableFPN
from .resnet_hybrid import resnet_18, resnet_50


class BackboneWithCustomFPN(nn.Module):
    """
    Class này ghép nối Backbone (ResNet) và FPN (PrunableFPN).
    Nó thay thế cho torchvision.models.detection.backbone_utils.BackboneWithFPN
    để cho phép dùng Custom FPN.
    """

    def __init__(self, backbone, return_layers, fpn, out_channels):
        super(BackboneWithCustomFPN, self).__init__()
        # Trích xuất các layer từ backbone (C1, C2, C3, C4...)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # Gắn module FPN tuỳ chỉnh vào
        self.fpn = fpn
        # FasterRCNN yêu cầu thuộc tính này để biết output size
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def _create_faster_rcnn_hybrid(backbone_body, num_classes, weights_backbone, fpn_compress_rate, **kwargs):
    """Hàm helper dùng chung cho cả R18 và R50"""

    # 1. Xác định số kênh input cho FPN
    dummy = torch.randn(1, 3, 224, 224)
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    # Dùng IntermediateLayerGetter để chạy thử và lấy số channels
    body_extractor = IntermediateLayerGetter(backbone_body, return_layers=return_layers)
    with torch.no_grad():
        feats = body_extractor(dummy)
        # feats là dict, ta lấy values và lấy shape[1] (số channels)
        in_channels_list = [v.shape[1] for k, v in feats.items()]

    # 2. Khởi tạo Custom FPN (Prunable)
    custom_fpn = PrunableFPN(in_channels_list, out_channels=256, compress_rate=fpn_compress_rate)

    # 3. Ghép Backbone + FPN bằng Class tự viết (ĐÃ SỬA LỖI TẠI ĐÂY)
    # Không dùng BackboneWithFPN của torchvision nữa
    backbone_with_fpn = BackboneWithCustomFPN(
        backbone=backbone_body,
        return_layers=return_layers,
        fpn=custom_fpn,
        out_channels=256
    )

    # 4. Tạo Faster R-CNN
    model = FasterRCNN(backbone_with_fpn, num_classes=num_classes, **kwargs)

    # 5. Load weights Backbone (nếu có)
    if weights_backbone is not None and str(weights_backbone) != "DEFAULT":
        print(f"Loading weights from: {weights_backbone}")
        try:
            state_dict = torch.load(weights_backbone, map_location='cpu')
            if 'model' in state_dict: state_dict = state_dict['model']

            # Load flexible (bỏ qua layer không khớp nếu cần)
            model.backbone.body.load_state_dict(state_dict, strict=False)
            print("Backbone weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load weights fully ({e})")

    return model


def fasterrcnn_resnet18_fpn(weights_backbone=None, num_classes=91, compress_rate=None, fpn_compress_rate=None,
                            **kwargs):
    # Init Backbone R18
    init_weights = "DEFAULT" if weights_backbone is None else None
    backbone = resnet_18(compress_rate=compress_rate, weights=init_weights)
    return _create_faster_rcnn_hybrid(backbone, num_classes, weights_backbone, fpn_compress_rate, **kwargs)


def fasterrcnn_resnet50_fpn(weights_backbone=None, num_classes=91, compress_rate=None, fpn_compress_rate=None,
                            **kwargs):
    # Init Backbone R50
    init_weights = "DEFAULT" if weights_backbone is None else None
    backbone = resnet_50(compress_rate=compress_rate, weights=init_weights)
    return _create_faster_rcnn_hybrid(backbone, num_classes, weights_backbone, fpn_compress_rate, **kwargs)