import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FasterRCNN, TwoMLPHead, FastRCNNPredictor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from .custom_fpn import PrunableFPN
from .resnet_hybrid import resnet_18, resnet_50


class BackboneWithCustomFPN(nn.Module):
    """
    Class này ghép nối Backbone (ResNet) và FPN (PrunableFPN).
    """

    def __init__(self, backbone, return_layers, fpn, out_channels):
        super(BackboneWithCustomFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = fpn
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def _create_faster_rcnn_hybrid(backbone_body, num_classes, weights_backbone, fpn_compress_rate, **kwargs):
    """Hàm helper dùng chung cho cả R18 và R50"""
    dummy = torch.randn(1, 3, 320, 320)
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    body_extractor = IntermediateLayerGetter(backbone_body, return_layers=return_layers)
    with torch.no_grad():
        feats = body_extractor(dummy)
        in_channels_list = [v.shape[1] for k, v in feats.items()]

    custom_fpn = PrunableFPN(in_channels_list, out_channels=256, compress_rate=fpn_compress_rate)
    backbone_with_fpn = BackboneWithCustomFPN(
        backbone=backbone_body,
        return_layers=return_layers,
        fpn=custom_fpn,
        out_channels=256
    )


    # Lấy box_head_dim ra khỏi kwargs để ép mô hình dùng size nhỏ (256)
    box_head_dim = kwargs.pop('box_head_dim', 256)

    # Anchor thu nhỏ lại để bắt được cá nhỏ trên ảnh 320x320
    # anchor_generator = AnchorGenerator(
    #     sizes=((16,), (32,), (64,), (128,), (256,)),
    #     aspect_ratios=((0.5, 1.0, 2.0),) * 5
    # )
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),  # Thu nhỏ toàn bộ dàn anchor
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    resolution = box_roi_pool.output_size[0]
    box_in_channels = backbone_with_fpn.out_channels * resolution * resolution
    box_head = TwoMLPHead(in_channels=box_in_channels, representation_size=box_head_dim)
    box_predictor = FastRCNNPredictor(in_channels=box_head_dim, num_classes=num_classes)

    # Khởi tạo mô hình FasterRCNN với các component đã ép kích thước
    model = FasterRCNN(
        backbone_with_fpn,
        # num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        box_predictor=box_predictor,
        **kwargs  # Lúc này kwargs chỉ còn min_size=320, max_size=320
    )
    # =========================================================================

    # 5. Load weights Backbone
    if weights_backbone is not None and str(weights_backbone) != "DEFAULT":
        print(f"Loading weights from: {weights_backbone}")
        try:
            state_dict = torch.load(weights_backbone, map_location='cpu')
            if 'model' in state_dict: state_dict = state_dict['model']

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