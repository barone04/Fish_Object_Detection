import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor

from models.custom_fpn import PrunableFPN


class BackboneWithFPN(nn.Module):
    """
    Wrap MobileNet backbone + custom PrunableFPN.
    FasterRCNN yêu cầu backbone có thuộc tính out_channels.
    """

    def __init__(self, backbone, return_layers, in_channels_list, fpn_out_channels=256, fpn_compress_rate=None):
        super().__init__()
        self.body = IntermediateLayerGetter(backbone.features, return_layers)
        self.fpn = PrunableFPN(
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels,
            compress_rate=fpn_compress_rate
        )
        self.out_channels = fpn_out_channels

    def forward(self, x):
        x = self.body(x)   # OrderedDict {"0":..., "1":..., "2":..., "3":...}
        x = self.fpn(x)    # OrderedDict {"0","1","2","3","pool"}
        return x


def _mobilenet_v3_large_weights(pretrained_backbone: bool):
    if not pretrained_backbone:
        return None

    # Torchvision mới: dùng Weights enum
    try:
        from torchvision.models import MobileNet_V3_Large_Weights
        return MobileNet_V3_Large_Weights.DEFAULT
    except Exception:
        # Torchvision cũ: fallback
        return None


def fasterrcnn_mobilenetv3_custom(
    num_classes: int = 2,
    fpn_compress_rate=None,
    pretrained_backbone: bool = True,
    freeze_backbone: bool = True,
    fpn_out_channels: int = 256,
    min_size: int = 320,
    max_size: int = 320,
    box_head_dim: int = 1024,
):
    weights = _mobilenet_v3_large_weights(pretrained_backbone)
    print(f"Loading MobileNetV3 Large (pretrained_backbone={pretrained_backbone}, weights={type(weights)})...")

    if weights is not None:
        backbone_model = torchvision.models.mobilenet_v3_large(weights=weights)
    else:
        # fallback cho torchvision cũ
        backbone_model = torchvision.models.mobilenet_v3_large(pretrained=pretrained_backbone)

    if freeze_backbone:
        for p in backbone_model.parameters():
            p.requires_grad = False
        print(" -> Backbone FROZEN (requires_grad=False).")

    # ====== Chọn 4 feature levels để khớp chuẩn FasterRCNN-FPN ======
    # Lưu ý: keys là index của backbone.features (string)
    # Đây là lựa chọn phổ biến (stride tăng dần); nếu torchvision thay đổi, vẫn OK vì ta sẽ đo channels bằng dummy.
    return_layers = {"2": "0", "5": "1", "12": "2", "16": "3"}

    # ====== Tự đo in_channels_list để tránh sai channels do version torchvision ======
    body = IntermediateLayerGetter(backbone_model.features, return_layers=return_layers)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        feats = body(dummy)
        in_channels_list = [v.shape[1] for v in feats.values()]

    print(f" -> return_layers={return_layers} | in_channels_list={in_channels_list}")

    backbone_with_fpn = BackboneWithFPN(
        backbone=backbone_model,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        fpn_out_channels=fpn_out_channels,
        fpn_compress_rate=fpn_compress_rate
    )

    # ====== RPN anchor generator: phải khớp 5 feature maps (0,1,2,3,pool) ======
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # ====== ROI Align chỉ dùng 4 maps (0..3), KHÔNG dùng pool ======
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2
    )

    # ====== Box head: giảm box_head_dim sẽ giảm size rất mạnh (vì FC layers cực lớn) ======
    resolution = box_roi_pool.output_size[0]
    box_in_channels = backbone_with_fpn.out_channels * resolution * resolution
    box_head = TwoMLPHead(in_channels=box_in_channels, representation_size=box_head_dim)
    box_predictor = FastRCNNPredictor(in_channels=box_head_dim, num_classes=num_classes)

    model = FasterRCNN(
        backbone_with_fpn,
        num_classes=None,  # vì ta đã truyền box_predictor
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        box_predictor=box_predictor,
        min_size=min_size,
        max_size=max_size,
    )

    return model
