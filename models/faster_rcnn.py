import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models._utils import IntermediateLayerGetter
from .custom_fpn import PrunableFPN
from .resnet_hybrid import resnet_18, resnet_50


def _create_faster_rcnn_hybrid(backbone_body, num_classes, weights_backbone, fpn_compress_rate, **kwargs):
    """Hàm helper dùng chung cho cả R18 và R50"""
    dummy = torch.randn(1, 3, 224, 224)
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    body_extractor = IntermediateLayerGetter(backbone_body, return_layers=return_layers)
    with torch.no_grad():
        feats = body_extractor(dummy)
        in_channels_list = [v.shape[1] for k, v in feats.items()]

    custom_fpn = PrunableFPN(in_channels_list, out_channels=256, compress_rate=fpn_compress_rate)
    backbone_with_fpn = BackboneWithFPN(backbone_body, return_layers, [0, 1, 2, 3], None, custom_fpn)
    model = FasterRCNN(backbone_with_fpn, num_classes=num_classes, **kwargs)

    if weights_backbone is not None and str(weights_backbone) != "DEFAULT":
        print(f"Loading weights from: {weights_backbone}")
        try:
            state_dict = torch.load(weights_backbone, map_location='cpu')
            if 'model' in state_dict: state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)
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