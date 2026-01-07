import torch
import torch.nn as nn
import json
from models.resnet_hybrid import Bottleneck, BasicBlock
from models.conv_bn_relu import ConvBNReLU
from models.faster_rcnn import fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn
from models.mobilenet_custom import fasterrcnn_mobilenetv3_custom


def get_kept_indices(mask_handler):
    if mask_handler is None or (not hasattr(mask_handler, "s_mask")) or mask_handler.s_mask is None:
        return None
    mask = mask_handler.s_mask.mask
    idx = torch.nonzero(mask).squeeze()
    if idx.dim() == 0:
        idx = idx.unsqueeze(0)
    return idx


def get_layer(model, name):
    try:
        return dict(model.named_modules())[name]
    except KeyError:
        return None


def get_input_mask_resnet(masked_model, current_layer_name):
    # ResNet-only helper; MobileNet sẽ return None
    if "layer1.0.conv1" in current_layer_name:
        stem_conv = get_layer(masked_model, "backbone.body.conv1")
        return get_kept_indices(stem_conv)

    for i in range(2, 5):
        if f"layer{i}.0.conv1" in current_layer_name:
            try:
                prev_stage = getattr(masked_model.backbone.body, f"layer{i - 1}")
                last_block = prev_stage[-1]
                if isinstance(last_block, BasicBlock):
                    return get_kept_indices(last_block.conv2)
                elif isinstance(last_block, Bottleneck):
                    return get_kept_indices(last_block.conv3)
            except AttributeError:
                return None

    if "conv2" in current_layer_name:
        sibling_name = current_layer_name.replace("conv2", "conv1")
        sibling_module_name = ".".join(sibling_name.split(".")[:-2])
        mod = get_layer(masked_model, sibling_module_name)
        return get_kept_indices(mod)

    if "conv3" in current_layer_name:
        sibling_name = current_layer_name.replace("conv3", "conv2")
        sibling_module_name = ".".join(sibling_name.split(".")[:-2])
        mod = get_layer(masked_model, sibling_module_name)
        return get_kept_indices(mod)

    return None


def convert_to_lean_model(masked_model, save_path=None):
    print("Starting Model Surgery...")

    # ===== 1) extract rates =====
    backbone_compress_rates = []
    if hasattr(masked_model, "backbone") and hasattr(masked_model.backbone, "body"):
        for m in masked_model.backbone.body.modules():
            if isinstance(m, ConvBNReLU):
                if m.s_mask is not None:
                    mask = m.s_mask.mask
                    kept = mask.sum().item()
                    total = mask.numel()
                    backbone_compress_rates.append(1.0 - (kept / total))
                else:
                    backbone_compress_rates.append(0.0)

    fpn_compress_rates = []
    if hasattr(masked_model.backbone, "fpn") and hasattr(masked_model.backbone.fpn, "layer_blocks"):
        for layer_block in masked_model.backbone.fpn.layer_blocks:
            if hasattr(layer_block, "compress_layer"):
                m = layer_block.compress_layer
                if m.s_mask is not None:
                    mask = m.s_mask.mask
                    kept = mask.sum().item()
                    total = mask.numel()
                    fpn_compress_rates.append(1.0 - (kept / total))
                else:
                    fpn_compress_rates.append(0.0)

    print(f"Rates extracted: Backbone={len(backbone_compress_rates)}, FPN={len(fpn_compress_rates)}")

    # ===== 2) infer num_classes + box_head_dim =====
    num_classes = 2
    if hasattr(masked_model, "roi_heads"):
        num_classes = masked_model.roi_heads.box_predictor.cls_score.out_features

    box_head_dim = 1024
    try:
        box_head_dim = masked_model.roi_heads.box_head.fc6.out_features
    except Exception:
        pass

    # ===== 3) detect arch + create lean model =====
    first_block = None
    try:
        first_block = masked_model.backbone.body.layer1[0]
    except Exception:
        first_block = None

    try:
        if first_block and isinstance(first_block, Bottleneck):
            print("Detected ResNet50 Architecture")
            lean_model = fasterrcnn_resnet50_fpn(
                num_classes=num_classes,
                compress_rate=backbone_compress_rates,
                fpn_compress_rate=fpn_compress_rates
            )
        elif first_block and isinstance(first_block, BasicBlock):
            print("Detected ResNet18 Architecture")
            lean_model = fasterrcnn_resnet18_fpn(
                num_classes=num_classes,
                compress_rate=backbone_compress_rates,
                fpn_compress_rate=fpn_compress_rates
            )
        else:
            print("Detected MobileNetV3 (default)")
            lean_model = fasterrcnn_mobilenetv3_custom(
                num_classes=num_classes,
                fpn_compress_rate=fpn_compress_rates,
                pretrained_backbone=False,
                freeze_backbone=True,
                box_head_dim=box_head_dim
            )
    except Exception as e:
        print(f"Error creating lean model: {e}")
        return None

    # ===== 4) copy weights =====
    lean_sd = lean_model.state_dict()
    masked_sd = masked_model.state_dict()

    for name, lean_param in lean_sd.items():
        if name not in masked_sd:
            continue
        masked_param = masked_sd[name]

        # scalar (num_batches_tracked)
        if lean_param.dim() == 0:
            lean_param.data.copy_(masked_param.data)
            continue

        module_name = ".".join(name.split(".")[:-1])
        lean_module = get_layer(lean_model, module_name)

        # ---- FPN handling ----
        if "fpn.layer_blocks" in name:
            block_name = ".".join(module_name.split(".")[:-1])
            masked_block = get_layer(masked_model, block_name)

            if "dw_conv" in name:
                out_idx = get_kept_indices(masked_block.compress_layer)
                if out_idx is not None:
                    lean_param.data.copy_(masked_param.data[out_idx, ...])
                else:
                    lean_param.data.copy_(masked_param.data)
                continue

            if "dw_bn" in name:
                out_idx = get_kept_indices(masked_block.compress_layer)
                if out_idx is not None and lean_param.shape[0] == len(out_idx):
                    lean_param.data.copy_(masked_param.data[out_idx])
                else:
                    lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])
                continue

            if "expand_conv" in name:
                out_idx = get_kept_indices(masked_block.compress_layer)
                if out_idx is not None and lean_param.shape[1] == len(out_idx):
                    lean_param.data.copy_(masked_param.data[:, out_idx, :, :])
                else:
                    lean_param.data.copy_(masked_param.data[:, :lean_param.shape[1], :, :])
                continue

            if "expand_bn" in name:
                lean_param.data.copy_(masked_param.data)
                continue

        # ---- Standard Conv/BN ----
        if isinstance(lean_module, nn.Conv2d):
            # nếu shape khớp (MobileNet backbone freeze), copy thẳng
            if lean_param.shape == masked_param.shape:
                lean_param.data.copy_(masked_param.data)
                continue

            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)
            out_idx = get_kept_indices(parent_masked_module)

            if name.endswith(".bias"):
                if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                    lean_param.data.copy_(masked_param.data[out_idx])
                else:
                    lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])
                continue

            w_temp = masked_param.data
            if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                w_temp = w_temp[out_idx, :, :, :]
            else:
                w_temp = w_temp[:lean_param.shape[0], :, :, :]

            if lean_param.shape[1] < masked_param.shape[1]:
                in_idx = get_input_mask_resnet(masked_model, name)
                if in_idx is not None and len(in_idx) == lean_param.shape[1]:
                    lean_param.data.copy_(w_temp[:, in_idx, :, :])
                else:
                    lean_param.data.copy_(w_temp[:, :lean_param.shape[1], :, :])
            else:
                lean_param.data.copy_(w_temp)

        elif isinstance(lean_module, nn.BatchNorm2d):
            if lean_param.shape == masked_param.shape:
                lean_param.data.copy_(masked_param.data)
                continue

            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)
            out_idx = get_kept_indices(parent_masked_module)

            if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                lean_param.data.copy_(masked_param.data[out_idx])
            else:
                lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])

        else:
            if lean_param.shape == masked_param.shape:
                lean_param.data.copy_(masked_param.data)

    # ===== 5) save =====
    if save_path:
        torch.save(lean_model.state_dict(), save_path)
        cfg = {"backbone": backbone_compress_rates, "fpn": fpn_compress_rates}
        json_path = save_path.replace(".pth", ".json")
        with open(json_path, "w") as f:
            json.dump(cfg, f)
        print(f"Lean config saved to: {json_path}")

    print(f"Surgery Completed. Lean model saved to {save_path}")
    return lean_model
