import torch
import torch.nn as nn
import json
import os
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
    if "layer1.0.conv1" in current_layer_name:
        stem_conv = get_layer(masked_model, "backbone.body.conv1")
        return get_kept_indices(stem_conv)

    for i in range(2, 5):
        if f"layer{i}.0.conv1" in current_layer_name:
            try:
                prev_stage = getattr(masked_model.backbone.body, f"layer{i - 1}")
                last_block = list(prev_stage.children())[-1]
                # ResNet BasicBlock vs Bottleneck
                if isinstance(last_block, BasicBlock):
                    return get_kept_indices(last_block.conv2)
                elif isinstance(last_block, Bottleneck):
                    return get_kept_indices(last_block.conv3)
            except:
                return None
    return None


def convert_to_lean_model(masked_model, save_path=None):
    masked_model.eval()
    device = next(masked_model.parameters()).device

    backbone_compress_rates = []
    fpn_compress_rates = []

    is_mobilenet = "MobileNet" in str(type(masked_model.backbone)) or (
                hasattr(masked_model.backbone, 'body') and "MobileNet" in str(type(masked_model.backbone.body)))

    print(f"Surgery detected model type: {'MobileNetV3' if is_mobilenet else 'ResNet'}")

    if not is_mobilenet:
        if hasattr(masked_model.backbone, "body"):
            body = masked_model.backbone.body
            for stage_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if hasattr(body, stage_name):
                    stage = getattr(body, stage_name)
                    for block in stage:
                        if isinstance(block, Bottleneck):
                            # Conv1
                            idx1 = get_kept_indices(block.conv1)
                            rate1 = 1.0 - (len(idx1) / block.conv1.out_channels) if idx1 is not None else 0.0
                            backbone_compress_rates.append(rate1)
                            # Conv2
                            idx2 = get_kept_indices(block.conv2)
                            rate2 = 1.0 - (len(idx2) / block.conv2.out_channels) if idx2 is not None else 0.0
                            backbone_compress_rates.append(rate2)

                        elif isinstance(block, BasicBlock):
                            idx1 = get_kept_indices(block.conv1)
                            rate1 = 1.0 - (len(idx1) / block.conv1.out_channels) if idx1 is not None else 0.0
                            backbone_compress_rates.append(rate1)
    else:
        backbone_compress_rates = None

    if hasattr(masked_model.backbone, "fpn"):
        fpn = masked_model.backbone.fpn
        if hasattr(fpn, "layer_blocks"):
            for block in fpn.layer_blocks:
                if hasattr(block, "compress_layer"):
                    idx = get_kept_indices(block.compress_layer)
                    full_dim = block.compress_layer.out_channels
                    if idx is not None:
                        current_dim = len(idx)
                        rate = 1.0 - (float(current_dim) / float(full_dim))
                    else:
                        rate = 0.0
                    fpn_compress_rates.append(rate)

    print(f"Calculated FPN Compress Rates: {fpn_compress_rates}")

    try:
        if hasattr(masked_model, 'roi_heads'):
            num_classes = masked_model.roi_heads.box_predictor.cls_score.out_features
        else:
            num_classes = 2
    except:
        num_classes = 2

    if is_mobilenet:
        print("Initializing Lean MobileNetV3...")
        lean_model = fasterrcnn_mobilenetv3_custom(
            num_classes=num_classes,
            fpn_compress_rate=fpn_compress_rates,
            pretrained_backbone=False
        )
    else:
        # Logic cũ cho ResNet
        if len(backbone_compress_rates) > 20:
            print("Initializing Lean ResNet50...")
            lean_model = fasterrcnn_resnet50_fpn(
                num_classes=num_classes,
                compress_rate=backbone_compress_rates,
                fpn_compress_rate=fpn_compress_rates,
                weights_backbone=None
            )
        else:
            print("Initializing Lean ResNet18...")
            lean_model = fasterrcnn_resnet18_fpn(
                num_classes=num_classes,
                compress_rate=backbone_compress_rates,
                fpn_compress_rate=fpn_compress_rates,
                weights_backbone=None
            )

    lean_model.to(device)
    lean_model.eval()

    print("Copying weights from Masked Model to Lean Model...")

    with torch.no_grad():
        for name, lean_param in lean_model.named_parameters():
            masked_param = get_layer(masked_model, name)
            if masked_param is None:
                if name in masked_model.state_dict():
                    masked_param = masked_model.state_dict()[name]
                else:
                    print(f"Warning: {name} not found in masked model. Skipping.")
                    continue
            else:
                pass

        lean_modules = dict(lean_model.named_modules())

        for module_name, lean_module in lean_modules.items():
            if not isinstance(lean_module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                continue

            masked_module = get_layer(masked_model, module_name)
            if masked_module is None: continue

            if isinstance(lean_module, nn.Conv2d):
                out_idx = get_kept_indices(masked_module)

                if isinstance(masked_module, ConvBNReLU):
                    pass

                parent_name = ".".join(module_name.split(".")[:-1])
                parent_module = get_layer(masked_model, parent_name)

                if isinstance(parent_module, ConvBNReLU):
                    out_idx = get_kept_indices(parent_module)
                else:
                    out_idx = None

                in_idx = None

                if "dw_conv" in module_name:
                    block_name = ".".join(module_name.split(".")[:-1])
                    block = get_layer(masked_model, block_name)
                    if block and hasattr(block, "compress_layer"):
                        in_idx = get_kept_indices(block.compress_layer)

                elif "expand_conv" in module_name:
                    block_name = ".".join(module_name.split(".")[:-1])
                    block = get_layer(masked_model, block_name)
                    if block and hasattr(block, "compress_layer"):
                        in_idx = get_kept_indices(block.compress_layer)

                elif not is_mobilenet:
                    in_idx = get_input_mask_resnet(masked_model, module_name)

                w_masked = masked_module.weight.data
                w_lean = lean_module.weight.data

                # Copy theo Out Channel
                if out_idx is not None and len(out_idx) == w_lean.shape[0]:
                    w_temp = w_masked[out_idx, :, :, :]
                else:
                    w_temp = w_masked  # Không bị prune output

                # Copy theo In Channel
                if in_idx is not None:
                    if lean_module.groups > 1 and lean_module.groups == lean_module.in_channels:
                        pass
                    elif len(in_idx) == w_lean.shape[1]:
                        w_temp = w_temp[:, in_idx, :, :]

                if w_temp.shape == w_lean.shape:
                    w_lean.copy_(w_temp)
                else:
                    if w_temp.shape == w_lean.shape:
                        w_lean.copy_(w_temp)
                    else:
                        min_out = min(w_temp.shape[0], w_lean.shape[0])
                        min_in = min(w_temp.shape[1], w_lean.shape[1])
                        w_lean[:min_out, :min_in, :, :].copy_(w_temp[:min_out, :min_in, :, :])

                if lean_module.bias is not None and masked_module.bias is not None:
                    b_masked = masked_module.bias.data
                    if out_idx is not None and len(out_idx) == lean_module.bias.shape[0]:
                        lean_module.bias.data.copy_(b_masked[out_idx])
                    else:
                        min_b = min(b_masked.shape[0], lean_module.bias.shape[0])
                        lean_module.bias.data[:min_b].copy_(b_masked[:min_b])

            elif isinstance(lean_module, nn.BatchNorm2d):
                parent_name = ".".join(module_name.split(".")[:-1])
                parent_module = get_layer(masked_model, parent_name)

                mask_idx = None

                if isinstance(parent_module, ConvBNReLU):
                    mask_idx = get_kept_indices(parent_module)
                elif "dw_bn" in module_name:
                    block_name = ".".join(module_name.split(".")[:-1])
                    block = get_layer(masked_model, block_name)
                    if block and hasattr(block, "compress_layer"):
                        mask_idx = get_kept_indices(block.compress_layer)

                for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                    src = getattr(masked_module, attr)
                    dst = getattr(lean_module, attr)

                    if mask_idx is not None and len(mask_idx) == dst.shape[0]:
                        dst.data.copy_(src.data[mask_idx])
                    else:
                        dst.data.copy_(src.data)

            elif isinstance(lean_module, nn.Linear):
                if lean_module.weight.shape == masked_module.weight.shape:
                    lean_module.weight.data.copy_(masked_module.weight.data)
                    if lean_module.bias is not None:
                        lean_module.bias.data.copy_(masked_module.bias.data)

    if save_path:
        print(f"Saving Lean Model to {save_path}")
        torch.save(lean_model.state_dict(), save_path)

        config_data = {'backbone': backbone_compress_rates, 'fpn': fpn_compress_rates}
        json_path = save_path.replace('.pth', '.json')
        with open(json_path, 'w') as f:
            json.dump(config_data, f)
        print(f"Config saved to {json_path}")

    return lean_model