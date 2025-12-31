import torch
import torch.nn as nn
import json
import os
from models.resnet_hybrid import Bottleneck, BasicBlock
from models.conv_bn_relu import ConvBNReLU
from models.faster_rcnn import fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn


def get_kept_indices(mask_handler):
    if mask_handler is None or mask_handler.s_mask is None:
        return None
    mask = mask_handler.s_mask.mask
    indices = torch.nonzero(mask).squeeze()
    if indices.dim() == 0: indices = indices.unsqueeze(0)
    return indices


def get_layer(model, name):
    return dict(model.named_modules())[name]


def convert_to_lean_model(masked_model, save_path=None):
    print("Starting Model Surgery...")

    backbone_compress_rates = []
    if hasattr(masked_model, 'backbone') and hasattr(masked_model.backbone, 'body'):
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
    if hasattr(masked_model.backbone, 'fpn') and hasattr(masked_model.backbone.fpn, 'layer_blocks'):
        for layer_block in masked_model.backbone.fpn.layer_blocks:
            # layer_block là BottleneckFPNBlock -> chứa compress_layer
            if hasattr(layer_block, 'compress_layer'):
                m = layer_block.compress_layer
                if m.s_mask is not None:
                    mask = m.s_mask.mask
                    kept = mask.sum().item()
                    total = mask.numel()
                    fpn_compress_rates.append(1.0 - (kept / total))
                else:
                    fpn_compress_rates.append(0.0)

    print(f"Rates extracted: Backbone={len(backbone_compress_rates)}, FPN={len(fpn_compress_rates)}")

    first_block = masked_model.backbone.body.layer1[0]
    if hasattr(masked_model, 'roi_heads'):
        num_classes = masked_model.roi_heads.box_predictor.cls_score.out_features
    else:
        num_classes = 2  # Fallback

    try:
        if isinstance(first_block, Bottleneck):
            print("Detected ResNet50 Architecture")
            lean_model = fasterrcnn_resnet50_fpn(
                num_classes=num_classes,
                compress_rate=backbone_compress_rates,
                fpn_compress_rate=fpn_compress_rates
            )
        elif isinstance(first_block, BasicBlock):
            print("Detected ResNet18 Architecture")
            lean_model = fasterrcnn_resnet18_fpn(
                num_classes=num_classes,
                compress_rate=backbone_compress_rates,
                fpn_compress_rate=fpn_compress_rates
            )
        else:
            print("Unknown architecture!")
            return None
    except Exception as e:
        print(f"Error creating lean model: {e}")
        return None

    lean_state_dict = lean_model.state_dict()
    masked_state_dict = masked_model.state_dict()

    for name, lean_param in lean_state_dict.items():
        if name not in masked_state_dict:
            continue

        masked_param = masked_state_dict[name]
        module_name = ".".join(name.split(".")[:-1])

        if "fpn.layer_blocks" in name:
            block_name = ".".join(module_name.split(".")[:-1])
            masked_block = get_layer(masked_model, block_name)

            if "compress_layer" in name and "conv" in name:
                pass

            elif "dw_conv" in name:
                out_idx = get_kept_indices(masked_block.compress_layer)
                if out_idx is not None:
                    if lean_param.shape[0] == len(out_idx):
                        lean_param.data.copy_(masked_param.data[out_idx, :, :, :])
                    else:
                        # Fallback an toàn
                        lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])
                else:
                    lean_param.data.copy_(masked_param.data)
                continue

            elif "dw_bn" in name:
                out_idx = get_kept_indices(masked_block.compress_layer)
                if out_idx is not None and lean_param.shape[0] == len(out_idx):
                    lean_param.data.copy_(masked_param.data[out_idx])
                else:
                    lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])
                continue

            elif "expand_conv" in name:
                out_idx = get_kept_indices(masked_block.compress_layer)
                if out_idx is not None and lean_param.shape[1] == len(out_idx):
                    lean_param.data.copy_(masked_param.data[:, out_idx, :, :])
                else:
                    lean_param.data.copy_(masked_param.data[:, :lean_param.shape[1], :, :])
                continue

        lean_module = get_layer(lean_model, module_name)

        if lean_param.shape == masked_param.shape:
            lean_param.data.copy_(masked_param.data)
            continue

        if isinstance(lean_module, nn.Conv2d):
            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)

            out_idx = get_kept_indices(parent_masked_module)

            if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                w_temp = masked_param.data[out_idx, :, :, :]

                if lean_param.shape[1] < masked_param.shape[1]:
                    lean_param.data.copy_(w_temp[:, :lean_param.shape[1], :, :])
                else:
                    lean_param.data.copy_(w_temp)
            else:
                # Fallback: Slice góc trái trên
                print(f"Warning: Slicing weights for {name}")
                lean_param.data.copy_(masked_param.data[:lean_param.shape[0], :lean_param.shape[1], :, :])

        # Case 3: BatchNorm
        elif isinstance(lean_module, nn.BatchNorm2d):
            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)
            out_idx = get_kept_indices(parent_masked_module)

            if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                lean_param.data.copy_(masked_param.data[out_idx])
            else:
                lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])

    if save_path:
        # Save Model
        torch.save(lean_model.state_dict(), save_path)

        # Save Config JSON (Format mới)
        config_data = {
            'backbone': backbone_compress_rates,
            'fpn': fpn_compress_rates
        }
        json_path = save_path.replace('.pth', '.json')
        with open(json_path, 'w') as f:
            json.dump(config_data, f)
        print(f"Lean config saved to: {json_path}")

    print(f"Surgery Completed. Lean model saved to {save_path}")
    return lean_model