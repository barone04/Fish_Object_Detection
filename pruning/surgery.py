import os
import json
import torch
import torch.nn as nn

from models.resnet_hybrid import Bottleneck, BasicBlock
from models.conv_bn_relu import ConvBNReLU
from models.faster_rcnn import fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn
from models.mobilenet_custom import fasterrcnn_mobilenetv3_custom


# -------------------------
# Helpers
# -------------------------
def get_kept_indices(mask_handler):
    """
    Trả về indices của các channel còn sống theo s_mask.
    Hỗ trợ ConvBNReLU (có s_mask) hoặc các object có trực tiếp `.mask`.
    """
    if mask_handler is None:
        return None

    # ConvBNReLU style
    if hasattr(mask_handler, "s_mask") and mask_handler.s_mask is not None:
        mask = mask_handler.s_mask.mask
    # Direct mask style
    elif hasattr(mask_handler, "mask") and mask_handler.mask is not None:
        mask = mask_handler.mask
    else:
        return None

    idx = torch.nonzero(mask).squeeze()
    if idx.dim() == 0:
        idx = idx.unsqueeze(0)
    return idx


def get_layer(model, name):
    try:
        return dict(model.named_modules())[name]
    except KeyError:
        return None


def infer_is_mobilenet(masked_model) -> bool:
    """
    Detect MobileNetV3 robustly (tránh case MobileNet bị wrapper làm mất chữ 'MobileNet').
    - ResNet-hybrid có backbone.body.layer1[0] là BasicBlock/Bottleneck.
    - MobileNet backbone.body thường là IntermediateLayerGetter/ModuleDict và KHÔNG có layer1.
    """
    bb = getattr(masked_model, "backbone", None)
    if bb is None:
        return False

    body = getattr(bb, "body", None)
    if body is None:
        # Một số cấu trúc backbone không có body -> coi như mobilenet/custom
        return True

    # ResNet-hybrid: có layer1
    if hasattr(body, "layer1"):
        return False

    # Không có layer1 -> gần như chắc MobileNet (IntermediateLayerGetter/ModuleDict)
    return True


def get_input_mask_resnet(masked_model, current_layer_name):
    """
    Logic cũ topology-aware input mask cho ResNet18/50.
    """
    # 1) layer1.0.conv1 nhận input từ stem conv1
    if "layer1.0.conv1" in current_layer_name:
        stem_conv = get_layer(masked_model, "backbone.body.conv1")
        return get_kept_indices(stem_conv)

    # 2) layer2.0.conv1 nhận input từ output cuối stage trước
    for i in range(2, 5):
        if f"layer{i}.0.conv1" in current_layer_name:
            prev_stage = getattr(masked_model.backbone.body, f"layer{i - 1}")
            last_block = prev_stage[-1]
            if isinstance(last_block, BasicBlock):
                return get_kept_indices(last_block.conv2)
            elif isinstance(last_block, Bottleneck):
                return get_kept_indices(last_block.conv3)

    # 3) fallback nội bộ block
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


def infer_box_head_dim(masked_model, default=1024) -> int:
    """
    Lấy box_head_dim thực tế từ model masked để tránh mismatch (1024 vs 256).
    """
    try:
        box_head = masked_model.roi_heads.box_head
        if hasattr(box_head, "fc6"):
            return int(box_head.fc6.out_features)
    except Exception:
        pass
    return int(default)


# -------------------------
# Main: convert_to_lean_model
# -------------------------
def convert_to_lean_model(masked_model, save_path=None):
    """
    - ResNet18/50: giữ nguyên logic cũ (extract rates + init + copy)
    - MobileNetV3: detect đúng + init lean mobilenet + copy FPN theo mask
    - Save: model_lean.pth + model_lean.json
    """
    print("Starting Model Surgery...")
    masked_model.eval()
    device = next(masked_model.parameters()).device

    is_mobilenet = infer_is_mobilenet(masked_model)
    print(f"Surgery detected model type: {'MobileNetV3' if is_mobilenet else 'ResNet'}")

    # --- 1) EXTRACT CONFIG/RATES (giữ nguyên logic cũ cho ResNet) ---
    backbone_compress_rates = []
    if not is_mobilenet:
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
    else:
        backbone_compress_rates = None  # MobileNet freeze backbone

    fpn_compress_rates = []
    if hasattr(masked_model, "backbone") and hasattr(masked_model.backbone, "fpn") and hasattr(masked_model.backbone.fpn, "layer_blocks"):
        for layer_block in masked_model.backbone.fpn.layer_blocks:
            if hasattr(layer_block, "compress_layer"):
                m = layer_block.compress_layer
                if hasattr(m, "s_mask") and m.s_mask is not None:
                    mask = m.s_mask.mask
                    kept = mask.sum().item()
                    total = mask.numel()
                    fpn_compress_rates.append(1.0 - (kept / total))
                else:
                    fpn_compress_rates.append(0.0)

    print(f"Rates extracted: Backbone={(len(backbone_compress_rates) if backbone_compress_rates is not None else 0)}, FPN={len(fpn_compress_rates)}")

    # --- 2) INIT LEAN MODEL ---
    num_classes = 2
    if hasattr(masked_model, "roi_heads"):
        num_classes = masked_model.roi_heads.box_predictor.cls_score.out_features

    box_head_dim = infer_box_head_dim(masked_model, default=1024)

    try:
        if is_mobilenet:
            print(f"Initializing Lean MobileNetV3 (box_head_dim={box_head_dim})...")
            # NOTE: builder mobilenet_custom PHẢI hỗ trợ box_head_dim để tránh mismatch
            lean_model = fasterrcnn_mobilenetv3_custom(
                num_classes=num_classes,
                fpn_compress_rate=fpn_compress_rates,
                pretrained_backbone=False,
                freeze_backbone=True,
                box_head_dim=box_head_dim,
                min_size=320,
                max_size=320
            )
        else:
            # GIỮ NGUYÊN LOGIC CŨ: xác định ResNet18/50 bằng type block
            first_block = masked_model.backbone.body.layer1[0]
            if isinstance(first_block, Bottleneck):
                print("Initializing Lean ResNet50...")
                lean_model = fasterrcnn_resnet50_fpn(
                    num_classes=num_classes,
                    compress_rate=backbone_compress_rates,
                    fpn_compress_rate=fpn_compress_rates,
                    min_size=320,
                    max_size=320
                )
            elif isinstance(first_block, BasicBlock):
                print("Initializing Lean ResNet18...")
                lean_model = fasterrcnn_resnet18_fpn(
                    num_classes=num_classes,
                    compress_rate=backbone_compress_rates,
                    fpn_compress_rate=fpn_compress_rates,
                    min_size=320,
                    max_size=320
                )
            else:
                print("Unknown architecture in ResNet branch!")
                return None
    except Exception as e:
        print(f"Error creating lean model: {e}")
        return None

    lean_model.to(device)
    lean_model.eval()

    # --- 3) COPY WEIGHTS (GIỮ NGUYÊN STYLE CŨ THEO state_dict; chỉ thêm MobileNet-safe copy) ---
    lean_state_dict = lean_model.state_dict()
    masked_state_dict = masked_model.state_dict()

    for name, lean_param in lean_state_dict.items():
        if name not in masked_state_dict:
            continue

        masked_param = masked_state_dict[name]

        # Scalar protection (num_batches_tracked)
        if lean_param.dim() == 0:
            lean_param.data.copy_(masked_param.data)
            continue

        module_name = ".".join(name.split(".")[:-1])
        lean_module = get_layer(lean_model, module_name)

        # -------------------------
        # A) FPN HANDLING (giữ logic cũ; dùng mask từ compress_layer)
        # -------------------------
        if "fpn.layer_blocks" in name:
            block_name = ".".join(module_name.split(".")[:-1])  # e.g. backbone.fpn.layer_blocks.0
            masked_block = get_layer(masked_model, block_name)

            # compress_layer.conv / dw_conv / expand_conv... tùy naming trong custom_fpn
            if masked_block is None or not hasattr(masked_block, "compress_layer"):
                # fallback: nếu không bắt được block, copy naive nếu shape khớp
                if lean_param.shape == masked_param.shape:
                    lean_param.data.copy_(masked_param.data)
                continue

            out_idx = get_kept_indices(masked_block.compress_layer)

            # Depthwise conv prune theo output channels
            if "dw_conv" in name:
                if out_idx is not None:
                    if lean_param.dim() == 4:
                        lean_param.data.copy_(masked_param.data[out_idx, :, :, :])
                    else:
                        lean_param.data.copy_(masked_param.data[out_idx])
                else:
                    lean_param.data.copy_(masked_param.data)
                continue

            # Depthwise BN prune theo out_idx
            if "dw_bn" in name:
                if out_idx is None:
                    lean_param.data.copy_(masked_param.data[: lean_param.shape[0]])
                else:
                    if lean_param.shape[0] == len(out_idx):
                        lean_param.data.copy_(masked_param.data[out_idx])
                    else:
                        # fallback safe slice
                        lean_param.data.copy_(masked_param.data[: lean_param.shape[0]])
                continue

            # expand_conv: prune theo input channels (dim=1)
            if "expand_conv" in name:
                if out_idx is not None and lean_param.dim() == 4:
                    if lean_param.shape[1] == len(out_idx):
                        lean_param.data.copy_(masked_param.data[:, out_idx, :, :])
                    else:
                        lean_param.data.copy_(masked_param.data[:, : lean_param.shape[1], :, :])
                else:
                    # no pruning info -> copy min slice
                    if lean_param.shape == masked_param.shape:
                        lean_param.data.copy_(masked_param.data)
                    else:
                        lean_param.data.copy_(masked_param.data[:, : lean_param.shape[1], :, :])
                continue

            # expand_bn: thường không prune theo compress_layer -> copy slice-safe
            if "expand_bn" in name:
                if lean_param.shape == masked_param.shape:
                    lean_param.data.copy_(masked_param.data)
                else:
                    lean_param.data.copy_(masked_param.data[: lean_param.shape[0]])
                continue

            # compress_layer conv/bn: (nếu naming có)
            if "compress_layer" in name:
                if out_idx is not None:
                    # conv weight
                    if lean_param.dim() == 4 and lean_param.shape[0] == len(out_idx):
                        lean_param.data.copy_(masked_param.data[out_idx, :, :, :])
                    # bn vectors
                    elif lean_param.dim() == 1 and lean_param.shape[0] == len(out_idx):
                        lean_param.data.copy_(masked_param.data[out_idx])
                    else:
                        # fallback
                        lean_param.data.copy_(masked_param.data[: lean_param.shape[0]])
                else:
                    # no mask => copy as much as possible
                    if lean_param.shape == masked_param.shape:
                        lean_param.data.copy_(masked_param.data)
                    else:
                        lean_param.data.copy_(masked_param.data[: lean_param.shape[0]])
                continue

            # Default fallback for any other fpn params
            if lean_param.shape == masked_param.shape:
                lean_param.data.copy_(masked_param.data)
            else:
                # minimal safe slice on dim0
                lean_param.data.copy_(masked_param.data[: lean_param.shape[0]])
            continue

        # -------------------------
        # B) STANDARD CONV/BN HANDLING
        #    - ResNet: giữ nguyên logic cũ (output mask + input mask topology-aware)
        #    - MobileNet: copy thẳng nếu shape khớp (backbone freeze)
        # -------------------------
        if isinstance(lean_module, nn.Conv2d):
            if is_mobilenet:
                # MobileNet backbone không prune -> shapes nên khớp
                if lean_param.shape == masked_param.shape:
                    lean_param.data.copy_(masked_param.data)
                else:
                    # fallback: slice-safe theo min dims
                    if lean_param.dim() == 4 and masked_param.dim() == 4:
                        oc = min(lean_param.shape[0], masked_param.shape[0])
                        ic = min(lean_param.shape[1], masked_param.shape[1])
                        lean_param.data.copy_(masked_param.data[:oc, :ic, :, :])
                continue

            # -------- ResNet old logic --------
            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)
            out_idx = get_kept_indices(parent_masked_module)

            if "bias" in name:
                if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                    lean_param.data.copy_(masked_param.data[out_idx])
                else:
                    lean_param.data.copy_(masked_param.data[: lean_param.shape[0]])
                continue

            w_temp = masked_param.data
            if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                w_temp = w_temp[out_idx, :, :, :]
            else:
                w_temp = w_temp[: lean_param.shape[0], :, :, :]

            if lean_param.shape[1] < masked_param.shape[1]:
                in_idx = get_input_mask_resnet(masked_model, name)
                if in_idx is not None and len(in_idx) == lean_param.shape[1]:
                    lean_param.data.copy_(w_temp[:, in_idx, :, :])
                else:
                    # fallback slice
                    lean_param.data.copy_(w_temp[:, : lean_param.shape[1], :, :])
            else:
                lean_param.data.copy_(w_temp)

        elif isinstance(lean_module, nn.BatchNorm2d):
            if is_mobilenet:
                if lean_param.shape == masked_param.shape:
                    lean_param.data.copy_(masked_param.data)
                else:
                    lean_param.data.copy_(masked_param.data[: lean_param.shape[0]])
                continue

            # -------- ResNet old logic --------
            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)
            out_idx = get_kept_indices(parent_masked_module)

            if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                lean_param.data.copy_(masked_param.data[out_idx])
            else:
                lean_param.data.copy_(masked_param.data[: lean_param.shape[0]])

        else:
            # Linear / others: copy if shape matches
            if lean_param.shape == masked_param.shape:
                lean_param.data.copy_(masked_param.data)

    # --- 4) SAVE ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(lean_model.state_dict(), save_path)

        config_data = {
            "backbone": backbone_compress_rates,
            "fpn": fpn_compress_rates
        }
        json_path = save_path.replace(".pth", ".json")
        with open(json_path, "w") as f:
            json.dump(config_data, f)

        print(f"Lean model saved to: {save_path}")
        print(f"Lean config saved to: {json_path}")

    print("Surgery Completed.")
    return lean_model
