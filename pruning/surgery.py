import torch
import torch.nn as nn
import json
import os
from models.resnet_hybrid import Bottleneck, BasicBlock
from models.conv_bn_relu import ConvBNReLU
# Import builder để tạo model đầy đủ
from models.faster_rcnn import fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn


def get_kept_indices(mask_handler):
    if mask_handler is None or mask_handler.s_mask is None:
        return None
    mask = mask_handler.s_mask.mask
    indices = torch.nonzero(mask).squeeze()
    if indices.dim() == 0: indices = indices.unsqueeze(0)
    return indices


def get_layer(model, name):
    try:
        return dict(model.named_modules())[name]
    except KeyError:
        return None


def get_input_mask_resnet(masked_model, current_layer_name):
    """
    Hàm thông minh giúp tìm Mask đầu vào cho các layer đặc biệt trong ResNet.
    Xử lý mối quan hệ giữa các Stage: Stem -> Layer1 -> Layer2...
    """
    # 1. Nếu là Layer đầu tiên của Block 1 (Stem -> Layer1)
    if "layer1.0.conv1" in current_layer_name:
        # Input đến từ Stem Conv (backbone.body.conv1)
        stem_conv = get_layer(masked_model, "backbone.body.conv1")
        return get_kept_indices(stem_conv)

    # 2. Nếu là Layer đầu của các Block tiếp theo (Layer(i-1) -> Layer(i))
    # Ví dụ: layer2.0.conv1 nhận input từ layer1 (Output cuối cùng của layer1)
    # Logic này áp dụng cho ResNet18 (BasicBlock) và ResNet50 (Bottleneck)
    for i in range(2, 5):
        if f"layer{i}.0.conv1" in current_layer_name:
            prev_stage = getattr(masked_model.backbone.body, f"layer{i - 1}")
            last_block = prev_stage[-1]

            # Nếu là BasicBlock (R18), output là conv2
            if isinstance(last_block, BasicBlock):
                return get_kept_indices(last_block.conv2)
            # Nếu là Bottleneck (R50), output là conv3
            elif isinstance(last_block, Bottleneck):
                return get_kept_indices(last_block.conv3)

    # 3. Logic nội bộ trong Block (đã xử lý ở hàm chính, nhưng thêm fallback ở đây)
    parts = current_layer_name.split(".")
    if "conv2" in current_layer_name:  # conv2 nhận từ conv1
        sibling_name = current_layer_name.replace("conv2", "conv1")
        # Cắt bỏ phần ".conv.weight" để lấy module
        sibling_module_name = ".".join(sibling_name.split(".")[:-2])
        mod = get_layer(masked_model, sibling_module_name)
        return get_kept_indices(mod)

    if "conv3" in current_layer_name:  # conv3 nhận từ conv2 (Bottleneck)
        sibling_name = current_layer_name.replace("conv3", "conv2")
        sibling_module_name = ".".join(sibling_name.split(".")[:-2])
        mod = get_layer(masked_model, sibling_module_name)
        return get_kept_indices(mod)

    return None


def convert_to_lean_model(masked_model, save_path=None):
    print("Starting Model Surgery...")

    # --- 1. EXTRACT CONFIG (Giữ nguyên) ---
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

    # --- 2. INIT LEAN MODEL ---
    first_block = masked_model.backbone.body.layer1[0]
    num_classes = 2  # Default fallback
    if hasattr(masked_model, 'roi_heads'):
        num_classes = masked_model.roi_heads.box_predictor.cls_score.out_features

    try:
        if isinstance(first_block, Bottleneck):
            print("Detected ResNet50 Architecture")
            lean_model = fasterrcnn_resnet50_fpn(num_classes=num_classes, compress_rate=backbone_compress_rates,
                                                 fpn_compress_rate=fpn_compress_rates)
        elif isinstance(first_block, BasicBlock):
            print("Detected ResNet18 Architecture")
            lean_model = fasterrcnn_resnet18_fpn(num_classes=num_classes, compress_rate=backbone_compress_rates,
                                                 fpn_compress_rate=fpn_compress_rates)
        else:
            print("Unknown architecture!")
            return None
    except Exception as e:
        print(f"Error creating lean model: {e}")
        return None

    # --- 3. COPY WEIGHTS ---
    lean_state_dict = lean_model.state_dict()
    masked_state_dict = masked_model.state_dict()

    for name, lean_param in lean_state_dict.items():
        if name not in masked_state_dict:
            continue

        masked_param = masked_state_dict[name]

        # --- FIX 1: SCALAR PROTECTION (Chống lỗi num_batches_tracked) ---
        if lean_param.dim() == 0:
            lean_param.data.copy_(masked_param.data)
            continue

        module_name = ".".join(name.split(".")[:-1])
        lean_module = get_layer(lean_model, module_name)

        # A. FPN HANDLING
        if "fpn.layer_blocks" in name:
            block_name = ".".join(module_name.split(".")[:-1])
            masked_block = get_layer(masked_model, block_name)

            # Logic FPN giữ nguyên vì đã đúng
            if "compress_layer" in name and "conv" in name:
                pass
            elif "dw_conv" in name:
                out_idx = get_kept_indices(masked_block.compress_layer)
                if out_idx is not None:
                    if lean_param.dim() == 4:
                        lean_param.data.copy_(masked_param.data[out_idx, :, :, :])
                    else:
                        lean_param.data.copy_(masked_param.data[out_idx])
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

        # B. STANDARD CONV/BN HANDLING
        if isinstance(lean_module, nn.Conv2d):
            # 1. Output Mask (Prune chiều 0)
            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)
            out_idx = get_kept_indices(parent_masked_module)

            # Xử lý Bias
            if "bias" in name:
                if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                    lean_param.data.copy_(masked_param.data[out_idx])
                else:
                    lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])
                continue

            # Xử lý Weight (Prune chiều 0 - Output)
            w_temp = masked_param.data
            if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                w_temp = w_temp[out_idx, :, :, :]
            else:
                w_temp = w_temp[:lean_param.shape[0], :, :, :]

            # 2. Input Mask (Prune chiều 1 - Input)
            if lean_param.shape[1] < masked_param.shape[1]:
                # --- FIX 2: TOPOLOGY AWARE INPUT MASK ---
                in_idx = get_input_mask_resnet(masked_model, name)

                if in_idx is not None and len(in_idx) == lean_param.shape[1]:
                    lean_param.data.copy_(w_temp[:, in_idx, :, :])
                else:
                    # Chỉ fallback slice nếu thực sự bó tay
                    print(
                        f"Warning: Fallback slicing input for {name}. Shapes: Lean{lean_param.shape} vs Masked{masked_param.shape}")
                    lean_param.data.copy_(w_temp[:, :lean_param.shape[1], :, :])
            else:
                lean_param.data.copy_(w_temp)

        elif isinstance(lean_module, nn.BatchNorm2d):
            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)
            out_idx = get_kept_indices(parent_masked_module)

            # Weight, Bias, Running_Mean, Running_Var đều là vector 1D -> Prune giống nhau
            if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                lean_param.data.copy_(masked_param.data[out_idx])
            else:
                lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])

        else:
            # Các layer khác (Linear, v.v.) copy nguyên
            if lean_param.shape == masked_param.shape:
                lean_param.data.copy_(masked_param.data)

    # --- 4. SAVE ---
    if save_path:
        torch.save(lean_model.state_dict(), save_path)
        config_data = {'backbone': backbone_compress_rates, 'fpn': fpn_compress_rates}
        json_path = save_path.replace('.pth', '.json')
        with open(json_path, 'w') as f:
            json.dump(config_data, f)
        print(f"Lean config saved to: {json_path}")

    print(f"Surgery Completed. Lean model saved to {save_path}")
    return lean_model