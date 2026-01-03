import torch
import torch.nn as nn
import json
import os
from models.resnet_hybrid import resnet_50, Bottleneck, resnet_18, BasicBlock
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
    return dict(model.named_modules())[name]


def convert_to_lean_model(masked_model, save_path=None):
    print("Starting Model Surgery...")

    # --- 1. TRÍCH XUẤT CONFIG BACKBONE ---
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

    # --- 2. TRÍCH XUẤT CONFIG FPN ---
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

    # --- 3. KHỞI TẠO LEAN MODEL ---
    first_block = masked_model.backbone.body.layer1[0]
    if hasattr(masked_model, 'roi_heads'):
        num_classes = masked_model.roi_heads.box_predictor.cls_score.out_features
    else:
        num_classes = 2

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

    # --- 4. COPY WEIGHTS ---
    lean_state_dict = lean_model.state_dict()
    masked_state_dict = masked_model.state_dict()

    for name, lean_param in lean_state_dict.items():
        if name not in masked_state_dict:
            continue

        masked_param = masked_state_dict[name]
        module_name = ".".join(name.split(".")[:-1])

        # --- A. XỬ LÝ FPN (GIỮ NGUYÊN CODE FPN CŨ) ---
        if "fpn.layer_blocks" in name:
            block_name = ".".join(module_name.split(".")[:-1])
            masked_block = get_layer(masked_model, block_name)

            if "compress_layer" in name and "conv" in name:
                pass  # Để Logic B xử lý
            elif "dw_conv" in name:
                # DW Conv: Cắt theo mask của compress_layer
                out_idx = get_kept_indices(masked_block.compress_layer)
                if out_idx is not None:
                    if lean_param.dim() == 4:  # Weight
                        lean_param.data.copy_(masked_param.data[out_idx, :, :, :])
                    else:  # Bias
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

        # --- B. LOGIC CHUNG (BACKBONE + OTHERS) ---
        lean_module = get_layer(lean_model, module_name)

        # 1. Nếu shape khớp hoàn toàn -> Copy ngay (Nhanh gọn)
        if lean_param.shape == masked_param.shape:
            lean_param.data.copy_(masked_param.data)
            continue

        # 2. Xử lý Conv2d (Trọng số và Bias)
        if isinstance(lean_module, nn.Conv2d):
            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)  # ConvBNReLU

            # Lấy Output Mask (Prune chiều 0)
            out_idx = get_kept_indices(parent_masked_module)

            # --- FIX ERROR 1: Xử lý Bias (Chỉ cắt chiều 0) ---
            if "bias" in name:
                if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                    lean_param.data.copy_(masked_param.data[out_idx])
                else:
                    lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])
                continue

            # --- Xử lý Weight (Cắt chiều 0 và chiều 1) ---
            # B1. Cắt chiều Output (Dim 0)
            w_temp = masked_param.data
            if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                w_temp = w_temp[out_idx, :, :, :]
            else:
                w_temp = w_temp[:lean_param.shape[0], :, :, :]

            # B2. Cắt chiều Input (Dim 1) - FIX ERROR 2: Slicing Warning
            if lean_param.shape[1] < masked_param.shape[1]:
                in_idx = None

                # Logic thông minh cho ResNet BasicBlock/Bottleneck
                # Nếu đây là conv2, input mask chính là output mask của conv1
                if "conv2" in module_name:
                    sibling_name = parent_name.replace("conv2", "conv1")
                    try:
                        sibling_module = get_layer(masked_model, sibling_name)
                        in_idx = get_kept_indices(sibling_module)
                    except:
                        pass

                # Nếu đây là conv3 (Bottleneck), input là conv2
                elif "conv3" in module_name:
                    sibling_name = parent_name.replace("conv3", "conv2")
                    try:
                        sibling_module = get_layer(masked_model, sibling_name)
                        in_idx = get_kept_indices(sibling_module)
                    except:
                        pass

                # Thực hiện cắt Input
                if in_idx is not None and len(in_idx) == lean_param.shape[1]:
                    # Cắt chuẩn theo Mask
                    lean_param.data.copy_(w_temp[:, in_idx, :, :])
                else:
                    # Fallback (Vẫn phải slice nếu không tìm thấy mask - ví dụ layer đầu block)
                    # Nhưng với conv2/conv3 logic trên đã xử lý triệt để
                    print(
                        f"Warning: Naive slicing input for {name}. Shapes: Lean{lean_param.shape} vs Masked{masked_param.shape}")
                    lean_param.data.copy_(w_temp[:, :lean_param.shape[1], :, :])
            else:
                # Input không đổi
                lean_param.data.copy_(w_temp)

        # 3. Xử lý BatchNorm
        elif isinstance(lean_module, nn.BatchNorm2d):
            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)
            out_idx = get_kept_indices(parent_masked_module)

            if out_idx is not None and len(out_idx) == lean_param.shape[0]:
                lean_param.data.copy_(masked_param.data[out_idx])
            else:
                lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])

    # --- 5. SAVE ---
    if save_path:
        torch.save(lean_model.state_dict(), save_path)

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