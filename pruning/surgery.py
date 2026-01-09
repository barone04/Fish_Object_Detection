# import os
# import json
# import torch
# import torch.nn as nn
#
# from models.resnet_hybrid import Bottleneck, BasicBlock
# from models.conv_bn_relu import ConvBNReLU
# from models.faster_rcnn import fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn
#
# try:
#     from models.mobilenet_custom import fasterrcnn_mobilenetv3_custom
#     _HAS_MOBILENET = True
# except Exception:
#     fasterrcnn_mobilenetv3_custom = None
#     _HAS_MOBILENET = False
#
#
# def get_kept_indices(mask_handler):
#     if mask_handler is None or (not hasattr(mask_handler, "s_mask")) or mask_handler.s_mask is None:
#         return None
#     mask = mask_handler.s_mask.mask
#     indices = torch.nonzero(mask).squeeze()
#     if indices.dim() == 0:
#         indices = indices.unsqueeze(0)
#     return indices
#
#
# def get_layer(model, name):
#     try:
#         return dict(model.named_modules())[name]
#     except KeyError:
#         return None
#
#
# def get_input_mask_resnet(masked_model, current_layer_name):
#     """
#     ResNet-only: tìm mask đầu vào hợp lý theo topology (Stem -> Layer1 -> Layer2...).
#     Giữ nguyên logic code cũ.
#     """
#     if "layer1.0.conv1" in current_layer_name:
#         stem_conv = get_layer(masked_model, "backbone.body.conv1")
#         return get_kept_indices(stem_conv)
#
#     for i in range(2, 5):
#         if f"layer{i}.0.conv1" in current_layer_name:
#             prev_stage = getattr(masked_model.backbone.body, f"layer{i - 1}")
#             last_block = prev_stage[-1]
#             if isinstance(last_block, BasicBlock):
#                 return get_kept_indices(last_block.conv2)
#             elif isinstance(last_block, Bottleneck):
#                 return get_kept_indices(last_block.conv3)
#
#     if "conv2" in current_layer_name:
#         sibling_name = current_layer_name.replace("conv2", "conv1")
#         sibling_module_name = ".".join(sibling_name.split(".")[:-2])
#         mod = get_layer(masked_model, sibling_module_name)
#         return get_kept_indices(mod)
#
#     if "conv3" in current_layer_name:
#         sibling_name = current_layer_name.replace("conv3", "conv2")
#         sibling_module_name = ".".join(sibling_name.split(".")[:-2])
#         mod = get_layer(masked_model, sibling_module_name)
#         return get_kept_indices(mod)
#
#     return None
#
#
# def _safe_mkdir_for_file(path: str):
#     if not path:
#         return
#     d = os.path.dirname(os.path.abspath(path))
#     if d and (not os.path.exists(d)):
#         os.makedirs(d, exist_ok=True)
#
#
# def _infer_num_classes(masked_model, default=2):
#     if hasattr(masked_model, "roi_heads") and hasattr(masked_model.roi_heads, "box_predictor"):
#         bp = masked_model.roi_heads.box_predictor
#         if hasattr(bp, "cls_score") and hasattr(bp.cls_score, "out_features"):
#             return int(bp.cls_score.out_features)
#     return int(default)
#
#
# def _infer_box_head_dim(masked_model, default=1024):
#     """
#     MobileNet builder của bạn có box_head_dim.
#     Nếu model có fc6 => suy ra out_features để build lean khớp.
#     """
#     try:
#         return int(masked_model.roi_heads.box_head.fc6.out_features)
#     except Exception:
#         return int(default)
#
#
# def convert_to_lean_model(masked_model, save_path=None):
#     print("Starting Model Surgery...")
#
#     # --- 1. EXTRACT CONFIG (GIỮ NGUYÊN LOGIC CŨ CHO RESNET) ---
#     backbone_compress_rates = []
#     if hasattr(masked_model, 'backbone') and hasattr(masked_model.backbone, 'body'):
#         for m in masked_model.backbone.body.modules():
#             if isinstance(m, ConvBNReLU):
#                 if m.s_mask is not None:
#                     mask = m.s_mask.mask
#                     kept = mask.sum().item()
#                     total = mask.numel()
#                     backbone_compress_rates.append(1.0 - (kept / total))
#                 else:
#                     backbone_compress_rates.append(0.0)
#
#     fpn_compress_rates = []
#     if hasattr(masked_model, 'backbone') and hasattr(masked_model.backbone, 'fpn') and hasattr(masked_model.backbone.fpn, 'layer_blocks'):
#         for layer_block in masked_model.backbone.fpn.layer_blocks:
#             if hasattr(layer_block, 'compress_layer'):
#                 m = layer_block.compress_layer
#                 if m.s_mask is not None:
#                     mask = m.s_mask.mask
#                     kept = mask.sum().item()
#                     total = mask.numel()
#                     fpn_compress_rates.append(1.0 - (kept / total))
#                 else:
#                     fpn_compress_rates.append(0.0)
#
#     print(f"Rates extracted: Backbone={len(backbone_compress_rates)}, FPN={len(fpn_compress_rates)}")
#
#     # --- 2. INIT LEAN MODEL ---
#     num_classes = _infer_num_classes(masked_model, default=2)
#
#     # Detect ResNet (giữ nguyên logic cũ: dựa vào layer1[0])
#     first_block = None
#     is_resnet = False
#     try:
#         if hasattr(masked_model, "backbone") and hasattr(masked_model.backbone, "body") and hasattr(masked_model.backbone.body, "layer1"):
#             first_block = masked_model.backbone.body.layer1[0]
#             is_resnet = True
#     except Exception:
#         is_resnet = False
#
#     try:
#         if is_resnet and isinstance(first_block, Bottleneck):
#             print("Detected ResNet50 Architecture")
#             lean_model = fasterrcnn_resnet50_fpn(
#                 num_classes=num_classes,
#                 compress_rate=backbone_compress_rates,
#                 fpn_compress_rate=fpn_compress_rates
#             )
#         elif is_resnet and isinstance(first_block, BasicBlock):
#             print("Detected ResNet18 Architecture")
#             lean_model = fasterrcnn_resnet18_fpn(
#                 num_classes=num_classes,
#                 compress_rate=backbone_compress_rates,
#                 fpn_compress_rate=fpn_compress_rates
#             )
#         else:
#             # MobileNetV3 path (KHÔNG ẢNH HƯỞNG RESNET)
#             if not _HAS_MOBILENET:
#                 print("Unknown architecture and MobileNet builder not available!")
#                 return None
#             print("Detected MobileNetV3 Architecture")
#             box_head_dim = _infer_box_head_dim(masked_model, default=1024)
#             lean_model = fasterrcnn_mobilenetv3_custom(
#                 num_classes=num_classes,
#                 fpn_compress_rate=fpn_compress_rates,
#                 pretrained_backbone=False,  # weights sẽ load từ masked_model qua copy
#                 freeze_backbone=True,
#                 box_head_dim=box_head_dim
#             )
#     except Exception as e:
#         print(f"Error creating lean model: {e}")
#         return None
#
#     lean_state_dict = lean_model.state_dict()
#     masked_state_dict = masked_model.state_dict()
#
#     for name, lean_param in lean_state_dict.items():
#         if name not in masked_state_dict:
#             continue
#
#         masked_param = masked_state_dict[name]
#
#         if lean_param.dim() == 0:
#             lean_param.data.copy_(masked_param.data)
#             continue
#
#         module_name = ".".join(name.split(".")[:-1])
#         lean_module = get_layer(lean_model, module_name)
#
#         if "fpn.layer_blocks" in name:
#             block_name = ".".join(module_name.split(".")[:-1])
#             masked_block = get_layer(masked_model, block_name)
#
#             if masked_block is None:
#                 if lean_param.shape == masked_param.shape:
#                     lean_param.data.copy_(masked_param.data)
#                 continue
#
#             if "compress_layer" in name and "conv" in name:
#                 pass
#
#             elif "dw_conv" in name:
#                 out_idx = get_kept_indices(getattr(masked_block, "compress_layer", None))
#                 if out_idx is not None:
#                     if lean_param.dim() == 4:
#                         lean_param.data.copy_(masked_param.data[out_idx, :, :, :])
#                     else:
#                         lean_param.data.copy_(masked_param.data[out_idx])
#                 else:
#                     if lean_param.shape == masked_param.shape:
#                         lean_param.data.copy_(masked_param.data)
#                     else:
#                         lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])
#                 continue
#
#             elif "dw_bn" in name:
#                 out_idx = get_kept_indices(getattr(masked_block, "compress_layer", None))
#                 if out_idx is not None and lean_param.shape[0] == len(out_idx):
#                     lean_param.data.copy_(masked_param.data[out_idx])
#                 else:
#                     lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])
#                 continue
#
#             elif "expand_conv" in name:
#                 out_idx = get_kept_indices(getattr(masked_block, "compress_layer", None))
#                 if out_idx is not None and lean_param.shape[1] == len(out_idx):
#                     lean_param.data.copy_(masked_param.data[:, out_idx, :, :])
#                 else:
#                     lean_param.data.copy_(masked_param.data[:, :lean_param.shape[1], :, :])
#                 continue
#
#             elif "expand_bn" in name:
#                 if lean_param.shape == masked_param.shape:
#                     lean_param.data.copy_(masked_param.data)
#                 else:
#                     lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])
#                 continue
#
#         if isinstance(lean_module, nn.Conv2d):
#             parent_name = ".".join(module_name.split(".")[:-1])
#             parent_masked_module = get_layer(masked_model, parent_name)
#             out_idx = get_kept_indices(parent_masked_module)
#
#             # Bias
#             if "bias" in name:
#                 if out_idx is not None and len(out_idx) == lean_param.shape[0]:
#                     lean_param.data.copy_(masked_param.data[out_idx])
#                 else:
#                     lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])
#                 continue
#
#             w_temp = masked_param.data
#             if out_idx is not None and len(out_idx) == lean_param.shape[0]:
#                 w_temp = w_temp[out_idx, :, :, :]
#             else:
#                 w_temp = w_temp[:lean_param.shape[0], :, :, :]
#
#             if lean_param.shape[1] < masked_param.shape[1]:
#                 in_idx = get_input_mask_resnet(masked_model, name)
#                 if in_idx is not None and len(in_idx) == lean_param.shape[1]:
#                     lean_param.data.copy_(w_temp[:, in_idx, :, :])
#                 else:
#                     lean_param.data.copy_(w_temp[:, :lean_param.shape[1], :, :])
#             else:
#                 lean_param.data.copy_(w_temp)
#
#         elif isinstance(lean_module, nn.BatchNorm2d):
#             parent_name = ".".join(module_name.split(".")[:-1])
#             parent_masked_module = get_layer(masked_model, parent_name)
#             out_idx = get_kept_indices(parent_masked_module)
#
#             # Weight/Bias/Running stats (1D)
#             if out_idx is not None and len(out_idx) == lean_param.shape[0]:
#                 lean_param.data.copy_(masked_param.data[out_idx])
#             else:
#                 lean_param.data.copy_(masked_param.data[:lean_param.shape[0]])
#
#         else:
#             if lean_param.shape == masked_param.shape:
#                 lean_param.data.copy_(masked_param.data)
#
#     if save_path:
#         _safe_mkdir_for_file(save_path)
#         torch.save(lean_model.state_dict(), save_path)
#
#         config_data = {"backbone": backbone_compress_rates, "fpn": fpn_compress_rates}
#         json_path = save_path.replace(".pth", ".json")
#         with open(json_path, "w") as f:
#             json.dump(config_data, f)
#
#         print(f"Lean model saved to: {save_path}")
#         print(f"Lean config saved to: {json_path}")
#
#     print("Surgery Completed.")
#     return lean_model

import torch
import torch.nn as nn
import json
import os
from models.resnet_hybrid import Bottleneck, BasicBlock
from models.conv_bn_relu import ConvBNReLU
# Import các builder
from models.faster_rcnn import fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn
from models.mobilenet_custom import fasterrcnn_mobilenetv3_custom


def get_kept_indices(mask_handler):
    """Lấy indices của các kênh được giữ lại từ ConvBNReLU/MaskProxy"""
    if mask_handler is None: return None
    # Nếu là ConvBNReLU wrapper
    if hasattr(mask_handler, "s_mask") and mask_handler.s_mask is not None:
        mask = mask_handler.s_mask.mask
    # Nếu là MaskProxy (đôi khi wrapper trả về cái này)
    elif hasattr(mask_handler, "mask"):
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


def get_input_mask_resnet(masked_model, current_layer_name):
    """Logic tìm mask đầu vào dành riêng cho ResNet (giữ nguyên logic cũ đã hoạt động tốt)"""
    if "layer1.0.conv1" in current_layer_name:
        stem_conv = get_layer(masked_model, "backbone.body.conv1")
        return get_kept_indices(stem_conv)

    for i in range(2, 5):
        if f"layer{i}.0.conv1" in current_layer_name:
            try:
                prev_stage = getattr(masked_model.backbone.body, f"layer{i - 1}")
                last_block = list(prev_stage.children())[-1]
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

    # Check loại Model
    is_mobilenet = "MobileNet" in str(type(masked_model.backbone)) or \
                   (hasattr(masked_model.backbone, 'body') and "MobileNet" in str(type(masked_model.backbone.body)))

    print(f"Surgery detected model type: {'MobileNetV3' if is_mobilenet else 'ResNet'}")

    # --- 1. EXTRACT RATES (FIX: Dùng cách quét module đệ quy cho ResNet) ---
    backbone_compress_rates = []
    fpn_compress_rates = []

    # A. BACKBONE RATES
    if is_mobilenet:
        backbone_compress_rates = None  # MobileNet freeze, không prune
    else:
        # [FIX 1] Quét toàn bộ ConvBNReLU trong body để đảm bảo đúng thứ tự/số lượng cho Builder ResNet cũ
        if hasattr(masked_model.backbone, "body"):
            for m in masked_model.backbone.body.modules():
                if isinstance(m, ConvBNReLU):
                    if m.s_mask is not None:
                        mask = m.s_mask.mask
                        rate = 1.0 - (mask.sum().item() / mask.numel())
                        backbone_compress_rates.append(rate)
                    else:
                        backbone_compress_rates.append(0.0)

    # B. FPN RATES
    if hasattr(masked_model.backbone, "fpn") and hasattr(masked_model.backbone.fpn, "layer_blocks"):
        for block in masked_model.backbone.fpn.layer_blocks:
            # Block custom FPN: compress_layer là ConvBNReLU
            if hasattr(block, "compress_layer"):
                m = block.compress_layer
                if hasattr(m, "s_mask") and m.s_mask is not None:
                    mask = m.s_mask.mask
                    rate = 1.0 - (mask.sum().item() / mask.numel())
                    fpn_compress_rates.append(rate)
                else:
                    fpn_compress_rates.append(0.0)

    print(
        f"Extracted Rates -> Backbone: {len(backbone_compress_rates) if backbone_compress_rates else 0}, FPN: {len(fpn_compress_rates)}")

    # --- 2. INIT LEAN MODEL ---
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
        # ResNet Logic
        if len(backbone_compress_rates) > 20:  # ResNet50
            print("Initializing Lean ResNet50...")
            lean_model = fasterrcnn_resnet50_fpn(
                num_classes=num_classes,
                compress_rate=backbone_compress_rates,
                fpn_compress_rate=fpn_compress_rates,
                weights_backbone=None
            )
        else:  # ResNet18
            print("Initializing Lean ResNet18...")
            lean_model = fasterrcnn_resnet18_fpn(
                num_classes=num_classes,
                compress_rate=backbone_compress_rates,
                fpn_compress_rate=fpn_compress_rates,
                weights_backbone=None
            )

    lean_model.to(device)
    lean_model.eval()

    # --- 3. COPY WEIGHTS (FIX: Logic Pruning chuẩn cho FPN) ---
    print("Copying weights...")

    with torch.no_grad():
        lean_modules = dict(lean_model.named_modules())

        for name, lean_module in lean_modules.items():
            # Chỉ xử lý các layer có trọng số
            if not isinstance(lean_module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                continue

            masked_module = get_layer(masked_model, name)
            if masked_module is None:
                continue

            # === NHÁNH 1: XỬ LÝ FPN (Logic riêng biệt) ===
            if "fpn.layer_blocks" in name:
                # Tìm block chứa layer này (Ví dụ: backbone.fpn.layer_blocks.0)
                block_name = ".".join(name.split(".")[:-1])
                masked_block = get_layer(masked_model, block_name)

                # Lấy indices từ compress_layer của block đó (Đây là nguồn mask duy nhất của block)
                mask_idx = None
                if hasattr(masked_block, "compress_layer"):
                    mask_idx = get_kept_indices(masked_block.compress_layer)

                # [Case A] Compress Layer (ConvBNReLU -> conv)
                if "compress_layer.conv" in name:
                    # Prune Output (dim 0)
                    if mask_idx is not None:
                        lean_module.weight.data.copy_(masked_module.weight.data[mask_idx, :, :, :])
                        if lean_module.bias is not None:
                            lean_module.bias.data.copy_(masked_module.bias.data[mask_idx])
                    else:
                        lean_module.weight.data.copy_(masked_module.weight.data)

                # [Case B] Compress Layer (ConvBNReLU -> bn)
                elif "compress_layer.bn" in name:
                    if mask_idx is not None:
                        lean_module.weight.data.copy_(masked_module.weight.data[mask_idx])
                        lean_module.bias.data.copy_(masked_module.bias.data[mask_idx])
                        lean_module.running_mean.data.copy_(masked_module.running_mean.data[mask_idx])
                        lean_module.running_var.data.copy_(masked_module.running_var.data[mask_idx])
                    else:
                        lean_module.weight.data.copy_(masked_module.weight.data)
                        # ... copy các cái khác tương tự

                # [Case C] Depthwise Conv (dw_conv)
                elif "dw_conv" in name:
                    # Input pruned theo compress_layer
                    # Với DW Conv, groups = in_channels. Prune in_channels đồng nghĩa prune luôn filters (dim 0)
                    if mask_idx is not None:
                        # weight shape: [C, 1, 3, 3]. Prune dim 0.
                        lean_module.weight.data.copy_(masked_module.weight.data[mask_idx, :, :, :])
                    else:
                        lean_module.weight.data.copy_(masked_module.weight.data)

                # [Case D] Depthwise BN (dw_bn)
                elif "dw_bn" in name:
                    if mask_idx is not None:
                        lean_module.weight.data.copy_(masked_module.weight.data[mask_idx])
                        lean_module.bias.data.copy_(masked_module.bias.data[mask_idx])
                        lean_module.running_mean.data.copy_(masked_module.running_mean.data[mask_idx])
                        lean_module.running_var.data.copy_(masked_module.running_var.data[mask_idx])
                    else:
                        lean_module.weight.data.copy_(masked_module.weight.data)

                # [Case E] Expand Conv (expand_conv)
                elif "expand_conv" in name:
                    # Prune Input (dim 1) theo compress_layer
                    if mask_idx is not None:
                        lean_module.weight.data.copy_(masked_module.weight.data[:, mask_idx, :, :])
                    else:
                        lean_module.weight.data.copy_(masked_module.weight.data)

                # [Case F] Expand BN (expand_bn) - Không Prune (Vì output về 256)
                elif "expand_bn" in name:
                    lean_module.weight.data.copy_(masked_module.weight.data)
                    lean_module.bias.data.copy_(masked_module.bias.data)
                    lean_module.running_mean.data.copy_(masked_module.running_mean.data)
                    lean_module.running_var.data.copy_(masked_module.running_var.data)

            # === NHÁNH 2: XỬ LÝ RESNET BACKBONE (Logic cũ) ===
            elif isinstance(lean_module, nn.Conv2d):
                # Nếu shape khớp (MobileNet Backbone, hoặc Layer ko prune), copy thẳng
                if lean_module.weight.shape == masked_module.weight.shape:
                    lean_module.weight.data.copy_(masked_module.weight.data)
                    if lean_module.bias is not None: lean_module.bias.data.copy_(masked_module.bias.data)
                    continue

                # Nếu không khớp -> ResNet Pruning logic
                # 1. Prune Output
                parent_name = ".".join(name.split(".")[:-1])
                parent_module = get_layer(masked_model, parent_name)
                out_idx = get_kept_indices(parent_module)

                w_temp = masked_module.weight.data
                if out_idx is not None and len(out_idx) == lean_module.out_channels:
                    w_temp = w_temp[out_idx, :, :, :]
                    # Bias
                    if lean_module.bias is not None:
                        lean_module.bias.data.copy_(masked_module.bias.data[out_idx])

                # 2. Prune Input
                in_idx = get_input_mask_resnet(masked_model, name)
                if in_idx is not None and len(in_idx) == lean_module.in_channels:
                    w_temp = w_temp[:, in_idx, :, :]

                # Copy final
                if w_temp.shape == lean_module.weight.shape:
                    lean_module.weight.data.copy_(w_temp)
                else:
                    # Fallback slicing (Chỉ khi thực sự cần)
                    print(f"Fallback slicing for {name}")
                    min_out = min(w_temp.shape[0], lean_module.weight.shape[0])
                    min_in = min(w_temp.shape[1], lean_module.weight.shape[1])
                    lean_module.weight.data[:min_out, :min_in].copy_(w_temp[:min_out, :min_in])

            # === NHÁNH 3: XỬ LÝ BATCHNORM KHÁC (Của ResNet) ===
            elif isinstance(lean_module, nn.BatchNorm2d):
                # Nếu shape khớp, copy thẳng
                if lean_module.weight.shape == masked_module.weight.shape:
                    lean_module.weight.data.copy_(masked_module.weight.data)
                    lean_module.bias.data.copy_(masked_module.bias.data)
                    lean_module.running_mean.data.copy_(masked_module.running_mean.data)
                    lean_module.running_var.data.copy_(masked_module.running_var.data)
                    continue

                # Prune theo layer Conv liền trước
                parent_name = ".".join(name.split(".")[:-1])
                parent_module = get_layer(masked_model, parent_name)
                out_idx = get_kept_indices(parent_module)

                if out_idx is not None and len(out_idx) == lean_module.num_features:
                    lean_module.weight.data.copy_(masked_module.weight.data[out_idx])
                    lean_module.bias.data.copy_(masked_module.bias.data[out_idx])
                    lean_module.running_mean.data.copy_(masked_module.running_mean.data[out_idx])
                    lean_module.running_var.data.copy_(masked_module.running_var.data[out_idx])

            # === NHÁNH 4: CÁC LAYER KHÁC (Linear...) ===
            else:
                if lean_module.weight.shape == masked_module.weight.shape:
                    lean_module.weight.data.copy_(masked_module.weight.data)
                    if getattr(lean_module, 'bias', None) is not None:
                        lean_module.bias.data.copy_(masked_module.bias.data)

    # --- 4. SAVE ---
    if save_path:
        print(f"Saving Lean Model to {save_path}")
        torch.save(lean_model.state_dict(), save_path)

        # Save JSON
        config_data = {'backbone': backbone_compress_rates, 'fpn': fpn_compress_rates}
        json_path = save_path.replace('.pth', '.json')
        with open(json_path, 'w') as f:
            json.dump(config_data, f)

    return lean_model