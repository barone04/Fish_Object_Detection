# import torch
# import torch.nn as nn
# from models.resnet_hybrid import resnet_50, Bottleneck, resnet_18, BasicBlock
# from models.conv_bn_relu import ConvBNReLU
#
#
# def get_kept_indices(mask_handler):
#     """
#     Trả về danh sách index của các filter được giữ lại (Mask = 1).
#     """
#     if mask_handler is None or mask_handler.s_mask is None:
#         return None
#
#     mask = mask_handler.s_mask.mask
#     # Lấy các vị trí có giá trị = 1
#     indices = torch.nonzero(mask).squeeze()
#     if indices.dim() == 0: indices = indices.unsqueeze(0)
#     return indices
#
#
# def convert_to_lean_model(masked_model, save_path=None):
#     print("Starting Model Surgery (Physical Removal)...")
#
#     # 1. Trích xuất Compress Rates
#     compress_rates = []
#     # Cần duyệt model theo thứ tự đúng để list compress_rates khớp với logic init của ResNet
#     # Ta duyệt qua các modules, tìm ConvBNReLU
#     for m in masked_model.modules():
#         if isinstance(m, ConvBNReLU):
#             if m.s_mask is not None:
#                 mask = m.s_mask.mask
#                 kept = mask.sum().item()
#                 total = mask.numel()
#                 compress_rates.append(1.0 - (kept / total))
#             else:
#                 compress_rates.append(0.0)
#
#     print(f"Extracted {len(compress_rates)} layers. Creating Lean Model...")
#
#     num_classes = masked_model.fc.out_features if hasattr(masked_model, 'fc') else 1000
#     try:
#         lean_model = resnet_50(compress_rate=compress_rates, num_classes=num_classes)
#     except Exception as e:
#         print(f"Error creating lean model: {e}")
#         return None
#
#     print("Copying weights based on masks...")
#
#     # Helper để lấy layer từ model bằng tên (vd: layer1[0].conv1)
#     def get_layer(model, name):
#         return dict(model.named_modules())[name]
#
#     # Map state_dict của lean model
#     lean_state_dict = lean_model.state_dict()
#     masked_state_dict = masked_model.state_dict()
#
#     # Chúng ta cần theo dõi 'indices' của layer trước đó để cắt chiều Input
#     # Với ResNet, luồng dữ liệu phức tạp (Skip connection).
#     # Để đơn giản và hiệu quả, ta sẽ duyệt theo cấu trúc Block.
#
#     # A. Copy các layer không bị prune (FC, Stem đầu vào nếu ko prune)
#     # Lưu ý: Stem (conv1) input là ảnh (3 kênh) -> Ko cần cắt chiều In.
#
#     # Ta duyệt qua từng tên tham số trong Lean Model
#     for name, param in lean_model.named_parameters():
#         if name not in masked_state_dict:
#             continue
#
#         masked_param = masked_state_dict[name]
#
#         # Tách tên module và tên tham số (vd: layer1.0.conv1.conv.weight -> layer1.0.conv1.conv, weight)
#         module_name = ".".join(name.split(".")[:-1])
#         param_type = name.split(".")[-1]  # weight, bias
#
#         lean_module = get_layer(lean_model, module_name)
#         masked_module = get_layer(masked_model, module_name)
#
#         # 1. Xử lý Conv2d
#         if isinstance(lean_module, nn.Conv2d):
#             # Tìm mask của chính layer này (Output Mask)
#             # Vì lean_module nằm trong ConvBNReLU của lean_model, ta cần tìm ConvBNReLU cha
#             # Nhưng ở đây ta đang iterate param của nn.Conv2d.
#             # Parent của nn.Conv2d chính là ConvBNReLU trong code resnet_hybrid.
#
#             # Để đơn giản: Ta so sánh shape.
#             # Nếu shape khớp -> Copy thẳng.
#             if param.shape == masked_param.shape:
#                 param.data.copy_(masked_param.data)
#                 continue
#
#             # Lấy parent module (ConvBNReLU) bên Masked Model
#             # Tên module_name ví dụ: layer1.0.conv1.conv
#             parent_name = ".".join(module_name.split(".")[:-1])  # layer1.0.conv1
#             parent_masked_module = get_layer(masked_model, parent_name)
#
#             # Lấy Indices Output (Filter cần giữ)
#             out_idx = get_kept_indices(parent_masked_module)
#
#
#             if out_idx is not None and len(out_idx) == param.shape[0]:
#                 # Cắt chiều Output
#                 w_temp = masked_param.data[out_idx, :, :, :]
#
#                 # Cắt chiều Input
#                 # Nếu số kênh input cũng bị giảm (lean < masked)
#                 if param.shape[1] < masked_param.shape[1]:
#                     # Cần tìm in_idx.
#                     # Với ConvBNReLU trong Block:
#                     # conv2 input là output conv1.
#                     # conv3 input là output conv2.
#                     # conv1 input là output block trước.
#
#                     param.data.copy_(w_temp[:, :param.shape[1], :, :])
#                 else:
#                     param.data.copy_(w_temp)
#             else:
#                 # Fallback: Copy phần góc trên bên trái (Sub-tensor)
#                 # Đây là cách tệ nhất nhưng đảm bảo code chạy ko lỗi
#                 print(f"Shape mismatch heavy at {name}: {param.shape} vs {masked_param.shape}. Slicing...")
#                 param.data.copy_(masked_param.data[:param.shape[0], :param.shape[1], :, :])
#
#         # 2. Xử lý BatchNorm (weight, bias, running_mean, running_var)
#         elif isinstance(lean_module, nn.BatchNorm2d):
#             # BN chỉ có 1 chiều (theo số kênh Output)
#             # Ta cần tìm Mask tương ứng của Conv liền trước nó.
#
#             # Tên module: layer1.0.conv1.bn
#             parent_name = ".".join(module_name.split(".")[:-1])  # layer1.0.conv1
#             parent_masked_module = get_layer(masked_model, parent_name)
#
#             out_idx = get_kept_indices(parent_masked_module)
#
#             if out_idx is not None and len(out_idx) == param.shape[0]:
#                 param.data.copy_(masked_param.data[out_idx])
#             else:
#                 param.data.copy_(masked_param.data[:param.shape[0]])
#
#         # 3. Các layer khác (FC)
#         else:
#             if param.shape == masked_param.shape:
#                 param.data.copy_(masked_param.data)
#
#     if save_path:
#         torch.save(lean_model.state_dict(), save_path)
#         import json
#         with open(save_path.replace('.pth', '.json'), 'w') as f:
#             json.dump(compress_rates, f)
#
#     return lean_model
#
#
#
import torch
import torch.nn as nn
from models.resnet_hybrid import resnet_50, Bottleneck, resnet_18, BasicBlock
from models.conv_bn_relu import ConvBNReLU


def get_kept_indices(mask_handler):
    """
    Trả về danh sách index của các filter được giữ lại (Mask = 1).
    """
    if mask_handler is None or mask_handler.s_mask is None:
        return None

    mask = mask_handler.s_mask.mask
    indices = torch.nonzero(mask).squeeze()
    if indices.dim() == 0: indices = indices.unsqueeze(0)
    return indices


def convert_to_lean_model(masked_model, save_path=None):
    print("Starting Model Surgery (Physical Removal)...")

    compress_rates = []
    for m in masked_model.modules():
        if isinstance(m, ConvBNReLU):
            if m.s_mask is not None:
                mask = m.s_mask.mask
                kept = mask.sum().item()
                total = mask.numel()
                compress_rates.append(1.0 - (kept / total))
            else:
                compress_rates.append(0.0)

    print(f"Extracted {len(compress_rates)} layers. Creating Lean Model...")

    num_classes = masked_model.fc.out_features if hasattr(masked_model, 'fc') else 1000

    first_layer_block = masked_model.layer1[0]
    try:
        if isinstance(first_layer_block, Bottleneck):
            print("ResNet50")
            lean_model = resnet_50(compress_rate=compress_rates, num_classes=num_classes)
        elif isinstance(first_layer_block, BasicBlock):
            print("ResNet18")
            lean_model = resnet_18(compress_rate=compress_rates, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown block type: {type(first_layer_block)}")

    except Exception as e:
        print(f"Error creating lean model: {e}")
        return None

    print("Copying weights based on masks...")

    def get_layer(model, name):
        return dict(model.named_modules())[name]

    lean_state_dict = lean_model.state_dict()
    masked_state_dict = masked_model.state_dict()

    for name, param in lean_model.named_parameters():
        if name not in masked_state_dict:
            continue

        masked_param = masked_state_dict[name]
        module_name = ".".join(name.split(".")[:-1])

        lean_module = get_layer(lean_model, module_name)

        if isinstance(lean_module, nn.Conv2d):
            if param.shape == masked_param.shape:
                param.data.copy_(masked_param.data)
                continue

            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)

            out_idx = get_kept_indices(parent_masked_module)

            if out_idx is not None and len(out_idx) == param.shape[0]:
                w_temp = masked_param.data[out_idx, :, :, :]

                if param.shape[1] < masked_param.shape[1]:
                    param.data.copy_(w_temp[:, :param.shape[1], :, :])
                else:
                    param.data.copy_(w_temp)
            else:
                print(f"Shape mismatch heavy at {name}: {param.shape} vs {masked_param.shape}. Slicing...")
                param.data.copy_(masked_param.data[:param.shape[0], :param.shape[1], :, :])

        elif isinstance(lean_module, nn.BatchNorm2d):
            parent_name = ".".join(module_name.split(".")[:-1])
            parent_masked_module = get_layer(masked_model, parent_name)
            out_idx = get_kept_indices(parent_masked_module)

            if out_idx is not None and len(out_idx) == param.shape[0]:
                param.data.copy_(masked_param.data[out_idx])
            else:
                param.data.copy_(masked_param.data[:param.shape[0]])

        else:
            if param.shape == masked_param.shape:
                param.data.copy_(masked_param.data)

    if save_path:
        torch.save(lean_model.state_dict(), save_path)
        import json
        with open(save_path.replace('.pth', '.json'), 'w') as f:
            json.dump(compress_rates, f)

    print(f"Surgery Completed. Lean model saved to {save_path}")
    return lean_model