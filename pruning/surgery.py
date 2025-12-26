import torch
import torch.nn as nn
from models.conv_bn_relu import ConvBNReLU
# Note: ResNet-specific lean reconstruction removed; support EfficientNet by fallback.


def get_kept_indices(mask_handler):
    """
    Trả về danh sách index của các filter được giữ lại (Mask = 1).
    """
    if mask_handler is None or mask_handler.s_mask is None:
        return None  # Giữ lại tất cả

    mask = mask_handler.s_mask.mask
    # Lấy các vị trí có giá trị = 1
    indices = torch.nonzero(mask).squeeze()
    if indices.dim() == 0: indices = indices.unsqueeze(0)
    return indices


def convert_to_lean_model(masked_model, save_path=None):
    print("Starting Model Surgery (Physical Removal)...")

    # 1. Trích xuất Compress Rates từ các ConvBNReLU có mask
    compress_rates = []
    for m in masked_model.modules():
        if isinstance(m, ConvBNReLU):
            if m.s_mask is not None and hasattr(m.s_mask, 'mask') and m.s_mask.mask is not None:
                mask = m.s_mask.mask
                kept = mask.sum().item()
                total = mask.numel()
                compress_rates.append(1.0 - (kept / total))
            else:
                compress_rates.append(0.0)

    print(f"Extracted {len(compress_rates)} masked convs.")

    # 2. Thử tái tạo lean model nếu là ResNet-hybrid; nếu không, fallback cho EfficientNet
    lean_model = None
    print("EfficientNet pipeline: skipping structural rebuild; exporting compress_rates only.")
    lean_model = None

    # 3. Nếu không thể tái cấu trúc, chỉ lưu compress_rates (nếu cần) và trả về None để caller xử lý
    if lean_model is None:
        if save_path:
            import json
            with open(save_path.replace('.pth', '.json'), 'w') as f:
                json.dump(compress_rates, f)
            print(f"Saved compress_rates config to {save_path.replace('.pth', '.json')}")
        return None

    # 3. Copy Weights (Smart Copy)
    print("Copying weights based on masks...")

    # Helper để lấy layer từ model bằng tên (vd: layer1[0].conv1)
    def get_layer(model, name):
        return dict(model.named_modules())[name]

    # Map state_dict của lean model
    lean_state_dict = lean_model.state_dict()
    masked_state_dict = masked_model.state_dict()

    # Chúng ta cần theo dõi 'indices' của layer trước đó để cắt chiều Input
    # Với ResNet, luồng dữ liệu phức tạp (Skip connection).
    # Để đơn giản và hiệu quả, ta sẽ duyệt theo cấu trúc Block.

    # A. Copy các layer không bị prune (FC, Stem đầu vào nếu ko prune)
    # Lưu ý: Stem (conv1) input là ảnh (3 kênh) -> Ko cần cắt chiều In.

    # Ta duyệt qua từng tên tham số trong Lean Model
    for name, param in lean_model.named_parameters():
        if name not in masked_state_dict:
            continue

        masked_param = masked_state_dict[name]

        # Tách tên module và tên tham số (vd: layer1.0.conv1.conv.weight -> layer1.0.conv1.conv, weight)
        module_name = ".".join(name.split(".")[:-1])
        param_type = name.split(".")[-1]  # weight, bias

        lean_module = get_layer(lean_model, module_name)
        masked_module = get_layer(masked_model, module_name)

        # 1. Xử lý Conv2d
        if isinstance(lean_module, nn.Conv2d):
            # Tìm mask của chính layer này (Output Mask)
            # Vì lean_module nằm trong ConvBNReLU của lean_model, ta cần tìm ConvBNReLU cha
            # Nhưng ở đây ta đang iterate param của nn.Conv2d.
            # Parent của nn.Conv2d chính là ConvBNReLU trong code resnet_hybrid.

            # Để đơn giản: Ta so sánh shape.
            # Nếu shape khớp -> Copy thẳng.
            if param.shape == masked_param.shape:
                param.data.copy_(masked_param.data)
                continue

            # Nếu shape không khớp -> Cần cắt.
            # Ta cần tìm Mask Out và Mask In.

            # Lấy parent module (ConvBNReLU) bên Masked Model
            # Tên module_name ví dụ: layer1.0.conv1.conv
            parent_name = ".".join(module_name.split(".")[:-1])  # layer1.0.conv1
            parent_masked_module = get_layer(masked_model, parent_name)

            # Lấy Indices Output (Filter cần giữ)
            out_idx = get_kept_indices(parent_masked_module)


            if out_idx is not None and len(out_idx) == param.shape[0]:
                # Cắt chiều Output
                w_temp = masked_param.data[out_idx, :, :, :]

                # Cắt chiều Input
                # Nếu số kênh input cũng bị giảm (lean < masked)
                if param.shape[1] < masked_param.shape[1]:
                    param.data.copy_(w_temp[:, :param.shape[1], :, :])
                else:
                    param.data.copy_(w_temp)
            else:
                print(f"Shape mismatch heavy at {name}: {param.shape} vs {masked_param.shape}. Slicing...")
                param.data.copy_(masked_param.data[:param.shape[0], :param.shape[1], :, :])

        # 2. Xử lý BatchNorm (weight, bias, running_mean, running_var)
        elif isinstance(lean_module, nn.BatchNorm2d):

            # Tên module: layer1.0.conv1.bn
            parent_name = ".".join(module_name.split(".")[:-1])  # layer1.0.conv1
            parent_masked_module = get_layer(masked_model, parent_name)

            out_idx = get_kept_indices(parent_masked_module)

            if out_idx is not None and len(out_idx) == param.shape[0]:
                param.data.copy_(masked_param.data[out_idx])
            else:
                param.data.copy_(masked_param.data[:param.shape[0]])

        # 3. Các layer khác (FC)
        else:
            if param.shape == masked_param.shape:
                param.data.copy_(masked_param.data)

    if save_path:
        torch.save(lean_model.state_dict(), save_path)
        import json
        with open(save_path.replace('.pth', '.json'), 'w') as f:
            json.dump(compress_rates, f)

    return lean_model
