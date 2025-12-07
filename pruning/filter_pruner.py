import torch
import torch.nn as nn
from .utils import get_weight, l1inftyinfty, l1inftyinfty_distance


class StructuredPruner:
    def __init__(self, model):
        self.model = model

    def compute_distance_matrix(self, layer):
        """
        Tính ma trận khoảng cách giữa các filter trong layer.
        """
        weight = get_weight(layer)
        n = weight.shape[0]  # Out channels

        # Nếu số filter quá lớn (>1024), có thể tốn time O(N^2), nhưng ResNet50 ok.
        D = torch.zeros(n, n, device=weight.device)

        # Duyệt qua các cặp
        for i in range(n):
            for j in range(i + 1, n):
                dist = l1inftyinfty_distance(weight[i], weight[j])
                D[i, j] = dist
                D[j, i] = dist  # Symmetric

        return D

    def prune(self, prune_ratio=0.3):
        """
        Prune các filter trùng lặp/yếu dựa trên ratio.
        prune_ratio: Tỷ lệ số filter muốn cắt bỏ (VD: 0.3 = cắt 30%).
        """
        print(f"Executing Filter Pruning (Ratio={prune_ratio})...")

        # Lấy danh sách các layer hỗ trợ structured pruning (ConvBNReLU)
        # Lưu ý: Trong ResNet, ta thường chỉ prune Conv1 và Conv2 của Bottleneck
        # Conv3 (Expansion) thường phải giữ nguyên channel để cộng với Shortcut (trừ khi prune cả shortcut)
        # Để đơn giản và an toàn, ta prune tất cả layers trả về từ model
        convs = self.model.get_prunable_layers(pruning_type="structured")

        total_pruned = 0

        for layer in convs:
            # 1. Tính ma trận khoảng cách
            D = self.compute_distance_matrix(layer)
            weight = get_weight(layer)
            n = weight.shape[0]

            # 2. Xác định số lượng cần cắt
            num_to_remove = int(round(n * prune_ratio))
            if num_to_remove == 0:
                # Nếu không cắt, vẫn update mask toàn 1
                layer.mask_handler.update(torch.ones(n, device=weight.device))
                continue

            # 3. Tìm các cặp giống nhau nhất (khoảng cách nhỏ nhất)
            # Lấy tam giác trên của ma trận D để không lặp
            mask_tri = torch.triu(torch.ones_like(D), diagonal=1)
            # D_flat chứa (dist, i, j)
            # Lọc các phần tử > 0 (nếu dist=0 tức là filter chết sẵn, ta xử lý sau)

            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((D[i, j].item(), i, j))

            # Sort tăng dần khoảng cách (càng nhỏ càng giống)
            pairs.sort(key=lambda x: x[0])

            # 4. Logic chọn filter để cắt (Code của bạn)
            pruned_indices = set()
            processed_indices = set()
            count = 0

            # Ưu tiên cắt cặp giống nhau trước
            for dist, i, j in pairs:
                if count >= num_to_remove: break
                if i in processed_indices or j in processed_indices: continue

                # So sánh độ lớn (norm)
                norm_i = l1inftyinfty(weight[i])
                norm_j = l1inftyinfty(weight[j])

                # Giữ cái mạnh, diệt cái yếu
                if norm_i >= norm_j:
                    kill_idx = j
                else:
                    kill_idx = i

                pruned_indices.add(kill_idx)
                processed_indices.add(i)
                processed_indices.add(j)
                count += 1

            # Nếu vẫn chưa cắt đủ số lượng (do hết cặp rời rạc),
            # cắt tiếp các filter có norm nhỏ nhất còn lại (Magnitude Pruning filler)
            if count < num_to_remove:
                remaining_indices = [k for k in range(n) if k not in pruned_indices]
                norms = [(l1inftyinfty(weight[k]), k) for k in remaining_indices]
                norms.sort(key=lambda x: x[0])  # Tăng dần

                for _, idx in norms:
                    if count >= num_to_remove: break
                    pruned_indices.add(idx)
                    count += 1

            # 5. Tạo Mask vector
            mask_vec = torch.ones(n, device=weight.device)
            if pruned_indices:
                mask_vec[list(pruned_indices)] = 0.0

            # 6. Update vào StructuredMask handler
            layer.mask_handler.update(mask_vec)
            # Apply ngay (Zeroing row)
            layer.mask_handler.apply(layer.conv)

            total_pruned += count

        print(f"Total filters logically pruned: {total_pruned}")