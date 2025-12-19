import os
import torch
import torch.utils.data
from PIL import Image


class FishDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transforms=None):
        """
        Args:
            root (string): Đường dẫn gốc tới dataset (VD: './NewDeepfish/NewDeepfish')
            split (string): 'train', 'val' (hoặc 'val'), 'test'
            transforms (callable, optional): Các transform áp dụng lên ảnh và target
        """
        self.root = root
        self.transforms = transforms

        if split == 'val': split = 'val'

        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Path not found: {self.img_dir}. Check your dataset structure!")

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        self.imgs = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(valid_exts)])

        print(f"Found {len(self.imgs)} images in {self.img_dir}")

    def __getitem__(self, idx):
        # 1. Load Ảnh
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load ảnh bằng PIL và convert sang RGB
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Load Label (YOLO format: class_id cx cy bw bh)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(self.lbl_dir, lbl_name)

        boxes = []
        labels = []
        area = []

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    data = list(map(float, line.strip().split()))
                    if len(data) >= 5:
                        cls_id = int(data[0])  # Thường là 0 (vì dataset chỉ có 1 loại cá)
                        cx, cy, bw, bh = data[1], data[2], data[3], data[4]

                        # Convert YOLO (Center_X, Center_Y, W, H) -> COCO (X1, Y1, X2, Y2)
                        # Tọa độ YOLO là normalized (0-1), cần nhân với w, h
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h

                        # Clip to image boundaries
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)

                        # Chỉ lấy box hợp lệ (có diện tích > 0)
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            # Faster R-CNN quy ước: 0 là Background.
                            # Nên class vật thể bắt đầu từ 1.
                            labels.append(1)  # Luôn là 1 vì chỉ có 1 loài cá (class_agnostic)
                            area.append((x2 - x1) * (y2 - y1))

        target = {}
        target["image_id"] = torch.tensor([idx])

        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["area"] = torch.as_tensor(area, dtype=torch.float32)
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Negative sample
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        # 4. Apply Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    # Hàm hỗ trợ cho Sampler (nếu dùng GroupedBatchSampler để tối ưu size ảnh)
    def get_height_and_width(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        with Image.open(img_path) as img:
            return img.height, img.width


# Hàm collate_fn để gom batch (bắt buộc cho Detection vì size ảnh/box khác nhau)
def collate_fn(batch):
    return tuple(zip(*batch))