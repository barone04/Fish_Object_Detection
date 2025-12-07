import os
import torch
import torch.utils.data
import cv2
import numpy as np
from PIL import Image


class FishDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transforms=None):
        """
        Args:
            root (string): Đường dẫn gốc tới dataset (e.g., './dataset/Fish_Dataset')
            split (string): 'train', 'valid', hoặc 'test'
            transforms (callable, optional): Các transform áp dụng lên ảnh và target
        """
        self.root = root
        self.transforms = transforms

        # Mapping tên thư mục Kaggle
        # Cấu trúc mong đợi: root/train/images, root/train/labels
        if split == 'val': split = 'valid'  # Fix tên thư mục nếu cần

        self.img_dir = os.path.join(root, split, "images")
        self.lbl_dir = os.path.join(root, split, "labels")

        # Lấy danh sách file ảnh, lọc chỉ lấy .jpg hoặc .png
        self.imgs = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')])

    def __getitem__(self, idx):
        # 1. Load Ảnh
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        lbl_path = os.path.join(self.lbl_dir, self.imgs[idx].rsplit('.', 1)[0] + '.txt')

        # Dùng PIL để tương thích với torchvision transforms
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []
        area = []

        # 2. Load Label (YOLO Format)
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = list(map(float, line.strip().split()))
                    if len(data) < 5: continue

                    cls_id = int(data[0])
                    cx, cy, bw, bh = data[1], data[2], data[3], data[4]

                    # Convert YOLO (norm) -> XYXY (absolute)
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h

                    # Clip coordinates để không văng ra ngoài ảnh
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    # Loại bỏ các box bị lỗi (diện tích <= 0)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes.append([x1, y1, x2, y2])
                    # Faster R-CNN dành class 0 cho Background -> Fish class + 1
                    labels.append(cls_id + 1)
                    area.append((x2 - x1) * (y2 - y1))

        # 3. Tạo Target Dictionary
        target = {}

        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["area"] = torch.as_tensor(area, dtype=torch.float32)
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Xử lý trường hợp ảnh không có cá (Negative sample)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        target["image_id"] = torch.tensor([idx])

        # 4. Apply Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    # Hàm hỗ trợ GroupedBatchSampler (của CORING)
    def get_height_and_width(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        # Đọc header ảnh để lấy size nhanh mà không cần decode full ảnh
        img = Image.open(img_path)
        return img.height, img.width


# Hàm Collate (để gom batch các ảnh có kích thước khác nhau)
def collate_fn(batch):
    return tuple(zip(*batch))