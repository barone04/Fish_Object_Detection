import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class FishClassificationDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        self.root = root

        if split == 'val': split = 'valid'
        self.img_dir = os.path.join(root, split, "images")
        self.lbl_dir = os.path.join(root, split, "labels")
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])

        self.transforms = transforms
        if self.transforms is None:
            # Default transform cho classification (Resize về 224x224 chuẩn ResNet)
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

        # Pre-scan: Tạo danh sách tất cả các con cá (Crop instances)
        # Mỗi phần tử là: (img_path, box_xyxy, label)
        self.samples = []
        print(f"Scanning dataset for classification samples ({split})...")

        for img_file in self.img_files:
            lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
            lbl_path = os.path.join(self.lbl_dir, lbl_file)
            img_path = os.path.join(self.img_dir, img_file)

            if os.path.exists(lbl_path):
                # Cần đọc ảnh để biết width/height cho việc convert tọa độ
                # Lưu ý: Việc open này chỉ đọc header, chưa load pixel nên nhanh
                with Image.open(img_path) as im:
                    w, h = im.size

                with open(lbl_path, 'r') as f:
                    for line in f:
                        data = list(map(float, line.strip().split()))
                        cls_id = int(data[0])  # 0-12

                        # YOLO to XYXY
                        cx, cy, bw, bh = data[1], data[2], data[3], data[4]
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)

                        # Validate crop
                        x1 = max(0, x1);
                        y1 = max(0, y1)
                        x2 = min(w, x2);
                        y2 = min(h, y2)

                        if x2 > x1 and y2 > y1:
                            self.samples.append((img_path, (x1, y1, x2, y2), cls_id))

        print(f"Found {len(self.samples)} fish instances.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box, label = self.samples[idx]

        # Load ảnh gốc
        img = Image.open(img_path).convert("RGB")

        # Crop con cá ra
        crop_img = img.crop(box)  # box is (left, upper, right, lower)

        # Apply transforms (Resize -> Tensor)
        if self.transforms:
            crop_img = self.transforms(crop_img)

        return crop_img, label