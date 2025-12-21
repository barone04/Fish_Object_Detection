import copy
import torch
import torch.utils.data
from pycocotools.coco import COCO
from tqdm import tqdm


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # 1. Tạo dictionary cho images và annotations
    # Khởi tạo info ngay bên trong để định nghĩa kiểu dữ liệu hỗn hợp ngay từ đầu
    dataset = {
        'images': [],
        'categories': [],
        'annotations': [],
        'info': {
            "description": "Fish Detection Dataset",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": ""
        }
    }

    categories = set()
    img_id_map = {}  # Mapping từ dataset index sang image_id thực

    print("Converting dataset to COCO format for evaluation...")

    for i in tqdm(range(len(ds))):
        img, targets = ds[i]

        image_id = targets["image_id"].item()
        img_id_map[i] = image_id

        # Thêm thông tin ảnh
        h, w = img.shape[-2:]
        dataset['images'].append({
            'id': image_id,
            'height': h,
            'width': w,
            'file_name': f"{image_id}.jpg"
        })

        # Thêm thông tin box (annotations)
        bboxes = targets["boxes"]
        labels = targets["labels"]
        areas = targets["area"]
        iscrowd = targets["iscrowd"]

        for j in range(len(bboxes)):
            bbox = bboxes[j].tolist()
            # Convert XYXY (Torch) -> XYWH (COCO)
            xmin, ymin, xmax, ymax = bbox
            w_box, h_box = xmax - xmin, ymax - ymin
            bbox_coco = [xmin, ymin, w_box, h_box]

            category_id = labels[j].item()
            categories.add(category_id)

            ann = {
                'id': len(dataset['annotations']) + 1,
                'image_id': image_id,
                'bbox': bbox_coco,
                'category_id': category_id,
                'area': areas[j].item(),
                'iscrowd': iscrowd[j].item()
            }
            dataset['annotations'].append(ann)

    # Thêm categories
    for cat_id in categories:
        dataset['categories'].append({'id': cat_id, 'name': str(cat_id)})

    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    return convert_to_coco_api(dataset)