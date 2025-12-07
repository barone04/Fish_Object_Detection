import math
import sys
import time
import torch
from . import utils

# Import các công cụ đánh giá mAP
# Giả sử chúng nằm trong data/ hoặc thư mục gốc, tùy cách setup path
from data.coco_utils import get_coco_api_from_dataset
from data.coco_eval import CocoEvaluator


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,
                    scaler=None, pruner=None):
    """
    Huấn luyện 1 epoch cho Detection (Pipeline 2 End-to-End).
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # Warmup Learning Rate cho epoch đầu tiên (giúp ổn định RPN)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # Image list -> device
        images = list(image.to(device) for image in images)
        # Target dict -> device
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # AMP (Mixed Precision)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Kiểm tra loss có bị NaN/Inf không
        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        # --- PRUNING HOOK ---
        # Ép mask = 0 sau khi optimizer cập nhật trọng số
        if pruner is not None:
            pruner.apply_masks()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses.item(), **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Chạy Inference và tính mAP chuẩn COCO.
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # Tránh conflict CPU
    cpu_device = torch.device("cpu")
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # Chuẩn bị Evaluator
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]  # Chỉ quan tâm bbox, ko có mask
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        outputs = model(images)

        # Chuyển output về CPU để tính toán
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # Mapping output với image_id
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # Tổng hợp kết quả từ nhiều GPU (nếu distributed)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Tính mAP
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    torch.set_num_threads(n_threads)
    return coco_evaluator