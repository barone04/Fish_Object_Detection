# import math
# import sys
# import time
# import torch
# from . import utils
#
# # Import các công cụ đánh giá mAP
# # Giả sử chúng nằm trong data/ hoặc thư mục gốc, tùy cách setup path
# from data.coco_utils import get_coco_api_from_dataset
# from data.coco_eval import CocoEvaluator
#
#
# def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,
#                     scaler=None, pruner=None):
#     """
#     Huấn luyện 1 epoch cho Detection (Pipeline 2 End-to-End).
#     """
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     header = f"Epoch: [{epoch}]"
#
#     # Warmup Learning Rate cho epoch đầu tiên (giúp ổn định RPN)
#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000
#         warmup_iters = min(1000, len(data_loader) - 1)
#         lr_scheduler = torch.optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=warmup_factor, total_iters=warmup_iters
#         )
#
#     for images, targets in metric_logger.log_every(data_loader, print_freq, header):
#         # Image list -> device
#         images = list(image.to(device) for image in images)
#         # Target dict -> device
#         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
#
#         # AMP (Mixed Precision)
#         with torch.cuda.amp.autocast(enabled=scaler is not None):
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())
#
#         # Kiểm tra loss có bị NaN/Inf không
#         loss_value = losses.item()
#         if not math.isfinite(loss_value):
#             print(f"Loss is {loss_value}, stopping training")
#             sys.exit(1)
#
#         optimizer.zero_grad()
#
#         if scaler is not None:
#             scaler.scale(losses).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             losses.backward()
#             optimizer.step()
#
#         # --- PRUNING HOOK ---
#         # Ép mask = 0 sau khi optimizer cập nhật trọng số
#         if pruner is not None:
#             pruner.apply_masks()
#
#         if lr_scheduler is not None:
#             lr_scheduler.step()
#
#         metric_logger.update(loss=losses.item(), **loss_dict)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#
#     return metric_logger
#
#
# @torch.no_grad()
# def evaluate(model, data_loader, device):
#     """
#     Chạy Inference và tính mAP chuẩn COCO.
#     """
#     n_threads = torch.get_num_threads()
#     torch.set_num_threads(1)  # Tránh conflict CPU
#     cpu_device = torch.device("cpu")
#     model.eval()
#
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Test:"
#
#     # Chuẩn bị Evaluator
#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = ["bbox"]  # Chỉ quan tâm bbox, ko có mask
#     coco_evaluator = CocoEvaluator(coco, iou_types)
#
#     for images, targets in metric_logger.log_every(data_loader, 100, header):
#         images = list(img.to(device) for img in images)
#
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#
#         model_time = time.time()
#         outputs = model(images)
#
#         # Chuyển output về CPU để tính toán
#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time
#
#         # Mapping output với image_id
#         res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
#
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
#
#     # Tổng hợp kết quả từ nhiều GPU (nếu distributed)
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#
#     # Tính mAP
#     coco_evaluator.synchronize_between_processes()
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#
#     torch.set_num_threads(n_threads)
#     return coco_evaluator

import math
import sys
import time
import torch
import contextlib
import os
import datetime

from . import utils
from data.coco_utils import get_coco_api_from_dataset
from data.coco_eval import CocoEvaluator


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    # Header giống YOLO
    # Faster R-CNN không có dfl_loss, ta thay bằng rpn_loss (tổng objectness + rpn_box)
    header_titles = f"{'Epoch':>10} {'GPU_mem':>10} {'box_loss':>10} {'cls_loss':>10} {'rpn_loss':>10} {'Instances':>10} {'Size':>10}"

    # In header mỗi đầu epoch
    if utils.is_main_process():
        print(header_titles)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # Để đếm số lượng Instances (số cá trong batch)
    total_instances = 0

    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header="")):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Đếm instances
        batch_instances = sum(len(t["boxes"]) for t in targets)
        total_instances = batch_instances  # Lấy số hiện tại của batch

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Giảm loss dict qua các GPU (nếu có) để log cho đúng
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update metrics
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # --- CUSTOM PRINTING STYLE YOLO ---
        if utils.is_main_process():
            # CHỈ IN KHI i CHIA HẾT CHO print_freq (ví dụ 50)
            if i % print_freq == 0:
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.2f}G"
                box_loss = metric_logger.meters['loss_box_reg'].avg
                cls_loss = metric_logger.meters['loss_classifier'].avg
                rpn_loss = metric_logger.meters['loss_objectness'].avg + metric_logger.meters['loss_rpn_box_reg'].avg

                img_size = f"{images[0].shape[-2]}:{images[0].shape[-1]}"
                epoch_str = f"{epoch + 1}/{20}"  # Ví dụ tổng 20 epoch

                log_msg = f"{epoch_str:>10} {mem:>10} {box_loss:>10.4f} {cls_loss:>10.4f} {rpn_loss:>10.4f} {total_instances:>10} {img_size:>10}"
                print(log_msg)

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()

        # Bịt miệng COCO loadRes (nguyên nhân gây spam chính)
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_evaluator.update(res)

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Synchronize evaluator
    coco_evaluator.synchronize_between_processes()

    # Bịt miệng accumulate & summarize mặc định
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

    torch.set_num_threads(n_threads)

    if coco_evaluator.coco_eval['bbox'] is not None:
        stats = coco_evaluator.coco_eval['bbox'].stats

        map50_95 = stats[0]
        map50 = stats[1]
        recall = stats[8]  # AR maxDets=100

        # Precision trong COCO API không có sẵn trực tiếp như một con số tổng hợp đơn giản
        # Nhưng ta có thể để trống hoặc dùng mAP50 làm đại diện gần đúng cho chất lượng Box
        precision_display = "-"

        print("\n" + "=" * 90)
        print(f"{'Class':>20} {'Images':>10} {'Instances':>10} {'Box(P)':>10} {'R':>10} {'mAP50':>10} {'mAP50-95':>10}")
        print("-" * 90)

        # Dòng tổng hợp 'all'
        num_images = len(data_loader.dataset)
        # Instances đếm hơi khó nếu không duyệt lại, ta để '-' hoặc lấy xấp xỉ
        print(
            f"{'all':>20} {num_images:>10} {'-':>10} {precision_display:>10} {recall:>10.3f} {map50:>10.3f} {map50_95:>10.3f}")
        print("=" * 90 + "\n")

    return coco_evaluator