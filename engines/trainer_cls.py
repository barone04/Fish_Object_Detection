import torch
import torch.nn as nn
from . import utils


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq,
                    scaler=None, pruner=None):
    """
    Huấn luyện 1 epoch cho Classification.
    pruner: Nếu có, sẽ gọi apply_masks sau mỗi bước optimizer để giữ số 0.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        # Mixed Precision Training (FP16) cho H100
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # --- PRUNING HOOK (Quan trọng) ---
        # Nếu đang trong giai đoạn prune, phải ép weight về 0 ngay lập tức
        if pruner is not None:
            pruner.apply_masks()

        # Logging
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1.item(), acc5=acc5.item())


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.update(acc1=acc1.item(), acc5=acc5.item())

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res