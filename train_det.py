import argparse
import os
import json
import torch
import torch.multiprocessing

from engines import trainer_det, utils as engine_utils
from data.fish_det_dataset import FishDetectionDataset, collate_fn
from data import presets
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet18_fpn
from models.mobilenet_custom import fasterrcnn_mobilenetv3_custom

torch.multiprocessing.set_sharing_strategy('file_system')


def resolve_device(requested: str) -> torch.device:
    req = (requested or "auto").lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(req)
        print("[WARN] CUDA requested but not available. Using CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def get_args_parser():
    parser = argparse.ArgumentParser(description="Detection Training")
    parser.add_argument("--data-path", default="NewDeepfish/NewDeepfish", type=str)
    parser.add_argument("--model", default="mobilenet_v3", type=str,
                        help="resnet18, fasterrcnn_resnet18_fpn, resnet50, mobilenet_v3")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--output-dir", default=".", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)

    # NEW: giảm mạnh size FasterRCNN bằng cách giảm MLP head
    parser.add_argument("--box-head-dim", default=1024, type=int,
                        help="ROI box head hidden dim (fc6/fc7). Lower => MUCH smaller model.")

    # Pruning / loading
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--weights-backbone", default=None, type=str)
    parser.add_argument("--compress-rate", default=None, type=str)
    parser.add_argument("--test-only", dest="test_only", action="store_true")
    return parser


def main(args):
    engine_utils.init_distributed_mode(args)
    device = resolve_device(args.device)
    print("Using device:", device)

    print("Loading Data...")
    dataset_train = FishDetectionDataset(args.data_path, split='train',
                                         transforms=presets.DetectionPresetTrain(data_augmentation='hflip'))
    dataset_test = FishDetectionDataset(args.data_path, split='val',
                                        transforms=presets.DetectionPresetEval())

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, sampler=torch.utils.data.RandomSampler(dataset_train),
        num_workers=args.workers, collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=torch.utils.data.SequentialSampler(dataset_test),
        num_workers=args.workers, collate_fn=collate_fn
    )

    print(f"Creating Model: {args.model}...")

    backbone_rates = None
    fpn_rates = None
    if args.compress_rate and os.path.exists(args.compress_rate):
        with open(args.compress_rate, "r") as f:
            cfg = json.load(f)
        if isinstance(cfg, list):
            backbone_rates = cfg
        elif isinstance(cfg, dict):
            backbone_rates = cfg.get("backbone")
            fpn_rates = cfg.get("fpn")

    if args.model == "mobilenet_v3":
        model = fasterrcnn_mobilenetv3_custom(
            num_classes=2,
            fpn_compress_rate=fpn_rates,
            pretrained_backbone=True,
            freeze_backbone=True,
            box_head_dim=args.box_head_dim
        )

    elif args.model in ("fasterrcnn_resnet18_fpn", "resnet18"):
        model = fasterrcnn_resnet18_fpn(
            num_classes=2,
            weights_backbone=args.weights_backbone,
            compress_rate=backbone_rates,
            fpn_compress_rate=fpn_rates
        )

    elif args.model in ("fasterrcnn_resnet50_fpn", "resnet50"):
        model = fasterrcnn_resnet50_fpn(
            num_classes=2,
            weights_backbone=args.weights_backbone,
            compress_rate=backbone_rates,
            fpn_compress_rate=fpn_rates
        )
    else:
        raise ValueError(f"Unknown model name: {args.model}")

    # load full weights
    if args.weights and os.path.exists(args.weights):
        try:
            try:
                checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(args.weights, map_location="cpu")

            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded weights from {args.weights}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    n_parameters = sum(p.numel() for p in params)
    print(f"Number of trainable parameters: {n_parameters}")

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
    )

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("AMP enabled:", use_amp)

    # resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        try:
            try:
                checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(args.resume, map_location="cpu")

            model.load_state_dict(checkpoint["model"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "lr_scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Error resuming: {e}")
            start_epoch = 0

    if args.test_only:
        trainer_det.evaluate(model, data_loader_test, device=device)
        return

    best_map = 0.0
    for epoch in range(start_epoch, args.epochs):
        trainer_det.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50, scaler=scaler)
        lr_scheduler.step()

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.output_dir:
            engine_utils.save_on_master(ckpt, os.path.join(args.output_dir, "model_last.pth"))

        coco_evaluator = trainer_det.evaluate(model, data_loader_test, device=device)
        current_map = 0.0
        if coco_evaluator is not None and hasattr(coco_evaluator, "coco_eval"):
            current_map = coco_evaluator.coco_eval["bbox"].stats[0]

        print(f"Epoch {epoch}: Current mAP={current_map:.4f} | Best mAP={best_map:.4f}")
        if args.output_dir and current_map > best_map:
            best_map = current_map
            print(f"--> New Best ({best_map:.4f}). Saving model_best.pth")
            engine_utils.save_on_master(ckpt, os.path.join(args.output_dir, "model_best.pth"))

    print(f"Training Finished. Best mAP: {best_map:.4f}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    engine_utils.mkdir(args.output_dir)
    main(args)
