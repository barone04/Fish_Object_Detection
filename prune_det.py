import argparse
import os
import torch
import utils
from engines import trainer_det, utils as engine_utils
from data.fish_det_dataset import FishDetectionDataset, collate_fn
from data import presets
from models.faster_rcnn import fasterrcnn_resnet50_fpn


def get_args_parser():
    parser = argparse.ArgumentParser(description="Detection Training")
    parser.add_argument("--data-path", default="dataset/fish_detection", type=str)
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--output-dir", default=".", type=str)
    parser.add_argument("--resume", default="", type=str, help="path to checkpoint to resume")

    # Pruning specific
    parser.add_argument("--weights", default=None, type=str, help="path to .pth backbone or full model")
    parser.add_argument("--compress-rate", default=None, type=str, help="path to json config or string list")

    return parser


def main(args):
    engine_utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # 1. Data
    print("Loading Data...")
    dataset_train = FishDetectionDataset(args.data_path, split='train',
                                         transforms=presets.DetectionPresetTrain(data_augmentation='hflip'))
    dataset_test = FishDetectionDataset(args.data_path, split='val', transforms=presets.DetectionPresetEval())

    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler,
        num_workers=args.workers, collate_fn=collate_fn
    )

    # 2. Model
    print("Creating Model...")
    # Parse compress_rate if it's a file path or string
    cpr = args.compress_rate
    if cpr and os.path.isfile(cpr):
        import json
        with open(cpr, 'r') as f:
            cpr = json.load(f)
    elif cpr:
        cpr = eval(cpr)  # Cẩn thận: eval string list

    # weights ở đây dùng để load model full (cho finetune) hoặc backbone (cho transfer)
    # Ở đây ta giả định load full model nếu finetune
    model = fasterrcnn_resnet50_fpn(num_classes=14, compress_rate=cpr)

    if args.weights:
        checkpoint = torch.load(args.weights, map_location='cpu')
        # Load flexible (bỏ qua layer không khớp nếu cần)
        if 'model' in checkpoint: checkpoint = checkpoint['model']
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded weights from {args.weights}")

    model.to(device)

    # 3. Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # 4. Training Loop
    for epoch in range(args.epochs):
        trainer_det.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50, scaler=scaler)

        if args.output_dir:
            engine_utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            }, os.path.join(args.output_dir, f"model_{epoch}.pth"))

        # Eval & Save Best
        # (Bạn có thể thêm logic lưu best mAP ở đây)
        trainer_det.evaluate(model, data_loader_test, device=device)

    # Save final
    if args.output_dir:
        engine_utils.save_on_master({'model': model.state_dict()}, os.path.join(args.output_dir, "model_best.pth"))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    engine_utils.mkdir(args.output_dir)
    main(args)