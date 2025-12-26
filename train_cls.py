import argparse
import os
import torch
import torch.nn as nn
from data.fish_cls_dataset import FishClassificationDataset
from models.efficientnet_hybrid import efficientnet_b0_classifier
from engines import trainer_cls, utils as engine_utils


def main(args):
    device = torch.device(args.device)
    engine_utils.init_distributed_mode(args)

    print("Loading Data...")
    dataset_train = FishClassificationDataset(args.data_path, split='train')
    dataset_val = FishClassificationDataset(args.data_path, split='val')

    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True
    )

    print(f"Creating Model: {args.model}")
    # Dùng EfficientNet-B0 pretrained làm mặc định cho classification
    model = efficientnet_b0_classifier(num_classes=13, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # Scheduler: Giảm LR khi loss đi ngang
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    scaler = torch.cuda.amp.GradScaler()

    print("Start Training...")
    for epoch in range(args.epochs):
        trainer_cls.train_one_epoch(model, criterion, optimizer, loader_train, device, epoch, 50, scaler)
        acc1 = trainer_cls.evaluate(model, criterion, loader_val, device)

        lr_scheduler.step(acc1)

        if args.output_dir:
            # Chỉ lưu nếu là epoch tốt nhất hoặc epoch cuối
            engine_utils.save_on_master({'model': model.state_dict()}, os.path.join(args.output_dir, "model_best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output-dir', default='.')
    # THÊM DÒNG NÀY ĐỂ SỬA LỖI, Thêm weight = "DEFAULT" vào để khởi tạo trọng số
    # Load pretrained model, weight = "DEFAULT" -> training -> pruning -> finetuning -> export .pth
    # Build model from scratch -> training with initial weight -> pruning -> finetuning -> export .pth
    parser.add_argument('--model', default='efficientnet_b0', help='model name')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.1)

    args = parser.parse_args()
    engine_utils.mkdir(args.output_dir)
    main(args)