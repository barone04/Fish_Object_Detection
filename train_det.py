# import argparse
# import os
# import torch
# from engines import trainer_det, utils as engine_utils
# from data.fish_det_dataset import FishDetectionDataset, collate_fn
# from data import presets
# from models.faster_rcnn import fasterrcnn_resnet50_fpn
#
# # --- THÊM ĐOẠN NÀY VÀO ĐẦU FILE ---
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
#
# def get_args_parser():
#     parser = argparse.ArgumentParser(description="Detection Training")
#     parser.add_argument("--data-path", default="dataset/fish_detection", type=str)
#     parser.add_argument("--model", default="fasterrcnn_resnet50_fpn", type=str)
#     parser.add_argument("--device", default="cuda", type=str)
#     parser.add_argument("--batch-size", default=8, type=int)
#     parser.add_argument("--epochs", default=20, type=int)
#     parser.add_argument("--workers", default=4, type=int)
#     parser.add_argument("--lr", default=0.02, type=float)
#     parser.add_argument("--weight-decay", default=1e-4, type=float)
#     parser.add_argument("--momentum", default=0.9, type=float)
#     parser.add_argument("--output-dir", default=".", type=str)
#     parser.add_argument("--resume", default="", type=str, help="path to checkpoint to resume")
#
#     # Pruning specific
#     parser.add_argument("--weights", default=None, type=str, help="path to full model")
#     # THÊM DÒNG NÀY:
#     parser.add_argument("--weights-backbone", default=None, type=str, help="path to backbone only")
#     parser.add_argument("--compress-rate", default=None, type=str, help="path to json config or string list")
#
#     return parser
#
#
# def main(args):
#     engine_utils.init_distributed_mode(args)
#     device = torch.device(args.device)
#
#     # 1. Data
#     print("Loading Data...")
#     dataset_train = FishDetectionDataset(args.data_path, split='train',
#                                          transforms=presets.DetectionPresetTrain(data_augmentation='hflip'))
#     dataset_test = FishDetectionDataset(args.data_path, split='val', transforms=presets.DetectionPresetEval())
#
#     train_sampler = torch.utils.data.RandomSampler(dataset_train)
#     test_sampler = torch.utils.data.SequentialSampler(dataset_test)
#
#     data_loader_train = torch.utils.data.DataLoader(
#         dataset_train, batch_size=args.batch_size, sampler=train_sampler,
#         num_workers=args.workers, collate_fn=collate_fn
#     )
#     data_loader_test = torch.utils.data.DataLoader(
#         dataset_test, batch_size=1, sampler=test_sampler,
#         num_workers=args.workers, collate_fn=collate_fn
#     )
#
#     # 2. Model
#     print("Creating Model...")
#
#     cpr = args.compress_rate
#     compress_rate_list = None  # Mặc định là None (không nén)
#
#     if cpr:
#         # Trường hợp 1: Là đường dẫn file JSON tồn tại
#         if os.path.exists(cpr) and os.path.isfile(cpr):
#             print(f"Loading compress_rate from file: {cpr}")
#             import json
#             with open(cpr, 'r') as f:
#                 compress_rate_list = json.load(f)
#
#         # Trường hợp 2: Là chuỗi list string "[0.1, 0.5...]"
#         elif cpr.startswith('[') and cpr.endswith(']'):
#             print(f"Loading compress_rate from string: {cpr}")
#             try:
#                 compress_rate_list = eval(cpr)
#             except:
#                 print("Error parsing compress_rate string!")
#                 raise
#
#         # Trường hợp 3: Đường dẫn file nhưng code không tìm thấy
#         else:
#             print(f"Warning: compress_rate provided '{cpr}' but file not found or invalid format!")
#             # Kiểm tra xem có phải file json mà surgery sinh ra không
#             # Trong pipeline1_transfer.sh, ta truyền config.json nhưng prune_cls lại sinh ra backbone_lean.json
#             # Hãy thử fix đuôi file
#             potential_path = cpr.replace('config.json', 'backbone_lean.json')
#             if os.path.exists(potential_path):
#                 print(f"Found alternative config file: {potential_path}")
#                 import json
#                 with open(potential_path, 'r') as f:
#                     compress_rate_list = json.load(f)
#             else:
#                 raise FileNotFoundError(f"Cannot find compress_rate config at {cpr}")
#
#     # Logic ưu tiên: Nếu có weights-backbone thì dùng nó, không thì dùng weights
#     w_backbone = args.weights_backbone
#
#     # Gọi hàm dựng model
#     model = fasterrcnn_resnet50_fpn(
#         num_classes=14,
#         compress_rate=compress_rate_list,
#         weights_backbone=w_backbone  # Truyền đúng tham số này
#     )
#
#     # Nếu muốn load full model (cho finetune pipeline 2)
#     if args.weights:
#         checkpoint = torch.load(args.weights, map_location='cpu')
#         # Load flexible (bỏ qua layer không khớp nếu cần)
#         if 'model' in checkpoint: checkpoint = checkpoint['model']
#         model.load_state_dict(checkpoint, strict=False)
#         print(f"Loaded weights from {args.weights}")
#
#     model.to(device)
#
#     # 3. Optimizer
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#     scaler = torch.cuda.amp.GradScaler()
#
#     # 4. Training Loop
#     for epoch in range(args.epochs):
#         trainer_det.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50, scaler=scaler)
#
#         if args.output_dir:
#             engine_utils.save_on_master({
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'args': args
#             }, os.path.join(args.output_dir, f"model_{epoch}.pth"))
#
#         # Eval & Save Best
#         # (Bạn có thể thêm logic lưu best mAP ở đây)
#         trainer_det.evaluate(model, data_loader_test, device=device)
#
#     # Save final
#     if args.output_dir:
#         engine_utils.save_on_master({'model': model.state_dict()}, os.path.join(args.output_dir, "model_best.pth"))
#
#
# if __name__ == "__main__":
#     args = get_args_parser().parse_args()
#     engine_utils.mkdir(args.output_dir)
#     main(args)

import argparse
import os
import torch
from engines import trainer_det, utils as engine_utils
from data.fish_det_dataset import FishDetectionDataset, collate_fn
from data import presets
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet18_fpn
import json

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')



def get_args_parser():
    parser = argparse.ArgumentParser(description="Detection Training")
    parser.add_argument("--data-path", default="NewDeepfish/NewDeepfish", type=str)
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
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')

    # Pruning specific
    parser.add_argument("--weights", default=None, type=str, help="path to full model")
    parser.add_argument("--weights-backbone", default=None, type=str, help="path to backbone only")
    parser.add_argument("--compress-rate", default=None, type=str, help="path to json config or string list")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    return parser


def main(args):
    engine_utils.init_distributed_mode(args)
    device = torch.device(args.device)

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

    print("Creating Model...")

    backbone_rates = None
    fpn_rates = None

    # Load Config từ JSON
    if args.compress_rate:
        print(f"Loading pruning config from: {args.compress_rate}")
        with open(args.compress_rate, 'r') as f:
            config_data = json.load(f)

            # Xử lý tương thích ngược
            if isinstance(config_data, list):
                # Format cũ: Chỉ có backbone
                backbone_rates = config_data
                fpn_rates = None
            elif isinstance(config_data, dict):
                # Format mới: {backbone: [], fpn: []}
                backbone_rates = config_data.get('backbone')
                fpn_rates = config_data.get('fpn')

    if args.model == 'fasterrcnn_resnet18_fpn':
        model = fasterrcnn_resnet18_fpn(
            num_classes=2,
            weights_backbone=args.weights_backbone,
            compress_rate=backbone_rates,
            fpn_compress_rate=fpn_rates
        )
    else:  # ResNet50
        model = fasterrcnn_resnet50_fpn(
            num_classes=2,
            weights_backbone=args.weights_backbone,
            compress_rate=backbone_rates,
            fpn_compress_rate=fpn_rates
        )

    if args.weights:
        checkpoint = torch.load(args.weights, map_location='cpu')
        if 'model' in checkpoint: checkpoint = checkpoint['model']
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded weights from {args.weights}")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_steps,
        gamma=args.lr_gamma
    )
    scaler = torch.cuda.amp.GradScaler()


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        if 'lr_scheduler' in checkpoint and lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    if args.test_only:
        print("Running in Test-Only mode...")
        trainer_det.evaluate(model, data_loader_test, device=device)
        return

    best_map = 0.0

    for epoch in range(start_epoch, args.epochs):
        trainer_det.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50, scaler=scaler)
        lr_scheduler.step()

        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args
        }

        if args.output_dir:
            engine_utils.save_on_master(checkpoint_dict, os.path.join(args.output_dir, "model_last.pth"))

        coco_evaluator = trainer_det.evaluate(model, data_loader_test, device=device)

        current_map = 0.0
        if coco_evaluator is not None and hasattr(coco_evaluator, 'coco_eval'):
            current_map = coco_evaluator.coco_eval['bbox'].stats[0]

        print(f"Epoch {epoch}: Current mAP = {current_map:.4f} | Best mAP = {best_map:.4f}")

        # Logic so sánh và lưu Best Model
        if args.output_dir and current_map > best_map:
            best_map = current_map
            print(f"--> Found New Best Model ({best_map:.4f})! Saving to model_best.pth")
            engine_utils.save_on_master(checkpoint_dict, os.path.join(args.output_dir, "model_best.pth"))

    print(f"Training Finished. Best mAP achieved: {best_map:.4f}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    engine_utils.mkdir(args.output_dir)
    main(args)