import argparse
import os
import torch
import copy
from engines import trainer_det, utils as engine_utils
from data.fish_det_dataset import FishDetectionDataset, collate_fn
from data import presets
from models.faster_rcnn import fasterrcnn_resnet50_fpn

# module Pruning
from pruning.songhan_pruner import UnstructuredPruner
from pruning.filter_pruner import StructuredPruner
from pruning import surgery

# Fix lỗi multiprocessing
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser(description="Pruning Detection (Pipeline 2)")

    # Dataset & Model
    parser.add_argument("--data-path", default="./NewDeepfish/NewDeepfish", type=str)
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to Dense Detection Model (Pipeline 2 Step 1)")

    # Training Params (Finetune)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.005, type=float, help="Low LR for finetuning")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)

    # Pruning Params
    parser.add_argument("--target-sparsity", default=0.4, type=float, help="Final target sparsity (e.g. 0.4)")
    parser.add_argument("--prune-iters", default=5, type=int, help="Number of iterative pruning steps")
    parser.add_argument("--finetune-epochs", default=3, type=int, help="Epochs to finetune after each prune step")

    parser.add_argument("--output-dir", default="./output/pipeline2_pruned", type=str)

    return parser


def main(args):
    engine_utils.init_distributed_mode(args)
    engine_utils.mkdir(args.output_dir)
    device = torch.device(args.device)

    # 1. Prepare Data
    print("Loading Data...")
    # Lưu ý: Pipeline 2 dataset có 2 class (1 cá + 1 nền)
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

    # 2. Load Dense Model (Baseline)
    print(f"Loading Dense Model from {args.checkpoint}...")
    # Khởi tạo model full (Dense)
    model = fasterrcnn_resnet50_fpn(num_classes=2)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' in checkpoint: checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)
    model.to(device)

    # 3. Setup Pruners
    # Quan trọng: Faster R-CNN bọc backbone bên trong 'model.backbone.body'
    backbone_module = model.backbone.body

    u_pruner = UnstructuredPruner(backbone_module)
    s_pruner = StructuredPruner(backbone_module)

    # Optimizer (Chỉ optimize các params yêu cầu grad)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # 4. Pruning Loop (Iterative)
    print(f"Start Pruning Loop: Target={args.target_sparsity}, Iters={args.prune_iters}")

    # Baseline Eval
    print("Evaluating Baseline...")
    trainer_det.evaluate(model, data_loader_test, device=device)

    for i in range(args.prune_iters):
        print(f"\n--- Pruning Iteration {i + 1}/{args.prune_iters} ---")

        # Target 0.4, 5 iters -> 0.08, 0.16, 0.24, 0.32, 0.40
        current_sparsity = args.target_sparsity * (i + 1) / args.prune_iters

        # A. Song Han (Unstructured)
        # Sensitivity tăng dần (ví dụ từ 0.2 lên 1.0)
        # Heuristic: sensitivity ~ 2 * current_sparsity
        sensitivity = 2.0 * current_sparsity
        u_pruner.prune(sensitivity=sensitivity)

        # B. Filter Pruning (Structured)
        s_pruner.prune(prune_ratio=current_sparsity)

        # C. Finetune to Recover
        print(f"Finetuning for {args.finetune_epochs} epochs...")
        for epoch in range(args.finetune_epochs):
            # Pass u_pruner vào để nó ép weight=0 sau mỗi step (nếu cần thiết)
            # Tuy nhiên ConvBNReLU forward đã handle việc nhân mask rồi.
            trainer_det.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50,
                                        scaler=scaler)

        # D. Evaluate
        trainer_det.evaluate(model, data_loader_test, device=device)

    # 5. Model Surgery (Tạo file backbone_lean.json và backbone_lean.pth)
    print("\n--- Performing Model Surgery ---")
    save_path = os.path.join(args.output_dir, "backbone_lean.pth")

    # Trích xuất backbone nén từ model to
    lean_backbone = surgery.convert_to_lean_model(backbone_module, save_path)

    if lean_backbone is not None:
        print("Surgery Successful!")
        print(f"Lean Backbone saved to: {save_path}")
        print(f"Configuration saved to: {save_path.replace('.pth', '.json')}")
    else:
        print("Surgery Failed!")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)