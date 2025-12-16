import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data.fish_cls_dataset import FishClassificationDataset
from models.resnet_hybrid import resnet_50
from pruning.songhan_pruner import UnstructuredPruner
from pruning.filter_pruner import StructuredPruner
from pruning.surgery import convert_to_lean_model
from engines import trainer_cls, utils as engine_utils
from thop import profile

def get_args_parser():
    parser = argparse.ArgumentParser(description="Classification Backbone Pruning")

    # Paths
    parser.add_argument("--data-path", required=True, help="Path to root dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to Dense Classification Model (.pth)")
    parser.add_argument("--output-dir", default="./output/pruned_cls", help="Where to save lean model")

    # Pruning Hyperparams
    parser.add_argument("--target-sparsity", default=0.6, type=float, help="Target filter removal ratio (0.0-1.0)")
    parser.add_argument("--prune-iters", default=10, type=int, help="Number of iterative pruning steps")
    parser.add_argument("--finetune-epochs", default=5, type=int, help="Epochs to retrain between prune steps")
    parser.add_argument("--sensitivity-mult", default=2.0, type=float,
                        help="Multiplier for SongHan threshold sensitivity")

    # Training Hyperparams
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for finetuning")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)

    return parser

# def measure_model(model, input_size=(1, 3, 224, 224), device='cuda'):
#     dummy_input = torch.randn(input_size).to(device)
#     flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
#     return flops, params


def main(args):
    engine_utils.init_distributed_mode(args)
    device = torch.device(args.device)
    engine_utils.mkdir(args.output_dir)

    print("=======================================================")
    print(f"STARTING BACKBONE PRUNING (Pipeline 1)")
    print(f"Target Sparsity: {args.target_sparsity} | Iters: {args.prune_iters}")
    print("=======================================================")

    # 1. Load Data (Classification)
    # Dùng tập train để finetune hồi phục, tập val để kiểm tra accuracy
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

    # 2. Load Model Dense
    # ResNet50 Hybrid (chưa nén)
    print(f"Loading dense backbone from {args.checkpoint}...")
    model = resnet_50(compress_rate=None, num_classes=13)  # 13 class cá

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']

    # Load weight, bỏ qua lỗi nếu key không khớp nhẹ (ví dụ wrap DDP)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    # Đánh giá Baseline trước khi cắt
    print("Evaluating Baseline...")
    acc1 = trainer_cls.evaluate(model, nn.CrossEntropyLoss(), loader_val, device)
    print(f"Baseline Acc@1: {acc1:.2f}%")

    # flops_dense, params_dense = measure_model(model, device)
    # print(f"Dense Model: FLOPs={flops_dense / 1e9:.2f}G, Params={params_dense / 1e6:.2f}M")

    # 3. Setup Pruners
    # Với Classification, ta prune trực tiếp model (vì model chính là backbone)
    sp_pruner = UnstructuredPruner(model)  # Song Han
    fp_pruner = StructuredPruner(model)  # Filter (L1-inf-inf)

    # Optimizer & Loss cho quá trình hồi phục
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler()  # Hỗ trợ H100 FP16

    # 4. Pruning Loop (Vòng lặp cắt tỉa)
    for i in range(args.prune_iters):
        print(f"\n--- Pruning Iteration {i + 1}/{args.prune_iters} ---")

        # A. Tính toán mức độ cắt tăng dần
        # Ví dụ: Target 0.6, iter 1 cắt 0.06, iter 2 cắt 0.12...
        current_sparsity = (i + 1) * (args.target_sparsity / args.prune_iters)

        # B. Apply Logic Cắt Tỉa
        # Song Han: Sensitivity tăng dần theo độ thưa
        sp_pruner.prune(sensitivity=current_sparsity * args.sensitivity_mult)
        # Filter: Ratio tăng dần
        fp_pruner.prune(prune_ratio=current_sparsity)

        # C. Retrain (Hồi phục)
        print(f"Finetuning to recover (Sparsity ~ {current_sparsity:.2f})...")
        for epoch in range(args.finetune_epochs):
            # Truyền sp_pruner vào để nó ép weight=0 sau mỗi step optimizer
            trainer_cls.train_one_epoch(
                model, criterion, optimizer, loader_train, device,
                epoch, print_freq=50, scaler=scaler, pruner=sp_pruner
            )

        # D. Validate nhanh
        acc1 = trainer_cls.evaluate(model, criterion, loader_val, device)
        print(f"Iter {i + 1} Acc@1: {acc1:.2f}%")

    # 5. Save Masked Model (Tùy chọn, để debug)
    # masked_path = os.path.join(args.output_dir, "backbone_masked.pth")
    # engine_utils.save_on_master({'model': model.state_dict()}, masked_path)

    # 6. Model Surgery (QUAN TRỌNG NHẤT)
    # Trích xuất và tạo model vật lý nhỏ gọn
    print("\n--- Performing Model Surgery ---")
    save_lean_path = os.path.join(args.output_dir, "backbone_lean.pth")

    # Hàm này sẽ tự động:
    # 1. Quét mask
    # 2. Tạo ResNet mới với channel nhỏ hơn
    # 3. Copy weight
    # 4. Lưu .pth và .json config
    lean_model = convert_to_lean_model(model, save_path=save_lean_path)

    # if lean_model is not None:
    #     lean_model.to(device)
    #     flops_lean, params_lean = measure_model(lean_model, device)
    #     print(f"Lean Model: FLOPs={flops_lean / 1e9:.2f}G, Params={params_lean / 1e6:.2f}M")
    #     print(
    #         f"Reduction: FLOPs -{(1 - flops_lean / flops_dense) * 100:.1f}%, Params -{(1 - params_lean / params_dense) * 100:.1f}%")

    if lean_model is not None:
        print(f"Surgery Successful!")
        print(f"Lean Backbone saved to: {save_lean_path}")
        print(f"Configuration saved to: {save_lean_path.replace('.pth', '.json')}")

        # (Optional) Verify lean model accuracy
        # Cần move lean model lên GPU để test
        lean_model.to(device)
        print("Verifying Lean Model Accuracy...")
        acc_final = trainer_cls.evaluate(lean_model, criterion, loader_val, device)
        print(f"Final Lean Model Acc@1: {acc_final:.2f}%")
    else:
        print("Surgery Failed!")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)