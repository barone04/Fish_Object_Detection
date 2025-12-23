import argparse
import os
import torch
import copy
from engines import trainer_det, utils as engine_utils
from data.fish_det_dataset import FishDetectionDataset, collate_fn
from data import presets
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet18_fpn
import types

from pruning.songhan_pruner import UnstructuredPruner
from pruning.filter_pruner import StructuredPruner
from pruning import surgery

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


# Trích xuất layer từ IntermediateLayerGetter
# def get_prunable_layers_wrapper(self, pruning_type="unstructured"):
#     convs = []
#     # 1. Stem (conv1)
#     if hasattr(self, 'conv1'):
#         convs.append(self.conv1)
#
#     # 2. Các Stage (layer1 -> layer4)
#     # IntermediateLayerGetter chứa các layer dưới dạng thuộc tính
#     for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
#         if hasattr(self, layer_name):
#             stage = getattr(self, layer_name)
#             for block in stage:
#                 # Gọi đệ quy vào từng block (Bottleneck)
#                 if hasattr(block, 'get_prunable_layers'):
#                     convs.extend(block.get_prunable_layers(pruning_type))
#     return convs
def get_prunable_layers_wrapper(self, pruning_type="unstructured"):
    convs = []

    if hasattr(self, 'conv1'):
        if hasattr(self.conv1, 'get_prunable_layers'):
            convs.extend(self.conv1.get_prunable_layers(pruning_type))
        else:
            convs.append(self.conv1)

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(self, layer_name):
            stage = getattr(self, layer_name)
            for block in stage:
                # Gọi đệ quy vào từng block (Bottleneck)
                if hasattr(block, 'get_prunable_layers'):
                    convs.extend(block.get_prunable_layers(pruning_type))

    return convs


def main(args):
    engine_utils.init_distributed_mode(args)
    engine_utils.mkdir(args.output_dir)
    device = torch.device(args.device)

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

    print(f"Loading Dense Model from {args.checkpoint}...")
    # Khởi tạo model full (Dense)
    # model = fasterrcnn_resnet50_fpn(num_classes=2)
    model = fasterrcnn_resnet18_fpn(num_classes=2)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' in checkpoint: checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)
    model.to(device)

    backbone_module = model.backbone.body
    backbone_module.get_prunable_layers = types.MethodType(get_prunable_layers_wrapper, backbone_module)

    u_pruner = UnstructuredPruner(backbone_module)
    s_pruner = StructuredPruner(backbone_module)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Start Pruning Loop: Target={args.target_sparsity}, Iters={args.prune_iters}")

    # Baseline Eval
    print("Evaluating Baseline...")
    trainer_det.evaluate(model, data_loader_test, device=device)

    for i in range(args.prune_iters):
        print(f"\n--- Pruning Iteration {i + 1}/{args.prune_iters} ---")

        current_sparsity = args.target_sparsity * (i + 1) / args.prune_iters

        # Sensitivity tăng dần (ví dụ từ 0.2 lên 1.0)
        # Heuristic: sensitivity ~ 2 * current_sparsity
        sensitivity = 2.0 * current_sparsity
        u_pruner.prune(sensitivity=sensitivity)

        s_pruner.prune(prune_ratio=current_sparsity)

        print(f"Finetuning for {args.finetune_epochs} epochs...")
        for epoch in range(args.finetune_epochs):
            trainer_det.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50,
                                        scaler=scaler)

        trainer_det.evaluate(model, data_loader_test, device=device)

    print("\n--- Performing Model Surgery ---")
    save_path = os.path.join(args.output_dir, "backbone_lean.pth")

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