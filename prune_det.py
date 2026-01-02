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

    parser.add_argument("--prune-fpn", action="store_true", help="Enable pruning for FPN layers")
    parser.add_argument("--freeze-backbone", action="store_true", help="Case 2: Freeze backbone, only prune FPN")
    return parser


class LayerProvider:
    """
    Class này giả lập behavior của Model.
    Khi Pruner gọi .get_prunable_layers(), nó sẽ trả về list layers chúng ta đã chọn sẵn.
    """

    def __init__(self, layers):
        self.layers = layers

    def get_prunable_layers(self, pruning_type="unstructured"):
        return self.layers


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
    if args.model == 'fasterrcnn_resnet50_fpn':
        model = fasterrcnn_resnet50_fpn(num_classes=2)
    else:
        model = fasterrcnn_resnet18_fpn(num_classes=2)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint: checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    # --- LOGIC CHỌN LAYER ---
    prunable_layers = []

    # 1. Backbone
    if not args.freeze_backbone:
        backbone_module = model.backbone.body
        if hasattr(backbone_module, 'get_prunable_layers_wrapper'):
            backbone_module.get_prunable_layers = types.MethodType(get_prunable_layers_wrapper, backbone_module)
            prunable_layers.extend(backbone_module.get_prunable_layers())
        elif hasattr(backbone_module, 'get_prunable_layers'):
            prunable_layers.extend(backbone_module.get_prunable_layers())
        print(f"Added Backbone layers. Current Total: {len(prunable_layers)}")

    # 2. FPN
    if args.prune_fpn:
        if hasattr(model.backbone, 'fpn') and hasattr(model.backbone.fpn, 'layer_blocks'):
            for block in model.backbone.fpn.layer_blocks:
                if hasattr(block, 'get_prunable_layers'):
                    prunable_layers.extend(block.get_prunable_layers())
            print(f"Added FPN layers. Current Total: {len(prunable_layers)}")
        else:
            print("Warning: --prune-fpn set but FPN modules not found or not prunable!")

    if len(prunable_layers) == 0:
        print("Error: No layers selected for pruning! Check --prune-fpn or --freeze-backbone flags.")
        return

    # Thay vì truyền None, ta truyền một object "Fake" cung cấp layers
    provider = LayerProvider(prunable_layers)

    u_pruner = UnstructuredPruner(provider)
    s_pruner = StructuredPruner(provider)
    # ---------------------

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

        sensitivity = 2.0 * current_sparsity
        # Bây giờ hàm prune sẽ gọi provider.get_prunable_layers() -> Trả về list đúng
        u_pruner.prune(sensitivity=sensitivity)
        s_pruner.prune(prune_ratio=current_sparsity)

        print(f"Finetuning for {args.finetune_epochs} epochs...")
        for epoch in range(args.finetune_epochs):
            trainer_det.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50,
                                        scaler=scaler)

        trainer_det.evaluate(model, data_loader_test, device=device)

    print("\n--- Performing Model Surgery ---")
    save_path = os.path.join(args.output_dir, "model_lean.pth")

    lean_model = surgery.convert_to_lean_model(model, save_path)

    if lean_model is not None:
        print("Surgery Successful!")
        print(f"Lean Backbone saved to: {save_path}")
        print(f"Configuration saved to: {save_path.replace('.pth', '.json')}")
    else:
        print("Surgery Failed!")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)