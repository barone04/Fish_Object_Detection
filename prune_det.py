import argparse
import os
import torch
import torch.multiprocessing

from engines import trainer_det, utils as engine_utils
from data.fish_det_dataset import FishDetectionDataset, collate_fn
from data import presets
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet18_fpn
from models.mobilenet_custom import fasterrcnn_mobilenetv3_custom

from models.conv_bn_relu import ConvBNReLU, MaskProxy, UnstructuredMask, StructuredMask
from pruning.songhan_pruner import UnstructuredPruner
from pruning.filter_pruner import StructuredPruner
from pruning import surgery

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
    parser = argparse.ArgumentParser(description="Pruning Detection (Pipeline 2)")
    parser.add_argument("--data-path", default="./NewDeepfish/NewDeepfish", type=str)

    parser.add_argument("--model", default="mobilenet_v3", type=str, help="resnet18, resnet50, mobilenet_v3")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--checkpoint", required=True, type=str)

    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)

    parser.add_argument("--target-sparsity", default=0.4, type=float)
    parser.add_argument("--prune-iters", default=5, type=int)
    parser.add_argument("--finetune-epochs", default=3, type=int)
    parser.add_argument("--output-dir", default="./output/pipeline2_pruned", type=str)

    parser.add_argument("--prune-fpn", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")

    # để builder MobileNet khớp với training
    parser.add_argument("--box-head-dim", default=1024, type=int)
    return parser


class LayerProvider:
    def __init__(self, layers, device):
        self.layers = layers
        self.device = device

    def get_prunable_layers(self, pruning_type="unstructured"):
        proxies = []
        for layer in self.layers:
            if not isinstance(layer, ConvBNReLU):
                continue

            if pruning_type == "unstructured":
                if layer.u_mask is None:
                    layer.u_mask = UnstructuredMask(layer.weight.shape).to(self.device)
            elif pruning_type in ("structured", "filter"):
                if layer.s_mask is None:
                    layer.s_mask = StructuredMask(layer.out_channels).to(self.device)

            proxies.append(MaskProxy(layer, pruning_type))
        return proxies


def get_prunable_layers_recursive(module):
    convs = []
    if isinstance(module, ConvBNReLU):
        convs.append(module)
    for child in module.children():
        convs.extend(get_prunable_layers_recursive(child))
    return convs


def main(args):
    engine_utils.init_distributed_mode(args)
    engine_utils.mkdir(args.output_dir)
    device = resolve_device(args.device)
    print("Using device:", device)

    print("Loading Data...")
    dataset_train = FishDetectionDataset(args.data_path, split="train",
                                        transforms=presets.DetectionPresetTrain(data_augmentation="hflip"))
    dataset_test = FishDetectionDataset(args.data_path, split="val",
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

    if args.model == "mobilenet_v3":
        model = fasterrcnn_mobilenetv3_custom(
            num_classes=2,
            pretrained_backbone=False,   # load checkpoint ngay sau
            freeze_backbone=True,
            box_head_dim=args.box_head_dim
        )
    elif args.model in ("fasterrcnn_resnet50_fpn", "resnet50"):
        model = fasterrcnn_resnet50_fpn(num_classes=2)
    else:
        model = fasterrcnn_resnet18_fpn(num_classes=2)

    # load checkpoint
    try:
        try:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        model.load_state_dict(checkpoint, strict=False)
        print(" -> Checkpoint loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR loading checkpoint: {e}")
        raise SystemExit(1)

    model.to(device)

    # collect prunable layers
    prunable_layers = []
    if args.model == "mobilenet_v3":
        print("--- Mode: MobileNetV3 (FREEZE backbone, PRUNE FPN ONLY) ---")
        if hasattr(model.backbone, "fpn"):
            prunable_layers.extend(get_prunable_layers_recursive(model.backbone.fpn))
    else:
        print("--- Mode: ResNet ---")
        if not args.freeze_backbone:
            prunable_layers.extend(get_prunable_layers_recursive(model.backbone.body))
        if args.prune_fpn and hasattr(model.backbone, "fpn"):
            prunable_layers.extend(get_prunable_layers_recursive(model.backbone.fpn))

    print(f"Total ConvBNReLU layers found: {len(prunable_layers)}")
    if len(prunable_layers) == 0:
        print("Error: No layers selected!")
        raise SystemExit(1)

    provider = LayerProvider(prunable_layers, device)
    u_pruner = UnstructuredPruner(provider)
    s_pruner = StructuredPruner(provider)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print("AMP enabled:", use_amp)

    print(f"Start Pruning Loop: Target={args.target_sparsity}, Iters={args.prune_iters}")

    print("Evaluating Baseline...")
    trainer_det.evaluate(model, data_loader_test, device=device)

    for i in range(args.prune_iters):
        print(f"\n--- Pruning Iteration {i + 1}/{args.prune_iters} ---")
        current_sparsity = args.target_sparsity * (i + 1) / args.prune_iters
        sensitivity = 2.0 * current_sparsity

        u_pruner.prune(sensitivity=sensitivity)
        s_pruner.prune(prune_ratio=current_sparsity)

        print(f"Finetuning for {args.finetune_epochs} epochs...")
        for epoch in range(args.finetune_epochs):
            trainer_det.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50, scaler=scaler)

        trainer_det.evaluate(model, data_loader_test, device=device)

    print("\n--- Performing Model Surgery ---")
    save_path = os.path.join(args.output_dir, "model_lean.pth")
    lean_model = surgery.convert_to_lean_model(model, save_path)

    if lean_model is not None:
        print("Surgery Successful!")
        print(f"Lean Model saved to: {save_path}")
    else:
        print("Surgery Failed!")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
