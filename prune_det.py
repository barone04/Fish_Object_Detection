import argparse
import os
import torch
import copy
import types
import torch.multiprocessing

from engines import trainer_det, utils as engine_utils
from data.fish_det_dataset import FishDetectionDataset, collate_fn
from data import presets
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet18_fpn
# IMPORT QUAN TRỌNG: Lấy class gốc để check instance
from models.conv_bn_relu import ConvBNReLU, MaskProxy, UnstructuredMask, StructuredMask

from pruning.songhan_pruner import UnstructuredPruner
from pruning.filter_pruner import StructuredPruner
from pruning import surgery

torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser(description="Pruning Detection (Pipeline 2)")
    parser.add_argument("--data-path", default="./NewDeepfish/NewDeepfish", type=str)
    # Chọn Model
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to Dense Detection Model (Pipeline 2 Step 1)")

    # Training Params
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)

    # Pruning Params
    parser.add_argument("--target-sparsity", default=0.4, type=float)
    parser.add_argument("--prune-iters", default=5, type=int)
    parser.add_argument("--finetune-epochs", default=3, type=int)
    parser.add_argument("--output-dir", default="./output/pipeline2_pruned", type=str)

    # Flags chọn vùng prune
    parser.add_argument("--prune-fpn", action="store_true", help="Enable pruning for FPN layers")
    parser.add_argument("--freeze-backbone", action="store_true", help="Case 2: Freeze backbone")
    return parser


# --- CLASS PROVIDER (Đã fix lỗi NoneType Mask) ---
class LayerProvider:
    def __init__(self, layers, device):
        self.layers = layers
        self.device = device

    def get_prunable_layers(self, pruning_type="unstructured"):
        proxies = []
        for layer in self.layers:
            # Check kỹ lần cuối để đảm bảo an toàn
            if not isinstance(layer, ConvBNReLU):
                continue

            # 1. LAZY INIT: Chỉ tạo mask khi chưa có
            if pruning_type == "unstructured":
                if layer.u_mask is None:
                    layer.u_mask = UnstructuredMask(layer.weight.shape).to(self.device)
            elif pruning_type == "structured" or pruning_type == "filter":
                if layer.s_mask is None:
                    layer.s_mask = StructuredMask(layer.out_channels).to(self.device)

            # 2. WRAP: Đóng gói vào Proxy
            proxies.append(MaskProxy(layer, pruning_type))

        return proxies


# --- HÀM THU THẬP LAYER (Đã fix logic đệ quy) ---
def get_prunable_layers_recursive(module):
    """
    Hàm này duyệt cây module.
    - Nếu gặp ConvBNReLU -> Lấy ngay.
    - Nếu gặp Container (Bottleneck, BasicBlock, Sequential...) -> Đào tiếp vào con.
    """
    convs = []

    # ĐIỂM SỬA QUAN TRỌNG NHẤT: Dùng isinstance thay vì hasattr
    if isinstance(module, ConvBNReLU):
        convs.append(module)

    # Luôn luôn duyệt con (để tìm ConvBNReLU lẩn trốn bên trong Container)
    # Lưu ý: ConvBNReLU không có con nào cần duyệt nữa nên logic này an toàn
    for child in module.children():
        convs.extend(get_prunable_layers_recursive(child))

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
    # Khởi tạo đúng loại model
    if args.model == 'fasterrcnn_resnet50_fpn':
        model = fasterrcnn_resnet50_fpn(num_classes=2)
    else:
        model = fasterrcnn_resnet18_fpn(num_classes=2)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint: checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    # --- THU THẬP LAYER ---
    prunable_layers = []

    # 1. Thu thập từ Backbone
    if not args.freeze_backbone:
        print("Collecting Backbone layers...")
        # Gọi hàm đệ quy mới vào phần body
        backbone_layers = get_prunable_layers_recursive(model.backbone.body)
        prunable_layers.extend(backbone_layers)

    # 2. Thu thập từ FPN
    if args.prune_fpn:
        print("Collecting FPN layers...")
        if hasattr(model.backbone, 'fpn'):
            fpn_layers = get_prunable_layers_recursive(model.backbone.fpn)
            prunable_layers.extend(fpn_layers)

    # Log kiểm tra
    print(f"Total ConvBNReLU layers found: {len(prunable_layers)}")

    # Kiểm tra sanity check (Debug)
    for i, l in enumerate(prunable_layers):
        if not isinstance(l, ConvBNReLU):
            print(f"CRITICAL ERROR: Layer {i} is NOT ConvBNReLU! It is {type(l)}")
            return  # Dừng ngay lập tức nếu sai

    if len(prunable_layers) == 0:
        print("Error: No layers selected! Check --prune-fpn or --freeze-backbone.")
        return

    # --- SETUP PRUNER ---
    provider = LayerProvider(prunable_layers, device)

    u_pruner = UnstructuredPruner(provider)
    s_pruner = StructuredPruner(provider)

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

        # Gọi Prune (sẽ qua Provider -> Proxy -> Mask)
        u_pruner.prune(sensitivity=sensitivity)
        s_pruner.prune(prune_ratio=current_sparsity)

        print(f"Finetuning for {args.finetune_epochs} epochs...")
        for epoch in range(args.finetune_epochs):
            trainer_det.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50,
                                        scaler=scaler)

        trainer_det.evaluate(model, data_loader_test, device=device)

    print("\n--- Performing Model Surgery ---")
    save_path = os.path.join(args.output_dir, "model_lean.pth")

    # Gọi Surgery
    lean_model = surgery.convert_to_lean_model(model, save_path)

    if lean_model is not None:
        print("Surgery Successful!")
        print(f"Lean Model saved to: {save_path}")
    else:
        print("Surgery Failed!")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)