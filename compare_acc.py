import torch
import os
import json
import argparse
from engines import trainer_det, utils as engine_utils
from data.fish_det_dataset import FishDetectionDataset, collate_fn
from data import presets

from models.faster_rcnn import fasterrcnn_resnet50_fpn


def load_model(args, checkpoint_path, config_json_path=None, num_classes=2, device='cuda'):
    print(f"\nLoading Model from: {checkpoint_path}")

    compress_rate_list = None
    if config_json_path:
        print(f"Loading pruning config: {config_json_path}")
        with open(config_json_path, 'r') as f:
            compress_rate_list = json.load(f)
    else:
        print("No pruning config provided. Loading as DENSE model.")

    model = fasterrcnn_resnet50_fpn(
        num_classes=num_classes,
        compress_rate=compress_rate_list,
        weights_backbone=None
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"WARNING: Issue loading weights: {e}")

    model.to(device)
    return model


def evaluate_performance(model, data_loader, device):
    """Chạy evaluate và trả về các chỉ số mAP"""
    coco_evaluator = trainer_det.evaluate(model, data_loader, device=device)

    stats = coco_evaluator.coco_eval['bbox'].stats
    return stats[0], stats[1]


def main():
    # Configs
    DATA_PATH = "NewDeepfish/NewDeepfish"
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    BATCH_SIZE = 8
    NUM_CLASSES = 2  # Background + Fish

    DENSE_MODEL = "output/step1_dense_det/model_best.pth"

    PRUNED_MODEL = "output/step3_final_result/model_best.pth"
    PRUNED_CONFIG = "output/step2_pruned_det/backbone_lean.json"


    dataset_test = FishDetectionDataset(DATA_PATH, split='val', transforms=presets.DetectionPresetEval())
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=BATCH_SIZE, sampler=test_sampler,
        num_workers=4, collate_fn=collate_fn
    )

    model_dense = load_model(None, DENSE_MODEL, config_json_path=None, num_classes=NUM_CLASSES, device=DEVICE)
    map_dense, map50_dense = evaluate_performance(model_dense, data_loader_test, DEVICE)
    del model_dense
    torch.cuda.empty_cache()

    model_pruned = load_model(None, PRUNED_MODEL, config_json_path=PRUNED_CONFIG, num_classes=NUM_CLASSES,
                              device=DEVICE)
    map_pruned, map50_pruned = evaluate_performance(model_pruned, data_loader_test, DEVICE)

    print("\n" + "=" * 60)
    print(f"{'METRIC':<20} | {'DENSE MODEL':<15} | {'PRUNED MODEL':<15} | {'CHANGE'}")
    print("-" * 60)

    diff_map = map_pruned - map_dense
    diff_map50 = map50_pruned - map50_dense

    print(f"{'mAP (0.5:0.95)':<20} | {map_dense:.4f}{'':<9} | {map_pruned:.4f}{'':<9} | {diff_map:+.4f}")
    print(f"{'mAP (0.5)':<20} | {map50_dense:.4f}{'':<9} | {map50_pruned:.4f}{'':<9} | {diff_map50:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()