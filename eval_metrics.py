import os
import time
import csv
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from dataset_loader import AgricultureObjectDetectionDataset


def box_iou(boxes1, boxes2):
    # boxes are [N,4] in xmin,ymin,xmax,ymax
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.empty((0, 0), device=boxes1.device)
    # convert to x1,y1,x2,y2 if already
    # compute intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-8)
    return iou


def collate_batch(batch):
    # batch: list of (inputs, targets)
    inputs_batch = {}
    targets_batch = []
    keys = batch[0][0].keys()
    for k in keys:
        vals = [b[0][k].unsqueeze(0) for b in batch]
        inputs_batch[k] = torch.cat(vals, dim=0)
    targets_batch = [b[1] for b in batch]
    return inputs_batch, targets_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--log-file", type=str, default="federated_metrics_eval.csv")
    parser.add_argument("--checkpoint", type=str, default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = args.device
    print("Device:", device)

    print("Loading model and processor...")
    model = Owlv2ForObjectDetection.from_pretrained(args.checkpoint)
    processor = Owlv2Processor.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    # Build dataset
    METAFRUIT_ROOT = r"C:\Users\balaj\OneDrive\Documents\project sem 5\archive\MetaFruit"
    FRUIT_FOLDERS = ["apple", "grapefruit", "lemon", "orange", "tangerine"]
    query_texts = [[ "apple", "grapefruit", "lemon", "orange", "tangerine", "leaf", "person" ]]

    all_datasets = []
    for fruit in FRUIT_FOLDERS:
        data_dir = os.path.join(METAFRUIT_ROOT, fruit)
        if os.path.isdir(data_dir):
            ds = AgricultureObjectDetectionDataset(data_dir=data_dir, processor=processor, texts_to_find=query_texts, fruit_type=fruit)
            if len(ds) > 0:
                all_datasets.append(ds)
    if not all_datasets:
        print("No datasets found; aborting")
        return
    full = ConcatDataset(all_datasets)
    subset = min(args.max_samples, len(full))
    items = [full[i] for i in range(subset)]
    loader = DataLoader(items, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    total_loss = 0.0
    total_count = 0
    total_correct_samples = 0
    total_samples_with_gt = 0
    batches = 0
    start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            # move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pred_boxes = outputs.pred_boxes  # [B, Q, 4] in x0,y0,x1,y1
            B = pred_boxes.shape[0]
            K = args.top_k

            batch_matches = 0
            batch_samples_with_gt = 0
            batch_samples_matched = 0
            batch_loss = 0.0
            num = 0
            for i in range(B):
                gt = targets[i]['boxes']
                if len(gt) == 0:
                    continue
                preds_k = pred_boxes[i, :K, :].to(device)
                gt = gt.to(device)
                ious = box_iou(preds_k, gt)
                if ious.numel() == 0:
                    continue
                ious_clone = ious.clone()
                matched = 0
                while True:
                    max_val = ious_clone.max()
                    if max_val < args.iou_threshold:
                        break
                    idx = (ious_clone == max_val).nonzero(as_tuple=False)[0]
                    p_idx = int(idx[0].item())
                    g_idx = int(idx[1].item())
                    pb = preds_k[p_idx].float()
                    tb = gt[g_idx].float()
                    batch_loss += F.mse_loss(pb, tb, reduction='mean').item()
                    num += 1
                    matched += 1
                    ious_clone[p_idx, :] = -1.0
                    ious_clone[:, g_idx] = -1.0
                batch_matches += matched
                batch_samples_with_gt += 1
                if matched > 0:
                    batch_samples_matched += 1

            if num > 0:
                avg_batch_loss = batch_loss / num
                total_loss += avg_batch_loss
                total_count += 1

            total_correct_samples += batch_samples_matched
            total_samples_with_gt += batch_samples_with_gt
            batches += 1

    elapsed = time.time() - start
    avg_loss = (total_loss / total_count) if total_count > 0 else 0.0
    accuracy = (total_correct_samples / max(total_samples_with_gt, 1)) if total_samples_with_gt > 0 else 0.0

    print(f"Eval finished — avg_loss={avg_loss:.6f}, accuracy={accuracy:.6f}, elapsed={elapsed:.1f}s")

    # append to CSV
    header = ["epoch", "avg_loss", "accuracy", "seconds"]
    exists = os.path.exists(args.log_file)
    with open(args.log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([0, f"{avg_loss:.6f}", f"{accuracy:.6f}", f"{elapsed:.1f}"])


if __name__ == '__main__':
    main()
