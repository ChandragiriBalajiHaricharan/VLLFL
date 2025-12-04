import argparse
import os
import json
import math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from peft import PeftModel

from dataset_loader import AgricultureObjectDetectionDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=r"C:\Users\balaj\OneDrive\Documents\project sem 5\archive\MetaFruit")
    p.add_argument("--fruits", nargs="+", default=["apple","grapefruit","lemon","orange","tangerine"])
    p.add_argument("--checkpoint", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--adapters", default="vllfl_adapters_efficient")
    p.add_argument("--device", default="auto", choices=["auto","cuda","cpu"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--viz-n", type=int, default=10, help="Number of example visualizations to save")
    p.add_argument("--out-dir", default="eval_results")
    return p.parse_args()


def cxcywh_to_xyxy(boxes):
    # boxes: [N,4] with cx,cy,w,h (normalized)
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=1)


def xyxy_iou(boxes1, boxes2):
    # boxes: [N,4] and [M,4] in same coords (normalized or pixels)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-9)
    return iou


def save_viz(image_path, gt_boxes, pred_boxes, matches, out_path):
    # gt_boxes: [n,4] in absolute pixel xyxy
    # pred_boxes: [m,4] in absolute pixel xyxy
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    # draw GT boxes in green
    for b in gt_boxes:
        draw.rectangle([b[0], b[1], b[2], b[3]], outline=(0, 255, 0), width=2)
    # draw predicted; matched indices in blue, unmatched in red
    for j, b in enumerate(pred_boxes):
        color = (255, 0, 0)
        if j in matches:
            color = (0, 0, 255)
        draw.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path)


def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    print(f"Device: {device}")

    print("Loading processor and base model...")
    processor = Owlv2Processor.from_pretrained(args.checkpoint)
    base_model = Owlv2ForObjectDetection.from_pretrained(args.checkpoint)

    print("Loading adapters...")
    model = PeftModel.from_pretrained(base_model, args.adapters)
    model.to(device)
    model.eval()

    # Build dataset
    print("Building datasets...")
    all_datasets = []
    query_texts = [["apple","grapefruit","lemon","orange","tangerine","leaf","person"]]
    for fruit in args.fruits:
        d_path = os.path.join(args.data_root, fruit)
        if os.path.exists(d_path):
            ds = AgricultureObjectDetectionDataset(d_path, processor, query_texts, fruit_type=fruit)
            if len(ds) > 0:
                all_datasets.append(ds)
                print(f"  Loaded {len(ds)} from {fruit}")
    if not all_datasets:
        print("No datasets found. Check data root.")
        return
    dataset = torch.utils.data.ConcatDataset(all_datasets)
    N = len(dataset)
    n_eval = min(args.max_samples, N)
    torch.manual_seed(42)
    indices = torch.randperm(N)[:n_eval].tolist()
    from torch.utils.data import Subset
    val_ds = Subset(dataset, indices)
    print(f"Evaluating on {len(val_ds)} samples")

    def collate(batch):
        batch = [b for b in batch if b is not None]
        pixel_values = torch.stack([item[0]['pixel_values'] for item in batch])
        input_ids_list = []
        attention_mask_list = []
        MAX_LENGTH = 16
        for item in batch:
            ids = item[0]['input_ids'].reshape(-1)[:MAX_LENGTH]
            mask = item[0]['attention_mask'].reshape(-1)[:MAX_LENGTH]
            if len(ids) < MAX_LENGTH:
                pad_len = MAX_LENGTH - len(ids)
                ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            input_ids_list.append(ids)
            attention_mask_list.append(mask)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        encoding = {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask}
        labels = [item[1] for item in batch]
        return encoding, labels

    loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_mse = 0.0
    matched_pairs = 0

    out_dir = Path(args.out_dir)
    viz_dir = out_dir / "viz"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    example_i = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        inputs, targets = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        pred_boxes = outputs.pred_boxes  # [B, Q, 4] normalized cxcywh
        logits = getattr(outputs, 'logits', None)

        B = len(targets)
        for i in range(B):
            t = targets[i]
            orig_h, orig_w = int(t['orig_size'][0]), int(t['orig_size'][1])
            gt_boxes_xyxy = t['boxes']  # absolute xyxy
            if gt_boxes_xyxy.numel() == 0:
                gt_boxes_xyxy = torch.empty((0,4))

            # convert gt to normalized cxcywh
            if gt_boxes_xyxy.shape[0] > 0:
                gt_norm = gt_boxes_xyxy / torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32)
                wh = gt_norm[:, 2:] - gt_norm[:, :2]
                cxcy = (gt_norm[:, 2:] + gt_norm[:, :2]) / 2
                gt_cxcywh = torch.cat([cxcy, wh], dim=1)
            else:
                gt_cxcywh = torch.empty((0,4))

            preds = pred_boxes[i].cpu()
            # get prediction confidences if available
            if logits is not None:
                scores = torch.softmax(logits[i], dim=-1)
                max_scores, _ = scores.max(dim=-1)
            else:
                max_scores = torch.ones(preds.shape[0]) * 0.1

            # filter preds by score threshold
            score_thresh = 0.05
            # ensure mask and preds are on same device (work on CPU for indexing)
            max_scores_cpu = max_scores.cpu()
            keep = (max_scores_cpu >= score_thresh)
            if keep.sum() == 0:
                kept_preds = preds
                kept_scores = max_scores_cpu
            else:
                kept_preds = preds[keep]
                kept_scores = max_scores_cpu[keep]

            # convert both to xyxy normalized
            pred_xyxy = cxcywh_to_xyxy(kept_preds)
            gt_xyxy_norm = torch.empty((0,4))
            if gt_cxcywh.shape[0] > 0:
                gt_xyxy_norm = cxcywh_to_xyxy(gt_cxcywh)

            iou_mat = xyxy_iou(pred_xyxy, gt_xyxy_norm)

            # greedy matching: find pairs with iou >= threshold
            matches_pred = set()
            matches_gt = set()
            for _ in range(min(iou_mat.shape[0], iou_mat.shape[1]) if iou_mat.numel()>0 else 0):
                max_val, idx = torch.max(iou_mat.reshape(-1), dim=0)
                if max_val.item() < args.iou_threshold:
                    break
                pred_idx = int(idx.item() // iou_mat.shape[1])
                gt_idx = int(idx.item() % iou_mat.shape[1])
                matches_pred.add(pred_idx)
                matches_gt.add(gt_idx)
                # zero out row and col
                iou_mat[pred_idx, :] = -1
                iou_mat[:, gt_idx] = -1

            tp = len(matches_pred)
            fp = kept_preds.shape[0] - tp
            fn = (gt_cxcywh.shape[0] - tp)
            total_tp += tp
            total_fp += max(0, fp)
            total_fn += max(0, fn)

            # compute mse for matched pairs (in normalized cxcywh space)
            if gt_cxcywh.shape[0] > 0 and kept_preds.shape[0] > 0 and len(matches_pred) > 0:
                # map matched preds to matched gts
                for pred_idx in matches_pred:
                    # find corresponding gt by computing iou between this pred and all gt and taking max
                    single_pred = pred_xyxy[pred_idx].unsqueeze(0)
                    ious = xyxy_iou(single_pred, gt_xyxy_norm).squeeze(0)
                    gt_match_idx = int(torch.argmax(ious).item())
                    gt_box = gt_cxcywh[gt_match_idx].to(preds.dtype)
                    pred_box = kept_preds[pred_idx]
                    total_mse += F.mse_loss(pred_box, gt_box).item()
                    matched_pairs += 1

            # visualization for first N examples
            if example_i < args.viz_n:
                # resolve original image path via underlying dataset
                # dataset is ConcatDataset of fruit datasets
                # need to find which underlying dataset and index
                global_idx = indices[batch_idx * args.batch_size + i]
                # map into underlying
                subds = dataset.datasets
                rem = global_idx
                img_path = None
                for sub in subds:
                    if rem < len(sub):
                        fname = sub.image_files[rem]
                        img_path = os.path.join(sub.data_dir, fname)
                        break
                    rem -= len(sub)
                if img_path is not None:
                    # convert normalized boxes to absolute pixel xyxy for viz
                    abs_gt = []
                    for b in gt_boxes_xyxy:
                        abs_gt.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                    abs_preds = []
                    for pb in pred_xyxy.cpu():
                        x1 = float(pb[0] * orig_w)
                        y1 = float(pb[1] * orig_h)
                        x2 = float(pb[2] * orig_w)
                        y2 = float(pb[3] * orig_h)
                        abs_preds.append([x1, y1, x2, y2])
                    viz_path = viz_dir / f"example_{example_i}.png"
                    save_viz(img_path, abs_gt, abs_preds, matches_pred, str(viz_path))
                    example_i += 1

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    avg_mse = (total_mse / matched_pairs) if matched_pairs > 0 else None

    results = {
        'n_eval': n_eval,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': float(precision),
        'recall': float(recall),
        'avg_matched_mse': float(avg_mse) if avg_mse is not None else None,
    }
    with open(out_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete. Results:")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
import os
import argparse
import random
import math
from contextlib import nullcontext
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont

from dataset_loader import AgricultureObjectDetectionDataset


# Collate (picklable top-level)
COLLATE_MAX_LENGTH = 16

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    pixel_values = torch.stack([item[0]['pixel_values'] for item in batch])
    input_ids_list = []
    attention_mask_list = []
    MAX_LENGTH = COLLATE_MAX_LENGTH
    for item in batch:
        ids = item[0]['input_ids'].reshape(-1)
        mask = item[0]['attention_mask'].reshape(-1)
        ids = ids[:MAX_LENGTH]
        mask = mask[:MAX_LENGTH]
        if len(ids) < MAX_LENGTH:
            pad_len = MAX_LENGTH - len(ids)
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
        input_ids_list.append(ids)
        attention_mask_list.append(mask)
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    encoding = {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask}
    labels = []
    for item in batch:
        target = item[1]
        orig_h, orig_w = target['orig_size']
        boxes_xyxy = target['boxes']
        if len(boxes_xyxy) == 0:
            boxes_cxcywh = torch.zeros(0, 4, dtype=torch.float32)
        else:
            boxes_norm = boxes_xyxy / torch.tensor([orig_w, orig_h, orig_w, orig_h])
            wh = boxes_norm[:, 2:] - boxes_norm[:, :2]
            cxcy = (boxes_norm[:, 2:] + boxes_norm[:, :2]) / 2
            boxes_cxcywh = torch.cat([cxcy, wh], dim=1).clamp(0, 1)
        labels.append({"boxes": boxes_cxcywh, "orig_size": (int(orig_h.item()), int(orig_w.item()))})
    return encoding, labels


def cxcywh_to_xyxy(boxes):
    # boxes: (N,4) cx,cy,w,h
    if boxes.numel() == 0:
        return boxes
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def iou_matrix(boxes1, boxes2):
    # both boxes are (N,4) xyxy normalized
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
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
    iou = inter / (union + 1e-9)
    return iou


def get_image_from_concat_dataset(concat_ds, idx):
    # Find which child dataset contains idx
    if hasattr(concat_ds, 'datasets'):
        sizes = [len(d) for d in concat_ds.datasets]
        cum = 0
        for ds_idx, s in enumerate(sizes):
            if idx < cum + s:
                local_idx = idx - cum
                data_dir = concat_ds.datasets[ds_idx].data_dir
                img_name = concat_ds.datasets[ds_idx].image_files[local_idx]
                img_path = os.path.join(data_dir, img_name)
                return Image.open(img_path).convert('RGB')
            cum += s
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default=r"C:\Users\balaj\OneDrive\Documents\project sem 5\archive\MetaFruit")
    p.add_argument('--fruits', nargs='+', default=["apple","grapefruit","lemon","orange","tangerine"])
    p.add_argument('--checkpoint', default='google/owlv2-base-patch16-ensemble')
    p.add_argument('--adapters-dir', default='vllfl_adapters_efficient')
    p.add_argument('--device', default='auto', choices=['auto','cuda','cpu'])
    p.add_argument('--val-samples', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--save-examples', default='eval_examples')
    p.add_argument('--iou-thresh', type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) or args.device == 'cuda' else 'cpu'
    print('Device:', device)

    print('Loading model and processor...')
    base_model = Owlv2ForObjectDetection.from_pretrained(args.checkpoint)
    processor = Owlv2Processor.from_pretrained(args.checkpoint)

    # Load PEFT adapters into base model
    model = PeftModel.from_pretrained(base_model, args.adapters_dir)
    model.to(device)
    model.eval()

    # Build dataset
    print('Building datasets...')
    all_datasets = []
    query_texts = [["apple","grapefruit","lemon","orange","tangerine","leaf","person"]]
    for fruit in args.fruits:
        d_path = os.path.join(args.data_root, fruit)
        if os.path.exists(d_path):
            ds = AgricultureObjectDetectionDataset(d_path, processor, query_texts, fruit_type=fruit)
            if len(ds) > 0:
                all_datasets.append(ds)
                print(f'  Loaded {len(ds)} from {fruit}')
    if not all_datasets:
        print('No datasets found.'); return
    concat_ds = ConcatDataset(all_datasets)
    N = len(concat_ds)
    val_n = min(args.val_samples, N)
    rng = torch.Generator()
    indices = torch.randperm(N, generator=rng)[:val_n].tolist()

    from torch.utils.data import Subset
    val_dataset = Subset(concat_ds, indices)
    print(f'Val dataset size: {len(val_dataset)}')

    # collate global length
    global COLLATE_MAX_LENGTH
    COLLATE_MAX_LENGTH = 16

    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # Metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_loss = 0.0
    count = 0

    os.makedirs(args.save_examples, exist_ok=True)
    saved_examples = 0

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        inputs, targets = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if device == 'cuda':
                ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float32)
            else:
                ctx = nullcontext()
            with ctx:
                outputs = model(**inputs)

        pred_boxes = outputs.pred_boxes.detach().cpu()  # [B, Q, 4] cxcywh normalized
        logits = outputs.logits.detach().cpu() if hasattr(outputs, 'logits') else None

        B = pred_boxes.shape[0]
        for i in range(B):
            gt = targets[i]
            gt_boxes = gt['boxes']  # (n,4) cxcywh normalized
            gt_xy = cxcywh_to_xyxy(gt_boxes) if gt_boxes.numel() else torch.empty((0,4))

            pb = pred_boxes[i]
            pb_xy = cxcywh_to_xyxy(pb)

            if logits is not None:
                scores = F.softmax(logits[i], dim=-1).max(dim=-1).values
            else:
                scores = torch.ones(pb_xy.shape[0])

            # sort by score
            order = torch.argsort(scores, descending=True)
            matched_gt = set()
            tp = 0
            fp = 0
            for idx_pred in order.tolist():
                pbox = pb_xy[idx_pred].unsqueeze(0)
                if gt_xy.numel() == 0:
                    fp += 1
                    continue
                ious = iou_matrix(pbox, gt_xy)  # (1, n_gt)
                best_iou, best_idx = torch.max(ious, dim=1)
                if best_iou.item() >= args.iou_thresh and best_idx.item() not in matched_gt:
                    tp += 1
                    matched_gt.add(best_idx.item())
                else:
                    fp += 1
            fn = gt_xy.shape[0] - len(matched_gt)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # val loss (MSE with up to K targets)
            K = 3
            loss = torch.tensor(0.0)
            num = 0
            for j in range(min(K, gt_boxes.shape[0])):
                tb = gt_boxes[j]
                pb_j = pb[j]
                loss = loss + F.mse_loss(pb_j, tb)
                num += 1
            if num > 0:
                total_loss += (loss.item() / num)
                count += 1

            # save example visuals for first 20 images
            if saved_examples < 20:
                # retrieve raw image from concat dataset
                global_idx = indices[batch_idx * args.batch_size + i] if isinstance(val_dataset, Subset) else None
                img = get_image_from_concat_dataset(concat_ds, global_idx) if global_idx is not None else None
                if img is not None:
                    draw = ImageDraw.Draw(img)
                    # draw GT in green
                    for g in gt_xy:
                        x1, y1, x2, y2 = [float(v) * s for v, s in zip(g, (gt['orig_size'][1], gt['orig_size'][0], gt['orig_size'][1], gt['orig_size'][0]))]
                        draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
                    # draw preds in red with score
                    for idx_pred in order.tolist()[:50]:
                        s = float(scores[idx_pred].item())
                        p = pb_xy[idx_pred]
                        x1, y1, x2, y2 = [float(v) * sdim for v, sdim in zip(p, (gt['orig_size'][1], gt['orig_size'][0], gt['orig_size'][1], gt['orig_size'][0]))]
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=1)
                        draw.text((x1, y1), f"{s:.2f}", fill='red')
                    img.save(os.path.join(args.save_examples, f'eval_{saved_examples}.jpg'))
                    saved_examples += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    avg_loss = total_loss / count if count > 0 else 0.0

    print('\nEvaluation results:')
    print(f' Val samples: {val_n}')
    print(f' Avg val loss (MSE on matched boxes): {avg_loss:.6f}')
    print(f' Precision@IoU{args.iou_thresh}: {precision:.4f}')
    print(f' Recall@IoU{args.iou_thresh}:    {recall:.4f}')

    # Save metrics JSON
    metrics = { 'val_samples': val_n, 'avg_val_loss': avg_loss, 'precision': precision, 'recall': recall }
    with open(os.path.join(args.save_examples, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
