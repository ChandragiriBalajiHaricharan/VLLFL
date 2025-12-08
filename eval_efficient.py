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

# Import your custom dataset loader
from dataset_loader import AgricultureObjectDetectionDataset

# --- Helper Functions ---

def cxcywh_to_xyxy(boxes):
    """Converts boxes from center (cx, cy, w, h) to corners (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=1)

def xyxy_iou(boxes1, boxes2):
    """Calculates Intersection over Union (IoU) matrix between two sets of boxes."""
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
    return inter / (union + 1e-9)

def collate_fn(batch):
    """Prepares a batch of data for the model."""
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
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

def save_viz(image_path, gt_boxes, pred_boxes, matches, out_path):
    """Draws boxes on images: Green = Truth, Blue = Correct Match, Red = Wrong Guess."""
    try:
        im = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        
        # Draw Ground Truth (Green)
        for b in gt_boxes:
            draw.rectangle(b, outline=(0, 255, 0), width=3)
            
        # Draw Predictions
        for j, b in enumerate(pred_boxes):
            # Blue if it matched a real fruit, Red if it's a "ghost" prediction
            color = (0, 0, 255) if j in matches else (255, 0, 0)
            draw.rectangle(b, outline=color, width=3)
            
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        im.save(out_path)
    except Exception as e:
        print(f"Error saving viz: {e}")

# --- Main Evaluation Loop ---

def parse_args():
    p = argparse.ArgumentParser()
    # Update this path to match your folder
    p.add_argument("--data-root", default=r"C:\Users\balaj\OneDrive\Documents\project sem 5\archive\MetaFruit")
    p.add_argument("--fruits", nargs="+", default=["apple","grapefruit","lemon","orange","tangerine"])
    p.add_argument("--checkpoint", default="google/owlv2-base-patch16-ensemble")
    # Make sure this points to your trained adapter folder
    p.add_argument("--adapters", default="vllfl_adapters_efficient")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--viz-n", type=int, default=20, help="Number of images to save")
    p.add_argument("--out-dir", default="eval_results")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    print(f"Running evaluation on: {device}")

    # 1. Load Model
    print("Loading base model and adapters...")
    processor = Owlv2Processor.from_pretrained(args.checkpoint)
    base_model = Owlv2ForObjectDetection.from_pretrained(args.checkpoint)
    
    # Load your trained LoRA weights
    model = PeftModel.from_pretrained(base_model, args.adapters)
    model.to(device)
    model.eval()

    # 2. Prepare Data
    print("Building datasets...")
    all_datasets = []
    # Queries used during training
    query_texts = [["apple","grapefruit","lemon","orange","tangerine","leaf","person"]]
    
    for fruit in args.fruits:
        d_path = os.path.join(args.data_root, fruit)
        if os.path.exists(d_path):
            ds = AgricultureObjectDetectionDataset(d_path, processor, query_texts, fruit_type=fruit)
            if len(ds) > 0:
                all_datasets.append(ds)
                print(f"  Loaded {len(ds)} images from {fruit}")

    if not all_datasets:
        print("Error: No datasets found. Check your --data-root path.")
        return

    dataset = torch.utils.data.ConcatDataset(all_datasets)
    
    # Subset for faster evaluation
    N = len(dataset)
    n_eval = min(args.max_samples, N)
    torch.manual_seed(42)
    indices = torch.randperm(N)[:n_eval].tolist()
    from torch.utils.data import Subset
    val_ds = Subset(dataset, indices)
    
    print(f"Evaluating on {len(val_ds)} random samples...")

    loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=collate_fn
    )

    # 3. Evaluation Loop
    total_tp, total_fp, total_fn = 0, 0, 0
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
            
        pred_boxes = outputs.pred_boxes
        logits = getattr(outputs, 'logits', None)

        B = len(targets)
        for i in range(B):
            # Process Ground Truth
            t = targets[i]
            orig_h, orig_w = int(t['orig_size'][0]), int(t['orig_size'][1])
            gt_boxes_xyxy = t['boxes'] # absolute pixels
            
            # Normalize GT for IoU calc
            if gt_boxes_xyxy.shape[0] > 0:
                gt_norm = gt_boxes_xyxy / torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32)
                wh = gt_norm[:, 2:] - gt_norm[:, :2]
                cxcy = (gt_norm[:, 2:] + gt_norm[:, :2]) / 2
                gt_cxcywh = torch.cat([cxcy, wh], dim=1)
                gt_xyxy_norm = cxcywh_to_xyxy(gt_cxcywh)
            else:
                gt_cxcywh = torch.empty((0,4))
                gt_xyxy_norm = torch.empty((0,4))

            # Process Predictions
            preds = pred_boxes[i].cpu() # normalized cxcywh
            if logits is not None:
                scores = torch.softmax(logits[i], dim=-1)
                max_scores, _ = scores.max(dim=-1)
            else:
                max_scores = torch.ones(preds.shape[0])

            # Filter low confidence
            score_thresh = 0.05
            keep = (max_scores.cpu() >= score_thresh)
            kept_preds = preds[keep]
            kept_scores = max_scores.cpu()[keep]

            pred_xyxy_norm = cxcywh_to_xyxy(kept_preds)

            # Match Predictions to Truth
            iou_mat = xyxy_iou(pred_xyxy_norm, gt_xyxy_norm)
            
            matches_pred = set()
            matches_gt = set()
            
            # Simple Greedy Matching
            if iou_mat.numel() > 0:
                for _ in range(min(iou_mat.shape)):
                    max_val, idx = torch.max(iou_mat.reshape(-1), dim=0)
                    if max_val.item() < args.iou_threshold:
                        break
                    
                    idx = idx.item()
                    pred_idx = idx // iou_mat.shape[1]
                    gt_idx = idx % iou_mat.shape[1]
                    
                    matches_pred.add(pred_idx)
                    matches_gt.add(gt_idx)
                    
                    iou_mat[pred_idx, :] = -1
                    iou_mat[:, gt_idx] = -1

            # Stats
            tp = len(matches_pred)
            fp = len(kept_preds) - tp
            fn = len(gt_cxcywh) - tp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Visualization
            if example_i < args.viz_n:
                # Find the original image path from the dataset
                global_idx = indices[batch_idx * args.batch_size + i]
                # Helper to find image path in ConcatDataset
                rem = global_idx
                img_path = None
                for sub in dataset.datasets:
                    if rem < len(sub):
                        img_path = os.path.join(sub.data_dir, sub.image_files[rem])
                        break
                    rem -= len(sub)
                
                if img_path:
                    # Convert normalized preds to pixels for drawing
                    abs_preds = []
                    for pb in pred_xyxy_norm:
                        x1, y1, x2, y2 = pb * torch.tensor([orig_w, orig_h, orig_w, orig_h])
                        abs_preds.append([x1.item(), y1.item(), x2.item(), y2.item()])
                    
                    # GT is already in pixels
                    abs_gt = gt_boxes_xyxy.tolist()
                    
                    viz_path = viz_dir / f"eval_{example_i}.png"
                    save_viz(img_path, abs_gt, abs_preds, matches_pred, str(viz_path))
                    example_i += 1

    # 4. Final Metrics
    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    results = {
        'n_eval': n_eval,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }
    
    print("\n" + "="*30)
    print("Evaluation Complete!")
    print(f"Precision: {results['precision']}")
    print(f"Recall:    {results['recall']}")
    print(f"F1 Score:  {results['f1_score']}")
    print("="*30)
    
    with open(out_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_dir}/results.json")
    print(f"Visualizations saved to {out_dir}/viz/")

if __name__ == '__main__':
    main()