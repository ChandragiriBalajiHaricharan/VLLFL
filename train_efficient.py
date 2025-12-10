import argparse
import os
import gc
import time
import sys
import math
from contextlib import nullcontext

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, ConcatDataset
    from torchvision.ops import sigmoid_focal_loss, box_convert
except Exception as e:
    raise ImportError(f"Failed importing torch/torchvision: {e}")

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from dataset_loader import AgricultureObjectDetectionDataset

# --- MONKEY PATCH (Fixes LoRA for OWL-ViT) ---
# This allows the PEFT library to find the input embeddings of the OWL-ViT model
def get_input_embeddings(self):
    return self.owlv2.text_model.embeddings.token_embedding

def set_input_embeddings(self, value):
    self.owlv2.text_model.embeddings.token_embedding = value

Owlv2ForObjectDetection.get_input_embeddings = get_input_embeddings
Owlv2ForObjectDetection.set_input_embeddings = set_input_embeddings

# --- LOSS ENGINE (Focal Loss + Hungarian Matcher) ---

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["logits"].shape[:2]
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["class_labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        if sum(sizes) == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]
            
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx[0], idx[1], target_classes_o] = 1.0
        loss_ce = sigmoid_focal_loss(src_logits, target_classes, alpha=0.25, gamma=2.0, reduction="mean")
        return {'loss_ce': loss_ce * src_logits.shape[1]} 

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {'labels': self.loss_labels, 'boxes': self.loss_boxes}
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=r"C:\Users\balaj\OneDrive\Documents\project sem 5\archive\MetaFruit")
    p.add_argument("--fruits", nargs="+", default=["apple","grapefruit","lemon","orange","tangerine"])
    p.add_argument("--checkpoint", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum-steps", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--save-dir", default="vllfl_adapters_efficient")
    p.add_argument("--device", default="cpu") 
    p.add_argument("--log-file", type=str, default="training_metrics.csv")
    return p.parse_args()

# --- COLLATE FN ---
COLLATE_MAX_LENGTH = 16

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    pixel_values = torch.stack([item[0]['pixel_values'] for item in batch])
    
    input_ids_list = []
    attention_mask_list = []
    MAX_LENGTH = COLLATE_MAX_LENGTH
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
    
    labels = []
    for item in batch:
        target = item[1]
        orig_h, orig_w = target['orig_size']
        boxes_xyxy = target['boxes']
        
        if len(boxes_xyxy) == 0:
            boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_norm = boxes_xyxy / torch.tensor([orig_w, orig_h, orig_w, orig_h])
            wh = boxes_norm[:, 2:] - boxes_norm[:, :2]
            cxcy = (boxes_norm[:, 2:] + boxes_norm[:, :2]) / 2
            boxes_cxcywh = torch.cat([cxcy, wh], dim=1).clamp(0, 1)
            class_labels = torch.zeros(len(boxes_cxcywh), dtype=torch.int64)

        labels.append({"boxes": boxes_cxcywh, "class_labels": class_labels})
        
    return encoding, labels

def main():
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    print("Loading model and processor...")
    base_model = Owlv2ForObjectDetection.from_pretrained(args.checkpoint)
    processor = Owlv2Processor.from_pretrained(args.checkpoint)

    if args.use_lora and args.lora_r > 0:
        print(f"Using LoRA with rank {args.lora_r}...")
        peft_config = LoraConfig(r=args.lora_r, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none")
        model = get_peft_model(base_model, peft_config)
        
        # CRITICAL FIX: Ensure inputs require grads so checkpointing works with frozen weights
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
        model.print_trainable_parameters()
    else:
        print("Training full model...")
        model = base_model
    
    model.to(device)
    model.train()

    # Enable Gradient Checkpointing
    if device == "cpu":
        try:
            model.gradient_checkpointing_enable()
        except: pass

    # Initialize Loss Criterion
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 2, 'loss_bbox': 5}
    losses = ['labels', 'boxes']
    criterion = SetCriterion(1, matcher, weight_dict, losses)
    criterion.to(device)

    print("Building datasets...")
    all_datasets = []
    for fruit in args.fruits:
        d_path = os.path.join(args.data_root, fruit)
        if os.path.exists(d_path):
            ds = AgricultureObjectDetectionDataset(d_path, processor, [[fruit]], fruit_type=fruit)
            if len(ds) > 0:
                all_datasets.append(ds)
                print(f"  Loaded {len(ds)} from {fruit}")
    
    if not all_datasets:
        print("No datasets found!")
        return
        
    full_dataset = ConcatDataset(all_datasets)
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        running_loss = 0.0
        count = 0
        pbar = tqdm(train_loader)

        for batch_idx, batch in enumerate(pbar):
            if batch is None: continue
            inputs, targets = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward Pass
            outputs = model(**inputs)
            
            output_dict = {
                "logits": outputs.logits,
                "pred_boxes": outputs.pred_boxes
            }
            target_list = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = criterion(output_dict, target_list)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, skipping batch")
                continue

            # Backward Pass
            loss.backward()
            
            if (batch_idx + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            count += 1
            pbar.set_postfix({'loss': f"{(running_loss / count):.4f}"})

        # Save Checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        print(f"Saved epoch {epoch+1} to {args.save_dir}")

if __name__ == '__main__':
    main()