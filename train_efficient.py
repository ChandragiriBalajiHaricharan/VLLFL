import argparse
import os
import gc
import time
from contextlib import nullcontext

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, ConcatDataset
except Exception as e:
    raise ImportError(f"Missing libraries. Error: {e}")

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from peft import get_peft_model, LoraConfig
from tqdm import tqdm

# Import your dataset loader (Ensure dataset_loader.py is in the repo!)
from dataset_loader import AgricultureObjectDetectionDataset

# --- MONKEY PATCHING FOR OWLV2 (Required for LoRA) ---
def get_input_embeddings(self):
    return self.owlv2.text_model.embeddings.token_embedding

def set_input_embeddings(self, value):
    self.owlv2.text_model.embeddings.token_embedding = value

Owlv2ForObjectDetection.get_input_embeddings = get_input_embeddings
Owlv2ForObjectDetection.set_input_embeddings = set_input_embeddings
# -----------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Default is set to generic "./data", but you can override this in Colab
    p.add_argument("--data-root", default="./data") 
    p.add_argument("--fruits", nargs="+", default=["apple","grapefruit","lemon","orange","tangerine"])
    p.add_argument("--checkpoint", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size (Try 4 for T4 GPU)")
    p.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=2) 
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    p.add_argument("--save-dir", default="./checkpoints")
    p.add_argument("--device", default="auto")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--log-file", type=str, default="training_metrics.csv")
    return p.parse_args()

# Global Collate config
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
            boxes_cxcywh = torch.zeros(1, 4, dtype=torch.float32)
        else:
            boxes_norm = boxes_xyxy / torch.tensor([orig_w, orig_h, orig_w, orig_h])
            wh = boxes_norm[:, 2:] - boxes_norm[:, :2]
            cxcy = (boxes_norm[:, 2:] + boxes_norm[:, :2]) / 2
            boxes_cxcywh = torch.cat([cxcy, wh], dim=1).clamp(0, 1)
        labels.append({"boxes": boxes_cxcywh})
    return encoding, labels

def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    
    def cxcywh_to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

    a = cxcywh_to_xyxy(boxes1)
    b = cxcywh_to_xyxy(boxes2)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting Training on {device} ---")

    # 1. Load Model
    print("Loading OwlViT model...")
    base_model = Owlv2ForObjectDetection.from_pretrained(args.checkpoint)
    processor = Owlv2Processor.from_pretrained(args.checkpoint)

    # 2. Setup LoRA (Only Train Adapters)
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(base_model, peft_config)
    model.to(device)
    model.train()
    
    model.print_trainable_parameters()

    # 3. Load Datasets
    print(f"Loading datasets from {args.data_root}...")
    all_datasets = []
    
    for fruit in args.fruits:
        d_path = os.path.join(args.data_root, fruit)
        if os.path.exists(d_path):
            try:
                # Queries list for OwlViT zero-shot guidance
                queries = [["apple","grapefruit","lemon","orange","tangerine","leaf","person"]]
                # Ensure the dataset loader accepts these arguments
                ds = AgricultureObjectDetectionDataset(d_path, processor, queries, fruit_type=fruit)
                if len(ds) > 0:
                    all_datasets.append(ds)
                    print(f"  > Loaded {len(ds)} samples for {fruit}")
            except Exception as e:
                print(f"  ! Error loading {fruit}: {e}")
        else:
            print(f"  ! Warning: Folder not found: {d_path}")

    if not all_datasets:
        print("ERROR: No datasets loaded. Please check your --data-root path.")
        return

    train_dataset = ConcatDataset(all_datasets)
    print(f"Total training samples: {len(train_dataset)}")

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == 'cuda')
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device='cuda') if device == "cuda" else None
    
    # Training Loop
    global_step = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        running_loss = 0.0
        pbar = tqdm(loader)
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None: continue
            
            inputs, targets = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16) if device == 'cuda' else nullcontext():
                outputs = model(**inputs)
                pred_boxes = outputs.pred_boxes
                
                loss = torch.tensor(0.0, device=device)
                count = 0
                for i, t in enumerate(targets):
                    gt = t['boxes'].to(device)
                    if len(gt) == 0: continue
                    preds = pred_boxes[i, :args.top_k, :]
                    ious = box_iou(preds, gt)
                    if ious.numel() > 0:
                        max_iou, _ = ious.max(dim=0)
                        loss += (1 - max_iou).mean()
                        count += 1
                
                if count > 0: loss = loss / count
                else: loss = torch.tensor(0.0, device=device, requires_grad=True)

            if scaler:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            global_step += 1

        # Save Checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(os.path.join(args.save_dir, f"epoch_{epoch+1}"))
        print(f"Saved checkpoint for Epoch {epoch+1}")

    print("Training Complete!")

if __name__ == '__main__':
    main()
