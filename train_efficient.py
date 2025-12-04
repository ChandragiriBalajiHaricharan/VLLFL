import argparse
import os
import gc
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from peft import get_peft_model, LoraConfig
from tqdm import tqdm

from dataset_loader import AgricultureObjectDetectionDataset

# Monkey patch (same as other scripts)
def get_input_embeddings(self):
    return self.owlv2.text_model.embeddings.token_embedding

def set_input_embeddings(self, value):
    self.owlv2.text_model.embeddings.token_embedding = value

Owlv2ForObjectDetection.get_input_embeddings = get_input_embeddings
Owlv2ForObjectDetection.set_input_embeddings = set_input_embeddings


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=r"C:\Users\balaj\OneDrive\Documents\project sem 5\archive\MetaFruit")
    p.add_argument("--fruits", nargs="+", default=["apple","grapefruit","lemon","orange","tangerine"])
    p.add_argument("--checkpoint", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--batch-size", type=int, default=1, help="Per-step batch size (use 1 for GPU with 6GB VRAM)")
    p.add_argument("--accum-steps", type=int, default=8, help="Gradient accumulation steps (simulates larger batch)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-batches", type=int, default=0, help="If >0, stop after this many batches (smoke test)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=0, help="If >0, limit dataset to this many samples per epoch")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (smaller=r for speed)")
    p.add_argument("--save-dir", default="vllfl_adapters_efficient")
    p.add_argument("--device", default="auto", choices=["cuda", "cpu", "auto"], help="Device to use (auto=cuda if available)")
    return p.parse_args()


# Use module-level globals for collate configuration so the function is picklable
COLLATE_MAX_LENGTH = 16
COLLATE_DEVICE = 'cpu'

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


def main():
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Load model and processor
    print("Loading model and processor...")
    base_model = Owlv2ForObjectDetection.from_pretrained(args.checkpoint)
    processor = Owlv2Processor.from_pretrained(args.checkpoint)

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(base_model, peft_config)
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    # Keep gradient checkpointing disabled for GPU - use gradient accumulation instead
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    # Dataset
    print("Building datasets...")
    all_datasets = []
    for fruit in args.fruits:
        d_path = os.path.join(args.data_root, fruit)
        if os.path.exists(d_path):
            ds = AgricultureObjectDetectionDataset(d_path, processor, [["apple","grapefruit","lemon","orange","tangerine","leaf","person"]], fruit_type=fruit)
            if len(ds) > 0:
                all_datasets.append(ds)
                print(f"  Loaded {len(ds)} from {fruit}")
    if not all_datasets:
        print("No datasets found. Check data root.")
        return
    train_dataset = ConcatDataset(all_datasets)
    print(f"Full dataset size: {len(train_dataset)}")

    # Optionally limit samples
    if args.max_samples > 0:
        from torch.utils.data import Subset
        indices = torch.randperm(len(train_dataset))[:args.max_samples].tolist()
        train_dataset = Subset(train_dataset, indices)
        print(f"Using subset size: {len(train_dataset)}")

    # Configure top-level collate globals so the collate function is picklable
    global COLLATE_MAX_LENGTH, COLLATE_DEVICE
    COLLATE_MAX_LENGTH = 16
    COLLATE_DEVICE = device
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if device == "cuda":
        scaler = torch.amp.GradScaler(device=device)
    else:
        scaler = None

    global_step = 0
    total_start = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        epoch_start = time.time()
        running_loss = 0.0
        count = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            inputs, targets = batch
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # forward + loss under AMP
            if device == "cuda":
                ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float32)
            else:
                ctx = nullcontext()
            with ctx:
                outputs = model(**inputs)
                pred_boxes = outputs.pred_boxes  # [B, Q, 4]
                loss = torch.tensor(0.0, device=device)
                num = 0
                # Use up to K predictions per sample (small K)
                K = 3
                for i, t in enumerate(targets):
                    if len(t['boxes']) == 0:
                        continue
                    n_targets = min(len(t['boxes']), K)
                    for j in range(n_targets):
                        tb = t['boxes'][j].to(device)
                        pb = pred_boxes[i, j, :]
                        loss = loss + F.mse_loss(pb, tb)
                        num += 1
                if num > 0:
                    loss = loss / num
                else:
                    # skip updates if no targets
                    loss = None

            if loss is None:
                # cleanup and continue
                del inputs, targets, outputs
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                continue

            # backward with accumulation
            if scaler is not None:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item()
            count += 1
            global_step += 1

            pbar.set_postfix({'loss': f"{(running_loss / count):.4f}", 'step': global_step})

            # Aggressive memory cleanup on GPU
            if device == "cuda" and (batch_idx + 1) % args.accum_steps == 0:
                torch.cuda.empty_cache()
                gc.collect()            # quick smoke-test exit
            if args.max_batches > 0 and global_step >= args.max_batches:
                print(f"Reached max batches ({args.max_batches}), stopping early (smoke test).")
                break

            # cleanup
            del inputs, targets, outputs
            torch.cuda.empty_cache()
            gc.collect()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} done — avg loss: {(running_loss / max(count,1)):.4f} — time: {epoch_time:.1f}s")

        # Save checkpoint each epoch
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"adapters_epoch{epoch+1}.pt")
        print(f"Saving checkpoint to {ckpt_path} (LoRA adapters)")
        model.save_pretrained(args.save_dir)

        if args.max_batches > 0 and global_step >= args.max_batches:
            break

    total_time = time.time() - total_start
    print(f"Total training time: {total_time:.1f}s")
    print("Done.")


if __name__ == '__main__':
    main()
