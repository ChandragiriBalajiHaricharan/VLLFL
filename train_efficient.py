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
    raise ImportError(
        "Failed importing torch. If you see an ImportError about typing_extensions or similar,\n"
        "please upgrade or install a compatible `typing_extensions` version. Example:\n"
        "  python -m pip install 'typing_extensions<4.6.0'\n"
        "Or recreate your env with compatible torch/torchaudio versions.\n"
        f"Original error: {e}") from e

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
    # Windows + OneDrive can be slow with multiple workers; default to 0 for safety.
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0, help="If >0, limit dataset to this many samples per epoch")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (smaller=r for speed)")
    p.add_argument("--save-dir", default="vllfl_adapters_efficient")
    p.add_argument("--device", default="auto", choices=["cuda", "cpu", "auto"], help="Device to use (auto=cuda if available)")
    p.add_argument("--top-k", type=int, default=3, help="Number of top predictions to consider per sample (K)")
    p.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold to count a correct detection (accuracy)")
    p.add_argument("--cache-dataset", action="store_true", help="Preload entire dataset into RAM (faster but uses memory)")
    p.add_argument("--persistent-workers", action="store_true", help="Use persistent workers in DataLoader (speeds up repeated epochs)")
    p.add_argument("--log-file", type=str, default="training_metrics.csv", help="CSV file to append epoch loss/accuracy for federated aggregation")
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


def box_iou(boxes1, boxes2):
    # boxes: [N,4] in cx,cy,w,h (normalized) -> convert to xyxy and compute pairwise IoU
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    def cxcywh_to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    a = cxcywh_to_xyxy(boxes1)
    b = cxcywh_to_xyxy(boxes2)

    lt = torch.max(a[:, None, :2], b[None, :, :2])  # left-top
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])  # right-bottom
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / (union + 1e-9)
    return iou


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

    # Optionally cache entire dataset in RAM for faster epochs (uses more memory)
    if args.cache_dataset:
        print("Caching dataset into RAM (this may use a lot of memory)...")
        items = []
        for i in range(len(train_dataset)):
            items.append(train_dataset[i])
        from torch.utils.data import Dataset as _TorchDataset
        class InMemoryDataset(_TorchDataset):
            def __init__(self, items):
                self.items = items
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                return self.items[idx]
        train_dataset = InMemoryDataset(items)
        print(f"Cached dataset size: {len(train_dataset)}")

    # Configure top-level collate globals so the collate function is picklable
    global COLLATE_MAX_LENGTH, COLLATE_DEVICE
    COLLATE_MAX_LENGTH = 16
    COLLATE_DEVICE = device
    pin_memory = True if device == 'cuda' else False
    persistent_workers = args.persistent_workers if args.num_workers > 0 else False
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

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
        epoch_correct = 0
        epoch_total = 0
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
                K = args.top_k
                # compute loss (MSE) over matched positions (simple pairing up to K)
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

                # Simple IoU-based accuracy (per-sample): a sample is correct if any of top-K preds
                # has IoU >= threshold with any ground-truth box.
                batch_correct = 0
                batch_total = 0
                for i, t in enumerate(targets):
                    gt = t['boxes']
                    if len(gt) == 0:
                        continue
                    preds_k = pred_boxes[i, :K, :].to(device)
                    gt = gt.to(device)
                    ious = box_iou(preds_k, gt)  # [K, G]
                    if ious.numel() == 0:
                        continue
                    max_iou = ious.max().item()
                    if max_iou >= args.iou_threshold:
                        batch_correct += 1
                    batch_total += 1

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

            # update epoch-level accuracy counters
            epoch_correct += batch_correct
            epoch_total += batch_total

            accuracy = (epoch_correct / max(epoch_total, 1)) if epoch_total > 0 else 0.0
            pbar.set_postfix({'loss': f"{(running_loss / count):.4f}", 'acc': f"{(accuracy*100):.2f}%", 'step': global_step})

            # Aggressive memory cleanup on GPU
            if device == "cuda" and (batch_idx + 1) % args.accum_steps == 0:
                torch.cuda.empty_cache()
                gc.collect()
            # quick smoke-test exit
            if args.max_batches > 0 and global_step >= args.max_batches:
                print(f"Reached max batches ({args.max_batches}), stopping early (smoke test).")
                break

            # cleanup
            del inputs, targets, outputs
            torch.cuda.empty_cache()
            gc.collect()

        epoch_time = time.time() - epoch_start
        avg_loss = (running_loss / max(count, 1))
        accuracy = (epoch_correct / max(epoch_total, 1)) if epoch_total > 0 else 0.0
        print(f"Epoch {epoch+1} done — avg loss: {avg_loss:.4f} — acc: {(accuracy*100):.2f}% — time: {epoch_time:.1f}s")

        # Append metrics to CSV for federated aggregation or later analysis
        try:
            write_header = not os.path.exists(args.log_file)
            with open(args.log_file, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("epoch,avg_loss,accuracy,seconds\n")
                f.write(f"{epoch+1},{avg_loss:.6f},{accuracy:.6f},{epoch_time:.1f}\n")
        except Exception as e:
            print(f"Warning: failed to write log file {args.log_file}: {e}")

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
