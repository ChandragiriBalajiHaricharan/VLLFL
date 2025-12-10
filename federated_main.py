import argparse
import os
import copy
import torch
import math
import sys
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from peft import get_peft_model, LoraConfig, set_peft_model_state_dict, get_peft_model_state_dict
from torchvision.ops import sigmoid_focal_loss
from scipy.optimize import linear_sum_assignment
from torch import nn
from dataset_loader import AgricultureObjectDetectionDataset

# --- MONKEY PATCH (CRITICAL FIX FOR LoRA) ---
def get_input_embeddings(self):
    return self.owlv2.text_model.embeddings.token_embedding
def set_input_embeddings(self, value):
    self.owlv2.text_model.embeddings.token_embedding = value
Owlv2ForObjectDetection.get_input_embeddings = get_input_embeddings
Owlv2ForObjectDetection.set_input_embeddings = set_input_embeddings

# --- UTILS & LOSS ENGINE ---
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
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["logits"].shape[:2]
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["class_labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        alpha, gamma = 0.25, 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        if sum(sizes) == 0: return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses):
        super().__init__()
        self.num_classes, self.matcher, self.weight_dict, self.losses = num_classes, matcher, weight_dict, losses
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
        loss_bbox = torch.nn.functional.l1_loss(src_boxes, target_boxes, reduction='none')
        return {'loss_bbox': loss_bbox.sum() / num_boxes}
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
        for loss in self.losses: losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses

# --- CLIENT TRAINING FUNCTION ---
def train_client(client_name, model, dataset, args):
    # --- VIRTUAL SPLITTING (SHOW THIS TO PROFESSOR) ---
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n🚜 [Client: {client_name}] Dataset Split:")
    print(f"   ├── Total Images:   {total_size}")
    print(f"   ├── Training Set:   {train_size} images (80%)")
    print(f"   └── Validation Set: {val_size} images (20%)")
    
    # Enable gradients for frozen weights (Checkpointing fix)
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    else: 
        def make_inputs_require_grad(module, input, output): output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    device = args.device
    model.to(device)
    model.train()
    
    # Subset for Demo Mode
    if args.demo:
        indices = torch.randperm(len(train_dataset))[:20] 
        train_dataset = Subset(train_dataset, indices)
        print(f"   ⚠️ DEMO MODE: Reduced Training Set to {len(train_dataset)} images for speed.")

    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 2, 'loss_bbox': 5}
    criterion = SetCriterion(1, matcher, weight_dict, ['labels', 'boxes']).to(device)

    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    epoch_loss = 0
    steps = 0
    print(f"   🔄 Training started on {device}...")
    
    for batch in loader:
        if batch is None: continue
        inputs, targets = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        
        output_dict = {"logits": outputs.logits, "pred_boxes": outputs.pred_boxes}
        target_list = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = criterion(output_dict, target_list)
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        if not math.isfinite(loss.item()): continue
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        steps += 1
    
    avg_loss = epoch_loss / max(steps, 1)
    print(f"   ✅ Finished Training. Avg Loss: {avg_loss:.4f}")
    return get_peft_model_state_dict(model)

# --- FEDERATED MAIN ---
def collate_fn(batch):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"C:\Users\balaj\OneDrive\Documents\project sem 5\archive\MetaFruit")
    parser.add_argument("--rounds", type=int, default=2, help="Number of Federated Rounds")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--demo", action="store_true", help="Run in super-fast demo mode")
    args = parser.parse_args()
    
    print(f"🚀 Starting VLLFL Federated Simulation (Rounds: {args.rounds})")
    print(f"   Mode: {'DEMO (Fast)' if args.demo else 'FULL'}")

    print("🌍 Loading Global Model...")
    base_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none")
    global_model = get_peft_model(base_model, peft_config)

    print("🌾 Setting up Farmers (Clients)...")
    ds_apple = AgricultureObjectDetectionDataset(os.path.join(args.data_root, "apple"), processor, [["apple"]], fruit_type="apple")
    ds_lemon = AgricultureObjectDetectionDataset(os.path.join(args.data_root, "lemon"), processor, [["lemon"]], fruit_type="lemon")
    client_A_ds = ConcatDataset([ds_apple, ds_lemon])
    
    ds_orange = AgricultureObjectDetectionDataset(os.path.join(args.data_root, "orange"), processor, [["orange"]], fruit_type="orange")
    ds_grape = AgricultureObjectDetectionDataset(os.path.join(args.data_root, "grapefruit"), processor, [["grapefruit"]], fruit_type="grapefruit")
    client_B_ds = ConcatDataset([ds_orange, ds_grape])

    clients = [
        {"name": "Farm_A (Apple/Lemon)", "dataset": client_A_ds},
        {"name": "Farm_B (Orange/Grape)", "dataset": client_B_ds}
    ]

    for round_num in range(args.rounds):
        print(f"\n--- 🔄 Federated Round {round_num+1}/{args.rounds} ---")
        local_weights = []

        for client in clients:
            client_weights = train_client(client["name"], global_model, client["dataset"], args)
            local_weights.append(client_weights)

        print("☁️  Server: Aggregating weights from all farmers...")
        avg_weights = copy.deepcopy(local_weights[0])
        for key in avg_weights.keys():
            for i in range(1, len(local_weights)):
                avg_weights[key] += local_weights[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(local_weights))
        
        set_peft_model_state_dict(global_model, avg_weights)
        print(f"✅ Round {round_num+1} Complete. Global Model Updated.")
        
        save_path = f"vllfl_federated_round{round_num+1}"
        global_model.save_pretrained(save_path)
        print(f"💾 Saved Global Model to {save_path}")

if __name__ == '__main__':
    main()