"""Train — supports local + Kaggle. Effective batch 8, AMP, EMA, ASL+ranking."""
import os, yaml, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from dataset import KneeDataset, resolve_data_root, LABEL_NAMES, load_meta
from text_proc import get_tokenizer
from transforms import get_train_transforms, get_val_transforms
from model_fusion import LabelAwareFusion, asymmetric_loss, ranking_loss

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    if HAS_TORCH: torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_folds(df, n_splits=5, seed=42):
    groups = df["StudyInstanceUID"] if "StudyInstanceUID" in df.columns else df.index
    # stratify approx: use first label or sum
    y = df[LABEL_NAMES[0]] if LABEL_NAMES[0] in df.columns else np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    folds = [-1]*len(df)
    for fold, (_, val_idx) in enumerate(gkf.split(df, y, groups)):
        for i in val_idx: folds[i]=fold
    df = df.copy(); df["fold"]=folds
    return df

def train_one_fold(cfg, df, fold=0, device="cuda"):
    print(f"[train] fold {fold} on {device}")
    use_text = cfg.get("use_reports", False) and cfg.get("text_model", "none") != "none"
    tokenizer = get_tokenizer(cfg.get("text_model","xlm-roberta-base")) if use_text else None
    tr_df = df[df["fold"]!=fold]
    va_df = df[df["fold"]==fold]
    if len(va_df)==0: va_df = tr_df.sample(frac=0.1, random_state=cfg["seed"])
    root = resolve_data_root()
    tr_ds = KneeDataset(tr_df, root, transform=get_train_transforms(cfg["image_size"]), tokenizer=tokenizer, max_len=cfg["max_report_len"], is_train=True, image_size=cfg["image_size"])
    va_ds = KneeDataset(va_df, root, transform=get_val_transforms(cfg["image_size"]), tokenizer=tokenizer, max_len=cfg["max_report_len"], is_train=False, image_size=cfg["image_size"])
    tr_loader = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

    model = LabelAwareFusion(num_labels=cfg["num_labels"], embed_dim=cfg["embed_dim"], text_model=cfg["text_model"], transformer_layers=cfg["label_transformer_layers"], heads=cfg["label_transformer_heads"]).to(device)
    # differential LR: backbone lower
    bb_params = list(model.image_branch.parameters())
    other = [p for n,p in model.named_parameters() if not n.startswith("image_branch")]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": cfg["lr_backbone"]}, {"params": other, "lr": cfg["lr_heads"]}], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] and device=="cuda")
    ema_decay = cfg.get("ema_decay", 0.9997)
    ema_state = None

    best_auc = 0
    for epoch in range(cfg["epochs"]):
        if hasattr(model.text_branch,"set_epoch"): model.text_branch.set_epoch(epoch)
        model.train()
        tot_loss=0
        for step, batch in enumerate(tr_loader):
            clips = batch["clips"].to(device)  # [B,8,3,H,W]
            labels = batch["labels"].to(device)
            input_ids = batch.get("input_ids")
            attn_mask = batch.get("attention_mask")
            if input_ids is not None and hasattr(input_ids,"to"): input_ids=input_ids.to(device); attn_mask=attn_mask.to(device)
            with torch.cuda.amp.autocast(enabled=cfg["amp"] and device=="cuda"):
                logits,_ = model(clips, input_ids, attn_mask, batch.get("planes"))
                loss_asl = asymmetric_loss(logits, labels, gamma_neg=cfg.get("asl_gamma_neg",4), clip=cfg.get("asl_clip",0.05))
                loss_rank = ranking_loss(logits, labels) * cfg.get("ranking_weight",0.15)
                loss = loss_asl * 0.85 + loss_rank
                loss = loss / cfg["grad_accum"]
            scaler.scale(loss).backward()
            if (step+1) % cfg["grad_accum"] == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad()
                # EMA update
                if ema_state is None: ema_state = {k: v.clone().detach() for k,v in model.state_dict().items()}
                else:
                    for k,v in model.state_dict().items(): ema_state[k].mul_(ema_decay).add_(v, alpha=1-ema_decay)
            tot_loss += loss.item()*cfg["grad_accum"]
            if step==0: print(f"  epoch {epoch} step {step} loss {loss.item()*cfg['grad_accum']:.4f}")
        sched.step()
        # val
        model.eval()
        all_logits=[]; all_labels=[]
        with torch.no_grad():
            for batch in va_loader:
                clips=batch["clips"].to(device); labels=batch["labels"]
                input_ids=batch.get("input_ids"); attn_mask=batch.get("attention_mask")
                if input_ids is not None and hasattr(input_ids,"to"): input_ids=input_ids.to(device); attn_mask=attn_mask.to(device)
                logits,_=model(clips, input_ids, attn_mask, batch.get("planes"))
                all_logits.append(logits.cpu()); all_labels.append(labels)
        if all_logits:
            logits = torch.cat(all_logits).numpy(); labels_np = torch.cat(all_labels).numpy()
            # compute mean AUROC over labels with both classes present
            aucs=[]
            for j in range(cfg["num_labels"]):
                m = labels_np[:,j]!=-1
                if m.sum()<10: continue
                y = labels_np[m,j]; p = 1/(1+np.exp(-logits[m,j]))
                if len(np.unique(y))<2: continue
                try: aucs.append(roc_auc_score(y,p))
                except: pass
            mean_auc = float(np.mean(aucs)) if aucs else 0
            print(f"  epoch {epoch} val mean AUROC {mean_auc:.4f} loss {tot_loss/len(tr_loader):.4f}")
            if mean_auc>best_auc:
                best_auc=mean_auc
                out = Path(cfg.get("model_dir","models"))/f"best_fold{fold}.pth"
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "ema": ema_state, "cfg": cfg, "auc": best_auc}, out)
                print(f"  saved {out}")
    print(f"[train] fold {fold} best AUROC {best_auc:.4f}")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--fold", type=int, default=0)
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config))
    cfg["model_dir"]=str(Path(args.config).parent.parent/"models")
    set_seed(cfg["seed"])
    df=load_meta(resolve_data_root())
    if "fold" not in df.columns or df["fold"].nunique()<=1:
        df=get_folds(df, n_splits=cfg["kfold"], seed=cfg["seed"])
    device="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    train_one_fold(cfg, df, fold=args.fold, device=device)
