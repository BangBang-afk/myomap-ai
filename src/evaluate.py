"""Evaluate — compute per-label AUROC, F1, conf matrix on val fold."""
import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from dataset import KneeDataset, resolve_data_root, LABEL_NAMES, load_meta
from text_proc import get_tokenizer
from transforms import get_val_transforms
from model_fusion import LabelAwareFusion

def evaluate(ckpt, config, fold=0, device="cuda"):
    cfg=yaml.safe_load(open(config))
    df=load_meta(resolve_data_root())
    # filter to fold
    if "fold" in df.columns: df=df[df["fold"]==fold]
    if len(df)==0: print("[evaluate] no data for fold, using mock"); return
    tok=get_tokenizer(cfg.get("text_model","xlm-roberta-base")) if cfg.get("use_reports") else None
    ds=KneeDataset(df, resolve_data_root(), transform=get_val_transforms(cfg["image_size"]), tokenizer=tok, max_len=cfg["max_report_len"], is_train=False, image_size=cfg["image_size"])
    loader=DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    model=LabelAwareFusion(num_labels=cfg["num_labels"], embed_dim=cfg["embed_dim"], text_model=cfg["text_model"]).to(device)
    ckpt_data=torch.load(ckpt, map_location=device)
    # prefer EMA if present
    state=ckpt_data.get("ema", ckpt_data.get("model", ckpt_data))
    # handle wrapped dict
    if isinstance(state, dict) and "model" in state: state=state["model"]
    try: model.load_state_dict(state, strict=False)
    except Exception as e: print(f"load warning {e}")
    model.eval()
    all_logits=[]; all_labels=[]
    with torch.no_grad():
        for batch in loader:
            clips=batch["clips"].to(device)
            input_ids=batch.get("input_ids"); attn=batch.get("attention_mask")
            if input_ids is not None and hasattr(input_ids,"to"): input_ids=input_ids.to(device); attn=attn.to(device)
            logits,_=model(clips, input_ids, attn, batch.get("planes"))
            all_logits.append(logits.cpu()); all_labels.append(batch["labels"])
    logits=torch.cat(all_logits).numpy(); labels=torch.cat(all_labels).numpy()
    probs=1/(1+np.exp(-logits))
    for j, name in enumerate(LABEL_NAMES):
        m=labels[:,j]!=-1
        if m.sum()==0: continue
        y=labels[m,j]; p=probs[m,j]
        if len(np.unique(y))<2: print(f"{name}: only one class, skip AUROC"); continue
        auc=roc_auc_score(y,p); f1=f1_score(y, (p>0.5).astype(int), zero_division=0)
        print(f"{name:22s} AUROC {auc:.3f} F1 {f1:.3f} n_pos {(y==1).sum()}/{len(y)}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--fold", type=int, default=0)
    args=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    evaluate(args.ckpt, args.config, args.fold, device)
