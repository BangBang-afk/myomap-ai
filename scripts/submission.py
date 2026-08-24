"""Generate Kaggle submission.csv — StudyInstanceUID + 12 probs."""
import argparse, yaml
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import KneeDataset, resolve_data_root, LABEL_NAMES, load_meta
from text_proc import get_tokenizer
from transforms import get_val_transforms
from model_fusion import LabelAwareFusion

def make_submission(ckpt, config, out="submission.csv", test_csv=None):
    cfg=yaml.safe_load(open(config))
    device="cuda" if torch.cuda.is_available() else "cpu"
    root=resolve_data_root()
    # load test meta
    if test_csv and Path(test_csv).exists():
        df=pd.read_csv(test_csv)
    else:
        # try Kaggle test
        for name in ["test.csv","sample_submission.csv"]:
            p=root/name
            if p.exists(): df=pd.read_csv(p); break
            hits=list(root.rglob(name))
            if hits: df=pd.read_csv(hits[0]); break
        else:
            print("[submission] no test.csv found, mock 5 rows")
            df=pd.DataFrame({"StudyInstanceUID":[f"mock_{i}" for i in range(5)]})

    tok=get_tokenizer(cfg.get("text_model","xlm-roberta-base")) if cfg.get("use_reports") else None
    ds=KneeDataset(df, root, transform=get_val_transforms(cfg["image_size"]), tokenizer=tok, max_len=cfg["max_report_len"], is_train=False, image_size=cfg["image_size"])
    loader=DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    model=LabelAwareFusion(num_labels=12, embed_dim=cfg["embed_dim"], text_model=cfg.get("text_model","xlm-roberta-base")).to(device)
    ckpt_data=torch.load(ckpt, map_location=device)
    state=ckpt_data.get("ema", ckpt_data.get("model", ckpt_data))
    if isinstance(state, dict) and "model" in state: state=state["model"]
    try: model.load_state_dict(state, strict=False)
    except Exception as e: print(f"load {e}")
    model.eval()
    rows=[]
    import numpy as np
    with torch.no_grad():
        for batch in loader:
            clips=batch["clips"].to(device)
            ids=batch.get("input_ids"); mask=batch.get("attention_mask")
            if ids is not None and hasattr(ids,"to"): ids=ids.to(device); mask=mask.to(device)
            logits,_=model(clips, ids, mask, batch.get("planes"))
            probs=torch.sigmoid(logits).cpu().numpy()
            for uid, p in zip(batch["study_uid"], probs):
                row={"StudyInstanceUID": uid}
                for n, v in zip(LABEL_NAMES, p): row[n]=float(v)
                rows.append(row)
    sub=pd.DataFrame(rows)
    # ensure column order
    cols=["StudyInstanceUID"]+LABEL_NAMES
    sub=sub[cols]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"wrote {out} shape {sub.shape}"); print(sub.head())

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--test_csv", default=None)
    args=ap.parse_args()
    make_submission(args.ckpt, args.config, args.out, args.test_csv)
