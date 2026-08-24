"""Infer single study — image + optional report → 12 probs + Grad-CAM stub."""
import torch, argparse, yaml
from pathlib import Path
import numpy as np
from dataset import LABEL_NAMES
from model_fusion import LabelAwareFusion

def infer_single(ckpt, clips_tensor, report_text="", config="configs/config.yaml", device="cuda"):
    cfg=yaml.safe_load(open(config))
    model=LabelAwareFusion(num_labels=12, embed_dim=cfg["embed_dim"], text_model=cfg.get("text_model","xlm-roberta-base")).to(device)
    ckpt_data=torch.load(ckpt, map_location=device)
    state=ckpt_data.get("ema", ckpt_data.get("model", ckpt_data))
    if isinstance(state, dict) and "model" in state: state=state["model"]
    try: model.load_state_dict(state, strict=False)
    except: pass
    model.eval()
    # clips_tensor: [N,3,H,W] or [1,N,3,H,W]
    if clips_tensor.dim()==4: clips_tensor=clips_tensor.unsqueeze(0)
    clips_tensor=clips_tensor.to(device)
    input_ids=attn=None
    if report_text:
        from text_proc import get_tokenizer, tokenize_report
        tok=get_tokenizer(cfg.get("text_model","xlm-roberta-base"))
        enc=tokenize_report(report_text, tok, cfg["max_report_len"])
        if "input_ids" in enc and hasattr(enc["input_ids"],"unsqueeze"):
            input_ids=enc["input_ids"].unsqueeze(0).to(device); attn=enc["attention_mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits, attn_map = model(clips_tensor, input_ids, attn)
        probs=torch.sigmoid(logits).cpu().numpy()[0]
    for n,p in zip(LABEL_NAMES, probs): print(f"{n:22s} {p:.3f}")
    return probs

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--report", default="")
    args=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    # mock clips demo
    clips=torch.randn(1,8,3,256,256)
    infer_single(args.ckpt, clips, args.report, args.config, device)
