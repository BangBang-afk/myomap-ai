"""Export fusion model to ONNX for optional mediorch integration."""
import torch, yaml, argparse
from pathlib import Path
from model_fusion import LabelAwareFusion

def export(ckpt, config, out="models/myomap.onnx"):
    cfg=yaml.safe_load(open(config))
    device="cpu"
    model=LabelAwareFusion(num_labels=12, embed_dim=cfg["embed_dim"], text_model=cfg.get("text_model","xlm-roberta-base")).to(device)
    ckpt_data=torch.load(ckpt, map_location=device)
    state=ckpt_data.get("ema", ckpt_data.get("model", ckpt_data))
    if isinstance(state, dict) and "model" in state: state=state["model"]
    try: model.load_state_dict(state, strict=False)
    except Exception as e: print(f"load {e}")
    model.eval()
    dummy_clips=torch.randn(1,8,3,256,256)
    dummy_ids=torch.randint(0,1000,(1,256))
    dummy_mask=torch.ones(1,256, dtype=torch.long)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (dummy_clips, dummy_ids, dummy_mask), out, input_names=["clips","input_ids","attention_mask"], output_names=["logits"], dynamic_axes={"clips":{0:"batch"},"input_ids":{0:"batch"},"attention_mask":{0:"batch"}}, opset_version=17)
    print(f"exported {out}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="models/myomap.onnx")
    args=ap.parse_args()
    export(args.ckpt, args.config, args.out)
