"""DINOv2-small backbone — self-supervised ViT, CLS token feature.

Uses local copy at myomap-ai/pretrained/dinov2-small for offline Kaggle (Internet OFF).
Falls back to HF hub if local not found.
"""
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None

class Dinov2Backbone(nn.Module):
    def __init__(self, model_path=None, embed_dim=384, dropout=0.2, freeze_layers=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.freeze_layers = freeze_layers
        if not HAS_TRANSFORMERS:
            self.encoder = None
            self.proj = None
            return
        # resolve local path: myomap-ai/pretrained/dinov2-small
        if model_path is None:
            here = Path(__file__).resolve().parents[1]
            local = here / "pretrained" / "dinov2-small"
            # Kaggle path after copy: /kaggle/working/myomap-ai/pretrained/dinov2-small
            candidates = [local, Path("/kaggle/working/myomap-ai/pretrained/dinov2-small"), Path("/kaggle/input/datasets/shubangsrivatsa/myomap-ai-rsna-knee/pretrained/dinov2-small")]
            for cand in candidates:
                if (cand / "config.json").exists():
                    model_path = str(cand)
                    break
            if model_path is None:
                model_path = "facebook/dinov2-small"
        try:
            self.encoder = AutoModel.from_pretrained(model_path, trust_remote_code=False)
            # freeze early layers
            if freeze_layers > 0:
                # ViT layers are in encoder.encoder.layer
                try:
                    layers = self.encoder.encoder.layer
                    for i in range(min(freeze_layers, len(layers))):
                        for p in layers[i].parameters():
                            p.requires_grad = False
                except Exception:
                    pass
            hid = self.encoder.config.hidden_size  # 384 for small
            if hid != embed_dim:
                self.proj = nn.Sequential(
                    nn.LayerNorm(hid),
                    nn.Linear(hid, embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            else:
                self.proj = nn.Identity()
            print(f"[dinov2] loaded {model_path} hid={hid} -> {embed_dim}, freeze {freeze_layers} layers")
        except Exception as e:
            print(f"[dinov2] load failed {e}, fallback zeros")
            self.encoder = None
            self.proj = None

    def forward(self, clips):
        # clips: [B, N, 3, H, W] -> use CLS per clip
        if self.encoder is None:
            B, N = clips.shape[:2]
            return torch.zeros(B, N, self.embed_dim, device=clips.device)
        B, N, C, H, W = clips.shape
        # DINOv2 expects 224x224, we ensure resize already done in dataset/transforms
        x = clips.view(B*N, C, H, W)
        # AutoModel for dinov2 returns last_hidden_state [B*N, seq, hid], CLS is [:,0]
        out = self.encoder(pixel_values=x)
        # handle different output types: BaseModelOutput vs dict
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state[:, 0, :]  # CLS
        else:
            h = out[0][:, 0, :]
        if self.proj is not None and not isinstance(self.proj, nn.Identity):
            h = self.proj(h)
        return h.view(B, N, self.embed_dim)
