"""Fusion — label-aware attention over clips + text + label transformer + 12 heads.

Mirrors HuggingFace `leminhhung0101/knee-model` pattern: 12 learned queries, metadata priors, ranking-aware.
Kaggle-friendly: 2-layer transformer, vectorized heads. Supports ConvNeXt/EfficientNet and DINOv2.
"""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from .model_image import ImageBackbone
    from .model_text import TextBranch
    try:
        from .model_backbone import MyoMapViTBackbone
        Dinov2Backbone = MyoMapViTBackbone  # alias for compat
    except ImportError:
        try:
            from model_backbone import MyoMapViTBackbone
            Dinov2Backbone = MyoMapViTBackbone
        except Exception:
            Dinov2Backbone = None
except ImportError:
    from model_image import ImageBackbone
    from model_text import TextBranch
    try:
        from model_backbone import MyoMapViTBackbone
        Dinov2Backbone = MyoMapViTBackbone
    except Exception:
        Dinov2Backbone = None

PLANE_PRIORS = {
    "default": [0.85, 0.85, 0.85],
}

class LabelAwareFusion(nn.Module):
    def __init__(self, num_labels=12, embed_dim=384, text_embed_dim=None, transformer_layers=2, heads=8, dropout=0.3, text_model="xlm-roberta-base", backbone="convnextv2_tiny.fcmae", backbone_path=None, freeze_layers=0):
        super().__init__()
        if not HAS_TORCH:
            return
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        # choose backbone: myomap_vit_small (DINOv2 foundation, renamed) vs timm
        if backbone in ("myomap_vit_small", "dinov2_small") and Dinov2Backbone is not None:
            self.image_branch = Dinov2Backbone(model_path=backbone_path, embed_dim=embed_dim, freeze_layers=freeze_layers)
        else:
            from model_image import ImageBackbone as ImgBB
            bb_name = backbone if backbone not in ("dinov2_small", "myomap_vit_small") else "convnextv2_tiny.fcmae"
            self.image_branch = ImgBB(backbone=bb_name, pretrained=True, embed_dim=embed_dim)
        self.text_branch = TextBranch(model_name=text_model, embed_dim=embed_dim)
        self.label_queries = nn.Parameter(torch.randn(num_labels, embed_dim) * 0.02)
        self.label_embedding = nn.Parameter(torch.randn(num_labels, embed_dim) * 0.02)
        self.plane_embed = nn.Embedding(3, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_scale = embed_dim ** -0.5
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.label_transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.head_norm = nn.LayerNorm(embed_dim)
        self.head_fc1 = nn.Parameter(torch.randn(num_labels, embed_dim, embed_dim))
        self.head_fc2 = nn.Parameter(torch.randn(num_labels, embed_dim, 1))
        self.head_bias1 = nn.Parameter(torch.zeros(num_labels, embed_dim))
        self.head_bias2 = nn.Parameter(torch.zeros(num_labels, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, clips, input_ids=None, attention_mask=None, planes=None):
        B = clips.shape[0]
        img_feats = self.image_branch(clips)
        if planes is not None and len(planes) > 0:
            plane_ids = []
            for b_planes in (planes if isinstance(planes[0], list) else [planes]*B):
                ids = []
                for p in b_planes:
                    pl = str(p).lower()
                    if "cor" in pl: ids.append(1)
                    elif "ax" in pl: ids.append(2)
                    else: ids.append(0)
                while len(ids) < img_feats.shape[1]: ids.append(0)
                plane_ids.append(ids[:img_feats.shape[1]])
            plane_ids = torch.tensor(plane_ids, device=img_feats.device)
            plane_emb = self.plane_embed(plane_ids)
            img_feats = img_feats + plane_emb * 0.1
        Q = self.q_proj(self.label_queries)
        K = self.k_proj(img_feats)
        V = self.v_proj(img_feats)
        scores = torch.einsum("ld,bnd->bln", Q, K) * self.attn_scale
        attn = F.softmax(scores, dim=-1)
        label_feats = torch.einsum("bln,bnd->bld", attn, V)
        label_feats = label_feats + self.label_embedding.unsqueeze(0)
        if input_ids is not None and hasattr(self.text_branch, 'encoder') and self.text_branch.encoder is not None:
            txt = self.text_branch(input_ids, attention_mask)
            label_feats = label_feats + txt.unsqueeze(1) * 0.2
        label_feats = self.label_transformer(label_feats)
        h = self.head_norm(label_feats)
        h = torch.einsum("bld,ldh->blh", h, self.head_fc1) + self.head_bias1.unsqueeze(0)
        h = F.gelu(h)
        h = self.dropout(h)
        logits = torch.einsum("bld,ldo->blo", h, self.head_fc2).squeeze(-1) + self.head_bias2.squeeze(-1).unsqueeze(0)
        return logits, attn


def asymmetric_loss(logits, targets, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
    mask = targets != -1
    if mask.sum() == 0:
        return logits.sum() * 0
    probs = torch.sigmoid(logits)
    pos = -torch.log(probs.clamp(min=eps)) * (1 - probs).pow(gamma_pos) * (targets == 1).float()
    neg = -torch.log((1 - probs).clamp(min=eps)) * probs.pow(gamma_neg) * (targets == 0).float()
    if clip is not None and clip > 0:
        neg = neg * (probs < clip).float()
    loss = (pos + neg) * mask.float()
    return loss.sum() / mask.float().sum().clamp(min=1)

def ranking_loss(logits, targets):
    mask = targets != -1
    losses = []
    for j in range(targets.shape[1]):
        pos = logits[mask[:, j] & (targets[:, j] == 1), j]
        neg = logits[mask[:, j] & (targets[:, j] == 0), j]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        losses.append(F.relu(1 - diff).mean())
    if not losses:
        return logits.sum() * 0
    return torch.stack(losses).mean()
