"""Image backbone — ConvNeXtV2-Tiny via timm, extracts per-clip features."""
try:
    import torch
    import torch.nn as nn
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    torch = None

class ImageBackbone(nn.Module):
    def __init__(self, backbone="convnextv2_tiny.fcmae", pretrained=True, embed_dim=384, dropout=0.3):
        super().__init__()
        if not HAS_TIMM:
            self.backbone = None
            self.embed_dim = embed_dim
            return
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features if hasattr(self.backbone, "num_features") else 768
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embed_dim = embed_dim

    def forward(self, clips):
        # clips: [B, N, 3, H, W] -> [B*N, 3, H, W]
        if self.backbone is None:
            B, N = clips.shape[:2]
            return torch.zeros(B, N, self.embed_dim, device=clips.device)
        B, N, C, H, W = clips.shape
        x = clips.view(B*N, C, H, W)
        f = self.backbone(x)  # [B*N, feat_dim]
        f = self.proj(f)      # [B*N, embed_dim]
        return f.view(B, N, self.embed_dim)
