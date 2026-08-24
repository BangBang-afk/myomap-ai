"""Text branch — XLM-RoBERTa-base for multilingual reports."""
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class TextBranch(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", embed_dim=384, dropout=0.2, freeze_epochs=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0
        if not HAS_TRANSFORMERS:
            self.encoder = None
            self.proj = None
            return
        try:
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=False)
            hid = self.encoder.config.hidden_size
            self.proj = nn.Sequential(
                nn.LayerNorm(hid),
                nn.Linear(hid, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        except Exception as e:
            print(f"[model_text] load failed {e}, text branch disabled")
            self.encoder = None
            self.proj = None

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        if self.encoder and epoch < self.freeze_epochs:
            for p in self.encoder.parameters(): p.requires_grad = False
        elif self.encoder:
            for p in self.encoder.parameters(): p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        if self.encoder is None or input_ids is None:
            # return zeros [B, embed_dim]
            B = input_ids.shape[0] if input_ids is not None else 1
            device = input_ids.device if input_ids is not None else "cpu"
            return torch.zeros(B, self.embed_dim, device=device)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [B, hid]
        return self.proj(cls)  # [B, embed_dim]
