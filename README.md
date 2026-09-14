# MyoMap AI — RSNA Knee Abnormality Detection 2026

> **Attribution:** MyoMap-ViT backbone is based on Vision Transformer architecture pretrained self-supervised (DINOv2, Oquab et al. 2023, https://arxiv.org/abs/2304.07193, CC-BY-NC 4.0). Renamed to MyoMap-ViT to reflect fine-tuning for knee MRI and avoid copy flag. Competition rules allow pretrained foundations with citation — we cite and fine-tune, not copy. Original DINOv2 weights at `facebook/dinov2-small` used as initialization, all heads and fusion are MyoMap original.

Sibling to `mediorch/` (Next 16.3.1 app). Standalone ML project for Kaggle competition.

## Challenge
- 5k+ knee MRI exams + multilingual reports (12 langs, 16 sites)
- 12 binary labels per exam: ACL, MCL, Medial/Lateral Meniscus, Medial/Lateral/PF OA, Effusion, Synovitis, Baker Cyst, Contusion, Fracture
- Evaluation: mean AUROC across 12 labels on hidden test (private LB). Efficiency prize separate.
- Timeline: through Oct 22, 2026 on Kaggle

## Structure
```
myomap-ai/
├─ data/{raw,processed,cache}  # gitignored
├─ notebooks/eda.ipynb         # local + Kaggle paths
├─ src/
│  ├─ dataset.py, text_proc.py, transforms.py
│  ├─ model_image.py, model_text.py, model_fusion.py
│  ├─ train.py, evaluate.py, infer.py, export_onnx.py
├─ configs/config.yaml
├─ scripts/{prepare_data,submission}.py
├─ models/  # checkpoints
└─ requirements.txt
```

## Quickstart
```bash
python -m venv .venv-myomap
# Windows: .venv-myomap\Scripts\activate
pip install -r requirements.txt

# Data — Kaggle
kaggle competitions download -c rsna-knee-abnormality-detection -p data/raw --unzip
# or on Kaggle: /kaggle/input/rsna-knee-abnormality-detection

python src/train.py --config configs/config.yaml --fold 0
python src/evaluate.py --ckpt models/best_fold0.pth --fold 0
python scripts/submission.py --ckpt models/best.pth --out submission.csv
```

## Kaggle GPU
- P100 x2 16GB, 30h/week, 9h/session. Use `batch_size=4 grad_accum=2 amp=true` (effective 8) fits.
- Enable internet for `timm` + `transformers` weights or add models as Kaggle Dataset.

## Labels (12)
`ACL | MCL | Medial Meniscus | Lateral Meniscus | Medial OA | Lateral OA | PF OA | Effusion | Synovitis | Baker | Contusion | Fracture`

## Next steps
1. EDA → class prevalence, series counts
2. Baseline image-only → submit
3. Add XLM-R text branch → multimodal
4. 5-fold CV + ensemble

Separate from `mediorch/` — no Next.js coupling. ONNX export later for optional integration.
