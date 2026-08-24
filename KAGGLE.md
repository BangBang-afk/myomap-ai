# MyoMap AI — Kaggle GPU Runbook (Internet ON)

## One-time setup (Kaggle Notebook, Internet ON)
1. Kaggle → Create Notebook → Add Input: `rsna-knee-abnormality-detection` competition data
2. Upload `myomap-ai/` folder as Dataset (or `Add → Upload` then `!cp -r /kaggle/input/myomap-ai myomap-ai`)
3. Settings (right panel) → **Internet ON** (required for first pip + HF weights), Accelerator **GPU P100** or **T4 x2**
4. Run cells in `notebooks/kaggle_train.ipynb` sequentially:
   - Cell 0: env check prints `Internet ON ✓`
   - Cell 1: `pip install timm transformers` (~2 min with internet)
   - Cell 3: EDA — confirms 12 label columns + prevalence
   - Cell 4-5: train fold 0 writes `myomap-ai/models/best_fold0.pth` to `/kaggle/working`
5. Weights cached: `convnextv2_tiny.fcmae` (~100MB) + `xlm-roberta-base` (~1GB) downloaded once per session.

## Train
```bash
python myomap-ai/src/train.py --config myomap-ai/configs/config.yaml --fold 0
# 5-fold CV
for i in 0 1 2 3 4; do python myomap-ai/src/train.py --config myomap-ai/configs/config.yaml --fold $i; done
```

## Evaluate + Submit
```bash
python myomap-ai/src/evaluate.py --ckpt myomap-ai/models/best_fold0.pth --config myomap-ai/configs/config.yaml --fold 0
python myomap-ai/scripts/submission.py --ckpt myomap-ai/models/best_fold0.pth --config myomap-ai/configs/config.yaml --out /kaggle/working/submission.csv
head /kaggle/working/submission.csv  # StudyInstanceUID + 12 cols
```

## Tips for 30h/week limit
- Effective batch 8 (batch 4 + grad_accum 2 + amp) fits P100 16GB. Don't increase or OOM.
- Save checkpoint to Kaggle Dataset: `Notebook → Save → Save model dataset` to persist across sessions.
- Second run: skip Cell 1 if deps cached, keep Internet ON only if need re-download. Final scored submission can run Internet OFF after weights cached.

## Local parity
```bash
pip install -r myomap-ai/requirements.txt
kaggle competitions download -c rsna-knee-abnormality-detection -p myomap-ai/data/raw --unzip
python myomap-ai/src/train.py --config myomap-ai/configs/config.yaml --fold 0
```

## If Internet OFF required
Pre-cache weights as Dataset: run `scripts/kaggle_setup.sh` once with Internet ON, then `tar -czf /kaggle/working/cache.tar.gz /root/.cache/huggingface`, upload as dataset, mount read-only.

