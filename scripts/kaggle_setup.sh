#!/bin/bash
# Kaggle setup — run with Internet ON (Kaggle Notebook Settings → Internet ON)
set -e
echo "MyoMap AI — Kaggle setup (Internet ON required)"
nvidia-smi | head -n 20 || echo "no nvidia-smi"
python --version
pip install -q --upgrade pip
pip install -q timm==1.0.9 transformers==4.44.2 albumentations==1.4.0 pydicom==2.4.4 pylibjpeg==1.4.0 pylibjpeg-libjpeg==1.3.2 accelerate==0.33.0 opencv-python-headless==4.10.0.84 2>&1 | tail -n 10
echo "deps installed ✓"
# pre-cache models (needs internet)
python -c "import timm; timm.create_model('convnextv2_tiny.fcmae', pretrained=True); print('convnext cached')"
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('xlm-roberta-base'); AutoModel.from_pretrained('xlm-roberta-base'); print('xlm-roberta cached')"
echo "weights cached to /root/.cache/huggingface"
echo "Ready: python myomap-ai/src/train.py --config myomap-ai/configs/config.yaml --fold 0"
