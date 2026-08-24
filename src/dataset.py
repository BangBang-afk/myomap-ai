"""RSNA Knee — dataset: DICOM series → 2.5D clips + multilingual reports.

Kaggle path: /kaggle/input/rsna-knee-abnormality-detection/
Local path: data/raw/
Handles missing data gracefully (mock tensors) so scaffold builds without download.
"""
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pydicom
    HAS_DICOM = True
except ImportError:
    HAS_DICOM = False

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # type: ignore

LABEL_NAMES = ["ACL","MCL","MedialMeniscus","LateralMeniscus","MedialOA","LateralOA","PatellofemoralOA","Effusion","Synovitis","BakerCyst","Contusion","Fracture"]

KAGGLE_INPUT = "/kaggle/input/rsna-knee-abnormality-detection"

def resolve_data_root(config_path: str = None) -> Path:
    if os.path.exists(KAGGLE_INPUT):
        return Path(KAGGLE_INPUT)
    # local sibling to mediorch
    here = Path(__file__).resolve().parents[1]
    return here / "data" / "raw"

def load_meta(root: Path) -> pd.DataFrame:
    # try common Kaggle file names
    for name in ["train.csv","train_folds.csv","train_folds_with_pseudo.csv","metadata.csv"]:
        p = root / name
        if p.exists():
            return pd.read_csv(p)
        # also search recursively
        found = list(root.rglob(name))
        if found:
            return pd.read_csv(found[0])
    # fallback empty with expected cols
    print(f"[dataset] no train.csv found under {root}, return empty meta (mock mode)")
    return pd.DataFrame(columns=["StudyInstanceUID"] + LABEL_NAMES + ["report_text"])

def dicom_to_array(dcm_path: str) -> np.ndarray:
    if not HAS_DICOM:
        return np.zeros((256,256), dtype=np.float32)
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        # window/limit normalize: simple rescale to [0,1]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        arr = (arr * 255).clip(0,255).astype(np.uint8)
        return arr
    except Exception:
        return np.zeros((256,256), dtype=np.uint8)

def build_25d_clip(slices: list, idx: int) -> np.ndarray:
    """Stack prev, center, next slice into 3 channels."""
    n = len(slices)
    prev = slices[max(0, idx-1)]
    ctr = slices[idx]
    nxt = slices[min(n-1, idx+1)]
    clip = np.stack([prev, ctr, nxt], axis=-1)  # HWC 3ch
    return clip

class KneeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform=None, tokenizer=None, max_len=256, is_train=True, image_size=256):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def _load_study_clips(self, study_uid: str):
        # Locate study folder — try multiple layouts
        patterns = [str(self.root / "**" / study_uid), str(self.root / study_uid), str(self.root / "train_images" / study_uid)]
        study_dir = None
        for pat in patterns:
            hits = glob.glob(pat, recursive=True)
            if hits:
                study_dir = Path(hits[0]); break
        if study_dir is None or not study_dir.exists():
            # mock: return 4 random clips
            return [np.random.randint(0,256,(self.image_size,self.image_size,3), dtype=np.uint8) for _ in range(4)], ["Axial"]*4

        # collect all dicoms under study
        dcm_files = sorted(study_dir.rglob("*.dcm"))
        if not dcm_files:
            return [np.random.randint(0,256,(self.image_size,self.image_size,3), dtype=np.uint8) for _ in range(4)], ["Sagittal"]*4

        # group by series (parent folder)
        from collections import defaultdict
        series = defaultdict(list)
        for f in dcm_files:
            series[str(f.parent)].append(str(f))
        clips = []
        planes = []
        for s_dir, files in series.items():
            files = sorted(files)
            slices = [dicom_to_array(f) for f in files[:32]]  # cap for memory
            # need resize slices to image_size
            try:
                import cv2
                slices = [cv2.resize(s, (self.image_size, self.image_size)) for s in slices]
            except Exception:
                import PIL.Image
                slices = [np.array(PIL.Image.fromarray(s).resize((self.image_size,self.image_size))) for s in slices]
            # sample center clips
            idxs = np.linspace(0, len(slices)-1, min(4, len(slices))).astype(int) if len(slices)>1 else [0]
            for idx in idxs:
                clips.append(build_25d_clip(slices, int(idx)))
            # infer plane from dirname
            name = s_dir.lower()
            if "sag" in name: planes.extend(["Sagittal"]*len(idxs))
            elif "cor" in name: planes.extend(["Coronal"]*len(idxs))
            elif "ax" in name: planes.extend(["Axial"]*len(idxs))
            else: planes.extend(["Sagittal"]*len(idxs))
        if not clips:
            clips = [np.random.randint(0,256,(self.image_size,self.image_size,3), dtype=np.uint8)]
            planes = ["Sagittal"]
        return clips, planes

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_uid = str(row.get("StudyInstanceUID", f"mock_{idx}"))
        clips, planes = self._load_study_clips(study_uid)

        # aggregate clips: pad/truncate to fixed 8, then mean-pool later in model
        import torch
        # apply transform per clip
        proc = []
        for c in clips[:8]:
            if self.transform:
                try:
                    c = self.transform(image=c)["image"]
                except Exception:
                    pass
            # HWC -> CHW, float
            if c.ndim == 3:
                c = np.transpose(c, (2,0,1))
            c = torch.from_numpy(c).float() / 255.0
            proc.append(c)
        while len(proc) < 8:
            proc.append(torch.zeros((3, self.image_size, self.image_size)))
        clips_tensor = torch.stack(proc)  # [8,3,H,W]

        # labels 12
        labels = []
        for n in LABEL_NAMES:
            v = row.get(n, np.nan)
            try:
                v = float(v)
                if np.isnan(v): v = -1  # missing
            except: v = -1
            labels.append(v)
        labels = torch.tensor(labels, dtype=torch.float32)

        # report text
        report = str(row.get("report_text", row.get("report", "")) or "")
        # tokenization deferred to text_proc if tokenizer provided
        item = {"study_uid": study_uid, "clips": clips_tensor, "planes": planes[:8], "labels": labels, "report": report}
        if self.tokenizer is not None:
            from .text_proc import tokenize_report
            tok = tokenize_report(report, self.tokenizer, self.max_len)
            # keep tensors
            for k in ["input_ids","attention_mask"]:
                if k in tok and hasattr(tok[k], 'shape'):
                    item[k] = tok[k]
        return item
