"""Albumentations transforms for 2.5D knee MRI clips."""
try:
    import albumentations as A
    HAS_ALB = True
except ImportError:
    HAS_ALB = False

def get_train_transforms(image_size: int = 256):
    if not HAS_ALB:
        return None
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CoarseDropout(p=0.2),  # 1.4 API: max_holes/max_height deprecated, use defaults
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def get_val_transforms(image_size: int = 256):
    if not HAS_ALB:
        return None
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
