from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine, RandomResizedCrop,
    RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, SmallestMaxSize, ElasticTransform
)
from torchvision import transforms
import torch
import cv2


def train_aug(cfg):
    aug = Compose([
        RandomResizedCrop(
            cfg.TRAIN.AUG.IMG_SIZE, cfg.TRAIN.AUG.IMG_SIZE,
            interpolation=cv2.INTER_CUBIC,
            scale=(0.5, 1)
        ),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        # ShiftScaleRotate(p=0.3),
        # PiecewiseAffine(p=0.5),
        ElasticTransform(p=0.5),
        HueSaturationValue(
            hue_shift_limit=4,
            sat_shift_limit=4,
            val_shift_limit=4,
            p=1.0
        ),
        RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(-0.2, 0.2),
            p=1.0
        ),
        OneOf(
            [OpticalDistortion(distort_limit=1.0),
             GridDistortion(num_steps=5, distort_limit=1.),
             # ElasticTransform(alpha=3),
             ],
            p=0.5
        ),
        Normalize(
            mean=cfg.TRAIN.AUG.MEAN,
            std=cfg.TRAIN.AUG.STD,
            max_pixel_value=255.0,
            p=1.0),
        ToTensorV2(p=1.0),
    ])
    return aug


def val_aug(cfg):
    aug = Compose([
        # SmallestMaxSize(CFG['img_size']),
        Resize(
            cfg.VALID.AUG.IMG_SIZE, cfg.VALID.AUG.IMG_SIZE,
            interpolation=cv2.INTER_CUBIC
        ),
        # CenterCrop(CFG['img_size'], CFG['img_size']),
        Normalize(
            mean=cfg.VALID.AUG.MEAN,
            std=cfg.VALID.AUG.STD,
            max_pixel_value=255.0,
            p=1.0),
        ToTensorV2(p=1.0),
    ])
    return aug


def test_aug(cfg):
    normalize = transforms.Normalize(mean=cfg.VALID.AUG.MEAN, std=cfg.VALID.AUG.STD)

    if cfg.TEST.AUG.TENCROP:
        aug = transforms.Compose([
            transforms.Resize(int(cfg.TEST.AUG.IMG_SIZE / 0.875)),
            transforms.TenCrop(cfg.TEST.AUG.IMG_SIZE),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
    else:
        aug = transforms.Compose([
            transforms.Resize(int(cfg.TEST.AUG.IMG_SIZE / 0.875)),
            transforms.CenterCrop(cfg.TEST.AUG.IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])

    return aug
