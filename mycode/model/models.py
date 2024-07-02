import sys

sys.path.append('/data/tbf/fungi')
import mytimm
import torch
from torch import nn


def convnext_large_dynamic_mlp(cfg):
    model = mytimm.create_model(
        'convnext_large_dynamic_mlp',
        pretrained=False,
        num_classes=10000,
        **cfg.BASIC.MODEL.PARAMS
    )
    model.load_state_dict(
        torch.load(cfg.BASIC.MODEL.PRETRAIN_PATH),
        strict=False
    )
    model.head.fc = nn.Linear(model.head.fc.in_features, cfg.BASIC.MODEL.NUM_CLASS)
    model.head.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.head.fc.bias.data.zero_()

    return model


def convnext_large_mlp(cfg):
    model = mytimm.create_model(
        'convnext_large_mlp',
        pretrained=False,
        num_classes=10000,
        **cfg.BASIC.MODEL.PARAMS
    )
    model.load_state_dict(
        torch.load(cfg.BASIC.MODEL.PRETRAIN_PATH),
        strict=True
    )
    model.head.fc = nn.Linear(model.head.fc.in_features, cfg.BASIC.MODEL.NUM_CLASS)
    model.head.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.head.fc.bias.data.zero_()

    return model


def volo_d4_448(cfg):
    model = mytimm.create_model(
        'volo_d4_448',
        pretrained=False,
        num_classes=1604
    )

    checkpoint = torch.load(cfg.BASIC.MODEL.PRETRAIN_PATH)
    for key in list(checkpoint):
        if key[:6] == 'module':
            checkpoint[key[7:]] = checkpoint[key]
            del checkpoint[key]

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    return model


def volo_d4_dynamic_mlp_448(cfg):
    model = mytimm.create_model(
        'volo_d4_dynamic_mlp_448',
        pretrained=False,
        num_classes=1604
    )

    checkpoint = torch.load(cfg.BASIC.MODEL.PRETRAIN_PATH)
    for key in list(checkpoint):
        if key[:6] == 'module':
            checkpoint[key[7:]] = checkpoint[key]
            del checkpoint[key]

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    return model

def vit_large_dynamic_mlp_patch14_clip_336(cfg):
    model = mytimm.create_model(
        'vit_large_dynamic_mlp_patch14_clip_336',
        pretrained=False,
        num_classes=10000,
        **cfg.BASIC.MODEL.PARAMS
    )

    model.load_state_dict(
        torch.load(cfg.BASIC.MODEL.PRETRAIN_PATH),
        strict=False
    )

    model.head = nn.Linear(model.head.in_features, cfg.BASIC.MODEL.NUM_CLASS)
    model.head.weight.data.normal_(mean=0.0, std=0.01)
    model.head.bias.data.zero_()

    return model


def eva02_large_dynamic_mlp_patch14_clip_336(cfg):
    model = mytimm.create_model(
        'eva02_large_dynamic_mlp_patch14_clip_336',
        pretrained=False,
        num_classes=10000,
        **cfg.BASIC.MODEL.PARAMS
    )

    model.load_state_dict(
        torch.load(cfg.BASIC.MODEL.PRETRAIN_PATH),
        strict=False
    )

    model.head = nn.Linear(model.head.in_features, cfg.BASIC.MODEL.NUM_CLASS)
    model.head.weight.data.normal_(mean=0.0, std=0.01)
    model.head.bias.data.zero_()

    return model


def vit_large_patch14_clip_336(cfg):
    model = mytimm.create_model(
        'vit_large_patch14_clip_336',
        pretrained=False,
        num_classes=10000,
        **cfg.BASIC.MODEL.PARAMS
    )

    model.load_state_dict(
        torch.load(cfg.BASIC.MODEL.PRETRAIN_PATH),
        strict=True
    )

    model.head = nn.Linear(model.head.in_features, cfg.BASIC.MODEL.NUM_CLASS)
    model.head.weight.data.normal_(mean=0.0, std=0.01)
    model.head.bias.data.zero_()

    return model


def eva02_large_patch14_clip_336(cfg):
    model = mytimm.create_model(
        'eva02_large_patch14_clip_336',
        pretrained=False,
        num_classes=10000,
        **cfg.BASIC.MODEL.PARAMS
    )

    model.load_state_dict(
        torch.load(cfg.BASIC.MODEL.PRETRAIN_PATH),
        strict=True
    )

    model.head = nn.Linear(model.head.in_features, cfg.BASIC.MODEL.NUM_CLASS)
    model.head.weight.data.normal_(mean=0.0, std=0.01)
    model.head.bias.data.zero_()

    return model
