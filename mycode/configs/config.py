# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.BASIC = CN()
_C.BASIC.NODES = 1
_C.BASIC.GPUS = 4
_C.BASIC.NR = 0
_C.BASIC.WORLD_SIZE = _C.BASIC.NODES * _C.BASIC.GPUS
_C.BASIC.DATASET = 'CUB'
_C.BASIC.DATA_PATH = r'/data/dataset/Cross_CUB/CUB_200_2011'
_C.BASIC.CHECKPOINT_PATH = r'./checkpoints'
_C.BASIC.LOG_PATH = r'./log'
_C.BASIC.LOG_NAME = ''
# _C.BASIC.BACKUP_PATH = r'./backup3'
_C.BASIC.RESUME = False
_C.BASIC.MODEL = CN(new_allowed=True)
_C.BASIC.MODEL.PARAMS = CN(new_allowed=True)
_C.BASIC.NUM_CLASS = 200
_C.BASIC.VALID_EVERY_EPOCH = 1 # 每隔多少个epoch验证一次
_C.BASIC.CHECK_EVERY_BATCH = 10
_C.BASIC.NUM_WORKERS = 16

_C.TRAIN = CN()
_C.TRAIN.AUG = CN(new_allowed=True)
_C.TRAIN.BATCH_SIZE = 256

_C.TRAIN.CLOSESET = CN()
_C.TRAIN.CLOSESET.EPOCHS = 15
_C.TRAIN.CLOSESET.TEMP = 1.0
_C.TRAIN.CLOSESET.MIX = CN()
_C.TRAIN.CLOSESET.MIX.PROB = 0.0
_C.TRAIN.CLOSESET.MIX.ALPHA = 0.4
_C.TRAIN.CLOSESET.SCHEDULER = CN(new_allowed=True)
_C.TRAIN.CLOSESET.OPTIMIZER = CN(new_allowed=True)
_C.TRAIN.CLOSESET.LOSS = CN(new_allowed=True)
_C.TRAIN.CLOSESET.LOSS.W = 1.0

_C.TRAIN.OPENSET = CN()
_C.TRAIN.OPENSET.EPOCHS = 5
_C.TRAIN.OPENSET.C = 64
_C.TRAIN.OPENSET.TEMP = 1.0
_C.TRAIN.OPENSET.MIX = CN()
_C.TRAIN.OPENSET.MIX.PROB = 0.0
_C.TRAIN.OPENSET.MIX.ALPHA = 0.4
_C.TRAIN.OPENSET.SCHEDULER = CN(new_allowed=True)
_C.TRAIN.OPENSET.OPTIMIZER = CN(new_allowed=True)
_C.TRAIN.OPENSET.LOSS = CN(new_allowed=True)



_C.VALID = CN()
_C.VALID.AUG = CN(new_allowed=True)
_C.VALID.BATCH_SIZE = 256



def get_cfg_defaults():
    return _C


def update_from_args(cfg, args):
    key_value = []
    for key, value in vars(args).items():
        if key != 'config_path' and value is not None:  # 确保有值才更新
            key_value.append(key.upper())
            key_value.append(value)
    cfg.merge_from_list(key_value)
    return cfg
