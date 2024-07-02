import sys

sys.path.append('/data/tbf/fungi')
from warmup import WarmUpScheduler
import argparse
import os
import random
from pathlib import Path
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import SyncBatchNorm
from torch.optim import lr_scheduler
from mycode.utils import get_logger
from torch import optim
from mycode import augments, dataset
from mycode.model import models
from configs.config import get_cfg_defaults, update_from_args
from train_closeset import train_one_epoch
from val import validate
import numpy as np
import loss
from torch import nn
import torch
from mycode.model.proser import get_openset_head
# from loss import ProserLoss
from train_openset import train_openset_one_epoch


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# gpu和args的参数位置都不可以变, 但是可以不要args, 只要gpu参数也可以运行
def run(gpu, cfg):
    # nr是当前是第几台机器，这里的rank是计算当前gpu是第几个进程
    rank = cfg.BASIC.NR * cfg.BASIC.GPUS + gpu
    # 对进程池进行初始化
    dist.init_process_group(
        backend='nccl',  # 后端的协议，nccl是基于gpu最好的协议, 但是Windows不支持, 要换成gloo
        init_method='env://',
        world_size=cfg.BASIC.WORLD_SIZE,
        rank=rank  # 这里给进程池中的每个进行命名，不能重名，所以用每个进程的rank
    )

    log_path = os.path.join(cfg.BASIC.LOG_PATH, f'{cfg.BASIC.LOG_NAME}.log')
    logger = get_logger(log_path)

    if dist.get_rank() == 0:
        logger.info(cfg)

    model = getattr(models, cfg.BASIC.MODEL.NAME)(cfg).cuda(gpu)

    batch_size = cfg.TRAIN.BATCH_SIZE // dist.get_world_size()
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = SyncBatchNorm.convert_sync_batchnorm(model)

    # 把模型放到ddp上
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if 'convnext' in cfg.BASIC.MODEL.NAME:
        original_head = ddp_model.module.head.fc
        dummy_head = get_openset_head(original_head, cfg.TRAIN.OPENSET.C, gpu)
        ddp_model.module.head.fc = dummy_head

    elif 'volo' in cfg.BASIC.MODEL.NAME:
        original_head = ddp_model.module.head
        original_aux_head = ddp_model.module.aux_head
        dummy_head = get_openset_head(original_head, cfg.TRAIN.OPENSET.C, gpu)
        dummy_aux_head = get_openset_head(original_aux_head, cfg.TRAIN.OPENSET.C, gpu)
        ddp_model.module.head = dummy_head
        ddp_model.module.aux_head = dummy_aux_head
    else:
        original_head = ddp_model.module.head
        dummy_head = get_openset_head(original_head, cfg.TRAIN.OPENSET.C, gpu)
        ddp_model.module.head = dummy_head

    open_optimizer = getattr(optim, cfg.TRAIN.OPENSET.OPTIMIZER.NAME)(
        ddp_model.parameters(), **cfg.TRAIN.OPENSET.OPTIMIZER.PARAMS
    )

    scaler = GradScaler()

    # Data loading code
    train_dataset = getattr(dataset, cfg.BASIC.DATASET)(
        data_path=cfg.BASIC.DATA_PATH,
        split='train',
        transform=augments.train_aug(cfg)
    )

    val_dataset = getattr(dataset, cfg.BASIC.DATASET)(
        data_path=cfg.BASIC.DATA_PATH,
        split='val',
        transform=augments.val_aug(cfg)
    )

    # sampler把数据分到不同的rank上
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=cfg.BASIC.WORLD_SIZE,
        rank=rank
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=cfg.BASIC.WORLD_SIZE,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        # sampler选项与shuffle互斥，使用了sampler就要把shuffle设置为false
        shuffle=False,
        num_workers=cfg.BASIC.NUM_WORKERS // dist.get_world_size(),
        pin_memory=True,
        drop_last=False,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        # sampler选项与shuffle互斥，使用了sampler就要把shuffle设置为false
        shuffle=False,
        num_workers=cfg.BASIC.NUM_WORKERS // dist.get_world_size(),
        pin_memory=True,
        sampler=val_sampler
    )

    checkpoint_folder = Path(f'{cfg.BASIC.CHECKPOINT_PATH}/{cfg.BASIC.LOG_NAME[:-4]}')

    checkpoint_folder.mkdir(exist_ok=True, parents=True)

    if len(cfg.TRAIN.OPENSET.SCHEDULER) > 0:
        open_scheduler = getattr(lr_scheduler, cfg.TRAIN.OPENSET.SCHEDULER.NAME)(
            optimizer=open_optimizer, **cfg.TRAIN.OPENSET.SCHEDULER.PARAMS
        )
    else:
        open_scheduler = None

    openset_loss_fn = getattr(loss, cfg.TRAIN.OPENSET.LOSS.NAME)().cuda(gpu)

    start_epoch = 0

    ddp_model.train()

    logger.info(f'gpu {dist.get_rank()} begins to train')

    best_track3 = 100000

    #################### train openset #########################

    for epoch in range(start_epoch, cfg.TRAIN.OPENSET.EPOCHS):

        train_loader.sampler.set_epoch(epoch)

        train_openset_one_epoch(ddp_model, open_optimizer, train_loader,
                                openset_loss_fn, epoch, gpu, scaler, logger, cfg)

        if open_scheduler is not None:
            open_scheduler.step()

        if (epoch + 1) % cfg.BASIC.VALID_EVERY_EPOCH == 0:
            track3 = validate(ddp_model, val_loader, gpu, logger, cfg)
            if track3 < best_track3 and dist.get_rank() == 0:
                best_track3 = track3
                logger.info(f'the best track3 on validation set is {track3:.4f}')
                checkpoint = {
                    'config': cfg,
                    'model_state_dict': ddp_model.module.state_dict(),
                }
                checkpoint_path = checkpoint_folder / f'openset_checkpoint.best.pt'
                torch.save(checkpoint, str(checkpoint_path))
    ############################################################


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--config-path', type=str,
                        default='/data/tbf/fungi/mycode/configs/volo-openset.yaml')
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'  # 设置master节点的地址
    os.environ['MASTER_PORT'] = '11133'  # 设置端口号，从而让所有节点能够相互通信

    same_seeds(100)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    cfg = update_from_args(cfg, args)

    mp.spawn(run, nprocs=cfg.BASIC.GPUS, args=(cfg,))


if __name__ == '__main__':
    main()
