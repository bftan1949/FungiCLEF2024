import time
from torch.cuda.amp import autocast as autocast
import torch
import torch.distributed as dist
from mycode.utils import AverageMeter, ProgressMeter
import numpy as np
from utils import mixup_diff_class
from utils import accuracy


def train_openset_one_epoch(ddp_model, optimizer, train_loader, cls_loss_fn, epoch, gpu, scaler, logger, cfg):
    ddp_model.train()
    data_time = AverageMeter("Data", ":6.3f")
    batch_time = AverageMeter("Time", ":6.3f")
    cls_loss_meter = AverageMeter("Loss", ":.4e")
    acc1_meter = AverageMeter("Acc1", ":6.2f")

    meters = [data_time, batch_time, cls_loss_meter, acc1_meter]

    progress = ProgressMeter(
        len(train_loader),
        meters,
        prefix="Openset-Epoch: [{}]".format(epoch),
        logger=logger
    )

    end = time.time()
    for batch, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if type(images) == list:
            assert len(images) == 2
            imgs = images[0].cuda(gpu, non_blocking=True)
            mf = images[1].cuda(gpu, non_blocking=True)
        else:
            imgs = images.cuda(gpu, non_blocking=True)
            mf = None

        labels = labels.cuda(gpu, non_blocking=True).to(torch.long)

        # if len(cfg.TRAIN.OPENSET.MIX) != 0 and np.random.rand(1) < cfg.TRAIN.OPENSET.MIX.PROB:
        if mf is not None:
            mixed_imgs, mixed_labels, mixed_mf = mixup_diff_class(imgs, labels, mf=mf,
                                                                  alpha=cfg.TRAIN.OPENSET.MIX.ALPHA)
            with autocast():
                mixed_logits = ddp_model(mixed_imgs, mixed_mf)
                loss1 = cls_loss_fn(mixed_logits / cfg.TRAIN.OPENSET.TEMP, mixed_labels)
                origin_logits, origin_labels = ddp_model(imgs, mf), labels
                loss2 = cls_loss_fn(origin_logits / cfg.TRAIN.OPENSET.TEMP, origin_labels)
                loss = loss1 + loss2
                logits = torch.concatenate([mixed_logits, origin_logits], dim=0)
                labels = torch.concatenate([mixed_labels, origin_labels], dim=0)

        else:
            mixed_imgs, mixed_labels = mixup_diff_class(imgs, labels, alpha=cfg.TRAIN.OPENSET.MIX.ALPHA)
            with autocast():
                mixed_logits = ddp_model(mixed_imgs)
                loss1 = cls_loss_fn(mixed_logits / cfg.TRAIN.OPENSET.TEMP, mixed_labels)
                origin_logits, origin_labels = ddp_model(imgs), labels
                loss2 = cls_loss_fn(origin_logits / cfg.TRAIN.OPENSET.TEMP, origin_labels)
                loss = loss1 + loss2
                logits = torch.concatenate([mixed_logits, origin_logits], dim=0)
                labels = torch.concatenate([mixed_labels, origin_labels], dim=0)

        # else:
        #     if meta_feat is not None:
        #         with autocast():
        #             logits = ddp_model(imgs, meta_feat)
        #             loss = cls_loss_fn(logits / cfg.TRAIN.OPENSET.TEMP, labels)
        #     else:
        #         with autocast():
        #             logits = ddp_model(imgs)
        #             loss = cls_loss_fn(logits / cfg.TRAIN.OPENSET.TEMP, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        (acc1,) = accuracy(logits, labels, topk=(1,))

        dist.barrier()

        gathered_info = torch.tensor([loss, acc1]).cuda(gpu)

        dist.all_reduce(gathered_info, op=torch.distributed.ReduceOp.AVG)
        gathered_info = gathered_info.tolist()
        cls_loss_meter.update(gathered_info[0])
        acc1_meter.update(gathered_info[1])

        batch_time.update(time.time() - end)
        end = time.time()

        if dist.get_rank() == 0:
            if (batch + 1) % cfg.BASIC.CHECK_EVERY_BATCH == 0 or (batch + 1) % len(train_loader) == 0:
                progress.display((batch + 1))

        time.sleep(0.003)
