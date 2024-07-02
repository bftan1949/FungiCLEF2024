import time
from torch.cuda.amp import autocast as autocast
import torch
import torch.distributed as dist
from mycode.utils import AverageMeter, ProgressMeter
import numpy as np
from utils import mixup
import random
from utils import accuracy


# def rand_bbox(size, lam):
#     W = size[2]
#     H = size[3]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = np.int32(W * cut_rat)
#     cut_h = np.int32(H * cut_rat)
#
#     # uniform
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)
#
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
#
#     return bbx1, bby1, bbx2, bby2
#
#
# def generate_mask_random(imgs, patch, mask_token_num_start, lam):
#     _, _, W, H = imgs.shape
#     assert W % patch == 0
#     assert H % patch == 0
#     p = W // patch
#
#     mask_ratio = 1 - lam
#     num_masking_patches = min(p ** 2, int(mask_ratio * (p ** 2)) + mask_token_num_start)
#     mask_idx = np.random.permutation(p ** 2)[:num_masking_patches]
#     lam = 1 - num_masking_patches / (p ** 2)
#     return mask_idx, lam
#
#
# def get_mixed_data(imgs, image_labels, cfg, gpu):
#     mix_lst = ['cutmix', 'mixup', 'tokenmix', 'randommix']
#
#     mix_type = cfg.TRAIN.MIX.NAME
#     patch = cfg.TRAIN.MIX.PARAMS.patch
#     mask_token_num_start = cfg.TRAIN.MIX.PARAMS.mask_token_num_start
#     lam = cfg.TRAIN.MIX.PARAMS.lam
#
#     assert mix_type in mix_lst, f'Not Supported mix type: {mix_type}'
#     if mix_type == 'randommix':
#         # select a mix_type randomly
#         mix_type = random.choice(mix_lst[:-1])
#     if mix_type == 'mixup':
#         alpha = 2.0
#         rand_index = torch.randperm(imgs.size()[0]).cuda(gpu)
#         target_a = image_labels
#         target_b = image_labels[rand_index]
#         lam = np.random.beta(alpha, alpha)
#         imgs = imgs * lam + imgs[rand_index] * (1 - lam)
#     elif mix_type == 'cutmix':
#         beta = 1.0
#         lam = np.random.beta(beta, beta)
#         rand_index = torch.randperm(imgs.size()[0]).cuda(gpu)
#         target_a = image_labels
#         target_b = image_labels[rand_index]
#         bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
#         imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
#         # adjust lambda to exactly match pixel ratio
#         lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
#     elif mix_type == 'tokenmix':
#         B, C, W, H = imgs.shape
#         mask_idx, lam = generate_mask_random(imgs, patch, mask_token_num_start, lam)
#         rand_index = torch.randperm(imgs.size()[0]).cuda()
#         p = W // patch
#         patch_w = patch
#         patch_h = patch
#         for idx in mask_idx:
#             row_s = idx // p
#             col_s = idx % p
#             x1 = patch_w * row_s
#             x2 = x1 + patch_w
#             y1 = patch_h * col_s
#             y2 = y1 + patch_h
#             imgs[:, :, x1:x2, y1:y2] = imgs[rand_index, :, x1:x2, y1:y2]
#
#         target_a = image_labels
#         target_b = image_labels[rand_index]
#
#     return imgs, target_a, target_b, lam


def train_one_epoch(ddp_model, optimizer, train_loader, cls_loss_fn, cost_loss_fn, epoch, gpu, scaler, logger, cfg):
    ddp_model.train()
    data_time = AverageMeter("Data", ":6.3f")
    batch_time = AverageMeter("Time", ":6.3f")
    cls_loss_meter = AverageMeter("Loss", ":.4e")
    acc1_meter = AverageMeter("Acc1", ":6.2f")

    if cost_loss_fn is not None:
        cost_loss_meter = AverageMeter("Cost", ":.4e")
        meters = [data_time, batch_time, cls_loss_meter, cost_loss_meter, acc1_meter]
    else:
        meters = [data_time, batch_time, cls_loss_meter, acc1_meter]

    progress = ProgressMeter(
        len(train_loader),
        meters,
        prefix="Closeset-Epoch: [{}]".format(epoch),
        logger=logger
    )

    end = time.time()
    for batch, (images, labels) in enumerate(train_loader):

        # with torch.autograd.set_detect_anomaly(True):

        data_time.update(time.time() - end)

        if type(images) == list:
            assert len(images) == 2
            imgs = images[0].cuda(gpu, non_blocking=True)
            meta_feat = images[1].cuda(gpu, non_blocking=True)
        else:
            imgs = images.cuda(gpu, non_blocking=True)
            meta_feat = None

        labels = labels.cuda(gpu, non_blocking=True)

        if len(cfg.TRAIN.CLOSESET.MIX) != 0 and np.random.rand(1) < cfg.TRAIN.CLOSESET.MIX.PROB:
            imgs, label_a, label_b, lam, index = mixup(imgs, labels, cfg.TRAIN.CLOSESET.MIX.ALPHA)
            if meta_feat != None:
                meta_feat = lam * meta_feat + (1 - lam) * meta_feat[index]
            with autocast():
                if meta_feat != None:
                    logits = ddp_model(imgs, meta_feat)
                else:
                    logits = ddp_model(imgs)

                cls_loss = (cls_loss_fn(logits / cfg.TRAIN.CLOSESET.TEMP, label_a) * lam
                            + cls_loss_fn(logits / cfg.TRAIN.CLOSESET.TEMP, label_b) * (1. - lam))
                if cost_loss_fn is not None:
                    cost_loss = (cost_loss_fn(logits / cfg.TRAIN.CLOSESET.TEMP, label_a) * lam
                                 + cost_loss_fn(logits / cfg.TRAIN.CLOSESET.TEMP, label_b) * (1. - lam))
                    loss = cls_loss + cfg.TRAIN.CLOSESET.LOSS.W * cost_loss
                else:
                    loss = cls_loss
        else:
            with autocast():
                if meta_feat != None:
                    logits = ddp_model(imgs, meta_feat)
                else:
                    logits = ddp_model(imgs)

                cls_loss = cls_loss_fn(logits / cfg.TRAIN.CLOSESET.TEMP, labels)
                if cost_loss_fn is not None:
                    cost_loss = cost_loss_fn(logits / cfg.TRAIN.CLOSESET.TEMP, labels)
                    loss = cls_loss + cfg.TRAIN.CLOSESET.LOSS.W * cost_loss
                else:
                    loss = cls_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        (acc1,) = accuracy(logits, labels, topk=(1,))

        dist.barrier()

        if cost_loss_fn is not None:
            gathered_info = torch.tensor([cls_loss, acc1, cost_loss]).cuda(gpu)
        else:
            gathered_info = torch.tensor([cls_loss, acc1]).cuda(gpu)

        dist.all_reduce(gathered_info, op=torch.distributed.ReduceOp.AVG)
        gathered_info = gathered_info.tolist()
        cls_loss_meter.update(gathered_info[0])
        acc1_meter.update(gathered_info[1])

        if cost_loss_fn is not None:
            cost_loss_meter.update(gathered_info[2])

        batch_time.update(time.time() - end)
        end = time.time()

        if dist.get_rank() == 0:
            if (batch + 1) % cfg.BASIC.CHECK_EVERY_BATCH == 0 or (batch + 1) % len(train_loader) == 0:
                progress.display((batch + 1))

        time.sleep(0.003)
