import time
import torch
import torch.distributed as dist
from mycode.utils import AverageMeter, ProgressMeter
from utils import accuracy, cal_track2, cal_track1


def validate(ddp_model, val_loader, gpu, logger, cfg):
    ddp_model.eval()
    data_time = AverageMeter("Data", ":6.3f")
    batch_time = AverageMeter("Time", ":6.3f")
    track1_meter = AverageMeter("Track1", ":6.2f")
    track2_meter = AverageMeter("Track2", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [data_time, batch_time, track1_meter, track2_meter],
        prefix="Validate:",
        logger=logger
    )

    with torch.no_grad():
        end = time.time()
        for batch, (images, labels) in enumerate(val_loader):

            data_time.update(time.time() - end)

            if type(images) == list:
                assert len(images) == 2
                imgs = images[0].cuda(gpu, non_blocking=True)
                meta_feat = images[1].cuda(gpu, non_blocking=True)
            else:
                imgs = images.cuda(gpu, non_blocking=True)
                meta_feat = None

            labels = labels.cuda(gpu, non_blocking=True)

            # 前向传播，这里的loss计算的就是当前rank进程内数据的loss，不是总loss
            if meta_feat != None:
                logits = ddp_model(imgs, meta_feat)
            else:
                logits = ddp_model(imgs)

            track1 = cal_track1(logits, labels)
            track2 = cal_track2(logits, labels)

            dist.barrier()

            gathered_info = torch.tensor([track1, track2]).cuda(gpu)
            dist.all_reduce(gathered_info, op=torch.distributed.ReduceOp.AVG)
            gathered_info = gathered_info.tolist()
            track1_meter.update(gathered_info[0])
            track2_meter.update(gathered_info[1])
            batch_time.update(time.time() - end)
            end = time.time()

            if dist.get_rank() == 0:
                if (batch + 1) % cfg.BASIC.CHECK_EVERY_BATCH == 0 or (batch + 1) % len(val_loader) == 0:
                    progress.display((batch + 1))

    return track1_meter.avg + track2_meter.avg
