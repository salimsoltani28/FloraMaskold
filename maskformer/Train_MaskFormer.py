import torch
import argparse
import torch.nn as nn
from tqdm.auto import tqdm
from collections import defaultdict
from utils import (get_config, get_logger, build_optimizer, build_scheduler, auto_resume_helper, save_checkpoint, get_grad_norm, parse_losses, reduce_tensor)
from dataset import build_loader
from models import build_model
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from torch.backends import cudnn
import os
import os.path as osp
from omegaconf import OmegaConf, read_write
from timm.utils import AverageMeter
import time
import datetime


def parse_args():
    parser = argparse.ArgumentParser('FloraMask training and evaluation script')
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')

    # To overwrite config file
    parser.add_argument('--batch-size', type=int, help='batch size for single GPU')
    parser.add_argument('--resume', help='resume from checkpoint')

    parser.add_argument(
        '--output', default='/mnt/gsdata/projects/panops/floramask/outputs/testrun1',type=str, help='root of output folder, '
        'the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', type=str, help='tag of experiment')
    parser.add_argument('--wandb', action='store_true', help='Use W&B to log experiments')
    parser.add_argument('--keep', type=int, help='Maximum checkpoint to keep')
    args = parser.parse_args()

    return args


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, device='cuda'):
    """
    Trains the model for one epoch.

    Args:
        config (dict): Configuration parameters.
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): The data loader for training data.
        optimizer (Optimizer): The optimizer for updating model parameters.
        epoch (int): The current epoch number.
        lr_scheduler (LRScheduler): The learning rate scheduler.
        
    Returns:
        dict: A dictionary containing the average loss and other metrics for the epoch.
    """
    logger = get_logger()
    dist.barrier()
    model.train()
    optimizer.zero_grad()
    if config.wandb and dist.get_rank() == 0:
        import wandb
    else:
        wandb = None

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    log_vars_meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()

    for idx, samples in enumerate(data_loader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        batch_size = config.data.batch_size
        #print device for model
        logger.debug(f'model device:{next(model.parameters()).device}')
        losses = model(samples, device)

        loss, log_vars = parse_losses(losses)
        loss.backward()
        
        if config.train.clip_grad > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
        else:
            grad_norm = get_grad_norm(model.parameters())
        
        optimizer.step()
        lr_scheduler.step_update(epoch*num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), batch_size)
        for loss_name in log_vars:
            log_vars_meters[loss_name].update(log_vars[loss_name], batch_size)
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0 * 1024.0) #Memory in GB
            etas = batch_time.avg * (num_steps - idx)
            log_vars_str = '\t'.join(f'{n} {m.val:.4f} ({m.avg:.4f})' for n, m in log_vars_meters.items())
            logger.info(f'Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}]\t'
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'{log_vars_str}\t'
                        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')
            if wandb is not None:
                log_stat = {f'iter/train_{n}': m.avg for n, m in log_vars_meters.items()}
                log_stat['iter/train_total_loss'] = loss_meter.avg
                log_stat['iter/learning_rate'] = lr
                wandb.log(log_stat)

    epoch_time = time.time() - start
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')
    result_dict = dict(total_loss=loss_meter.avg)
    for n, m in log_vars_meters.items():
        result_dict[n] = m.avg
    dist.barrier()
    return result_dict

def train(cfg):
    # Build data loaders
    if cfg.wandb and dist.get_rank() == 0:
        import wandb
        wandb.init(
            project='floramask',
            name=osp.join(cfg.model_name, cfg.tag),
            dir=cfg.output,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=cfg.checkpoint.auto_resume)
    else:
        wandb = None
      
    dist.barrier()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger = get_logger()  
    
    logger.info("Loading data loaders")
    train_subset, val_subset, train_loader, val_loader = build_loader(config)

    
    # Define the model
    model = build_model(config['model'])
    model.to(device)


    optimizer = build_optimizer(cfg.train, model)
    
    if cfg.wandb and dist.get_rank() == 0:
        wandb.watch(model, log="all")

    # Check the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters: {total_params / 1e6:.2f} million")

    model = MMDistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
    model_without_ddp = model.module

    lr_scheduler = build_scheduler(cfg.train, optimizer, len(train_loader))
  
    # Define the optimizer and learning rate scheduler
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['base_lr'], eps=config['train']['optimizer']['eps'], betas=tuple(config['train']['optimizer']['betas']))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['lr_scheduler']['decay_epochs'], gamma=config['train']['lr_scheduler']['cycle_limit'])

    # Mixed precision training
    #scaler = torch.cuda.amp.GradScaler()
    
    if cfg.checkpoint.auto_resume:
        resume_file = auto_resume_helper(cfg.output)
        if resume_file:
            if cfg.checkpoint.resume:
                logger.warning(f'auto-resume changing resume file from {cfg.checkpoint.resume} to {resume_file}')
            with read_write(cfg):
                cfg.checkpoint.resume = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {cfg.output}, ignoring auto resume')

    metrics = {'min_loss': float('inf'), 'best_epoch': -1}
    start_time = time.time()
    
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        torch.cuda.empty_cache()
        logger.debug(f'epoch:{epoch}')
        loss_train_dict = train_one_epoch(cfg, model, train_loader, optimizer, epoch, lr_scheduler, device= device)
        
        if dist.get_rank() == 0 and (epoch % cfg.checkpoint.save_freq == 0 or epoch == (cfg.train.epochs - 1)):
            save_checkpoint(cfg, epoch, model_without_ddp, {
                'min_loss': metrics['min_loss'],
            }, optimizer, lr_scheduler)
        dist.barrier()
        loss_train = loss_train_dict['total_loss']
        logger.info(f'Avg loss of the network on the {len(train_subset)} train images: {loss_train:.2f}')
        # evaluate
        if (epoch % cfg.evaluate.eval_freq == 0 or epoch == (cfg.train.epochs - 1)):
            loss = validate(cfg, val_loader, model)           
            dist.barrier()
            if cfg.evaluate.save_best and dist.get_rank() == 0 and loss < metrics['min_loss']:
                metrics['min_loss'] = min(metrics['min_loss'], loss)
                save_checkpoint(
                    cfg, epoch, model_without_ddp, metrics, optimizer, lr_scheduler, suffix='min_loss')


        if wandb is not None:
            log_stat = {f'epoch/train_{k}': v for k, v in loss_train_dict.items()}
            log_stat.update({
                'epoch/epoch': epoch,
                'epoch/n_parameters': total_params,
                'epoch/test_loss': loss,
            })
            wandb.log(log_stat)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    dist.barrier()
        



@torch.no_grad()
def validate(config, data_loader, model):
    logger = get_logger()
    dist.barrier()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    
    end = time.time()
    logger.info('Building Inference model for evaluation')

    for idx, samples in enumerate(data_loader):
        # compute output
        output = model(**samples, train=False) 
        # measure loss
        loss = output['loss']
        loss = reduce_tensor(loss)
        loss_meter.update(loss.item(), 1)
        
        #measure elapsed time
        logger.debug("all metrics updated")
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f'Test: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                         f'Mem {memory_used:.0f}MB\t'
                         )
    logger.info('Clearing zero shot classifier')
    torch.cuda.empty_cache()
    logger.info(f' loss {loss_meter.avg:.3f}')
    dist.barrier()
    return loss_meter.avg
   
   
if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Load config file
    config = get_config(args)
    # start faster ref: https://github.com/open-mmlab/mmdetection/pull/7036

    logger = get_logger(config)

    mp.set_start_method('fork', force=True)
    init_dist('pytorch')
    rank, world_size = get_dist_info()
    logger.info(f'RANK and WORLD_SIZE in environ: {rank}/{world_size}')

    dist.barrier()

    # Set random seed
    set_random_seed(config.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(config.output, exist_ok=True)

    train(config)

    dist.barrier()