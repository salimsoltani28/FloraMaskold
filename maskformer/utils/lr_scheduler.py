from timm.scheduler.cosine_lr import CosineLRScheduler

def build_scheduler(config, optimizer, n_iter_per_epoch):
    print(f'Building {config.lr_scheduler.name} scheduler with {n_iter_per_epoch} iterations per epoch')
    num_steps = int(config.lr_scheduler.decay_epochs * n_iter_per_epoch)
    warmup_steps = int(config.warmup_epochs * n_iter_per_epoch)

    lr_scheduler = None
    if config.lr_scheduler.name == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.min_lr,
            warmup_lr_init=config.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=config.lr_scheduler.cycle_limit,
            t_in_epochs=False,
        )
        print(f'lr_scheduler: {lr_scheduler}')
    else:
        raise NotImplementedError(f'lr scheduler {config.lr_scheduler.name} not implemented')
    return lr_scheduler