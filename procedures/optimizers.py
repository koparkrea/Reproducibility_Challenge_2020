import torch, math




class _LRScheduler(object):
    def __init__(self, optimizer, warmup_epochs, epochs):
        
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.max_epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.current_iter = 0
    
    def extra_repr(self):
        return ''
    
    def state_dict(self):
        return {key : value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
    
    def get_lr(self):
        raise NotImplementedError

    def get_min_lr(self):
        return min(self.get_lr())
    
    def update(self, cur_epoch, cur_iter):
        if cur_epoch is not None:
            assert cur_epoch >= 0, 'invalid cur_epoch = {:}'.format(cur_epoch)
            cur_epoch = int(cur_epoch)
            self.current_epoch = cur_epoch
        if cur_iter is not None:
            assert cur_iter >= 0, 'invalid cur_iter = {:}'.format(cur_iter)
            cur_iter = int(cur_iter)
            self.current_iter = cur_iter
        for param_groups, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_groups['lr'] = lr


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, epochs, T_max, eta_min):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, warmup_epochs, epochs)
    
    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.current_epoch >= self.warmup_epochs and self.current_epoch < self.max_epochs:
                last_epoch = self.current_epoch - self.warmup_epochs
                lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * last_epoch / self.T_max)) / 2
            elif self.current_epoch >= self.max_epochs:
                lr = self.eta_min
            else:
                lr = (self.current_epoch / self.warmup_epochs + self.current_iter / self.warmup_epochs) *base_lr
            lrs.append(lr)
        return lrs


############ NEED TO ADD OTHER CONDITIONS!!!!
def get_optim_scheduler(parameters, config):
    
    if config.optim == 'SGD':
        optim = torch.optim.SGD(parameters, lr = config.LR, momentum= config.momentum, weight_decay= config.decay, nesterov= config.nesterov)
    elif config.optim == 'RMSprop':
        optim = torch.optim.RMSprop(parameters, lr = config.LR, momentum= config.momentum, weight_decay= config.decay)
    else:
        raise ValueError('invalid optim = {:}'.format(config.optim))

    if config.scheduler == 'cos':
        T_max = getattr(config, 'T_max', config.epochs)
        scheduler = CosineAnnealingLR(optim, config.warmup, config.epochs, T_max, config.eta_min)
    else:
        raise ValueError('invalid scheduler = {:}'.format(config.scheduler))

    if config.criterion == 'Softmax':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError('invalid criterion = {:}'.format(config.criterion))
    return optim, scheduler, criterion
