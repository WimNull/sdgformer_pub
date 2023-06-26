import math


class LinearCosineLR:
    def __init__(self, optimizer, base_lr, total_epoch, steps_per_epoch=1, min_lr=1e-6, warmup_epoch=0, warmup_decay=1e-6, last_epoch=-1) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_step = total_epoch*steps_per_epoch
        self.warmup_step = warmup_epoch*steps_per_epoch
        self.warmup_factor = warmup_decay
        self.last_step = 0
        if last_epoch>0: self.last_step = steps_per_epoch*last_epoch
        for idx, group in enumerate(self.optimizer.param_groups):
            group['base_lr'] = base_lr
            group['base_wd'] = group['weight_decay']
        self.update_lr()
        
    def update_lr(self):
        lr = (self.base_lr-self.min_lr)*(1+math.cos(math.pi*self.last_step/self.total_step))/2 + self.min_lr
        if self.last_step<self.warmup_step:
            alpha = self.last_step / self.warmup_step
            warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
            lr *= warmup_factor
        for idx, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr
            # bdw = group['base_wd']
            # group['weight_decay'] = (bdw-1e-5)*math.cos(math.pi/2*self.last_step/self.total_step)+1e-5
        return lr

    def step(self):
        self.update_lr()
        self.last_step += 1

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict) -> None:
        self.__dict__.update(state_dict)
        self.update_lr()

