#encoding:utf-8
import os
from pathlib import Path
import numpy as np
import torch

class ModelCheckpoint(object):
    def __init__(self, checkpoint_dir,monitor,logger,
                 arch,mode='min',epoch_freq=1,best = None,
                 save_best_only = True,
                 ):
        if isinstance(checkpoint_dir,Path):
            self.base_path = checkpoint_dir
        else:
            self.base_path = Path(checkpoint_dir)

        self.arch = arch
        self.logger = logger
        self.monitor = monitor
        self.epoch_freq = epoch_freq
        self.save_best_only = save_best_only

        # two modes
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        
        if best:
            self.best = best

        if save_best_only:
            self.model_name = f"best_{arch}_model.pth"

    def epoch_step(self, state,current):
        
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
                self.best = current
                state['best'] = self.best
                best_path = self.base_path/ self.model_name
                torch.save(state, str(best_path))
        
        else:
            filename = self.base_path / f"epoch_{state['epoch']}_{state[self.monitor]}_{self.arch}_model.pth"
            if state['epoch'] % self.epoch_freq == 0:
                self.logger.info("\nEpoch %d: save model to disk."%(state['epoch']))
                torch.save(state, str(filename))
