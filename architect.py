import torch
import numpy as np
import torch.nn as nn


class Architect () :
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            weight_decay=args.arch_weight_decay , betas=(0.5, 0.999) , lr=0.007)#args.arch_lr )

    def step (self, input_valid, target_valid) :
        self.optimizer.zero_grad ()
        self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step (self, input_valid, target_valid) :
        loss = self.model._loss (input_valid, target_valid)
        loss.backward ()



