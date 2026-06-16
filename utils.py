"""Small training utilities."""
import random
import numpy as np
import torch


class AverageMeter(object):
    """Tracks a running average of a scalar."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_main_output(outputs):
    """Return the primary prediction whether the model emits a single tensor or
    a [main, aux...] deep-supervision list."""
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs
