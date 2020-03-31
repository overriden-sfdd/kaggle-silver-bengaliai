import copy
import sys

from fastai2.vision.all import *
from fastai2.callback.all import *
from fastai2.callback.tracker import *
from fastai2.basics import *
from ..train_utils.bengali_funcs import GridMask

from .utils.logger import log
gb_ns = sys.modules[__name__]


def get_object(obj):
    if hasattr(gb_ns, obj.name):
        return getattr(gb_ns, obj.name)
    else:
        return eval(obj.name)
    
    
def get_norm(tdl):
    xb,_ = tdl.one_batch()
    mega_batch_stats = xb.mean(), xb.std()
    return mega_batch_stats


def get_arch(cfg):
    return getattr(gb_ns, cfg.arch_name)


def get_loss(cfg):
    loss = getattr(gb_ns, cfg.loss.name)(**cfg.loss.params)
    log('loss: %s' % cfg.loss.name)
    return loss


def get_transforms(cfg):
    batch_tfms = [get_object(transform)(**transform.params) for transform in cfg.data.train.transforms]
    return batch_tfms


def get_cbs(cfg):
    aug_cbs = [get_object(ac)(**ac.params) for ac in cfg.data.train.aug_cbs]
    train_cbs = [get_object(tc)(**tc.params) for tc in cfg.data.train.train_cbs]
    return aug_cbs,train_cbs


def get_optim(cfg, parameters):
    optim = getattr(gb_ns, cfg.optim.name)(**cfg.optim.params)
    log(f'optim: {cfg.optim.name}')
    return optim