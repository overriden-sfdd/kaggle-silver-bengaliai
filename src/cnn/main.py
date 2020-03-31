import sys
import os
import time
import argparse
import random
import collections
import pickle

from fastai2.vision.all import *
from fastai2.callback.all import *
from fastai2.callback.tracker import *
from fastai2.basics import *

from . import factory
from .utils.logger import logger, log
from ..train_utils.bengali_funcs import *
from ..train_utils.bengali_augs import *
from ..train_utils.model import *
del globals()['Config']
from .utils.config import Config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('config')
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output') 
    return parser.parse_args()


def main():

    args = get_args()
    cfg = Config.fromfile(args.config)

    # copy command line args to cfg
    cfg.mode = args.mode
    cfg.pretrain = args.pretrain
    cfg.fold = args.fold
    cfg.output = args.output
    cfg.gpu = args.gpu

    logger.setup(cfg.workdir, name='%s_fold%d' % (cfg.mode, cfg.fold))

    log(f'mode: {cfg.mode}')
    log(f'workdir: {cfg.workdir}')
    log(f'fold: {cfg.fold}')
    log(f'batch size: {cfg.bs}')

    arch = factory.get_arch(cfg)

    if cfg.mode == 'train':
        train(cfg, arch)
    elif cfg.mode == 'test':
        test(cfg, model)


def train(cfg, arch):
    image_dir = cfg.data.train.imgdir
    df = pd.read_csv(cfg.data.train.labels_df)
    bs = cfg.bs
    
    vocab = construct_vocab(df[cfg.class_names])
    split_idx = IndexSplitter(df.loc[df.fold==cfg.fold].index)(df)
    
    get_image = lambda x: image_dir + f'/{x[0]}.png'
    type_tfms = [[get_image, PILImageBW.create, ToTensor], [get_labels, MEMCategorize(vocab=vocab)]]
    item_tfms = [ToTensor]
    batch_tfms = [IntToFloatTensor] + factory.get_transforms(cfg)
    
    dsrc = Datasets(df.values, type_tfms, splits=split_idx)
    tdl = TfmdDL(dsrc, bs=bs, after_item=item_tfms, after_batch=batch_tfms, device=default_device())
    
    batch_tfms += [Normalize.from_stats(*factory.get_norm(tdl))]
    tdl_train = TfmdDL(dsrc.train, bs=bs, after_item=item_tfms, after_batch=batch_tfms, device=default_device())
    tdl_valid = TfmdDL(dsrc.valid, bs=bs, after_item=item_tfms, after_batch=batch_tfms, device=default_device())
    
    dbch = DataLoaders(tdl_train, tdl_valid, device=default_device()) 
    model = CascadeModel(arch=arch, n=dbch.c, sz=cfg.sz, pre=cfg.pretrain)
    loss_function = factory.get_loss(cfg)
    optimizer = factory.get_optim(cfg)
    bot_lr,top_lr = cfg.sliced_lr
    aug_cbs,train_cbs = factory.get_cbs(cfg)
    
    learn = Learner(dbch, model, loss_func=loss_function, optimizer=optimizer, cbs=aug_cbs,
               metrics=[RecallPartial(a=i) for i in range(len(dbch.c))] + [RecallCombine()],
               splitter=lambda m: [list(m.body.parameters()), list(m.heads.parameters())],
               model_dir=args.output)
    
    if cfg.gpu: learn.to_fp16()
    learn.fit_one_cycle(cfg.epochs, lr_max=slice(bot_lr, top_lr), cbs=train_cbs)
    
    
if __name__ == '__main__':

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('benchmark', torch.backends.cudnn.benchmark)

    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')