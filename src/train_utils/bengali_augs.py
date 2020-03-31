from fastai2.basics import *
from fastai2.vision.all import *
from fastai2.callback.all import *
from fastai2.callback.tracker import *
from torch.distributions.beta import Beta


class MyTrackCallback(Callback):
    run_after,run_valid = [Normalize],False
    def __init__(self, augs, probs): self.augs,self.probs = augs,probs    
        
    def aug_tracker(self, augs, probs): return augs[int(np.random.choice(len(augs), 1, p=probs))]
    
    def begin_batch(self): self.learn.condition = self.aug_tracker(self.augs, self.probs)
        

def NoLoss(*o): pass
class CustomMixUp(Callback):
    run_after,run_valid = MyTrackCallback,False
    def __init__(self, alpha=0.4): self.distrib = Beta(tensor(alpha), tensor(alpha))
        
    def begin_fit(self):self.loss_func0 = self.learn.loss_func
        
    def begin_batch(self):
        if self.learn.condition != self.__class__.__name__: return
        self.dls.after_batch.fs[-1].p,self.dls.after_batch.fs[-2].p = 0.,0.
        self.learn.loss_func = NoLoss
        lam = self.distrib.sample((self.y[:, 0].size(0),)).squeeze().to(self.x[0].device)
        lam = torch.stack([lam, 1-lam], 1)
        self.lam = lam.max(1)[0]
        shuffle = torch.randperm(self.y[:, 0].size(0)).to(self.x.device)
        xb1 = tuple(L(self.xb).itemgot(shuffle))
        yb1 = tuple([self.yb[i][shuffle] for i in range(len(self.yb))])
        nx_dims = len(self.x.size())
        self.learn.xb = tuple(L(xb1,self.xb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=nx_dims-1)))
        self.learn.yb = yb1,self.yb
        
    def after_loss(self):
        if self.learn.condition != self.__class__.__name__: return
        loss0 = self.loss_func0(self.learn.pred, *self.learn.yb[0], cb_reduction='none')
        loss1 = self.loss_func0(self.learn.pred, *self.learn.yb[1], cb_reduction='none', index=self.loss_func0.index)
        # loss1 = self.loss_func0(self.learn.pred, *self.learn.yb[1], cb_reduction='none')
        self.learn.loss = torch.lerp(loss0, loss1, self.lam[self.loss_func0.index[1]]).mean()
        # self.learn.loss = torch.lerp(loss0, loss1, self.lam).mean()
        self.learn.loss_func = self.loss_func0
        
        
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1,bby1,bbx2,bby2


class CutMix(Callback):
    run_after,run_valid = MyTrackCallback,False
    def __init__(self, alpha=1., stack_y=True): self.alpha,self.stack_y = alpha,stack_y

    def begin_fit(self): self.loss_func0 = self.learn.loss_func

    def begin_batch(self):
        if self.learn.condition != self.__class__.__name__: return
        self.dls.after_batch.fs[-1].p,self.dls.after_batch.fs[-2].p = 0.,0.
        self.learn.loss_func = NoLoss
        lam = np.random.beta(self.alpha, self.alpha)
        shuffle = torch.randperm(self.y[:, 0].size(0)).to(self.x.device)
        yb1 = TensorCategory(*[self.yb[i][shuffle] for i in range(len(self.yb))])
        last_input_size = self.x.shape
        bbx1, bby1, bbx2, bby2 = rand_bbox(last_input_size, lam)
        new_input = self.x.clone()
        new_input[:, ..., bby1:bby2, bbx1:bbx2] = self.x[shuffle, ..., bby1:bby2, bbx1:bbx2]
        self.learn.xb = tuple([new_input])
        lam = self.x.new([lam])
        if self.stack_y:
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (last_input_size[-1] * last_input_size[-2]))
            lam = self.x.new([lam])
            self.learn.yb = tuple([torch.cat([self.y.float(), yb1.float(), lam.repeat(last_input_size[0]).unsqueeze(1).float()], 1)])
        else:
            if len(learn.y.shape) == 2:
                lam = lam.unsqueeze(1).float()
            self.learn.yb = tuple([self.yb.float() * lam + yb1.float() * (1-lam)])


    def after_loss(self):
        if self.learn.condition != self.__class__.__name__: return
        self.learn.loss = self.loss_func0(self.learn.pred, *self.learn.yb, cb_reduction='mean')
        self.learn.loss_func = self.loss_func0
        
        
class EraseCallback(Callback):
    run_after,run_valid = CustomMixUp,False

    def begin_fit(self): self.loss_func0 = self.learn.loss_func

    def begin_batch(self):
        self.mu,self.gm,self.cutout = self.learn.condition == 'CustomMixUp',self.learn.condition == 'GridMask',self.learn.condition == 'Cutout'
        if self.mu: return
        self.learn.loss_func = NoLoss
        
        if self.gm: 
            self.dls.after_batch.fs[-2].p=0.
            self.dls.after_batch.fs[-1].p=1.
        elif self.cutout: 
            self.dls.after_batch.fs[-1].p=0.
            self.dls.after_batch.fs[-2].p=1.

    def after_loss(self):
        if self.mu: return
        self.learn.loss = self.loss_func0(self.learn.pred, *self.learn.yb, cb_reduction='mean')
        self.learn.loss_func = self.loss_func0
