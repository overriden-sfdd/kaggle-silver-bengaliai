from fastai2.vision.all import *
from fastai2.basics import *
from sklearn.metrics import recall_score
    
    
def get_labels(x): return tensor(x[1:4].astype('uint8'))    
def construct_vocab(df):
    return L(L(o for o in df[class_name].unique() if o==o).sorted() for class_name in df.columns.tolist())
    
    
class MEMCategorize(Transform):
    loss_func,order=CrossEntropyLossFlat(),1
    def __init__(self, vocab): self.vocab,self.c = vocab,L(len(cls) for cls in vocab)
    def encodes(self, o): return TensorCategory(tensor(o).float())
    def decodes(self, o): return MultiCategory (tensor(o))
    

class GridMask(RandTransform):
    order = 101
    def __init__(self, p=0.5, num_grid=3, fill_value=0, rotate=0, mode=0):
        super().__init__(p=p)
        if isinstance(num_grid, int): num_grid = (num_grid, num_grid)
        if isinstance(rotate, int): rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = torch.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w)), dtype=torch.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def get_params(self, img):
        height, width = img.shape[-2:]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return mask, rand_h, rand_w, angle
    
    def encodes(self, image:TensorImage, **params):
        mask, rand_h, rand_w, angle = self.get_params(image)
        h, w = image.shape[-2:]
        mask = afunc.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        mask = tensor(mask.float()).to(default_device())
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w]
        return image

    
class Loss_combine(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target,reduction='mean'):
        x1,x2,x3 = input
        x1,x2,x3 = x1.float(),x2.float(),x3.float()
        y = target.long()
        return 0.7*F.cross_entropy(x1,y[:,0],reduction=reduction) + 0.1*F.cross_entropy(x2,y[:,1],reduction=reduction) + \
          0.2*F.cross_entropy(x3,y[:,2],reduction=reduction)
    
    
class OHEM(Module):
    def __init__(self, top_k=0.7, weights=[0.7, 0.1, 0.2]):
        super(OHEM, self).__init__()
        self.loss = F.cross_entropy
        self.top_k = top_k
        self.weights = weights
    
    def forward(self, input, target, cb_reduction='mean', index=None):
        y,loss = target.long(),0
        
        for idx, row in enumerate(input):
            gt = y[:, idx]
            loss += self.weights[idx] * self.loss(row, gt, reduction='none', ignore_index=-1)

        if self.top_k == 1: valid_loss = loss

        self.index = torch.topk(loss, int(self.top_k * loss.size()[0])) if index is None else index
        valid_loss = loss[self.index[1]]

        return valid_loss.mean() if cb_reduction == 'mean' else valid_loss
    
    
class RecallPartial(Metric):
    # based on AccumMetric
    "Stores predictions and targets on CPU in accumulate to perform final calculations with `func`."
    def __init__(self, df, a=0, **kwargs):
        self.func = partial(recall_score, average='macro')
        self.a = a
        self.df = df

    def reset(self): self.targs,self.preds = tensor([]), tensor([])

    def accumulate(self, learn):
        fp,sp,tp = learn.pred
        preds,targs = torch.stack((fp.argmax(-1),sp.argmax(-1),tp.argmax(-1)), dim=-1).float(),learn.y
        preds,targs = to_detach(preds),to_detach(targs)
        self.preds = torch.cat((self.preds, preds))
        self.targs = torch.cat((self.targs, targs))

    @property
    def value(self):
        if len(self.preds) == 0: return
        return self.func(self.targs[:, self.a], self.preds[:, self.a])

    @property
    def name(self): return self.df.columns[self.a+1]
    
class RecallCombine(Metric):
    def accumulate(self, learn):
        scores = [learn.metrics[i].value for i in range(3)]
        self.combine = np.average(scores, weights=[7,1,2])

    @property
    def value(self):
        return self.combine
    
    
class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)    