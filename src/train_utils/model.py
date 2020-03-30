from .bengali_funcs import MishFunction
from fastai2.vision.all import *
from fastai2.basics import *


class Head(Module):
    def __init__(self, nc, n, ps=0.5):
        self.fc = nn.Sequential(*[AdaptiveConcatPool2d(), Mish(), Flatten(),
             LinBnDrop(nc*2, 512, True, ps, Mish()),
             LinBnDrop(512, n, True, ps)])
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
        
    def forward(self, x):
        return self.fc(x)
    
    
class CascadeModel(Module):
    def __init__(self, arch, n, pre=True):
        m = arch(pre)
        m = nn.Sequential(*children_and_parameters(m)[:-4])
        conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        w = (m[0][0].weight.sum(1)).unsqueeze(1)
        conv.weight = nn.Parameter(w)
        m[0][0] = conv
        nc = m(torch.zeros(2, 1, sz, sz)).detach().shape[1]
        self.body = m
        self.heads = nn.ModuleList([Head(nc, c) for c in n])
        
    def forward(self, x):    
        x = self.body(x)
        return [f(x) for f in self.heads]