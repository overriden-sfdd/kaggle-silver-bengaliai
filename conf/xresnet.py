workdir = './model/xresnet50'


sz = (128, 128)
bs = 128
nfolds = 5 #keep the same split as the initial dataset
fold = 0
arch_name = 'xresnet50'
SEED = 1337
class_names = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
al,probs = ['Cutout', 'GridMask', 'CustomMixUp', 'CutMix'],[0.15, 0.15, 0.7, 0.]

loss = dict(
    name='OHEM',
    params=dict(),
)


optim = dict(
    name='Adam',
    params=dict(),
)


mixup = dict(name='CustomMixUp', params=dict())
cutmix = dict(name='CutMix', params=dict())
erase = dict(name='EraseCallback', params=dict())
tracker = dict(name='MyTrackCallback', params=dict(al=al, probs=probs))

rlrop = dict(name='ReduceLROnPlateou', params=dict(patience=3))
savemodel = dict(name='SaveModelCallback', params=dict(fname=f'{arch_name}'))

brightness = dict(name='Brightness', params=dict(max_lighting=0.3, p=0.5))
contrast = dict(name='Contrast', params=dict())
gridmask = dict(name='GridMask', params=dict(p=0., num_grid=(3,7))
erasing = dict(name='RandomErasing', params(p=0., sh=0.15))
warp = dict(name='Warp', params=dict(magnitude=0.1))
affine = dict(name='AffineCoordTfm', params=dict(size=sz))
              

data = dict(
    train=dict(
        labels_df='some_input_path_df.csv'
        imgdir='some_input_path_train',
        transforms=[brightness, contrast, gridmask, erasing, warp, affine],
        aug_cbs=[mixup, cutmix, erase, tracker],
        train_cbs=[rlrop, savemodel],
    ),
    test = dict(
        imgdir='some_input_path_test',
        imgsize=sz,
        pqt_files=[
        ],
    ),
)