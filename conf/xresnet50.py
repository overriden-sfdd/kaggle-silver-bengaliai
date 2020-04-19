workdir = 'model/xresnet50'


sz = 128
bs = 16
nfolds = 5 #keep the same split as the initial dataset
fold = 0
arch_name = 'xresnet50'
SEED = 1337
sliced_lr = (1e-3, 1e-2)

inp = './input'
md = inp + '/bengaliai-cv19'
labels = inp + '/folded_data'
images = inp + '/train_images'

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
tracker = dict(name='MyTrackCallback', params=dict(augs=al, probs=probs))

rlrop = dict(name='ReduceLROnPlateau', params=dict(patience=3))

brightness = dict(name='Brightness', params=dict(max_lighting=0.3, p=0.5))
contrast = dict(name='Contrast', params=dict())
gridmask = dict(name='GridMask', params=dict(p=0., num_grid=(3,7)))
erasing = dict(name='RandomErasing', params=dict(p=0., sh=0.15))
warp = dict(name='Warp', params=dict(magnitude=0.1))
affine = dict(name='AffineCoordTfm', params=dict(size=sz))
              

data = dict(
    train=dict(
        labels_df=labels + '/train_with_fold.csv',
        imgdir=images,
        transforms=[brightness, contrast, gridmask, erasing, warp, affine],
        aug_cbs=[mixup, cutmix, erase, tracker],
        train_cbs=[rlrop],
    ),
    test = dict(
        imgdir='some_input_path_test',
        imgsize=sz,
        pqt_files=[
            md + '/test_image_data_0.parquet',
            md + '/test_image_data_1.parquet',
            md + '/test_image_data_2.parquet',
            md + '/test_image_data_3.parquet',
        ],
    ),
)