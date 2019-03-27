from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.PSF = edict()
config.VALID = edict()

config.img_size = 176 # croppped LF 2D image size
config.size_factor = [1, 1] # real img_size = img_size * size_factor
config.PSF.n_slices = 101
config.PSF.Nnum = 11
config.n_channels = 1

## Adam
config.TRAIN.batch_size = 1
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## learning 
config.TRAIN.n_epoch = 4000
config.TRAIN.lr_decay = 0.1
#config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)
config.TRAIN.decay_every = 500

label = 'beads_20X[m100-100]_step2um'
#label = 'celegans_pan-neu_[m30-30]_step2um_multi_psf_unet'
#label = 'bead_[m15-15]_step2um_unet'
config.TRAIN.test_saving_path = "sample/test/{}/".format(label)
config.TRAIN.ckpt_saving_interval = 10
config.TRAIN.ckpt_dir = "checkpoint/{}/".format(label)
config.TRAIN.log_dir = "log/{}/".format(label)

config.TRAIN.device = 1
# whether the inputs is time-series or not. 
# If true, each batch of images will be regarded as a time series.
config.TRAIN.series_input = False
config.TRAIN.valid_on_the_fly = False
## train set location
#config.TRAIN.target3d_path = 'data/train/bead/[m15-15]step2um_lightsheet_20190224/WF/all/'
#config.TRAIN.lf2d_path = 'data/train/bead/[m15-15]step2um_lightsheet_20190224/LF/all/'
config.TRAIN.target3d_path = 'data/train/bead/[m100,100]step2um_by_zhaoqiang_20190326/WF/'
config.TRAIN.lf2d_path = 'data/train/bead/[m100,100]step2um_by_zhaoqiang_20190326/LF/'

## validate set location
config.VALID.lf2d_size = [935, 935]
config.VALID.lf2d_path = 'data/valid/'
config.VALID.saving_path = 'data/valid/recon/'