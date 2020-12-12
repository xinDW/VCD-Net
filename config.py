from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.PSF = edict()
config.VALID = edict()

config.img_size     = 176   # croppped LF 2D image size
config.size_factor  = [1, 1] # real img_size = img_size * size_factor
config.PSF.n_slices = 31  # z depth
config.PSF.Nnum     = 11
config.n_channels   = 1
config.use_batch_norm = False
config.n_interp       = 4  # num of subpixel layer before u-net 
config.n_base_filters = 128

config.normalize_mode = 'max'    # ['percentile', 'max']

config.TRAIN.using_edge_loss = False
config.TRAIN.using_vgg_loss  = False

label                        = 'bead_[m30-0]_step1um_unet'   
config.label = label

## train set location
config.TRAIN.target3d_path  = 'data/train/fish/confocal/zebrafish_heart_cmlc_mcherry_4dpf_[m50-50]_N11_step2um_20x_flipz_shallow_region_2/WF/'
config.TRAIN.lf2d_path      = 'data/train/fish/confocal/zebrafish_heart_cmlc_mcherry_4dpf_[m50-50]_N11_step2um_20x_flipz_shallow_region_2/LF/mixed_noise/'

## validate set location
config.VALID.lf2d_path      = 'example_data/valid/'   
config.VALID.bitdepth       = 16


config.VALID.saving_path = '{}/recon_{}/'.format(config.VALID.lf2d_path, label)
config.TRAIN.test_saving_path = "sample/test/{}/".format(label)
config.TRAIN.ckpt_saving_interval = 10
config.TRAIN.ckpt_dir = "checkpoint/{}/".format(label)
config.TRAIN.log_dir = "log/{}/".format(label)

config.TRAIN.device_num = 2
config.TRAIN.device     = 0
# whether the inputs is time-series or not. 
# If true, each batch of images will be regarded as a time series.
config.TRAIN.series_input = False
config.TRAIN.valid_on_the_fly = False


## Adam
config.TRAIN.batch_size = 4
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## learning 
config.TRAIN.n_epoch = 500
config.TRAIN.lr_decay = 0.5
#config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)
config.TRAIN.decay_every = 100