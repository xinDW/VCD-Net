from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.PSF = edict()
config.VALID = edict()

config.img_size     = 176                                                           # LF 2D image size for training
config.size_factor  = [1, 1]                                                        # Real img_size = img_size * size_factor
config.PSF.n_slices = 61                                                            # Number of z slices of the 3-D reconstruction
config.PSF.Nnum     = 11                                                            # N number of the light field psf
config.n_channels   = 1                                                             # Number of channels of the training and validation data

label                             = 'bead_40x_n11_[m30-30]_step1um_sparse'                  # Distingiushable label for saving the checkpoint files and validation result
# label                             = 'rbcDSRED_20x_n11_[m50-50]_step2um_sparse'
config.label                      = label     


## Training 
config.TRAIN.target3d_path        = '../data/example_data/train/bead/[m30-30]step1um/WF/'   # 3-D targets for training
config.TRAIN.lf2d_path            = '../data/example_data/train/bead/[m30-30]step1um/LF/'  # LF projections for training

config.TRAIN.test_saving_path     = "sample/test/{}/".format(label)
config.TRAIN.ckpt_saving_interval = 10
config.TRAIN.ckpt_dir             = "../data/checkpoint/{}/".format(label)
config.TRAIN.log_dir              = "log/{}/".format(label)
config.TRAIN.device               = 0                                               # gpu used for training, 0 means the 1st device is used.
config.TRAIN.valid_on_the_fly     = False
config.TRAIN.using_edge_loss      = False                                           # use the edges loss to promote the quality of the reconstructions

config.TRAIN.batch_size  = 1
config.TRAIN.lr_init     = 1e-4
config.TRAIN.beta1       = 0.9
config.TRAIN.n_epoch     = 200
config.TRAIN.lr_decay    = 0.1
config.TRAIN.decay_every = 50

## Inference
config.VALID.lf2d_path            = '../data/example_data/valid/bead(sparse)/'                           # location of LF measurements (Nnum = 11) to be reconstructed
config.VALID.saving_path          = '{}VCD_{}/'.format(config.VALID.lf2d_path, label)





