from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.Trans = edict()
config.PSF = edict()
config.VALID = edict()

config.img_size     = 176                                                           # LF 2D image size for training
config.size_factor  = [1, 1]                                                        # Real img_size = img_size * size_factor
config.PSF.n_slices = 31                                                            # Number of z slices of the 3-D reconstruction
config.PSF.Nnum     = 11                                                           # N number of the light field psf
config.n_channels   = 1                                                             # Number of channels of the training and validation data
config.n_base_filters = 128
config.normalize_mode = 'max' #['percentile', 'max']
# label                             = 'bead_40x_n11_[m30-30]_step1um_sparse'                  # Distingiushable label for saving the checkpoint files and validation result
# label                             = 'rbcDSRED_20x_n11_[m50-50]_step2um_sparse'
# label                             = 'cardiac_20x_n11_[m50-50]_step2um_dense'
# label                             = 'zebrafisch_heart_20X_[m50-50]_N11_step2um_lightsheet_xl8_sparse'
# label = 'nuclei_20x_n11_[m75-75]_step3um_conpen_xl9_out4_sparse'


# label                             = 'cardiac_20x_lightsheet_n11_[m50-50]_step2um_subpixel-head'    
# label                             = 'cardiac_20x_lightsheet_n11_[m50-50]_step2um_upconv-head' 
# label                             = 'cardiac_20x_lightsheet_n11_[m50-50]_step2um_subpixel-head_binary-loss'    
# label                             = 'cardiac_20x_lightsheet_n11_[m50-50]_step2um_upconv-head_binary-loss' 

#label = 'worm-a+b_40x_yyc_n11_[m30-0]_step1um_wide'
label = 'worm-a+b_40x_hmy_n11_[m30-0]_step1um_wide_bead_guided'  # n_base_filters = 128
# label = 'celegans_40x_a+b_[m30-0]step1um'  #  old net, best quality
# label = 'a+b_hmy_40x_[m30-0]_step1um_old-nobn' # old net, norm[0,1]
#label = 'zq_moving_worm_high_sbr_40x_n11_[m30-0]_step1um'
# label = 'celegans_40x_a+b_[m30-0]step1um_repeat' #norm[-1,1]  
# label = 'worm-a+b_40x_n11_[m30-0]_step1um_legacy'
# label = 'worm-a+b_40x_n11_[m30-0]_step1um_unetc_no-bn'
# label = 'bead_aniso-f2_[m30-0]_step1um' # best result, subpixel head, nearest+conv decoder
# label = 'worm-a+b_40x_n11_[m30-0]_step1um_16bit_various_intens_unetc_no-bn' 
label = 'bead_aniso-f2_[m30-0]_step1um_wide' 
# label = 'worm-a+b_40x_n11_[m30-0]_step1um_16bit_various_intens'

# label = 'panneuron_x40_step1um_[m30-0]'
# label = 'a+b_40x_hmy_n11_[m30-30]_step1um'
label = 'neuron_8um_simu_40x_n11_[m30-0]_step1um'
config.label                      = label     


## Training 
config.TRAIN.target3d_path        = 'I:/LFRnet/data/train/celegans/40X_A+B_[m30-0]_step1um/WF/'   # 3-D targets for training
config.TRAIN.lf2d_path            = 'I:/LFRnet/data/train/celegans/40X_A+B_[m30-0]_step1um/LF/'  # LF projections for training

config.TRAIN.target3d_path        = 'I:/LFRnet/data/train/celegans/20190828_40X_dx12_Nnum11_Celegans_A+B_[m30-0]_step1um_hmy/WF'   
config.TRAIN.lf2d_path            = 'I:/LFRnet/data/train/celegans/20190828_40X_dx12_Nnum11_Celegans_A+B_[m30-0]_step1um_hmy/LF'


config.TRAIN.target3d_path        = 'I:/LFRnet/data/train/bead/simu_anisotropic[m30-0]_step1um/WF/'  
config.TRAIN.lf2d_path            = 'I:/LFRnet/data/train/bead/simu_anisotropic[m30-0]_step1um/LF/'  

# config.TRAIN.target3d_path        = 'I:/LFRnet/data/train/fish/lightsheet/zebrafish_heart_cmlc_mcherry_4dpf_[50-m50]_N11_step2um_20x_shallow/WF/'  
# config.TRAIN.lf2d_path            = 'I:/LFRnet/data/train/fish/lightsheet/zebrafish_heart_cmlc_mcherry_4dpf_[50-m50]_N11_step2um_20x_shallow/LF/'  

# config.TRAIN.target3d_path        = 'I:/LFRnet/data/train/celegans/neuron_8um_simu_40x_n11_[m30-0]_step1um/WF'  
# config.TRAIN.lf2d_path            = 'I:/LFRnet/data/train/celegans/neuron_8um_simu_40x_n11_[m30-0]_step1um/LF/'

config.VALID.bitdepth             = 16
config.TRAIN.test_saving_path     = "sample/test/{}/".format(label)
config.TRAIN.ckpt_saving_interval = 100
config.TRAIN.ckpt_dir             = "checkpoint/{}/".format(label)

config.TRAIN.log_dir              = "log/{}/".format(label)
config.TRAIN.device               = 0                                             # gpu used for training, 0 means the 1st device is used.
config.TRAIN.device_nums          = 1
config.TRAIN.valid_on_the_fly     = False
config.TRAIN.using_edge_loss      = False                                           # use the edges loss to promote the quality of the reconstructions
config.TRAIN.using_binary_loss    = False
config.TRAIN.using_batch_norm     = False

config.TRAIN.batch_size  = 4
config.TRAIN.lr_init     = 1e-4
config.TRAIN.beta1       = 0.9
config.TRAIN.n_epoch     = 500
config.TRAIN.lr_decay    = 0.1
config.TRAIN.decay_every = 50

## Inference
#config.VALID.lf2d_path            = 'I:/LFRnet/data/tools/DataPre/SIsoftware/Data/02_Rectified/20201014_worm/sort_gfp_rotate_0p1/5_sub210/'              # location of LF measurements (Nnum = 11) to be reconstructed
config.VALID.lf2d_path            = 'E:/405backup/LFRNet/data/valid/celegans/20181129celegans_100fps/LightFiled/RFP_N11crop704/sbg'
# config.VALID.lf2d_path            = 'I:/LFRnet/data/valid/celegans/a+b_20201007/worm3/3x_sort_200ms_RFP_140-316_16bit_subbg120_rotate-0p3/'
# config.VALID.lf2d_path            = 'I:/LFRnet/data/valid/celegans/a+b_20201007/worm3/3X_sort_200ms_16bit_GFP_subbg115_141-316/'
# config.VALID.lf2d_path            = 'I:/LFRnet/data/tools/DataPre/SIsoftware/Data/02_Rectified/20201014_worm/sort_gfp_rotate_0p1/2_sub210/'