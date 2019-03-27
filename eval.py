import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import time

from config import *
from utils import *
from model import UNet

def read_valid_images(path):
    """
    Params:
        - 
    return images in shape [n_images, height=img_size/n_num, width=img_size/n_num, channels=n_num**2]
    """
    img_set = []
    img_list = sorted(tl.files.load_file_list(path=path, regx='.*.tif', printable=False))
            
    for img_file in img_list:
        print(path + img_file)
        img = get_lf_extra(img_file, path) 
        

        if (img.dtype != np.float32):
            img = img.astype(np.float32, casting='unsafe')
            
        print(img.shape)

        img_set.append(img)
    
    if (len(img_set) == 0):
        raise Exception("none of the images have been loaded")
    
    print('read %d from %s' % (len(img_set), path)) 
    img_set = np.asarray(img_set)
    print(img_set.shape)
    return img_set

n_num = config.PSF.Nnum

def evaluate(epoch, batch_size=1):
    #checkpoint_dir = "checkpoint/"

    checkpoint_dir = config.TRAIN.ckpt_dir
    lf_size = config.VALID.lf2d_size
    valid_lr_img_path = config.VALID.lf2d_path
    save_dir = config.VALID.saving_path
    tl.files.exists_or_mkdir(save_dir)
    
    start_time = time.time()
    valid_lf_extras = read_valid_images(valid_lr_img_path)
    
    t_image = tf.placeholder('float32', [batch_size, int(np.ceil(lf_size[0]/n_num)) , int(np.ceil(lf_size[1]/n_num)), n_num ** 2])
    net = UNet(t_image, config.PSF.n_slices, lf_size, is_train=True, reuse=False, name='unet') 
  
    ckpt_found = False

    filelist = os.listdir(checkpoint_dir)
    for file in filelist:
        if '.npz' in file and str(epoch) in file:
                ckpt_file = file
                ckpt_found = True
                break

    if not ckpt_found:
        raise Exception('no such checkpoint file')
            
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + ckpt_file, network=net)
        
        for idx in range(0,len(valid_lf_extras), batch_size):
            
            recon = sess.run(net.outputs, {t_image : valid_lf_extras[idx:idx+batch_size]})
            write3d(recon, save_dir+'net_%s_%06d_epoch%d.tif' % (label, idx, epoch))
            #write3d(out, save_dir+'/epoch{}_{:0>4}.tif'.format(epoch, idx//batch_size))
            print('writing %d / %d ...' % (idx + 1, len(valid_lf_extras)))

          
    print("time elapsed : %4.4fs " % (time.time() - start_time))
 
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=int, default=0)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()
    ckpt = args.ckpt
    batch_size = args.batch

    evaluate(ckpt, batch_size)
    
