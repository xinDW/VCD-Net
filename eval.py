import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import time

from config import config
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
    _, height, width, _ = img_set.shape
    
    return img_set, height, width

def __raise(info):
    raise Exception(info)

def evaluate(epoch, batch_size=1, use_cpu=False):
    '''
    def _eval_images_in_folder_root(folder, check_empty=False, sess=None):  
        valid_img_list = sorted(tl.files.load_file_list(path=folder, regx='.*.tif', printable=False))
        valid_lf2d_imgs = read_all_imgs(valid_img_list, path=folder, type='2d', n_threads=batch_size, check_empty=check_empty)

        if len(valid_lf2d_imgs) == 0: 
            return 

        save_dir_ = '{}/{}/'.format(save_dir, folder)  
        tl.files.exists_or_mkdir(save_dir_)
  
        cell_value = np.zeros([batch_size, cell_size[0], cell_size[1], 512])
        for idx in range(0, len(valid_img_list), batch_size):
            image = tl.prepro.threading_data(valid_lf2d_imgs[idx : idx + batch_size], fn=lf_extract_fn, mode='toChannel', padding=False)
            #image = get_img2d_fn(file, config.VALID.lf2d_path)
            #image = lf_extract_fn(image, mode='toChannel', padding=False)
            # print(image.shape)
            start_t = time.time()
            out, cell_value = sess.run([net.outputs, cell], {t_image: image, t_cell : cell_value})         
            print('time elapsed : %4.4fs' % (time.time() - start_t))  
            print("saving {}/{} ...".format(idx, len(valid_img_list)))
            
            write3d(out, save_dir_+'/{}_epoch{}_{:0>4}.tif'.format(tag, epoch, idx//batch_size))

    def _eval_recursively(root):
        print(root)
        _eval_images_in_folder_root(root, check_empty=False)
        
        dirs = os.listdir(root)
        for file in dirs:
            child_path = os.path.join(root, file) + '/'
            if os.path.isdir(child_path) and (not ('net_recon' in file)):
                _eval_recursively(child_path)

    '''
    #checkpoint_dir = "checkpoint/bead_simu_resolution_test/"

    checkpoint_dir = config.TRAIN.ckpt_dir
    lf_size = config.VALID.lf2d_size
    valid_lr_img_path = config.VALID.lf2d_path
    save_dir = config.VALID.saving_path
    tl.files.exists_or_mkdir(save_dir)
    
    n_num = config.PSF.Nnum
    devices_num = config.TRAIN.device_num

    
    valid_lf_extras, height, width = read_valid_images(valid_lr_img_path)
    
    #t_image = tf.placeholder('float32', [batch_size, int(np.ceil(lf_size[0]/n_num)) , int(np.ceil(lf_size[1]/n_num)), n_num ** 2])
    t_image = tf.placeholder('float32', [batch_size, height , width, n_num ** 2])

    device_str = '/gpu:0' if not use_cpu else '/cpu:0'

    with tf.device(device_str):
        net = UNet(t_image, config.PSF.n_slices, [height * n_num, width * n_num], is_train=True, reuse=False, name='unet') 
  
    ckpt_found = False
    filelist = os.listdir(checkpoint_dir)
    for file in filelist:
        if '.npz' in file and str(epoch) in file:
        #if '.npz' in file and str(epoch) and(config.label) in file:
            ckpt_file = file
            ckpt_found = True
            break

    ckpt_found or __raise('no such checkpoint file')
          
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + ckpt_file, network=net)
        
        for idx in range(0,len(valid_lf_extras), batch_size):
            start_time = time.time()  

            recon = sess.run(net.outputs, {t_image : valid_lf_extras[idx:idx+batch_size]})
            print("time elapsed : %4.4fs " % (time.time() - start_time))
            
            write3d(recon, save_dir+'net_%s_%06d_epoch%d.tif' % (config.label, idx, epoch))
            #write3d(out, save_dir+'/epoch{}_{:0>4}.tif'.format(epoch, idx//batch_size))
            print('writing %d / %d ...' % (idx + 1, len(valid_lf_extras)))

          
    
 
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument("-r", "--recursive", help="recursively eval all images under config.VALID.lf2d_path and its sub-folders",
                        action="store_true") #if the option is specified, assign True to args.recursive. Not specifying it implies False.

    parser.add_argument("--cpu", help="use CPU instead of GPU for inference",
                        action="store_true") 
    
    args = parser.parse_args()
    ckpt = args.ckpt
    batch_size = args.batch
    use_cpu = args.cpu

    evaluate(ckpt, batch_size)
    
