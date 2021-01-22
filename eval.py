import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import time

from config import config

from model import *
# from model.util.tf_pb import save_graph_as_pb
from utils import *


n_num      = config.PSF.Nnum
n_slices   = config.PSF.n_slices
label      = config.label
n_filters  = config.n_base_filters
bitdepth   = config.VALID.bitdepth
normalize_mode = config.normalize_mode
normalize_fn = normalize_percentile if normalize_mode == 'percentile' else normalize

def __raise(info):
    raise Exception(info)

def read_valid_images(path):
    """return images in shape [n_images, height=img_size/n_num, width=img_size/n_num, channels=n_num**2]
    """

    def __cast(im, dtype=np.float32):
        return im if im.dtype is np.float32 else im.astype(np.float32, casting='unsafe')

    img_list = sorted(tl.files.load_file_list(path=path, regx='.*.tif', printable=False))

    img_set  = [__cast(lfread_norm(img_file, path, n_num=n_num, normalize_fn=normalize_fn)) for img_file in img_list]

    len(img_set) != 0 or __raise("none of the images have been loaded")
    
    print('read %d from %s' % (len(img_set), path)) 
    img_set = np.asarray(img_set)
    _, height, width, _ = img_set.shape
    
    return img_set, img_list, height, width


def infer(epoch, batch_size=1, use_cpu=False, save_mean=False, save_max=False):
    """ Infer the 3-D images from the 2-D LF images using the trained VCD-Net

    Params:
        -epoch     : int, the epoch number of the checkpoint file to be loaded
        -batch_size: int, batch size of the VCD-Net  
        -use_cpu   : bool, whether to use cpu for inference. If false, gpu will be used.
    """
    
    epoch = 'best' if epoch == 0 else epoch

    checkpoint_dir    = config.TRAIN.ckpt_dir
    lf_2d_path        = config.VALID.lf2d_path
    save_dir          = os.path.join(lf_2d_path, 'vcd-%s' % label)

    tl.files.exists_or_mkdir(save_dir)
    
    valid_lf_extras, names, height, width = read_valid_images(lf_2d_path)
    t_image = tf.placeholder('float32', [batch_size, height , width, n_num ** 2])
    # t_image = tf.placeholder('float32', [batch_size, height , width, 1])

    device_str = '/gpu:0' if not use_cpu else '/cpu:0'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    with tf.device(device_str): 
        net = UNet_B(t_image, n_slices, [height * n_num, width * n_num], n_base_filters=n_filters, reuse=False, name='unet')
        # net = UNet_C(t_image, n_slices, [height * n_num, width * n_num], use_bn=False, is_train=False, reuse=False, name='unet')

    ckpt_file = [filename for filename in os.listdir(checkpoint_dir) if ('.npz' in filename and str(epoch) in filename) ] 
    len(ckpt_file) > 0 or __raise('no such checkpoint file')

    output_node_name = 'vcd_output'
    output = tf.identity(net.outputs, output_node_name)
    
    # bitdepth = config.VALID.bitdepth
    # assert(bitdepth in [8, 16, 32])

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        tl.layers.initialize_global_variables(sess)
        ckpt_file = checkpoint_dir + '/' + ckpt_file[0] 
        print('loading %s' % ckpt_file)
        tl.files.load_and_assign_npz(sess=sess, name=ckpt_file, network=net)
        
        # save_graph_as_pb(sess=sess, 
        #                     output_node_names=output_node_name, 
        #                     output_graph_file=os.path.join(checkpoint_dir, 'vcd_best.pb'))

        #im_buffer        = np.zeros([len(valid_lf_extras), height * n_num, width * n_num, config.PSF.n_slices])
        im_buffer        = []
        recon_start_time = time.time() 
        recon_time       = 0

        
        for idx in range(0,len(valid_lf_extras), batch_size):
            start_time = time.time()  
            # write3d(valid_lf_extras[idx:idx+batch_size], os.path.join(save_dir, 'lfe-%s.tif' % (names[idx])), bitdepth=16)
            lfe = valid_lf_extras[idx:idx+batch_size]
            recon = sess.run(net.outputs, {t_image : lfe})
            batch_time = time.time() - start_time
            recon_time = recon_time + batch_time
            print("%s [%.3f, %.3f] time elapsed (sess.run): %4.4fs " % (names[idx], np.min(lfe), np.max(lfe), time.time() - start_time))
            # im_buffer.append(recon)
            if save_mean:
                write3d(np.mean(recon, axis=-1, keepdims=True), os.path.join(save_dir, 'vcd-mean-%s' % (names[idx])), bitdepth=bitdepth)
            elif save_max:
                write3d(np.max(recon, axis=-1, keepdims=True), os.path.join(save_dir, 'vcd-mip-%s' % (names[idx])), bitdepth=bitdepth)
            else:
                write3d(recon, os.path.join(save_dir, 'vcd-%s' % (names[idx])), bitdepth=bitdepth)

            # if idx == 0:
            #     save_activations(save_dir, sess=sess, final_layer=net, feed_dict={t_image : lfe}, verbose=False)

        print("\nrecon time elapsed (sess.run): %4.4fs " % (recon_time))
        
        # print('saving results ... ')
        # io_start_time = time.time()
        # for idx, im in enumerate(im_buffer):
        #     write3d(im, os.path.join(save_dir, 'vcd-%s' % (names[idx])), bitdepth=8)
        # print("IO time elapsed (imwrite): %4.4fs " % (time.time() - io_start_time))

    
          
    
 
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=1)
    # parser.add_argument("-r", "--recursive", help="recursively eval all images under config.VALID.lf2d_path and its sub-folders",
    #                     action="store_true") #if the option is specified, assign True to args.recursive. Not specifying it implies False.


    parser.add_argument("--max", help="save max-intensity-projection instead of 3-D stack",
                        action="store_true") 

    parser.add_argument("--mean", help="save mean-intensity-projection instead of 3-D stack",
                        action="store_true") 

    parser.add_argument("--cpu", help="use CPU instead of GPU for inference",
                        action="store_true") 
    
    args = parser.parse_args()
    
    infer(args.ckpt, batch_size=args.batch, use_cpu=args.cpu, save_max=args.max, save_mean=args.mean)
    


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