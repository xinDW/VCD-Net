import time
import os
import tensorflow as tf
import tensorlayer as tl
import numpy as np

from model import UNet
from dataset import Dataset
from utils import *
from config import *

###====================== HYPER-PARAMETERS ===========================###
img_size = config.img_size * np.array(config.size_factor) # this is a numpy array, not a python list, cannot be concated with other list [] by "+"
n_slices = config.PSF.n_slices
n_num = config.PSF.Nnum
base_size = img_size // n_num # lateral size of lf_extra

n_channels = config.n_channels
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

## learning
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.ceil(np.sqrt(batch_size)))

series_input = config.TRAIN.series_input
time_series_len = 40

test_save_dir = config.TRAIN.test_saving_path

checkpoint_dir = config.TRAIN.ckpt_dir
ckpt_saving_interval = config.TRAIN.ckpt_saving_interval
log_dir = config.TRAIN.log_dir

    
def train(begin_epoch):
    ## create folders to save result images and trained model
    save_dir = test_save_dir
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir(checkpoint_dir)
    tl.files.exists_or_mkdir(log_dir)
    
    ###========================== DEFINE MODEL ============================###
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    t_lf_extra = tf.placeholder('float32', [batch_size, base_size[0], base_size[1], n_num ** 2], name='t_lf_extra_input')
    t_target3d = tf.placeholder('float32', [batch_size, img_size[0], img_size[1], n_slices], name='t_target3d')

    vars_tag = 'LFRNet'

    with tf.device('/gpu:{}'.format(config.TRAIN.device)):
        net = UNet(t_lf_extra, n_slices, img_size, is_train=True, name=vars_tag)

    net.print_params(False)   
    g_vars = tl.layers.get_variables_with_name(vars_tag, train_only=True, printable=True)

    #====================
    #loss function
    #=====================
    loss = tf.reduce_mean(tf.squared_difference(t_target3d, net.outputs))

    optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=g_vars)
    
    
    configProto = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    configProto.gpu_options.allow_growth = True
    
    sess = tf.Session(config=configProto)

    tf.summary.scalar('loss', loss)
    merge_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
       
    def __find_available_ckpt(end):
        begin = end
        while not os.path.exists(checkpoint_dir+'/{}_epoch{}.npz'.format(label, begin)):
            begin -= 10
            if begin < 0:
                return 0
        print('\n\ninit ckpt found at epoch %d\n\n' % begin)        
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/{}_epoch{}.npz'.format(label, begin), network=net) 
        return begin
        
    if (begin_epoch != 0):
      if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/{}_lfrnet_epoch{}.npz'.format(label, begin_epoch), network=net) is False:
        raise Exception('falied to load % s' % 'net_epoch{}.npz'.format(begin_epoch))
    else:
      __find_available_ckpt(n_epoch)
      
    sess.run(tf.assign(lr_v, lr_init))
    
    ###====================== LOAD DATA ===========================###
    training_dataset = Dataset(config.TRAIN.target3d_path, config.TRAIN.lf2d_path, n_slices, n_num, base_size)
    dataset_size = training_dataset.prepare(batch_size, n_epoch)
    
    test_target3d, test_lf_extra = training_dataset.for_test()
    write3d(test_target3d, save_dir+'/target3d.tif') 
    write3d(test_lf_extra, save_dir+'/lf_extra.tif') 
    

    while training_dataset.hasNext():
        step_time = time.time()
        HR_batch, LF_batch, cursor, epoch = training_dataset.iter()

        epoch += begin_epoch
        if epoch != 0 and (epoch % decay_every == 0) and cursor == batch_size:
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            print('\nlearning rate updated : %f\n' % (lr_init * new_lr_decay))

        error_ln,  _, batch_summary = sess.run([loss, optim, merge_op], {t_lf_extra : LF_batch, t_target3d : HR_batch})


        print("Epoch:[%d/%d] iter:[%d/%d] times: %4.3fs, loss:%.6f" % (epoch, n_epoch, cursor, dataset_size, time.time() - step_time, error_ln))
        summary_writer.add_summary(batch_summary, epoch * (dataset_size // batch_size - 1) + cursor / batch_size)

        if (epoch !=0) and (epoch%ckpt_saving_interval == 0) and (cursor == 0):
        #if (epoch%ckpt_saving_interval == 0) and (cursor == 0):   
            npz_file_name = checkpoint_dir+'/{}_lfrnet_epoch{}.npz'.format(label, epoch)
            tl.files.save_npz(net.all_params, name=npz_file_name, sess=sess)

            for idx in range(0, len(test_lf_extra), batch_size):
                if idx + batch_size <= len(test_lf_extra):
                    test_lr_batch = test_lf_extra[idx : idx + batch_size]
                    #test_hr_batch = test_target3d[idx : idx + batch_size]
                    out = sess.run(net.outputs, {t_lf_extra : test_lr_batch})
                    write3d(out, save_dir+'test_epoch{}_{}.tif'.format(epoch, idx))
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--ckpt', type=int, default=0, help='')
    
    args = parser.parse_args()
    train(args.ckpt)