import time
import os
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt

# from model import UNet_A, UNet_B, AtrousUNet, RDN
from model import *
from model.util.losses import *
from dataset import Dataset
from utils import write3d
from config import config
#from train_multi_gpu import MutilDeviceTrainer

###====================== HYPER-PARAMETERS ===========================###
img_size   = config.img_size * np.array(config.size_factor) # a numpy array, not a python list, cannot be concated with other list [] by "+"
n_slices   = config.PSF.n_slices
n_num      = config.PSF.Nnum
base_size  = img_size // n_num # lateral size of lf_extra

batch_size = config.TRAIN.batch_size
lr_init    = config.TRAIN.lr_init
beta1      = config.TRAIN.beta1
#n_devices  = config.TRAIN.device_nums
## learning
n_epoch     = config.TRAIN.n_epoch
lr_decay    = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

label                = config.label
test_saving_dir      = config.TRAIN.test_saving_path
checkpoint_dir       = config.TRAIN.ckpt_dir
ckpt_saving_interval = config.TRAIN.ckpt_saving_interval
log_dir              = config.TRAIN.log_dir

normalize_mode    = 'max'
using_edge_loss   = config.TRAIN.using_edge_loss
using_binary_loss = config.TRAIN.using_binary_loss
# using_vgg_loss  = config.TRAIN.using_vgg_loss

def __raise(e):
    raise(e)

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

class Trainer:   
    def __init__(self, dataset):
        self.dataset = dataset
        self.losses = {}

    def build_graph(self):
        ###========================== DEFINE MODEL ============================###
        with tf.variable_scope('learning_rate'):
            self.learning_rate = tf.Variable(lr_init, trainable=False)

        self.plchdr_lf = tf.placeholder('float32', [batch_size, base_size[0], base_size[1], n_num ** 2], name='t_lf_extra_input')
        # self.plchdr_lf = tf.placeholder('float32', [batch_size, img_size[0], img_size[1], 1], name='t_lf_input')
        self.plchdr_target3d = tf.placeholder('float32', [batch_size, img_size[0], img_size[1], n_slices], name='t_target3d')
        #self.plchdr_target3d = tf.placeholder('float32', [batch_size, base_size[0] * 8, base_size[1]* 8, n_slices], name='t_target3d')

        vars_tag = 'vcdnet'

        with tf.device('/gpu:{}'.format(config.TRAIN.device)):
            self.net = UNet_B(self.plchdr_lf, n_slices=n_slices, output_size=img_size, is_train=True, reuse=False, name=vars_tag)
           
            

        self.net.print_params(False)
        g_vars = tl.layers.get_variables_with_name(vars_tag, train_only=True, printable=True)

        #====================
        #loss function
        #=====================
        loss_ln    = l2_loss(self.plchdr_target3d, self.net.outputs)
        self.loss  = loss_ln
        self.loss_test = loss_ln
        self.losses.update({'ln_loss': loss_ln})

        if using_edge_loss:
            loss_edges = 1e-1 * (self.net.outputs, self.plchdr_target3d)
            self.loss += loss_edges
            self.losses.update({'edge_loss': loss_edges})
            

        if using_binary_loss:
            self.plchdr_mask3d = tf.placeholder('float32', [batch_size, img_size[0], img_size[1], n_slices], name='t_mask3d')
            target = self.plchdr_target3d
            vcd_out = self.net.outputs

            target_fg = tf.math.multiply(target, self.plchdr_mask3d)
            target_bg = tf.math.subtract(target, target_fg)
            vcd_fg = tf.math.multiply(vcd_out, self.plchdr_mask3d)
            vcd_bg = tf.math.subtract(vcd_out, vcd_fg)

            fg_loss = 10 * l2_loss(target_fg, vcd_fg)
            bg_loss = l2_loss(target_bg, vcd_bg)
            self.loss = fg_loss + bg_loss
            self.loss_test = fg_loss + bg_loss
            self.losses.update({'fg_loss': fg_loss})
            self.losses.update({'bg_loss': bg_loss})

        # if using_vgg_loss:
        #     vgg_loss       = self._get_vgg_loss(self.net.outputs, self.plchdr_target3d)
        #     self.losses.update({'vgg_loss' : vgg_loss})
        #     self.loss += vgg_loss

       

        configProto = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        configProto.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=configProto)
        self._make_summaries()
        self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.loss, var_list=g_vars)
        

    def _train(self, begin_epoch):
        """Train the VCD-Net

        Params
            -begin_epoch: int, if not 0, a checkpoint file will be loaded and the training will continue from there
        """
        ## create folders to save result images and trained model
        save_dir = test_saving_dir
        tl.files.exists_or_mkdir(save_dir)
        tl.files.exists_or_mkdir(checkpoint_dir)
        tl.files.exists_or_mkdir(log_dir)
        
        
        self.sess.run(tf.global_variables_initializer())
        
      
        if (begin_epoch != 0):
            ckpt_file = [filename for filename in os.listdir(checkpoint_dir) if ('.npz' in filename and str(begin_epoch) in filename) ] 
            if len(ckpt_file) == 0 or tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(checkpoint_dir, ckpt_file[0]), network=self.net) is False:
                raise(Exception('falied to load % s' % '{}/vcdnet_epoch{}.npz'.format(label, begin_epoch)))
        # else:
        #     self._find_available_ckpt(n_epoch)
        
        self.sess.run(tf.assign(self.learning_rate, lr_init))
        
        # if using_vgg_loss:
        #     self._load_vgg_model(sess=self.sess)    

        ### load data
        dataset_size     = self.dataset.prepare(batch_size)
        
        self._write_test_samples()
        

        fetches = self.losses
        fetches['optim'] = self.optim
        fetches['batch_summary'] = self.merge_op

        for epoch in range(begin_epoch, n_epoch):
            self._update_learning_rate(epoch, decay_every, lr_decay, lr_init)
            
            for HR_batch, LF_batch, cursor in self.dataset.data():
                step_time = time.time()
                if using_binary_loss:
                    feed_dict = {self.plchdr_lf : LF_batch, self.plchdr_target3d : HR_batch[0], self.plchdr_mask3d : HR_batch[1]}
                else:
                    feed_dict = {self.plchdr_lf : LF_batch, self.plchdr_target3d : HR_batch[0]}
                
                evaluated = self.sess.run(fetches, feed_dict)
                print("\rEpoch:[%d/%d] iter:[%d/%d] time: %4.3fs, " % (epoch, n_epoch, cursor, dataset_size, time.time() - step_time), end="")
                
                losses_val = {name : value for name, value in evaluated.items() if 'loss' in name}
                print(losses_val, end='')

                self.summary_writer.add_summary(evaluated['batch_summary'], epoch * (dataset_size // batch_size) + cursor / batch_size)

            self._record_avg_test_loss(epoch, self.sess)
            if epoch != 0 and (epoch%ckpt_saving_interval == 0):
                self._save_intermediate_ckpt(epoch, self.sess)

        '''
        final_cursor     = (dataset_size // batch_size - 1) * batch_size
        while self.dataset.hasNext():
            step_time = time.time()

            if not using_binary_loss:
                HR_batch, LF_batch, cursor, epoch = self.dataset.iter()
                feed_dict = {self.plchdr_lf : LF_batch, self.plchdr_target3d : HR_batch}
            else:
                HR_batch, mask_batch, LF_batch, cursor, epoch = self.dataset.iter()
                feed_dict = {self.plchdr_lf : LF_batch, self.plchdr_target3d : HR_batch, self.plchdr_mask3d : mask_batch}

            epoch += begin_epoch
            self._update_learning_rate(epoch, decay_every, lr_decay, lr_init)

            evaluated = self.sess.run(fetches, feed_dict)

            if cursor == final_cursor:
                self._record_avg_test_loss(epoch, self.sess)
                if epoch != 0 and (epoch%ckpt_saving_interval == 0):
                    self._save_intermediate_ckpt(epoch, self.sess)


            print("\rEpoch:[%d/%d] iter:[%d/%d] time: %4.3fs, " % (epoch, n_epoch, cursor, dataset_size, time.time() - step_time), end="")
            losses_val = {name : value for name, value in evaluated.items() if 'loss' in name}
            print(losses_val, end='')

            self.summary_writer.add_summary(evaluated['batch_summary'], epoch * (dataset_size // batch_size - 1) + cursor / batch_size)
        '''

    def _update_learning_rate(self, epoch, decay_every, lr_decay, lr_init):
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            self.sess.run(tf.assign(self.learning_rate, lr_init * new_lr_decay))
            print('\nlearning rate updated : %f\n' % (lr_init * new_lr_decay))

    def _make_summaries(self):
        for key, val in self.losses.items():
            tf.summary.scalar(key, val)

        self.merge_op       = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)

    def _write_test_samples(self):
        for hr_batch, lf_batch, _ in self.dataset.for_test():
            write3d(hr_batch[0], test_saving_dir+'/target3d.tif',bitdepth=8) 
            write3d(lf_batch, test_saving_dir+'/lf_extra.tif',bitdepth=8) 
            if using_binary_loss:
                write3d(hr_batch[1], test_saving_dir+'/mask3d.tif',bitdepth=8) 
            break

    def _save_intermediate_ckpt(self, tag, sess):
    
        tag = ('epoch%d' % tag) if is_number(tag) else tag

        npz_file_name = checkpoint_dir+'/vcdnet_{}.npz'.format(tag)
        tl.files.save_npz(self.net.all_params, name=npz_file_name, sess=sess)

        for _, lf_batch, _ in self.dataset.for_test():
            out = self.sess.run(self.net.outputs, {self.plchdr_lf : lf_batch})
            write3d(out, test_saving_dir+'test_{}.tif'.format(tag), bitdepth=8)
            break


    def _record_avg_test_loss(self, epoch, sess):

        print()
        test_loss = 0
        n_batchs = 0
        for hr_batch, lr_batch, idx in self.dataset.for_test():
            feed_test = {self.plchdr_lf : lr_batch, self.plchdr_target3d : hr_batch[0]}
            if using_binary_loss:
                feed_test.update({self.plchdr_mask3d : hr_batch[1]})

            test_loss_batch = sess.run(self.loss_test, feed_test)
            test_loss += test_loss_batch
            n_batchs += 1

        test_loss /= n_batchs     

        if 'min_test_loss' not in dir(self):
            self.min_test_loss = test_loss
            self.best_epoch    = epoch
            self.test_loss_plt = []

        if (test_loss < self.min_test_loss):
            self.min_test_loss = test_loss
            self.best_epoch    = epoch
            self._save_intermediate_ckpt(tag='best', sess=sess)
            # self._save_pb(sess)

        print('[validation] loss = %.6f best = %.6f (@epoch%d)' % (test_loss, self.min_test_loss, self.best_epoch))
        self.test_loss_plt.append([epoch, test_loss])


    def _plot_test_loss(self):
        loss = np.asarray(self.test_loss_plt)
        plt.figure()
        plt.plot(loss[:, 0], loss[:, 1])
        
        plt.savefig(test_saving_dir + 'test_loss.png', bbox_inches='tight')
        plt.show()

    def train(self, **kwargs):
        try:
            self._train(**kwargs)
        finally:
            self._plot_test_loss()   
        


    def _find_available_ckpt(self, end):
        begin = end
        while not os.path.exists(checkpoint_dir+'/vcdnet_epoch{}.npz'.format(begin)):
            begin -= 10
            if begin < 0:
                return 0
        print('\n\ninit ckpt found at epoch %d\n\n' % begin)        
        tl.files.load_and_assign_npz(sess=self.sess, name=checkpoint_dir+'/vcdnet_epoch{}.npz'.format(begin), network=self.net) 
        return begin
        
    def transfor_learning(self,**kwargs):
        try:
            self._train(**kwargs)
        finally:
            self._plot_test_loss()
    '''
    def _build_vgg(self, input):
        if 'net_vgg' not in dir(self):
            self.net_vgg = Vgg19_simple_api(input, reuse=False)
            return self.net_vgg
        else:
            return Vgg19_simple_api(input, reuse=True)

    def _get_vgg_loss(self, pred, target):
        n = pred.shape.as_list()[-1]
        mid = n // 2
        feat_p = self._build_vgg(pred[..., mid : mid + 1])
        feat_t = self._build_vgg(target[..., mid : mid + 1])
        loss =  1e-2 * tl.cost.mean_squared_error(feat_p.outputs, feat_t.outputs, is_mean=True)

        # loss = 0 
        # for i in range(n):
        #     feat_p = self._build_vgg(pred[..., i : i + 1])
        #     feat_t = self._build_vgg(target[..., i : i + 1])
        #     loss +=  tl.cost.mean_squared_error(feat_p.outputs, feat_t.outputs, is_mean=True)
        # loss *= 1. / n * 1e-2
        return loss

    def _load_vgg_model(self, sess):
        vgg19_npy_path = "model/vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(vgg19_npy_path, encoding='latin1').item()

        params = []
        for val in sorted( npz.items() ):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            
            if val[0] == 'conv1_1':
                W = W[:,:,0:1,:]
            if val[0] == 'conv5_1':
                break
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        tl.files.assign_params(sess, params, self.net_vgg)    
    '''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--ckpt', type=int, default=0, help='')
    args = parser.parse_args()
    
    training_dataset = Dataset(config.TRAIN.target3d_path, 
                            config.TRAIN.lf2d_path, 
                            n_slices, 
                            n_num, 
                            base_size, 
                            normalize_mode=normalize_mode,
                            bianry_mask=using_binary_loss)
    
    trainer = Trainer(training_dataset)
    trainer.build_graph()
    trainer.train(begin_epoch=args.ckpt)