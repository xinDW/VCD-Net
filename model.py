import tensorflow as tf
import numpy as np
import tensorlayer as tl

from tensorlayer.layers import InputLayer, Conv2d, Conv3dLayer, UpSampling2dLayer, SubpixelConv2d, ElementwiseLayer, BatchNormLayer, ConcatLayer
from config import *
from custom import *

__all__ = [
    'LapSRN',
    'UNet'
    ]

#img_size = config.img_size * np.array(config.size_factor)
#img_height, img_width = img_size

w_init = tf.random_normal_initializer(stddev=0.02)
b_init = None
g_init = tf.random_normal_initializer(1., 0.02)

def conv2d(layer, n_filter, filter_size=3, stride=1, act=tf.identity, W_init=w_init, b_init=b_init, name = 'conv2d'):
    return tl.layers.Conv2d(layer, n_filter=int(n_filter), filter_size=(filter_size, filter_size), strides=(stride, stride), act=act, padding='SAME', W_init=W_init, b_init=b_init, name=name)
        
def conv3d(layer, 
    act=tf.identity, 
    filter_shape=(2,2,2,3,32),  #Shape of the filters: (filter_depth, filter_height, filter_width, in_channels, out_channels)
    strides=(1, 1, 1, 1, 1), W_init=w_init, b_init=b_init, name='conv3d'): 
    
    return tl.layers.Conv3dLayer(layer, act=act, shape=filter_shape, strides=strides, padding='SAME', W_init=W_init, b_init=b_init, W_init_args=None, b_init_args=None, name=name)

def deconv2d(layer, out_channels, filter_size=3, stride=2, out_size=None, act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='deconv2d'):
    """
    up-sampling the layer in height and width by factor 2
    Parames:
        shape - shape of filter : [height, width, out_channels, in_channels]
        out_size : height and width of the outputs 
    """
    batch, h, w, in_channels = layer.outputs.get_shape().as_list()   
    filter_shape = (filter_size, filter_size, int(out_channels), int(in_channels))
    if out_size is None:
        output_shape = (batch, int(h * stride), int(w * stride), int(out_channels))
    else :
        output_shape = (batch, out_size[0], out_size[1], int(out_channels))
    strides = (1, stride, stride, 1)
    return tl.layers.DeConv2dLayer(layer, act=act, shape=filter_shape, output_shape=output_shape, strides=strides, padding=padding, W_init=W_init, b_init=b_init, W_init_args=None, b_init_args=None, name=name)

def merge(layers):
    '''
    merge two Layers by element-wise addition
    Params : 
        -layers : list of Layer instances to be merged : [layer1, layer2, ...]
    '''
    return tl.layers.ElementwiseLayer(layers, combine_fn=tf.add, name='merge')
    
def batch_norm(layer, act=tf.identity, is_train=True, gamma_init=g_init, name='bn'): 
    return tl.layers.BatchNormLayer(layer, act=act, is_train=is_train, gamma_init=gamma_init, name=name)
    
                               

def UpConv(layer, out_channels, filter_size=4, factor=2, name='upconv'):
    with tf.variable_scope(name):
        n = tl.layers.UpSampling2dLayer(layer, size=(factor, factor), is_scale=True, method=1, name = 'upsampling')
        '''
        - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
        - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
        - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
        - Index 3 ResizeMethod.AREA, Area interpolation.
        '''
        n = conv2d(n, n_filter=out_channels, filter_size=filter_size, name='conv1')
        return n

#=====================================================
# LapSRN (Laplacian Pyramid Nets for Super-Resolution)
#=====================================================
def conv_block(input, name):
    '''
    a sequential of conv layers. 
    Params :
        -input : a TensorLayer.Layer instance
    return : a Layer instance
    '''
    n_c = 64
    k_size = 3
    n_conv_layers = 10
    def _conv_lrelu(x, name):
        return conv2d(x, n_filter=n_c, filter_size=k_size, stride=1, act=tf.nn.leaky_relu, name = name)

    with tf.variable_scope(name):
        x = _conv_lrelu(input, 'conv1')
        for i in range(1, n_conv_layers):
            x = _conv_lrelu(x, 'conv%d' % (i + 1))
        x = deconv2d(x, out_channels=n_c, filter_size=4, stride=2, act=tf.nn.leaky_relu, name='deconv')
        return x

def LapSRN(lf_extra, n_slices,  name='lapsrn'):
    '''
    Implemtation of LapSRN, to reconstruct a light filed image (with different views re-arranged in 'channels' dimension) 
    into a 3-D object
    Params:
      lf_extra: [batch, height, width, channels=Nnum^2]
    '''
    act = tf.nn.leaky_relu
    with tf.variable_scope(name):
        with tf.variable_scope('s1'):
            n = InputLayer(lf_extra, 'lf_extra')
            n1 = conv2d(n, n_filter=64, filter_size=4, act=act, name='conv0') 

            convt_f1 = conv_block(n1, name='conv_block1')
            convt_r1 = conv2d(convt_f1, n_filter=n_slices, filter_size=3, name='conv1')  
            convt_i1 = deconv2d(n, out_channels=n_slices, filter_size=4, stride=2, name='deconv1')
            sr_2x = merge([convt_r1, convt_i1])

        with tf.variable_scope('s2'):
            convt_f2 = conv_block(convt_f1, name='conv_block2')
            convt_r2 = conv2d(convt_f2, n_filter=n_slices, filter_size=3, name='conv2') 
            convt_i2 = deconv2d(sr_2x, out_channels=n_slices, filter_size=4, stride=2, name='deconv2')
            sr_4x = merge([convt_r2, convt_i2])

        with tf.variable_scope('s3'):
            convt_f3 = conv_block(convt_f2, name='conv_block3')
            convt_r3 = conv2d(convt_f3, n_filter=n_slices, filter_size=3, name='conv3') 
            convt_i3 = deconv2d(sr_4x, out_channels=n_slices, filter_size=4, stride=2, name='deconv3')
            sr_8x = merge([convt_r3, convt_i3])
            sr_8x = conv2d(sr_8x, n_filter=n_slices, filter_size=3, act=tf.tanh, name='out')

        return sr_2x, sr_4x, sr_8x


def UNet(lf_extra, n_slices, img_size, is_train=True, reuse=False, name='unet'):
    '''
    # number of interpolations before output sizes reach config.img_size
    n_interp = 1; 
    while h * (2 ** n_interp) < img_size[0]:
        n_interp += 1;
    '''    
    n_interp = 4
    # _, w, h, _ = lf_extra.shape
    #channels_interp = in_channels.value
    channels_interp = 128
    act = tf.nn.relu
        
    with tf.variable_scope(name, reuse=reuse):
        n = InputLayer(lf_extra, 'lf_extra')
        n = conv2d(n, n_filter=channels_interp, filter_size=7, name='conv1')
        ## Up-scale input 
        for i in range(n_interp):
            channels_interp = channels_interp / 2
            n = SubpixelConv2d(n, scale=2, name='interp/subpixel%d' % i)
            n = conv2d(n, n_filter=channels_interp, filter_size=3, name='interp/conv%d' % i)
            #n = batch_norm(n, act=tf.nn.relu, is_train=is_train, name='interp/bn%d' % i)
            
        n = conv2d(n, n_filter=channels_interp, filter_size=3, act=act, name='interp/conv_final') # 176*176
        n = batch_norm(n, is_train=is_train, name='interp/bn_final')
        #n = UpSampling2dLayer(n, size=(img_size[0], img_size[1]), is_scale=False, name = 'interp/upsampling_final')
        
        pyramid_channels = [128, 256, 512, 512, 512] # output channels number of each conv layer in the encoder
        encoder_layers = []
        with tf.variable_scope('encoder'):
            n = conv2d(n, n_filter=64, filter_size=3, stride=2, name='conv0')
            
            for idx, nc in enumerate(pyramid_channels):
                encoder_layers.append(n) # append n0, n1, n2, n3, n4 (but without n5)to the layers list
                print('encoder %d : %s' % (idx, str(n.outputs.get_shape())))
                n = LReluLayer(n, name='lreu%d' % (idx + 1))
                n = conv2d(n, n_filter=nc, filter_size=3, stride=2, name='conv%d' % (idx + 1)) 
                n = batch_norm(n, is_train=is_train, name='bn%d' % (idx + 1))

        nl = len(encoder_layers)        
        with tf.variable_scope('decoder'):
            _, h, w, _ = encoder_layers[-1].outputs.shape.as_list()
            n = ReluLayer(n, name='relu1')
            n = deconv2d(n, pyramid_channels[-1], out_size=(h, w), padding='SAME', name='deconv1')
            n = batch_norm(n, is_train=is_train, name='bn1')
            
            for idx in range(nl - 1, -1, -1): # idx = 4,3,2,1,0
                if idx > 0:
                    _, h, w, _ = encoder_layers[idx - 1].outputs.shape.as_list()
                    out_size = (h, w)
                    out_channels = pyramid_channels[idx-1]
                else:
                    out_size = None
                    out_channels = n_slices

                print('decoder %d : %s' % (idx, str(n.outputs.get_shape())))
                n = ConcatLayer([encoder_layers[idx], n], concat_dim=-1, name='concat%d' % (nl - idx))
                n = ReluLayer(n, name='relu%d' % (nl - idx + 1))
                #n = UpConv(n, 512, filter_size=4, factor=2, name='upconv2')
                n = deconv2d(n, out_channels, out_size = out_size, padding='SAME', name='deconv%d' % (nl - idx + 1))
                n = batch_norm(n, is_train=is_train, name='bn%d' % (nl - idx + 1))
                #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='dropout1')

            if n.outputs.shape[1] != img_size[0]:
                n = UpSampling2dLayer(n, size=img_size, is_scale=False, name = 'resize_final')
           
            n.outputs = tf.tanh(n.outputs)
            #n = conv2d(n, n_filter=n_slices, filter_size=3, act=tf.tanh, name='out')  
            return n
    
def forward_projection(recon3d, psf, mask, n_num=11):
    '''
    tensorflow implementation of light field imaging process, i.e. the forward projection
    params:
        recon3d - A tensor that represents 3-D image in shape of [batch, height, width, channels=n_slices]
        psf     - A tensor in shape of [n_num, n_num, n_slices, psf_size, psf_size, 1, 1]
        mask    - A tensor aimed at extracting views from recon3d, [img_size, img_size, n_num, n_num]
    return: 
        light field projection 
    '''
 
    batch, height, width, n_slices = recon3d.shape
    projection = tf.zeros([batch, height, width, 1])
    with tf.variable_scope('forward_projection', reuse=True):
        '''
        for i in range(n_num):
            for j in range(n_num):
                tmp_mask = mask[:,:,i,j]
                for d in range(n_slices):
                    tmp_img = recon3d[:, :, :, d]
                    
                    tmp_img = tmp_img * tmp_mask
                    tmp_img = tf.expand_dims(tmp_img, axis=-1)
                    tmp_psf = psf[i, j, d, ...]  # [psf_size, psf_size, in_channels=1, out_channels=1]
                    tmp_proj = tf.nn.conv2d(tmp_img, tmp_psf, strides=[1,1,1,1], padding='SAME')
                    projection = projection + tmp_proj
        '''
        for d in range(n_slices):
            tmp_img0 = recon3d[:,:,:,d]
            for i in range(n_num):
                for j in range(n_num):
                    tmp_mask = mask[:,:,i,j]
                    tmp_img = tmp_img0 * tmp_mask
                    tmp_img = tf.expand_dims(tmp_img, axis=-1)
                    tmp_psf = psf[i,j,d,...]
                    tmp_proj = tf.nn.conv2d(tmp_img, tmp_psf, strides=[1,1,1,1], padding='SAME')
                    
                    projection = projection + tmp_proj
                    
        return projection
        
"""
@deprecated

def discriminator3d(input_images, is_train=True, reuse=False):
    
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    #layer, n_filter, filter_size=3, strides=(1,1), act=tf.identity, 
    with tf.device('/device:GPU:1'):
        with tf.variable_scope("discriminator3d", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            net_in = InputLayer(input_images, name='input/images')
            net_h0 = conv2d(net_in, df_dim, 4, 2, act=lrelu, name='h0/c')

            net_h1 = conv2d(net_h0, df_dim*2, 4, 2, act=None, name='h1/c')
            net_h1 = batch_norm(net_h1, act=lrelu, is_train=is_train, name='h1/bn')
            net_h2 = conv2d(net_h1, df_dim*4, 4, 2, act=None, name='h2/c')
            net_h2 = batch_norm(net_h2, act=lrelu, is_train=is_train, name='h2/bn')
            net_h3 = conv2d(net_h2, df_dim*8, 4, 2, act=None, name='h3/c')
            net_h3 = batch_norm(net_h3, act=lrelu, is_train=is_train, name='h3/bn')
            net_h4 = conv2d(net_h3, df_dim*16, 4, 2, act=None, name='h4/c')
            net_h4 = batch_norm(net_h4, act=lrelu, is_train=is_train, name='h4/bn')
            net_h5 = conv2d(net_h4, df_dim*32, 4, 2, act=None, name='h5/c')
            net_h5 = batch_norm(net_h5, act=lrelu, is_train=is_train, name='h5/bn')
            net_h6 = conv2d(net_h5, df_dim*16, 1, 1, act=None, name='h6/c')
            net_h6 = batch_norm(net_h6, act=lrelu, is_train=is_train, name='h6/bn')
            net_h7 = conv2d(net_h6, df_dim*8, 1, 1, act=None, name='h7/c')
            net_h7 = batch_norm(net_h7, is_train=is_train, name='h7/bn')

            net = conv2d(net_h7, df_dim*2, 1, 1, act=None, name='res/c')
            net = batch_norm(net, act=lrelu, is_train=is_train, name='res/bn')
            net = conv2d(net, df_dim*2, 3, 1, act=None, name='res/c2')
            net = batch_norm(net, act=lrelu, is_train=is_train, name='res/bn2')
            net = conv2d(net, df_dim*8, 3, 1, act=None, name='res/c3')
            net = batch_norm(net, is_train=is_train, name='res/bn3')
            net_h8 = ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='res/add')
            net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

            net_ho = FlattenLayer(net_h8, name='ho/flatten')
            net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init = w_init, name='ho/dense')
            logits = net_ho.outputs
            net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

            return net_ho, logits




def residual_block(x, reuse=False, is_train=True, name='res_block'):
    n_channels = x.outputs.get_shape().as_list()[-1]
    with tf.variable_scope(name, reuse=reuse):
        
        shortcut = x;
        
        x = conv2d(x, n_filter=n_channels, filter_size=(1,1), strides=(1,1), name="conv1")
        x = batch_norm(x, act=tf.nn.relu, is_train=is_train, name='bn1')
        x = conv2d(x, n_filter=64, filter_size=3, strides=(1,1), name="conv2")
        x = batch_norm(x, act=tf.nn.relu, is_train=is_train, name='bn2')
        x = conv2d(x, n_filter=n_channels, filter_size=(1,1), strides=(1,1), name="conv3")
        x = batch_norm(x, act=tf.nn.relu, is_train=is_train, name='bn3')
        
        x = ElementwiseLayer([x, shortcut], combine_fn=tf.add, name='jump_connection')
        
    return x

def residual_block3d(x, filter_size=3, reuse=False, is_train=True, name='res_block'):
    n_channels = x.outputs.get_shape().as_list()[-1]
    with tf.variable_scope(name, reuse=reuse):
        
        shortcut = x;
       
        x = conv3d(x, filter_shape=(1,filter_size,filter_size,n_channels,n_channels), name="conv1")
        x = batch_norm(x, act=tf.nn.relu, is_train=is_train, name='bn1')
        x = conv3d(x, filter_shape=(1,filter_size,filter_size,n_channels,n_channels), name="conv2")
        x = batch_norm(x, act=tf.nn.relu, is_train=is_train, name='bn2')
        
        x = ElementwiseLayer([x, shortcut], combine_fn=tf.add, name='jump_connection')
        
    return x
                                  
def LFRNet3(lf_extra, psf_config, is_train=False, reuse=False):

    '''
    Params:
        lf_extra : extracted light field image with dimension [batch, height=img_height/Nnum, width=img_width/Nnum, channels=Nnum**2]
    Return:
        out [batch, height=img_height, width=img_width, channels=n_slices]
    '''
    psf_size = psf_config.psf_size;
    n_slices = psf_config.n_slices;
    n_num = psf_config.Nnum
    
    batch, w, h, in_channels = lf_extra.shape
    
    print(lf_extra.shape)
    n_interp = 1; # number of interpolations before output sizes reach config.img_size
    while h * (2 ** n_interp) < img_height / 2:
        n_interp += 1;
        
    with tf.variable_scope("LFRNet3", reuse=reuse):
                
        n = InputLayer(lf_extra) # [batch, depth, height, width, channels] 
        n = conv2d(n, n_filter=64, filter_size=7, name='conv1')
        n = batch_norm(n, act=tf.nn.relu, is_train=is_train, name='bn1')
        n = conv2d(n, n_filter=128, filter_size=3, name='conv2')
        n = batch_norm(n, act=tf.nn.relu, is_train=is_train, name='bn2')
        n = conv2d(n, n_filter=256, filter_size=3, name='conv3')
        n = batch_norm(n, act=tf.nn.relu, is_train=is_train, name='bn3')
       
        shortcut = n
        for i in range(16):
            n = residual_block(n, reuse=reuse, is_train=is_train, name='res_block%d' % i)
        
        n = ElementwiseLayer([n, shortcut], combine_fn=tf.add, name='add')
        
        channels_interp = n.outputs.get_shape().as_list()[-1]
        
        for i in range(n_interp):
            channels_interp = channels_interp / 2;
            h = h * 2;
            w = w * 2;
            #n = deconv2d(n, out_channels=channels_interp, filter_size=3, out_size=(h, w), name = 'interp/deconv%d' % i)
            n = SubpixelConv2d(n, scale=2, name='interp/subpixel%d' % i)
            n = conv2d(n, n_filter=channels_interp, filter_size=3, name='interp/conv%d' % i)
            n = batch_norm(n, act=tf.nn.relu, is_train=is_train, name='interp/bn%d' % i)
        
        n = conv2d(n, n_filter=n_slices*2, filter_size=3, name='interp/conv_final')
        n = batch_norm(n, act=tf.nn.relu, is_train=is_train, name='interp/bn_final')
        #n = deconv2d(n, out_channels=n_slices, filter_size=3, out_size=(img_height, img_width), name = 'interp/deconv_final')
        n = UpSampling2dLayer(n, size=(img_height, img_width), is_scale=False, name = 'interp/upsampling_final')
        
        out =  conv2d(n, n_filter=n_slices, filter_size=(1,1), act=tf.tanh, name='out')
        return out
        
        
def LFRNet(lf_extra, psf_config, is_train=False, reuse=False):

    '''
    Params:
        lf_extra : extracted light field image with dimension [batch, depth, height, width, channels]
    '''
    psf_size = psf_config.psf_size;
    Nnum = psf_config.Nnum;
    n_quarter = (int(Nnum / 2) + 1) ** 2;
    n_slices = psf_config.n_slices;
    batch, depth, h, w, channels = lf_extra.shape
            
    with tf.variable_scope("LFRNet", reuse=reuse):
        #n = InputLayer(lf_extra, name='lf_extra')
        dummy = tf.zeros([batch, 1, h, w, channels], dtype=tf.float32, name='dummy')
        for d in range(n_slices):
            with tf.variable_scope('z%d' % d, reuse=reuse):
                ## adjust psf_size according to z index
                if (d != 0) and (d % 3 == 0):
                    psf_size = psf_size - 22;
                    
                slice = tf.zeros([batch, h, w, channels], dtype=tf.float32, name='slice%d' % d)
                for i in range(n_quarter):
                    with tf.variable_scope('n%d' % i, reuse = reuse):
                        '''
                        n = tf.layers.conv2d(lf2d, filters=16, kernel_size=(psf_size,psf_size)) # could be replaced with split-transfrom-merge strategy
                        n = tf.contrib.layers.batch_norm(n, is_train=is_train, reuse=reuse, scope=)
                        n = tf.layers.conv2d(n, filters=4, kernel_size=3)
                        n = tf.layers.conv2d(n, filters=1, kernel_size=3)
                        '''
               
                        n = InputLayer(lf_extra[:, i, :, :, :]) # 2-D conv is performed, so data format of [batch, height, width, channels] is required
                        n = conv2d(n, n_filter=16, filter_size=(psf_size, psf_size), name='conv1')
                        n = batch_norm(n, is_train=is_train, name='bn1')
                        n = conv2d(n, n_filter=32, filter_size=3, name='conv2')
                        n = batch_norm(n, is_train=is_train, name='bn2')
                        n = conv2d(n, n_filter=64, filter_size=3, name='conv3')
                        n = batch_norm(n, is_train=is_train, name='bn3')
                        
                        n = residual_block(n, reuse=reuse, is_train=is_train, name='res_block1')
                        
                        n = conv2d(n, n_filter=1, filter_size=3, name='final_conv')
                        slice = slice + n.outputs #[batch, h, w, c]
                    
                dummy = tf.concat([dummy, tf.expand_dims(slice, axis=1)], axis=1)  # expand the fromat of slice into [b, 1, h, w, c] and concat it to the stack in 'depth' dimension : [batch, depth, height, width, channels]  
            
        out = InputLayer(dummy[:, 1:, :, :, :], 'out') #exclude the first all-zero slice and return [b, d, h, w, c]
        return out        

def LFRNet2(lf_extra, psf_config, is_train=False, reuse=False):

    '''
    Params:
        lf_extra : light field 2-D image with dimension [batch, height, width, 1]
    '''
    psf_size = psf_config.psf_size;
    n_slices = psf_config.n_slices;
    batch, h, w, in_channels = lf_extra.shape
            
    with tf.variable_scope("LFRNet2", reuse=reuse):
        
        dummy = InputLayer(tf.zeros([batch, 1, h, w, 1], dtype=tf.float32, name='dummy'), name='dummy')
        for d in range(n_slices):
            with tf.variable_scope('z%d' % d, reuse=reuse):
                ## adjust psf_size according to z index
                #if (d != 0) and (d % 3 == 0) and (psf_size - 22 > 0):
                #    psf_size = psf_size - 22;
                
                n = InputLayer(lf_extra) # [batch, depth, height, width, channels] 
                
                n = conv2d(n, n_filter=64, filter_size=7, name='conv1')
                n = batch_norm(n, is_train=is_train, name='bn1')
                n = conv2d(n, n_filter=128, filter_size=3, name='conv2')
                n = batch_norm(n, is_train=is_train, name='bn2')
                n = conv2d(n, n_filter=256, filter_size=3, name='conv3')
                n = batch_norm(n, is_train=is_train, name='bn3')
                
               
                for i in range(4):
                    n = residual_block(n, reuse=reuse, is_train=is_train, name='res_block%d' % i)
                
                n = conv2d(n, n_filter=1, filter_size=3, name='final_conv')
                
                n = ExpandDimsLayer(n, axis=1) # [b, 1, h, w, c=1]
                dummy = ConcatLayer([dummy, n], concat_dim=1)  # concat the current slice to the stack in 'depth' dimension : [batch, depth, height, width, channels]  
            
        #out = InputLayer(dummy[:, 1:, :, :, :], 'out') #exclude the first all-zero slice and return [b, d, h, w, c]
        out = LambdaLayer(dummy, lambda x : x[:,1:,:,:,:], name='out')
        return out  
"""        