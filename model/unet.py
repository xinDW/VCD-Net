from .util.utils import *
import tensorlayer as tl
import tensorflow as tf

__all__ = ['UNet_A',
        'UNet_B',
        'UNet_C',
        ]

def upscale(layer, out_channels, out_size=None, scale=None, mode='upconv', name='upsale'):
    # if (out_size is None) and (scale is None):
    #     raise ValueError("At least one of out_size and scale must be non-None")

    if mode == 'upconv':
        if out_size is None:
            batch, height, width, _ = layer.outputs.get_shape()
            out_size = [int(height * scale), int(width * scale)]
        return upconv(layer, out_channels, out_size, filter_size=3, name=name)

    elif mode == 'deconv':
        return deconv2d(layer, out_channels=out_channels, out_size=out_size, name='%sdeconv' % name)
        
    elif mode == 'subpixel':
        if (scale is None):
            raise ValueError('scale cannot be None when mode==subpixel')

        n = SubpixelConv2d(layer, scale=scale, name='%s/subpixel' % name)
        return conv2d(n, n_filter=out_channels, filter_size=3, name='%s/conv' % name)
        
    else:
        raise ValueError('unknown mode: %s' % mode)

def conv(layer, n_filter, filter_size, stride=1, act=tf.identity, using_batch_norm=False, is_train=True, name='conv_block'):
    with tf.variable_scope(name):
        n = conv2d(layer, n_filter=n_filter, filter_size=filter_size, stride=stride, name='conv')
        if using_batch_norm:
            n = batch_norm(n, is_train=is_train, name='bn')
        n.outputs = act(n.outputs)
        return n


def UNet_A(lf_extra, n_slices, output_size, is_train=True, reuse=False, name='unet'):
    '''U-net based VCD-Net for light field reconstruction.
    Params:
        lf_extra: tf.tensor 
            In shape of [batch, height, width, n_num^2], the extracted views from the light field image
        n_slices: int
            The slices number of the 3-D reconstruction.
        output_size: list of int
            Lateral size of the 3-D reconstruction, i.e., [height, width].
        is_train: boolean 
            Sees tl.layers.BatchNormLayer.
        reuse: boolean 
            Whether to reuse the variables or not. See tf.variable_scope() for details.
        name: string
            The name of the variable scope.
    Return:
        The 3-D reconstruction in shape of [batch, height, width, depth=n_slices]
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
        with tf.variable_scope('interp'): 
            for i in range(n_interp):
                channels_interp = channels_interp / 2
                n = SubpixelConv2d(n, scale=2, name='interp/subpixel%d' % i)
                n = conv2d(n, n_filter=channels_interp, filter_size=3, name='conv%d' % i)
                
            n = conv2d(n, n_filter=channels_interp, filter_size=3, name='conv_final') # 176*176
            n = batch_norm(n, is_train=is_train, name='bn_final')
            n = ReluLayer(n, name='reul_final' )
        
        pyramid_channels = [128, 256, 512, 512, 512] # output channels number of each conv layer in the encoder
        encoder_layers = []
        with tf.variable_scope('encoder'):
            n = conv2d(n, n_filter=64, filter_size=3, stride=1, name='conv0')
            n = batch_norm(n, is_train=is_train, name='bn_0')
            n = ReluLayer(n, name='reul0' )
       

            for idx, nc in enumerate(pyramid_channels):
                encoder_layers.append(n) # append n0, n1, n2, n3, n4 (but without n5)to the layers list
                print('encoder %d : %s' % (idx, str(n.outputs.get_shape())))
                n = conv2d(n, n_filter=nc, filter_size=3, stride=1, name='conv%d' % (idx + 1))  
                n = batch_norm(n, is_train=is_train, name='bn%d' % (idx + 1))
                n = ReluLayer(n, name='reul%d' % (idx + 1))
                n1 = PadDepth(encoder_layers[-1],desired_channels=nc)
                n = merge([n,n1], name='add%d' % (idx + 1))
                n = tl.layers.MaxPool2d(n, filter_size=(3,3), strides=(2,2), name='maxplool%d' % (idx + 1))

        nl = len(encoder_layers)        
        with tf.variable_scope('decoder'):
            _, h, w, _ = encoder_layers[-1].outputs.shape.as_list()
            n = tl.layers.UpSampling2dLayer(n,size=(h, w),is_scale=False, name = 'upsamplimg')
            
            for idx in range(nl - 1, -1, -1): # idx = 4,3,2,1,0
                if idx > 0:
                    _, h, w, _ = encoder_layers[idx - 1].outputs.shape.as_list()
                    out_size = (h, w)
                    out_channels = pyramid_channels[idx-1]
                else:
                    #out_size = None
                    out_channels = n_slices

                print('decoder %d : %s' % (idx, str(n.outputs.get_shape())))
                n = ConcatLayer([encoder_layers[idx], n], concat_dim=-1, name='concat%d' % (nl - idx))
                n = conv2d(n, out_channels, filter_size=3, stride=1,name='conv%d' % (nl - idx + 1))
                n = ReluLayer(n, name='relu%d' % (nl - idx + 1))
                n = batch_norm(n, is_train=is_train, name='bn%d' % (nl - idx + 1))
                #n = UpConv(n, 512, filter_size=4, factor=2, name='upconv2')
                n = tl.layers.UpSampling2dLayer(n,size=out_size,is_scale=False, name = 'upsamplimg%d' % (nl - idx + 1))
                
                #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='dropout1')

            if n.outputs.shape[1] != output_size[0]:
                n = UpSampling2dLayer(n, size=output_size, is_scale=False, name = 'resize_final')
            #n = conv2d(n, n_slices, filter_size=3, stride=1,name='conv_final' )
            n.outputs = tf.tanh(n.outputs)
            #n.outputs = tf.nn.relu(n.outputs)
            #n = conv2d(n, n_filter=n_slices, filter_size=3, act=tf.tanh, name='out')  
            return n



def UNet_B(lf_extra, n_slices, output_size, 
            n_pyramid_levels=4,
            n_base_filters=64,
            using_batch_norm=False,
            is_train=False,
            last_act=tf.nn.relu,
            reuse=False, name='unet'):
    '''U-net based VCD-Net for sparse light field reconstruction, faster than UNet_A
    Params:
        lf_extra: tf.tensor 
            In shape of [batch, height, width, n_num^2], the extracted views from the light field image
        n_slices: int
            The slices number of the 3-D reconstruction.
        output_size: list of int
            Lateral size of the 3-D reconstruction, i.e., [height, width].
        using_batch_norm: boolean
            Whether using batch normalization after each convolutional layer. 
        is_train: boolean, only valid when using_batch_norm=True.
            Sees tl.layers.BatchNormLayer.
        last_act: tensorflow activation functions
            Acivation function applied to the final layer.
        reuse: boolean 
            Whether to reuse the variables or not. See tf.variable_scope() for details.
        name: string
            The name of the variable scope.
    Return:
        The 3-D reconstruction in shape of [batch, height, width, depth=n_slices]
    '''    
    n_interp = 4
    channels_interp = 128
    act = tf.nn.relu

     
    with tf.variable_scope(name, reuse=reuse):
        n = InputLayer(lf_extra, 'lf_extra')
        # n = conv2d(n, n_filter=channels_interp, filter_size=5, name='conv1')
        n = conv(n, n_filter=channels_interp, filter_size=5, act=act, using_batch_norm=using_batch_norm, is_train=is_train, name='conv1')

        ## Up-scale input
        with tf.variable_scope('interp'): 
            for i in range(n_interp):
                channels_interp = channels_interp / 2
                n = upscale(n, out_channels=channels_interp, scale=2, mode='subpixel', name='upsale%d' % i)
                
        pyramid_channels = [n_base_filters * i for i in range(1, n_pyramid_levels + 1)] # output channels number of each conv layer in the encoder
        encoder_layers = []
        with tf.variable_scope('encoder'):
            # n = conv2d(n, n_filter=64, filter_size=3, stride=2, name='conv0')
            n = conv(n, n_filter=64, filter_size=3, stride=2, act=act, using_batch_norm=using_batch_norm, is_train=is_train, name='conv0')

            for idx, nc in enumerate(pyramid_channels):
                encoder_layers.append(n) # append n0, n1, n2, n3, n4 (but without n5)to the layers list
                print('encoder %d : %s' % (idx, str(n.outputs.get_shape())))
                n = LReluLayer(n, name='relu%d' % (idx + 1))
                # n = conv2d(n, n_filter=nc, filter_size=3, name='conv%d' % (idx + 1)) 
                n = conv(n, n_filter=nc, filter_size=3, act=act, using_batch_norm=using_batch_norm, is_train=is_train, name='conv%d' % (idx + 1)) 
                n = max_pool2d(n, filter_size=2, stride=2)

        nl = len(encoder_layers)        
        with tf.variable_scope('decoder'):
            _, h, w, _ = encoder_layers[-1].outputs.shape.as_list()
            n = ReluLayer(n, name='relu1')
            n = upscale(n, out_channels=pyramid_channels[-1], out_size=(h, w), mode='upconv', name='upsale1')
            
            for idx in range(nl - 1, -1, -1): # idx = 4,3,2,1,0
                if idx > 0:
                    _, h, w, _ = encoder_layers[idx - 1].outputs.shape.as_list()
                    out_size = (h, w)
                    out_channels = pyramid_channels[idx-1]
                    
                else:
                    out_size = output_size
                    out_channels = n_base_filters
                
                print('decoder %d : %s' % (idx, str(n.outputs.get_shape())))
                n = ConcatLayer([encoder_layers[idx], n], concat_dim=-1, name='concat%d' % (nl - idx))
                n = ReluLayer(n, name='relu%d' % (nl - idx + 1))
                n = upscale(n, out_channels=out_channels, out_size=out_size, mode='upconv', name='upscale%d' % (nl - idx + 1))
                #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='dropout1')

            n = conv2d(n, n_filter=n_slices, filter_size=3, act=last_act, name='out')  
            return n

@deprecated
def UNet_C(lf_extra, n_slices, output_size, is_train=True, reuse=False, name='unet'):
    '''U-net based VCD-Net for sparse light field reconstruction.
    Params:
        lf_extra: tf.tensor 
            In shape of [batch, height, width, n_num^2], the extracted views from the light field image
        n_slices: int
            The slices number of the 3-D reconstruction.
        output_size: list of int
            Lateral size of the 3-D reconstruction, i.e., [height, width].
        is_train: boolean 
            Sees tl.layers.BatchNormLayer.
        reuse: boolean 
            Whether to reuse the variables or not. See tf.variable_scope() for details.
        name: string
            The name of the variable scope.
    Return:
        The 3-D reconstruction in shape of [batch, height, width, depth=n_slices]
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
        with tf.variable_scope('interp'): 
            for i in range(n_interp):
                channels_interp = channels_interp / 2
                n = SubpixelConv2d(n, scale=2, name='interp/subpixel%d' % i)
                #n = deconv2d(n, out_channels=channels_interp, name='deconv%d' % (i))
                n = conv2d(n, n_filter=channels_interp, filter_size=3, name='conv%d' % i)
                
            n = conv2d(n, n_filter=channels_interp, filter_size=3, act=act, name='conv_final') # 176*176
            n = batch_norm(n, is_train=is_train, name='bn_final')
        
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

            if n.outputs.shape[1] != output_size[0]:
                n = UpSampling2dLayer(n, size=output_size, is_scale=False, name = 'resize_final')
           
            n.outputs = tf.tanh(n.outputs)
            #n = conv2d(n, n_filter=n_slices, filter_size=3, act=tf.tanh, name='out')  
            return n
    
"""
def UNet_B(lf_extra, n_slices, output_size, is_train=True, reuse=False, name='unet'):
    '''U-net based VCD-Net for sparse light field reconstruction, faster than UNet_A
    Params:
        lf_extra: tf.tensor 
            In shape of [batch, height, width, n_num^2], the extracted views from the light field image
        n_slices: int
            The slices number of the 3-D reconstruction.
        output_size: list of int
            Lateral size of the 3-D reconstruction, i.e., [height, width].
        is_train: boolean 
            Sees tl.layers.BatchNormLayer.
        reuse: boolean 
            Whether to reuse the variables or not. See tf.variable_scope() for details.
        name: string
            The name of the variable scope.
    Return:
        The 3-D reconstruction in shape of [batch, height, width, depth=n_slices]
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
        with tf.variable_scope('interp'): 
            for i in range(n_interp):
                channels_interp = channels_interp / 2
                n = SubpixelConv2d(n, scale=2, name='interp/subpixel%d' % i)
                #n = deconv2d(n, out_channels=channels_interp, name='deconv%d' % (i))
                n = conv2d(n, n_filter=channels_interp, filter_size=3, name='conv%d' % i)
                
            n = conv2d(n, n_filter=channels_interp, filter_size=3, act=act, name='conv_final') # 176*176
            n = batch_norm(n, is_train=is_train, name='bn_final')
        
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

            if n.outputs.shape[1] != output_size[0]:
                n = UpSampling2dLayer(n, size=output_size, is_scale=False, name = 'resize_final')
           
            n.outputs = tf.tanh(n.outputs)
            #n = conv2d(n, n_filter=n_slices, filter_size=3, act=tf.tanh, name='out')  
            return n

def UNet_B(lf_extra, n_slices, output_size, series_input=False, cell=None, is_train=False, reuse=False, name='unet'):
    '''U-net based VCD-Net for dense light field reconstruction.
    Params:
        lf_extra: tf.tensor 
            In shape of [batch, height, width, n_num^2], the extracted views from the light field image
        n_slices: int
            The slices number of the 3-D reconstruction.
        output_size: list of int
            Lateral size of the 3-D reconstruction, i.e., [height, width].
        series_input: boolean
            Deprecated.
        cell: numpy.ndarray
            Deprecated.
        is_train: boolean 
            Sees tl.layers.BatchNormLayer.
        reuse: boolean 
            Whether to reuse the variables or not. See tf.variable_scope() for details.
        name: string
            The name of the variable scope.
    Return:
        The 3-D reconstruction in shape of [batch, height, width, depth=n_slices]
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
        with tf.variable_scope('interp'): 
            for i in range(n_interp):
                channels_interp = channels_interp / 2
                n = SubpixelConv2d(n, scale=2, name='interp/subpixel%d' % i)
                #n = deconv2d(n, out_channels=channels_interp, name='deconv%d' % (i))
                n = conv2d(n, n_filter=channels_interp, filter_size=3, name='conv%d' % i)
                
            n = conv2d(n, n_filter=channels_interp, filter_size=3, name='conv_final') # 176*176
            n = batch_norm(n, is_train=is_train, name='bn_final')
            n = ReluLayer(n, name='reul_final' )
        
        pyramid_channels = [128, 256, 512, 512, 512] # output channels number of each conv layer in the encoder
        encoder_layers = []
        with tf.variable_scope('encoder'):
            n = conv2d(n, n_filter=64, filter_size=5, stride=1, name='conv0')
            n = batch_norm(n, is_train=is_train, name='bn_0')
            n = ReluLayer(n, name='reul0' )

            for idx, nc in enumerate(pyramid_channels):
                encoder_layers.append(n) # append n0, n1, n2, n3, n4 (but without n5)to the layers list
                n = tl.layers.MaxPool2d(n, filter_size=(2,2), strides=(2,2), name='maxplool%d' % (idx + 1))
                print('encoder %d : %s' % (idx, str(n.outputs.get_shape())))
                n = conv2d(n, n_filter=nc, filter_size=5, stride=1, name='conv%d' % (idx + 1))  
                n = batch_norm(n, is_train=is_train, name='bn%d' % (idx + 1))
                n = ReluLayer(n, name='reul%d' % (idx + 1))
                

        nl = len(encoder_layers)        
        with tf.variable_scope('decoder'):
            _, h, w, _ = encoder_layers[-1].outputs.shape.as_list()
            n = tl.layers.UpSampling2dLayer(n,size=(h, w),is_scale=False, name = 'upsamplimg')
            
            for idx in range(nl - 1, -1, -1): # idx = 4,3,2,1,0
                if idx > 0:
                    _, h, w, _ = encoder_layers[idx - 1].outputs.shape.as_list()
                    out_size = (h, w)
                    out_channels = pyramid_channels[idx-1]
                else:
                    #out_size = None
                    out_channels = n_slices

                print('decoder %d : %s' % (idx, str(n.outputs.get_shape())))
                n = ConcatLayer([encoder_layers[idx], n], concat_dim=-1, name='concat%d' % (nl - idx))
                n = conv2d(n, out_channels, filter_size=5, stride=1,name='conv%d' % (nl - idx + 1))
                n = ReluLayer(n, name='relu%d' % (nl - idx + 1))
                n = batch_norm(n, is_train=is_train, name='bn%d' % (nl - idx + 1))
                #n = UpConv(n, 512, filter_size=4, factor=2, name='upconv2')
                n = tl.layers.UpSampling2dLayer(n,size=out_size,is_scale=False, name = 'upsamplimg%d' % (nl - idx + 1))
                
                #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='dropout1')

            if n.outputs.shape[1] != output_size[0]:
                n = UpSampling2dLayer(n, size=output_size, is_scale=False, name = 'resize_final')
           
            #n.outputs = tf.tanh(n.outputs)
            #n = conv2d(n, n_filter=n_slices, filter_size=3, act=tf.tanh, name='out')  
            return n

def UNet(lf_extra, n_slices, output_size, n_interp=4, n_channels=128, use_bn=False, is_train=True, reuse=False, name='unet'):
    '''U-net based VCD-Net for sparse light field reconstruction.
    Params:
        lf_extra: tf.tensor 
            In shape of [batch, height, width, n_num^2], the extracted views from the light field image
        n_slices: int
            The slices number of the 3-D reconstruction.
        output_size: list of int
            Lateral size of the 3-D reconstruction, i.e., [height, width].
        n_interp: int
            Number of the subpixel convolutional layers that upscale the lateral size of the input views. 
        n_channels: int 
            Number of the channels of the convolutional layers in the 'interp' part of the VCD-Net.
        use_bn: boolean
            Whether to add the batch normalization after each convolutional layer.
        is_train: boolean 
            Sees tl.layers.BatchNormLayer.
        reuse: boolean 
            Whether to reuse the variables or not. See tf.variable_scope() for details.
        name: string
            The name of the variable scope.
    Return:
        The 3-D reconstruction in shape of [batch, height, width, depth=n_slices]
    '''    
    
    # _, w, h, _ = lf_extra.shape
    channels_interp = n_channels
    act = tf.nn.relu

    def __batch_norm(n, is_train=True, name='bn', apply=True):
        return batch_norm(n, is_train=is_train, name=name) if apply else n

    with tf.variable_scope(name, reuse=reuse):

        n = InputLayer(lf_extra, 'lf_extra')

        n = conv2d(n, n_filter=64, filter_size=7, name='conv1')
        # n = conv2d(n, n_filter=128, filter_size=5, name='conv1')
        ## Up-scale input
        with tf.variable_scope('interp'): 
            for i in range(n_interp):
                #channels_interp = channels_interp / 2
                n = SubpixelConv2d(n, scale=2, name='interp/subpixel%d' % i)
                #n = deconv2d(n, out_channels=channels_interp, name='deconv%d' % (i))
                n = conv2d(n, n_filter=channels_interp, filter_size=3, name='conv%d' % i)
                
            n = conv2d(n, n_filter=channels_interp, filter_size=3, act=act, name='conv_final') # 176*176
            n = __batch_norm(n, is_train=is_train, name='bn_final', apply=use_bn)
        
        pyramid_channels = [1, 2, 4, 4, 4] # output channels number of each conv layer in the encoder
        pyramid_channels = [i * channels_interp for i in pyramid_channels]

        encoder_layers = []
        with tf.variable_scope('encoder'):
            # n = conv2d(n, n_filter=64, filter_size=3, stride=2, name='conv0')
            n = conv2d(n, n_filter=channels_interp, filter_size=3, stride=2, name='conv0')

            for idx, nc in enumerate(pyramid_channels):
                encoder_layers.append(n) # append n0, n1, n2, n3, n4 (but without n5)to the layers list
                print('encoder %d : %s' % (idx, str(n.outputs.get_shape())))
                n = LReluLayer(n, name='lreu%d' % (idx + 1))
                n = conv2d(n, n_filter=nc, filter_size=3, stride=2, name='conv%d' % (idx + 1)) 
                n = __batch_norm(n, is_train=is_train, name='bn%d' % (idx + 1), apply=use_bn)

        nl = len(encoder_layers)        
        with tf.variable_scope('decoder'):
            _, h, w, _ = encoder_layers[-1].outputs.shape.as_list()
            n = ReluLayer(n, name='relu1')
            n = deconv2d(n, pyramid_channels[-1], out_size=(h, w), padding='SAME', name='deconv1')
            n = __batch_norm(n, is_train=is_train, name='bn1', apply=use_bn)
            
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
                n = __batch_norm(n, is_train=is_train, name='bn%d' % (nl - idx + 1), apply=use_bn)
                #n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='dropout1')

            if n.outputs.shape[1] != output_size[0]:
                n = UpSampling2dLayer(n, size=output_size, is_scale=False, name = 'resize_final')
           
            if use_bn:
                n.outputs = tf.tanh(n.outputs)
            #n = conv2d(n, n_filter=n_slices, filter_size=3, act=tf.tanh, name='out')  
            return n

"""