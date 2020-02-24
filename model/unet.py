from .util.utils import *


def UNet_A(lf_extra, n_slices, out_size, n_interp=4, n_channels=128, use_bn=False, is_train=True, reuse=False, name='unet'):
    '''U-net based VCD-Net for sparse light field reconstruction.
    Params:
        lf_extra: tf.tensor 
            In shape of [batch, height, width, n_num^2], the extracted views from the light field image
        n_slices: int
            The slices number of the 3-D reconstruction.
        out_size: list of int
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

            if n.outputs.shape[1] != out_size[0]:
                n = UpSampling2dLayer(n, size=out_size, is_scale=False, name = 'resize_final')
           
            if use_bn:
                n.outputs = tf.tanh(n.outputs)
            #n = conv2d(n, n_filter=n_slices, filter_size=3, act=tf.tanh, name='out')  
            return n


def UNet_B(lf_extra, n_slices, out_size, series_input=False, cell=None, is_train=False, reuse=False, name='unet'):
    '''U-net based VCD-Net for dense light field reconstruction.
    Params:
        lf_extra: tf.tensor 
            In shape of [batch, height, width, n_num^2], the extracted views from the light field image
        n_slices: int
            The slices number of the 3-D reconstruction.
        out_size: list of int
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
    batch, w, h, in_channels = lf_extra.shape
    #channels_interp = in_channels.value
    channels_interp = 128

    #act = tf.nn.relu 
    act = None
    with tf.variable_scope(name, reuse=reuse):
        n = tl.layers.InputLayer(lf_extra, 'lf_extra')
        n = conv2d(n, n_filter=channels_interp, filter_size=7, name='conv1')
        ## Up-scale input 
        for i in range(n_interp):
            channels_interp = channels_interp / 2
            h = h * 2
            w = w * 2
            n = SubpixelConv2d(n, scale=2, name='interp/subpixel%d' % i)
            n = conv2d(n, n_filter=channels_interp, filter_size=3, act=act, name='interp/conv%d' % i)
            n = batch_norm(n, act=tf.nn.relu, is_train=is_train, name='interp/bn%d' % i)
            
        n = conv2d(n, n_filter=n_slices, filter_size=3, act=act, name='interp/conv_final') # 176*176
        n = batch_norm(n, act=tf.nn.relu, is_train=is_train, name='interp/bn_final')
        #n = UpSampling2dLayer(n, size=(img_height, img_width), is_scale=False, name = 'interp/upsampling_final')
        
        with tf.variable_scope('encoder'):
            n0 = conv2d(n, n_filter=64, filter_size=4, stride=2, name='conv0') # 88*88
            n1 = LReluLayer(n0, name='lreu1')
            n1 = conv2d(n1, n_filter=128, filter_size=4, stride=2, name='conv1') # 44*44
            n1 = batch_norm(n1, is_train=is_train, name='bn1')
            
            n2 = LReluLayer(n1, name='lrelu2')
            n2 = conv2d(n2, n_filter=256, filter_size=4, stride=2, name='conv2') # 22*22
            n2 = batch_norm(n2, is_train=is_train, name='bn2')
            
            n3 = LReluLayer(n2, name='lrelu3')
            n3 = conv2d(n3, n_filter=512, filter_size=4, stride=2, name='conv3') # 11*11
            n3 = batch_norm(n3, is_train=is_train, name='bn3')
            
            n4 = LReluLayer(n3, name='lrelu4')
            n4 = conv2d(n4, n_filter=512, filter_size=4, stride=2, name='conv4') # 5*5
            n4 = batch_norm(n4, is_train=is_train, name='bn4')
            
            n5 = LReluLayer(n4, name='lrelu5')
            n5 = conv2d(n5, n_filter=512, filter_size=4, stride=2, name='conv5') # 2*2
            n5 = batch_norm(n5, is_train=is_train, name='bn5')
            if series_input:
                n5.outputs = tf.concat([n5.outputs, cell], axis=-1, name='concat_cell')
                n5 = conv2d(n5, n_filter=512, filter_size=1, name='conv_cell')
                keep_rate = conv2d(n5, n_filter=512, filter_size=1, act=tf.nn.sigmoid, name='keep_rate')
                update = conv2d(n5, n_filter=512, filter_size=1, act=tf.tanh, name='update')
                if cell != None:
                    abandon_rate = conv2d(n5, n_filter=512, filter_size=1, act=tf.nn.sigmoid, name='abandon_rate')                   
                    new_cell = tf.multiply(cell, abandon_rate.outputs,) + tf.multiply(update.outputs, keep_rate.outputs)
                else:
                    new_cell = tf.multiply(update.outputs, keep_rate.outputs)
            else:
                new_cell = n5.outputs
                
        with tf.variable_scope('decoder'):
            n6 = ReluLayer(n5, name='relu1')
            #n6 = UpConv(n6, 512, filter_size=4, factor=2, name='upconv1')
            _, h, w, _ = n4.outputs.shape.as_list()
            n6 = deconv2d(n6, 512, filter_size=4, out_size=(h, w), name='deconv1')
            n6 = batch_norm(n6, is_train=is_train, name='bn1')
            
            n7 = ConcatLayer([n4, n6], concat_dim=-1, name='concat1')
            n7 = ReluLayer(n7, name='relu2')
            #n7 = UpConv(n7, 512, filter_size=4, factor=2, name='upconv2')
            _, h, w, _ = n3.outputs.shape.as_list()
            n7 = deconv2d(n7, 512, filter_size=4, out_size=(h, w), name='deconv2')
            n7 = batch_norm(n7, is_train=is_train, name='bn2')
            n7 = DropoutLayer(n7, keep=0.5, is_fix=True, is_train=is_train, name='dropout1')
            
            n8 = ConcatLayer([n3, n7], concat_dim=-1, name='concat2')
            n8 = ReluLayer(n8, name='relu3')
            #n8 = UpConv(n8, 256, filter_size=4, factor=2, name='upconv3')
            _, h, w, _ = n2.outputs.shape.as_list()
            n8 = deconv2d(n8, 256, filter_size=4, out_size=(h, w), name='deconv3')
            n8 = batch_norm(n8, is_train=is_train, name='bn3')
            n8 = DropoutLayer(n8, keep=0.5, is_fix=True, is_train=is_train, name='dropout2')
            
            n9 = ConcatLayer([n2, n8], concat_dim=-1, name='concat3')
            n9 = ReluLayer(n9, name='relu4')
            #n9 = UpConv(n9, 128, filter_size=4, factor=2, name='upconv4')
            _, h, w, _ = n1.outputs.shape.as_list()
            n9 = deconv2d(n9, 128, filter_size=4, out_size=(h, w), name='deconv4')
            n9 = batch_norm(n9, is_train=is_train, name='bn4')
            
            n10 = ConcatLayer([n1, n9], concat_dim=-1, name='concat4')
            n10 = ReluLayer(n10, name='relu5')
            #n10 = UpConv(n10, 64, filter_size=4, factor=2, name='upconv5')
            _, h, w, _ = n0.outputs.shape.as_list()
            n10 = deconv2d(n10, 64, filter_size=4, out_size=(h, w), name='deconv5')
            n10 = batch_norm(n10, is_train=is_train, name='bn5')
            
            n11 = ConcatLayer([n0, n10], concat_dim=-1, name='concat5')
            n11 = ReluLayer(n11, name='relu6')
            #n11 = UpConv(n11, n_slices, filter_size=4, factor=2, name='upconv6')
            n11 = deconv2d(n11, n_slices, filter_size=4, name='upconv6')
            n11 = batch_norm(n11, is_train=is_train, name='bn6')
            
            if n11.outputs.shape[1] != out_size[0]:
                out = UpSampling2dLayer(n11, size=(out_size[0], out_size[1]), is_scale=False, name = 'resize_final')
            else:
                out = n11
                
            out.outputs = tf.tanh(out.outputs)
            
            return out, new_cell
