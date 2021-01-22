import os
import re

import numpy as np
import imageio
import PIL.Image as pilimg
import tensorlayer as tl



__all__ = [
    'imread',
    'volread_HWD',
    'volread_HWD_norm',
    'rearrange3d',
    'imread_norm',
    'lfread_norm',
    'write3d',
    'normalize_percentile',
    'normalize',
    'save_activations',
    'fft',
    'spectrum2im',
    'PSFConfig',
]

def imread(filename, path):
    im = imageio.imread(os.path.join(path,filename))
    if im.ndim == 2:
        im = im[:,:, np.newaxis]
    return im

def volread_HWD(filename, path, make_mask=False):
    im = imageio.volread(os.path.join(path,filename)) # [depth, height, width]
    if make_mask:
        im = otsu_thresholding(im)

    return rearrange3d(im)

def imread_norm(filename, path, normalize_fn, **kwargs):
    return normalize_fn(imread(filename, path), **kwargs)

def volread_HWD_norm(filename, path, normalize_fn):
    """
    Parames:
        mode - Depth : read 3-D image in format [depth=slices, height, width, channels=1]
               Channels : [height, width, channels=slices]
    """
    image = volread_HWD(filename, path) # [depth, height, width]
    # image = image[..., np.newaxis] # [depth, height, width, channels]       
    return normalize_fn(image)
    
def rearrange3d(image):
    """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
    """
    
    image = np.squeeze(image) # remove channels dimension
    #print('reshape : ' + str(image.shape))
    depth, height, width = image.shape
    image_re = np.zeros([height, width, depth], dtype=image.dtype) 
    for d in range(depth):
        image_re[:,:,d] = image[d,:,:]
    return image_re    
    
def lfread_norm(filename, path, n_num, normalize_fn, padding=False, **kwargs):
    image = imread_norm(filename, path, normalize_fn, **kwargs)
    extra = extract_views(image, n_num=n_num, padding=padding)
    return extra


def normalize(im):  
    assert im.dtype in [np.uint8, np.uint16]
    
    x = im.astype(np.float32)
    max_ = 255. if im.dtype == np.uint8 else 65536.
    # x = x / (max_ / 2.) - 1.
    x = x / (max_)
    return x

def normalize_percentile(im, low=0.2, high=99.8):
    p_low  = np.percentile(im, low)
    p_high = np.percentile(im, high)

    eps = 1e-3
    x = (im - p_low) / (p_high - p_low + eps)
    # print('%.2f-%.2f' %  (np.min(x), np.max(x)))
    return x


def _write3d(x, path, bitdepth=8):
    """
    x : [depth, height, width, channels=1]
    """
    assert (bitdepth in [8, 16, 32])
    x = clamp(x, low=0, high=1)

    if bitdepth == 32:
         x = x.astype(np.float32)

    else:
        if bitdepth == 8:
            x = x * 255
            # x = (x + 1) * 127.5
            x = x.astype(np.uint8)  
        else:
            x = x * 65535
            # x = (x + 1) * 65535. /2
            x = x.astype(np.uint16) 
    
    imageio.volwrite(path, x[..., 0])
        
def write3d(x, path, bitdepth=32):
    """
    x : [batch, depth, height, width, channels] or [batch, height, width, channels>3]
    """
    
    #print(x.shape)
    dims = len(x.shape)
    
    if dims == 4:
        batch, height, width, n_channels = x.shape
        x_re = np.zeros([batch, n_channels, height, width, 1])
        for d in range(n_channels):
            slice = x[:,:,:,d]
            x_re[:,d,:,:,:] = slice[:,:,:,np.newaxis]
            
    elif dims == 5:
        x_re = x
    else:
        raise Exception('unsupported dims : %s' % str(x.shape))
    
    batch = x_re.shape[0]
    if batch == 1:
        _write3d(x_re[0], path, bitdepth) 
    else:  
        fragments = path.split('.')
        new_path = ''
        for i in range(len(fragments) - 1):
            new_path = new_path + fragments[i]
        for index, image in enumerate(x_re):
            #print(image.shape)
            _write3d(image, new_path + '_' + str(index) + '.' + fragments[-1], bitdepth) 

def otsu_thresholding(im):
    '''Otsu thresholding 
    Params:
        im: numpy.ndarray. im.dtype must be in [np.uint8, np.uint16]

    '''
    if not (im.dtype in [np.uint8, np.uint16]):
        raise (ValueError('image must be 8-bit or 16 bit'))
    n_bins = 256 if im.dtype == np.uint8 else 65536

    im_shape = im.shape
    n_pixels = 1
    for s in im.shape:
        n_pixels *= s

    histo = np.zeros(n_bins, np.float32)
    
    for p in np.reshape(im, [-1]):
        histo[p] += 1

    n_pixels_bg, n_pixels_fg = 0, n_pixels
    sum_bg, sum_fg = 0, np.sum(im)
    
    max_var_between = -1e10
    thres = 0
    for i in range(n_bins):
        n_pixels_bg += histo[i]
        n_pixels_fg -= histo[i]

        if n_pixels_bg == 0:
            continue
        if n_pixels_fg == 0:
            break

        sum_bg += histo[i] * i
        sum_fg -= histo[i] * i

        mu_bg = sum_bg / n_pixels_bg
        mu_fg = sum_fg / n_pixels_fg

        var = n_pixels_bg/n_pixels * n_pixels_fg/n_pixels * (mu_bg - mu_fg) * (mu_bg - mu_fg)
        if (var > max_var_between):
            max_var_between = var
            thres = i

    bw = np.zeros_like(im)
    bw[im > thres] = 1

    return bw

def resize_fn(x, size):
    '''
    Param:
        -size: [height, width]
    '''
    x = np.array(pilimg.fromarray(x).resize(size=(size[1], size[0]), resample=pilimg.BICUBIC))
    
    return x
    
def extract_views(lf2d, n_num=11, mode='toChannel', padding=False):
    """
    Extract different views from a single LF projection
    
    Params:
        -lf2d: numpy.array, 2-D light field projection in shape of [height, width, channels=1]
        -mode - 'toDepth' -- extract views to depth dimension (output format [depth=multi-slices, h, w, c=1])
                'toChannel' -- extract views to channel dimension (output format [h, w, c=multi-slices])
        -padding -   True : keep extracted views the same size as lf2d by padding zeros between valid pixels
                     False : shrink size of extracted views to (lf2d.shape / Nnum);
    Returns:
        ndarray [height, width, channels=n_num^2] if mode is 'toChannel' 
                or [depth=n_num^2, height, width, channels=1] if mode is 'toDepth'
    """
    n = n_num
    h, w, c = lf2d.shape
    if padding:
        if mode == 'toDepth':
            lf_extra = np.zeros([n*n, h, w, c]) # [depth, h, w, c]
            
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, i : h : n, j : w : n, :] = lf2d[i : h : n, j : w : n, :]
                    d += 1
        elif mode == 'toChannel':
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([h, w, n*n])
            
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[i : h : n, j : w : n, d] = lf2d[i : h : n, j : w : n]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)
    else:
        new_h = int(np.ceil(h / n))
        new_w = int(np.ceil(w / n))

        if mode == 'toChannel':
            
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([new_h, new_w, n*n])
            
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[:, : , d] = lf2d[i : h : n, j : w : n]
                    d += 1
                    
            
                        
                        
        elif mode == 'toDepth':
            lf_extra = np.zeros([n*n, new_h, new_w, c]) # [depth, h, w, c]
            
            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, :, :, :] = lf2d[i : h : n, j : w : n, :]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)
            
    return lf_extra
    
    
def clamp(x, low=0, high=1):
    # min_ = np.percentile(x, low)
    # max_ = np.max(x) if high == 100 else np.percentile(x, high)
    x = np.clip(x, low, high)
    return x


def fft(im):
    """
    Params:
        -im:  ndarray in shape [height, width, channels]

    """
    assert len(im.shape) == 3
    spec = np.fft.fft2(im, axes=(0, 1))
    return np.fft.fftshift(spec, axes=(0, 1))

def spectrum2im(fs):
    """
    Convert the Fourier spectrum into the original image
    Params:
        -fs: ndarray in shape [batch, height, width, channels]
    """
    fs = np.fft.fftshift(fs, axes=(1, 2))
    return np.fft.ifft2(fs, axes=(1, 2))

def load_psf(path, n_num=11, psf_size=155, n_slices=16):
    '''
    Return : [n_num, n_num, n_slices, psf_size, psf_size, 1, 1]
    '''
    print('loading psf...')
    file_list = sorted(tl.files.load_file_list(path=path, regx='.*.tif', printable=False))
    if len(file_list) != n_num ** 2:
        raise Exception('psf files number must be euqal to Nnum^2');
        
    psf5d = np.zeros([n_num, n_num, n_slices, psf_size, psf_size])
    for i in range(n_num):
        for j in range(n_num):
            idx = i * n_num + j
            psf = imageio.volread(path + file_list[idx]) # [depth=n_slices, psf_size, psf_size]
            psf5d[i,j,...] = psf
            
    print('load psf5d in shape %s' % str(psf5d.shape))        
    return psf5d[..., np.newaxis, np.newaxis]  
    
def generate_mask(n_num, img_size):
    '''
    Generate a mask that help to extract different view from a 3-D scene. Used in forward projection.
    Return: mask[img_size, img_size, n_num, n_num]
    '''
    mask = np.zeros([img_size, img_size, n_num, n_num])
    for i in range(n_num):
        for j in range(n_num):
            for h in range(0, img_size):
                for w in range(0, img_size):
                    if h % n_num == i and w % n_num == j:
                        mask[h, w, i, j] = 1
                        
    return mask

def save_activations(save_dir, sess, final_layer, feed_dict, verbose=False):
        """
        save all the feature maps (before the final_layer) into tif file.
        Params:
            -final_layer: a tl.layers.Layer instance
        """
        if not isinstance(final_layer, tl.layers.Layer):
            raise(ValueError('final_layer must be an instance of tl.layers.Layer'))
        fetches = {}
        for i, layer in enumerate(final_layer.all_layers):
            if verbose:
                print("  layer {:2}: {:40}  {:20}".format(i, str(layer.name), str(layer.get_shape())))
            name = re.sub(':', '', str(layer.name))
            name = re.sub('/', '_', name)
            fetches.update({name : layer})
        features = sess.run(fetches, feed_dict)

        layer_idx = 0
        
        for name, feat in features.items():
            save_path = os.path.join(save_dir, '%03d/' % layer_idx)
            tl.files.exists_or_mkdir(save_path, verbose=False)
            filename = os.path.join(save_path, '{}.tif'.format(name))
            write3d(feat, filename)
            layer_idx += 1

class PSFConfig(object):
    def __init__(self, size, n_num, n_slices):
        self.psf_size = size
        self.n_num = n_num
        self.n_slices = n_slices

    @property
    def N_num(self):
        return self.n_num

    @property
    def n_slices(self):
        return self.n_slices
