import numpy as np
import imageio
import PIL.Image as pilimg
import tensorlayer as tl



__all__ = [
    'get_img3d_fn',
    'rearrange3d_fn',
    'get_and_rearrange3d',
    'get_img2d_fn',
    'get_lf_extra',
    'lf_extract_fn',
    'write3d'
]

def get_img3d_fn(filename, path):
    """
    Parames:
        mode - Depth : read 3-D image in format [depth=slices, height, width, channels=1]
               Channels : [height, width, channels=slices]
    """
    image = imageio.volread(path + filename) # [depth, height, width]
    image = image[..., np.newaxis] # [depth, height, width, channels]
            
    return normalize_fn(image)
    
def rearrange3d_fn(image):
    """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
    """
    
    image = np.squeeze(image) # remove channels dimension
    #print('reshape : ' + str(image.shape))
    depth, height, width = image.shape
    image_re = np.zeros([height, width, depth]) 
    for d in range(depth):
        image_re[:,:,d] = image[d,:,:]
    return image_re    

def get_and_rearrange3d(filename, path):
    image = get_img3d_fn(filename, path)
    return rearrange3d_fn(image)
    
def get_img2d_fn(filename, path):
  
    image = imageio.imread(path + filename).astype(np.float)
    if image.ndim == 2:
        image = image[:,:, np.newaxis]
    #print(image.shape)
    return normalize_fn(image)

def get_lf_extra(filename, path, n_num=11):
    image = get_img2d_fn(filename, path)
    extra = lf_extract_fn(image, n_num=n_num)
    
    return extra
    
def resize_normalize_fn(x, size):
    x = resize_fn(x, size) 
    x = normalize_fn(x) 
    return x

def normalize_fn(x):   
    x = x / (255. / 2.)
    x = x - 1
    return x
    
def resize_fn(x, size):
    '''
    Param:
        -size: [height, width]
    '''
    x = np.array(pilimg.fromarray(x).resize(size=(size[1], size[0]), resample=pilimg.BICUBIC))
    
    return x
    
def lf_extract_fn(lf2d, n_num=11, mode='toChannel', padding=False):
    """
    Extract different views from a single LF projection
    
    Params:
        -lf2d - 2-D light field projection
        -mode - 'toDepth' -- extract views to depth dimension (output format [depth=multi-slices, h, w, c=1])
                'toChannel' -- extract views to channel dimension (output format [h, w, c=multi-slices])
        -padding -   True : keep extracted views the same same as lf2d by padding zeros between valid pixels
                     False : shrink size of extracted views to (lf2d.shape / Nnum);
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
    
    
def do_nothing(x):
    return x
    
def _write3d(x, path):
    """
    x : [depth, height, width, channels=1]
    """
    x = x + 1.  #[0, 2]
    x = x * 65535. / 2.
    x = x.astype(np.uint16)

    imageio.volwrite(path, x[..., 0])
        
def write3d(x, path):
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
        _write3d(x_re[0], path) 
    else:  
        fragments = path.split('.')
        new_path = ''
        for i in range(len(fragments) - 1):
            new_path = new_path + fragments[i]
        for index, image in enumerate(x_re):
            #print(image.shape)
            _write3d(image, new_path + '_' + str(index) + '.' + fragments[-1]) 

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
    
def retrieve_single_slice_from_file(file, path):
    x = get_img3d_fn(file, path)
    slice = 8
    tmp = x[slice, :, :, :] # height, width, channels
    return tmp
     
class PSFConfig:
    def __init__(self, size, n_num, n_slices):
        self.psf_size = size
        self.n_num = n_num
        self.n_slices = n_slices