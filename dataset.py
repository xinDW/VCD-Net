import tensorlayer as tl
import numpy as np
import os

from utils import *
import PIL.Image as pilImg

class Dataset:
    def __init__(self, train_hr3d_path, train_lf2d_path, 
                n_slices, n_num, 
                lf2d_base_size=None, 
                normalize_mode='max', 
                test_split=0.2,
                shuffle=True, 
                bianry_mask=False, 
                multi_scale=False):
        '''
        Params:
            n_slices      : int, depth of the 3d target images (the reconstructions)
            n_num         : int, Nnum of light filed imaging
            lf2d_base_size: 2-element list, [height, width], equals to (lf2d_size / n_num)
            normalize_mode: str, normalization mode of dataset in ['max', 'percentile']
            test_split    : float, ratio of training data left for testing
            shuffle       : boolean, whether to shuffle the training dataset
            bianry_mask   : boolean, whether to generate binary masks for HR targets (used for calculating the binaryMSE)
            multi_scale   : boolean, whether to generate multi-scale HRs
        '''
        self.train_lf2d_path = train_lf2d_path
        self.train_hr3d_path = train_hr3d_path

        self.lf2d_base_size   = lf2d_base_size
        self.n_slices         = n_slices
        self.n_num            = n_num
        self.test_split       = test_split
        self.shuffle          = shuffle
        self.make_binary_mask = bianry_mask
        self.multi_scale      = multi_scale
        
        self.normalize_fn = normalize_percentile if normalize_mode is 'percentile' else normalize

    def _load_dataset(self, shuffle=True, make_binary_mask=False):
        def _shuffle_in_unison(arr1, arr2):
            """shuffle elements in arr1 and arr2 in unison along the leading dimension 
            Params:
                -arr1, arr2: np.ndarray
                    must be in the same size in the leading dimension
            """
            assert (len(arr1) == len(arr2))
            new_idx = np.random.permutation(len(arr1)) 
            return arr1[new_idx], arr2[new_idx]

        def _load_imgs(path, fn, regx='.*.tif', printable=False, **kwargs):
            im_list = sorted(tl.files.load_file_list(path=path, regx=regx, printable=printable))
            
            sample = fn(im_list[0], path, **kwargs)
            ims = np.zeros(shape=[len(im_list)] + list(sample.shape), dtype=np.float32)
            
            for i, im_file in enumerate(im_list):
                im = fn(im_file, path, **kwargs) 
                if (im.dtype != np.float32):
                    im = im.astype(np.float32, casting='unsafe')
                print('\r%d %s : %s' % (i, im_file, str(im.shape)), end='')  

                ims[i] = im
            print()
            return ims

        training_hr3d = _load_imgs(self.train_hr3d_path, fn=volread_HWD_norm, normalize_fn=self.normalize_fn)
        training_lf2d = _load_imgs(self.train_lf2d_path, fn=get_lf_extra, n_num=self.n_num, normalize_fn=self.normalize_fn)
        # training_lf2d = _load_imgs(self.train_lf2d_path, fn=imread_norm, normalize_fn=self.normalize_fn)

        
        if (len(training_hr3d) == 0) or (len(training_lf2d) == 0) :
            raise Exception("none of the images have been loaded, please check the file directory in config")
            
        assert len(training_hr3d) == len(training_lf2d)

        if make_binary_mask:
            hr3d_mask = _load_imgs(self.train_hr3d_path, fn=volread_HWD, make_mask=make_binary_mask)
            self.training_hr3d_mask = hr3d_mask

        # [self.training_hr3d, self.training_lf2d] = _shuffle_in_unison(training_hr3d, training_lf2d) if shuffle else [training_hr3d, training_lf2d]
        [self.training_hr3d, self.training_lf2d] = training_hr3d, training_lf2d
        self.data_pair_num = len(self.training_hr3d)
        

    def _generate_hr_pyramid(self):
        def _resize_xy(img3d, size):
            '''
            img3d : [height, width, depth]
            size  : [new_height, new_width]
            '''
            h, w, depth = img3d.shape
            new_size = size + [depth]
            img_re = np.zeros(new_size)
            
            for d in range(0, depth):
                img_re[:,:,d] = np.array(pilImg.fromarray(img3d[:,:,d]).resize((size[1], size[0]), resample=pilImg.BICUBIC))
            return img_re

        '''
        hr_pyramid = []

        for idx in range(0, self.data_pair_num):
            hr = self.training_hr3d[idx]
            tmp = []
            for scale in range(1, 4):
                tmp.append(_resize_xy(hr, self.lf2d_base_size * (2**scale))
            hr_pyramid.append(tmp)

        self.trainig_data_hr_pyramid = np.asarray(hr_pyramid)
        '''
        hr_s1, hr_s2, hr_s3 = [], [], []
        base_height, base_width = self.lf2d_base_size

        for idx in range(0, self.data_pair_num):
            hr = self.training_hr3d[idx]
            hr_s1.append(_resize_xy(hr, [base_height * 2, base_width * 2]))
            hr_s2.append(_resize_xy(hr, [base_height * 4, base_width * 4]))
            hr_s3.append(_resize_xy(hr, [base_height * 8, base_width * 8]))

        self.hr_s1 = np.asarray(hr_s1)
        self.hr_s2 = np.asarray(hr_s2)
        self.hr_s3 = np.asarray(hr_s3)

    def _get_data_index(self, len, shuffle=True):
        indices = np.arange(len)
        if shuffle:
            indices = np.random.permutation(indices)
        return indices

    def prepare(self, batch_size):
        '''
        this function must be called after the Dataset instance is created
        '''
        if os.path.exists(self.train_lf2d_path) and os.path.exists(self.train_hr3d_path):
            self._load_dataset(shuffle=self.shuffle, make_binary_mask=self.make_binary_mask)
        else:
            raise Exception('image data path doesn\'t exist:\n%s\n%s' %(self.train_lf2d_path, self.train_hr3d_path))
        

        self.test_pair_num = int(self.data_pair_num * self.test_split)
        self.training_pair_num = self.data_pair_num - self.test_pair_num

        '''
        generate HR pyramid
        '''
        if self.multi_scale:
            self._generate_hr_pyramid()

        self.index = self._get_data_index(self.training_pair_num, shuffle=self.shuffle)
        
        self.batch_size = batch_size

        print('HR dataset : %d\nLF dataset: %d\n' % (len(self.training_hr3d), len(self.training_lf2d)))
        #print('batch size : %d \n%d batches available\n' % (self.batch_size, (self.data_pair_num - batch_size) // batch_size))
        return self.training_pair_num

    def for_test(self):
        n = self.test_pair_num
        b = self.batch_size

        for idx in range(0, n - b + 1, b):
            end = idx + b
            hr_batch = self.training_hr3d[idx : end]
            lf_batch = self.training_lf2d[idx : end]

            if self.make_binary_mask:
                mask_batch = self.training_hr3d_mask[idx : end]
                yield (hr_batch, mask_batch), lf_batch, idx
            else:
                yield (hr_batch, ), lf_batch, idx
    
    def data(self):
        '''
        return the next batch of the training data
        '''
        nt = self.test_pair_num
        data_index = self._get_data_index(self.training_pair_num, shuffle=self.shuffle) + nt

        # for epoch in range(1, self.n_epochs + 1):
        for cursor in range(0, len(data_index) - self.batch_size + 1, self.batch_size):
            batch_idx = data_index[cursor : cursor + self.batch_size]
            hr_batch = self.training_hr3d[batch_idx]
            lf_batch = self.training_lf2d[batch_idx]

            if self.make_binary_mask:
                mask_batch = self.training_hr3d_mask[batch_idx]
                yield (hr_batch, mask_batch), lf_batch, cursor
            else:
                yield (hr_batch, ), lf_batch, cursor

    """
    def hasNext(self):
        return True if self.epoch < self.n_epochs else False
             
    def iter(self):
        '''
        return the next batch of the training data
        '''
        nt = self.test_pair_num
        if self.epoch < self.n_epochs:
            if self.cursor + self.batch_size > self.data_pair_num :
                self.epoch += 1
                self.cursor = nt

            idx = self.cursor
            end = idx + self.batch_size
            self.cursor += self.batch_size

            hr_batch = self.training_hr3d[idx : end]
            lf_batch = self.training_lf2d[idx : end]

            if self.make_binary_mask:
                mask_batch = self.training_hr3d_mask[idx : end]
                return np.asarray(hr_batch), np.asarray(mask_batch), np.asarray(lf_batch), idx - nt, self.epoch
            else:
                return np.asarray(hr_batch), np.asarray(lf_batch), idx - nt, self.epoch
                
        raise Exception('epoch index out of bounds:%d/%d' %(self.epoch, self.n_epochs))
    """


if __name__ == "__main__":
    test_dir = 'data/train/bead/aniso-f2_[m30-0]_step1um_N11/test'
    dataset = Dataset(train_hr3d_path=os.path.join(test_dir, 'WF'),
                        train_lf2d_path=os.path.join(test_dir, 'LF'),
                        n_slices=31, 
                        n_num=11,
                        lf2d_base_size=16,
                        test_split=0.)
    dataset.prepare(batch_size=2)

    save_dir = os.path.join(test_dir, 'write')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(3):
        for hr, lf, idx in dataset.data():
            # print('\repoch%d idx%d' % (epoch, idx), end='')
            write3d(lf, os.path.join(save_dir, 'lf_epoch%04d_idx%04d.tif' % (epoch, idx)), bitdepth=8) 