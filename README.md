# VCDNet

Tensorflow implementation of training and using VCDNet for high-efficiency light field reconstruction. 

## Requirements

* Python 3.6.7
* (Optional but highly recommended) CUDA 10.2 and CUDNN 

## Install

Clone the code repository using Git or just download the zip file. Download and the example data and the trained model parameters from [Google Drive](https://drive.google.com/file/d/1oN83xto69ouzjqEOXaQWFJQG7AreZH5f/view?usp=sharing). The directory tree should be (after the example data and model parameters being downloaded and extracted): 
```    
.
├── model
├── tensorlayer
├── config.py
├── dataset.py
├── eval.py
├── train.py
├── utils.py
├── data
    └── checkpoint
        └── bead_40x_n11_[m30-30]_step1um_sparse
        └── cardiac_20x_n11_[m50-50]_step2um_dense
        └── rbcDSRED_20x_n11_[m50-50]_step2um_sparse
    └── example_data
        └── train
        └── valid
            └── bead(sparse)
                └── bead_40x_[m30-30].tif
            └── blood_cell(sparse)
                └── rbc_tail_001.tif
                └── rbc_tail_002.tif
                └── rbc_tail_003.tif
            └── cardiac_wall(dense)
                └── heart_001.tif
                └── heart_002.tif
                └── heart_003.tif
            └── expected_output.zip
        
```

To set up the environment, use:
```
pip install -r requirement.txt
```
VCDNet requires an old-version Tensorlayer(1.8.1, contained in this repository) to run.

## Usage

#### Training 

A training dataste for bead samples is contained in the example data. Download it from [Google Drive](https://drive.google.com/file/d/1oN83xto69ouzjqEOXaQWFJQG7AreZH5f/view?usp=sharing).
Then run

```
python train.py [options]
```
options:

* `-c <begin_epoch>`   Load the existing checkpoint file and continue training from epoch designated by begin_epoch   

To train the VCD-Net on your own dataset, the following should be done:
1. Change the settings of the training dataset in `config.py`, including:
```
    config.img_size=<lateral size (px) of the LFPs and 3-D targets>
    config.PSF.n_slices=<number of slices of the 3-D targets>                                                            
    config.PSF.Nnum     = <N number of the LF data>
    config.TRAIN.lf2d_path  = <Path-to-the-LF-Projections>     
    config.TRAIN.target3d_path = <Path-to-the-3D-targets>   
```
2. Name the label of this training in `config.py`, as a symbol of the trained model.
    label =  <'a-distinguishable-label'>   
We recommand that the label includes the following parameters that are most frequently changed based on our trial:
* The sample info  
* The magnification of the objective of the LFM system.
* The N number of the LF data.
* The physical z-range (um) of the 3-D reconstructions.
* The z-step size (um) of the 3-D reconstructions.
* The model type.

There are two variants of the VCDNet provided in current code, for the reconstructing of the sparse signals (e.g. beads, neurons, blood cells, etc.) and the dense signals (e.g. cardiomyocytes), respectively. With slight different layer parameters and structures, they proved to be efficient in our applications. The program will choose the appropriate one according to your label. 
For example,
 ```
 label=zebrafish_20x_N11_[m50-50]_step2um_dense
 ``` 
 the keyword 'dense' indicates that a VCDNet for dense signals will be trained on a dataset of the 20x zebrafish images, of which the LFs have the N number 11, and the 3-D targets with 2-um step size are located at -50um ~ 50um.  

The parameters of the trained model will be saved in the corresponding directory under the `repository_root/checkpoint/`. Later at the inference stage, the program will load the trained model parameters according to the designated label.

#### Inference

Example light-field measurements of beads and blood cells of a zebrafish tail can be found at `data/example_data/valid/`. To reconstruct a 3-D image, 
change the settings and the input path in `config.py`:

```
# To infer the bead sample:
config.PSF.n_slices          = 61
config.PSF.Nnum              = 11
label                        = 'bead_40x_n11_[m30-30]_step1um_sparse' 
config.VALID.lf2d_path       = '/data/example_data/valid/bead(sparse)/'  

# or to infer the blood cell sample:
config.PSF.n_slices          = 51
config.PSF.Nnum              = 11
label                        = 'rbcDSRED_20x_n11_[m50-50]_step2um_sparse' 
config.VALID.lf2d_path       = '/data/example_data/valid/blood_cell/'  
```

Then run :
```
python eval.py [options]
```

options: 

* `-b <batch_size>`  Batch size when infering, default is 1

*  `--cpu`           Use cpu instead of gpu for inference, if a CUDA-enabled GPU is not available. 

The results will be saved at a new folder under the directory of the input images. 

To run the reconstructing process on your own LF measurements using the VCDNet trained on your own dataset, change the label and the settings to appropriate ones (See  section "Training" for details), and change `config.VALID.lf2d_path` to the path of your LF measurements. Then run the command as above.
