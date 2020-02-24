# VCDNet

Tensorflow implementation of training and using VCDNet for high-efficiency light field reconstruction. 

## Requirements

* Python 3.6.7
* (Optional but highly recommended) CUDA 10.2 and CUDNN 

## Install

Download the repository and unzip. The directory tree should be: 
```    
.
├── config.py
├── custom.py
├── dataset.py
├── eval.py
├── model.py
├── rdn.py
├── tensorlayer
├── train.py
├── utils.py
├── example_data
    └── train
        └── bead
            └── [m30-30]step1um
                ├── LF
                └── WF
```

To set up the environment, use:
```
pip install -r requirement.txt
```
VCDNet requires an old-version Tensorlayer(1.8.1, contained in this repository) to run.

## Usage

#### Training 

Download the example training dataset from [Google Drive](https://drive.google.com/file/d/1vFP1dX7hKmctt7DVQ0-MQ8fBYwcDIYwc/view?usp=sharing) and unzip to the repository folder. 

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
    config.TRAIN.lf2d_path  = <Path-to-the-LFPs>     
    config.TRAIN.target3d_path = <Path-to-the-3D-targets>   
```
2. Name the label of this training in `config.py`, as a symbol of the trained model.
    label =  <'a-distinguishable-label'>   
We recommand that the label includes the following information:
* The sample type  
* The magnification of the objective of the LFM system.
* The N number of the LF data.
* The physical z-range (um) of the 3-D reconstructions.
* The z-step size (um) of the 3-D reconstructions.
* The model type.

There are two variants of the VCDNet, for the reconstructing of the sparse signals and the dense signals, respectively. The program will choose the appropriate one according to your label. 
For example,
 ```
 label=zebrafish_20x_N11_[m50-50]_step2um_dense
 ``` 
 indicates that a VCDNet for dense signals will be trained on a dataset of the 20x zebrafish images, of which the LFs have the N number 11, and the 3-D targets with 2-um step size are located at -50um ~ 50um.  

The parameters of the trained model will be saved in the corresponding directory under the `repository_root\checkpoint\`. At the inference stage, the program will load the trained model parameters according to the designated label.

#### Inference

An example light-field measurement can be found at `example_data/valid/`. To reconstruct a 3-D image, run :
```
python eval.py [options]
```

options: 

* `-b <batch_size>`  Batch size when infering, default is 1

*  `--cpu`           Use cpu instead of gpu for inference, if a CUDA-enabled GPU is not available. 

The results will be saved at `example_data/valid/VCD/`

To run the reconstructing process on your own LF measurements using the VCDNet trained on your own dataset, change the label to the appropriate one (See  section "Training" for details), and change `config.VALID.lf2d_path` to the path of your LF measurements. Then run the command as above.
