# LFRNet

Tensorflow implementation of training and using VCDNet for high-efficiency light field reconstruction. 

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
VCDNet requires an old-version Tensorlayer(contained in this repository) to run.

## Usage

#### Training 

Download the example training dataset from [Google Drive](https://drive.google.com/file/d/1vFP1dX7hKmctt7DVQ0-MQ8fBYwcDIYwc/view?usp=sharing) and unzip to the repository folder. 

Then run

```
python train.py [options]
```
options:

* `-c <begin_epoch>`   Load the existing checkpoint file and continue training from epoch designated by begin_epoch    

#### Inference

An example light-field measurement can be found at `example_data/valid/`. To reconstruct a 3-D image, run :
```
python eval.py [options]
```

options:

* `-c <ckpt_num>`    Epoch number of the checkpint file 

* `-b <batch_size>`  Batch size when infering, default is 1

*  `--cpu`           Use cpu instead of gpu for inference. 

The results will be saved at `example_data/valid/VCD/`
