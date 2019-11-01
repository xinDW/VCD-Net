# LFRNet

Tensorflow implementation of training and using VCDNet for high-efficiency light field reconstruction. 

## Install

```
pip install -r requirement.txt
```
VCDNet requires an old-version Tensorlayer(contained in this repository) to run.

## Usage

#### Training 

```
python train.py [options]
```
options:

* `-c <begin_epoch>`   Load the existing checkpoint file and continue training from epoch  `begin_epoch`    

#### Inference

```
python eval.py [options]
```

options:

* `-c <ckpt_num>`      Epoch number of the checkpint file 
* `-b <batch_size>`  Batch size when infering, default is 1
* `-r`  Recursively inference all the tiff images found in the `valid_lr_path` and its sub-folders
*  `--cpu` Use cpu instead of gpu for inference. 
