# Detector-Free Weakly Supervised Group Activity Recognition

### [Dongkeun Kim](https://dk-kim.github.io/), [Jinsung Lee](https://cvlab.postech.ac.kr/lab/members.php), [Minsu Cho](https://cvlab.postech.ac.kr/~mcho/), [Suha Kwak](https://suhakwak.github.io/)

### [Project Page](http://cvlab.postech.ac.kr/research/DFWSGAR/) | [Paper](https://arxiv.org/abs/2204.02139)

## Overview
This work introduces a detector-free approach for weakly supervised group activity recognition. 


This repository also contains the implementation used for *Exploring Channel and Temporal Attention Constraints for Weakly Supervised Group Activity Recognition*. The current release provides the Volleyball and NBA training and evaluation paths. Dataset files and model weights are not redistributed.

## Citation
If you find our code or paper useful, please consider citing our paper:

    @misc{Xiang2026ChannelTemporal,
      title  = {Exploring Channel and Temporal Attention Constraints for Weakly Supervised Group Activity Recognition},
      author = {Xiang, Ye and Sun, Xinyue and Ding, Ruixuan and Wu, Lifang},
      year   = {2026},
      note   = {Manuscript under review}
    }

## Requirements

- Ubuntu 16.04
- Python 3.8.5
- CUDA 11.0
- PyTorch 1.7.1

## Conda environment installation
    conda env create --file environment.yml

    conda activate gar
    
## Install additional package
    sh scripts/setup.sh
    
## Download dataset
- Volleyball dataset <br/>
Download Volleyball dataset from:   <br/> 
https://drive.google.com/file/d/1DaUE3ODT_H5mBFi8JzOVBNzVldxfbPbX/view?usp=sharing      
Dataset should be located following the file structure described below. <br/>

- NBA dataset <br/>
The dataset is available upon request to the authors of 
  "Social Adaptive Module for Weakly-supervised Group Activity Recognition (ECCV 2020)". 
  

## Trained weights

Pretrained weights are not included in this repository. Replace `your/checkpoint/path/model.pth` in the evaluation scripts with a local checkpoint.

If an official checkpoint URL and SHA-256 checksum are available, use:

    bash scripts/download_checkpoints.sh your/checkpoint/url your/checkpoint/sha256 checkpoints model.pth

## Run test scripts

Before running, replace `your/data/path/` and `your/checkpoint/path/model.pth` in the corresponding script.

- Volleyball Dataset (Original 8 class classification)

    sh scripts/test_volleyball.sh

- NBA Dataset

    sh scripts/test_nba.sh

## Run train scripts

Before running, replace `your/data/path/` in the corresponding script.

- Volleyball Dataset (Original 8 class classification)

    sh scripts/train_volleyball.sh

- NBA Dataset

    sh scripts/train_nba.sh

## File structure

│── Dataset/ <br/>
│   │── volleyball/ <br/>
│   │    └── videos/ <br/>
│   │── NBA_dataset/ <br/>
│   │    └── videos/ <br/>
│   │    └── train_video_ids <br/>
│   │    └── test_video_ids <br/>
│── scripts/ <br/>
│── dataloader/ <br/>
│── models/ <br/>
│── util/ <br/>
train.py <br/>
train_nba_now.py <br/>
test.py <br/>
test_NBA.py <br/>
README.md <br/> 
environment.yml <br/>


## Acknowledgement
This work was supported by the NRF grant and the IITP grant funded by Ministry of Science and ICT, Korea (NRF-2021R1A2C3012728, NRF-2018R1A5A1060031, IITP-2020-0-00842, IITP-2021-0-00537, No. 2019-0-01906 Artificial Intelligence Graduate School Program-POSTECH). 

