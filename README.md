# Exploring Channel and Temporal Attention Constraints for Weakly Supervised Group Activity Recognition


## Overview

This repository provides the implementation of
"Exploring Channel and Temporal Attention Constraints for Weakly Supervised
Group Activity Recognition."

The proposed framework introduces a Multi-Threshold Sparse Channel Attention
Module to suppress irrelevant visual information and select discriminative
channel features. It further employs a Temporal Constraint Mechanism and
Temporal Constraint Loss to guide temporal attention toward the ordered
occurrence of sub-activities.

The implementation supports experiments on the Volleyball and NBA datasets.
Dataset files and pretrained model weights are not redistributed.

## Citation

Publication information will be updated after the paper is published.

    @misc{xiang_exploring_channel_temporal,
      title  = {Exploring Channel and Temporal Attention Constraints for Weakly Supervised Group Activity Recognition},
      author = {  },
      note   = {Manuscript}
    }

## Requirements

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
README.md <br/> 
environment.yml <br/>


## Acknowledgement

This work was supported by the National Natural Science Foundation of China
(Grant No. 62106011).

## Codebase Acknowledgement

This codebase is built upon Detector-Free Weakly Supervised Group Activity
Recognition (DFWSGAR) by Dongkeun Kim, Jinsung Lee, Minsu Cho, and Suha Kwak.

- Original project: http://cvlab.postech.ac.kr/research/DFWSGAR/
- Original paper: https://arxiv.org/abs/2204.02139
