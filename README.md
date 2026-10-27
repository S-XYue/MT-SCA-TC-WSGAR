# Exploring Channel and Temporal Attention Constraints for Weakly Supervised Group Activity Recognition

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


