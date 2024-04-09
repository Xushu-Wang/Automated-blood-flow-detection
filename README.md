# Automated Blood Flow Detection

This repo contains a PyTorch implementation of U-Net related automated blood flow detection deep learning networks.

## Dependency

The following are packages needed for running this repo.


- PyTorch==2.0.1
- torchvision
- numpy
- pandas
- cv2
- matplotlib
- json

## Running the Experiment

1. For network architecture, add specific configuration in config folder


2. run the train script

```python train.py --config [configuration]```


Recommended hardware: 2 NVIDIA Tesla P-100 GPUs 


## Dataset Source

- The Breast Ultrasound Images Dataset (BUSI) : https://datasetninja.com/breast-ultasound-images
- Private Dataset (Not disclosed)

