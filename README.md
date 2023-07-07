<div align="center">    
 
# Few-Zero shot learning using CLIP architecture as backbone
## Proudly implemented in Pytorch Lightning

![CI testing](https://github.com/sergiuoprea/clip_with_few_shots/workflows/CI%20testing/badge.svg?branch=master&event=push)

</div>
 
## Description   
This is a simple 3-layer multi-class classifier working on top of features extracted from [CLIP architecture](https://github.com/openai/CLIP). The provided dataset is imbalanced, so we have used a WeightedRandomSampler to ensure the DataLoader retrieves the samples with equal probability. Also, training set was splitted into the train and validation splits using the latter to choice the best performing model. On the other side, we have tested ViT-B/32 and ViT-B/16 CLIP models. The latter provided the best results in our case.

## Goals
* Design a system to recognize novel object types from a few images used for training.
* Start from pretrained [CLIP architecture](https://github.com/openai/CLIP)
* Few shot task: develop a classification model trained and tested on the provided data.
* Zero shot task: as an extension of the few shot task, and using no data for training.

## Provided dataset
* For training: a dataset (/data/train) consisting of a small set of training images, i.e. 10-20 samples per object class.
* For testing: a dataset (/data/test) providing some images reserved to evaluate the system.

## How to run
### Requirement
* #### Torch env with GPU 
```
torch.cuda.is_available()
```
True
* #### Python 3.10
```
python --version
```
python = 3.10.11

### Install dependency
```
pip install -r requirement.txt
```
### Run with default config 
```
python main.py 
```

### Run with custom config
```
python main.py --path_to_data <datapath> --batch_size <batchsize> --max_epochs <max_epoch> --learning_rate <lr>
```


## Some results 

### Confusion matrix
The horizontal and vertical axes indices correspond with the following classes:

```
['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'truck']
```


We notice a confussion between cars and trucks due to the similarities between both.

### Training and validation accuracies
```
Check WandB
```
