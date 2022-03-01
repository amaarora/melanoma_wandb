# Melanoma Experiments + Weights and Biases Integration
Welcome to the repository! If you don't know what this repository is about, don't worry! That's what this README is for. This repository contains all working code for the [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification) kaggle competition that should get you in the top-5% of the leaderboard! 

I used W&B heavily to track all my experiments, run hyperparameter sweeps, store datasets as W&B tables, store model weights as model artifacts after every epoch and also use the embedding projector to interpret what model learned. 

I have also written four reports with detailed explanations to explain each step:

- [How to prepare the dataset for the Melanoma Classification?](https://wandb.ai/amanarora/melanoma/reports/How-to-prepare-the-dataset-for-the-Melanoma-Classification---VmlldzoxNjI4NTkz): This report showcases how to download and prepare the dataset for the Melanoma competition. We also store the dataset similar to the Weights and Biases tables as below called "Train data" and "Validation data".
- [How to track all your experiments using Microsoft Excel?](https://wandb.ai/amanarora/melanoma/reports/How-to-track-all-your-experiments-using-Microsoft-Excel---VmlldzoxNTY3MjQ2): In this report, I showcase how I used W&B to track all experiments that I ran as part of the Melanoma competition. We learn how to build beautiful dashboards such as the one below as part of this report.
- [How to save all your trained model weights locally after every epoch](https://wandb.ai/amanarora/melanoma/reports/How-to-save-all-your-trained-model-weights-locally-after-every-epoch--VmlldzoxNTkzNjY1): As part of this report I showcase how I utilized W&B artifacts to store all model weights for each of the experiments after every epoch. This makes it really easy for me later when ensembling models to pick the best performing models and create an ensemble for top-5 or top-10 models. 
- [How to Build a Robust Medical Model Using Weights & Biases](https://wandb.ai/amanarora/melanoma/reports/How-to-Build-a-Robust-Medical-Model-Using-Weights-Biases--VmlldzoxNTM3NzY3): As part of this report, I showcase how I integrated various products of Weights and Biases ecosystem such as Tables, Sweeps, Artifacts and Embedding Projector to really get in the top-5% of the leaderboard on Kaggle! 

You can also find an example dashboard that this GitHub repository creates here - [Melanoma W&B Dashboard](https://wandb.ai/amanarora/melanoma?workspace=user-amanarora).

## Downloading the Dataset
To download the dataset, we can use the Kaggle API. Please run the following line of code to get the Melanoma dataset:

```kaggle competitions download -c siim-isic-melanoma-classification```

## Data Preprocessing 
For a detailed explanation on data preprocessing - please refer to the report [How to prepare the dataset for the Melanoma Classification?](https://wandb.ai/amanarora/melanoma/reports/How-to-prepare-the-dataset-for-the-Melanoma-Classification---VmlldzoxNjI4NTkz). 

The line of code that you want to run once you've downloaded the dataset is below: 
```python 
python resize_images.py --input_folder <path to input data> --output_folder <path to output folder> --cc --mantain_aspect_ratio --sz 256
```

## Model Training 
Once you have downloaded and pre-processed the dataset, we are now ready to perform model training. The training script automatically logs all experiments to Weights and Biases. 

Please run the following line of code to kick-off model training. (Note that you will need a machine with a GPU to run model training)

```python 
python train.py --model_name efficient_net --arch_name efficientnet-b0 --device cuda --metric 'auc' --training_folds_csv /home/arora/git_repos/melanoma_wandb/data/train_folds.csv --train_data_dir /home/arora/git_repos/melanoma_wandb/data/usr/resized_train_256_cc --kfold 0 --pretrained imagenet --train_batch_size 64 --valid_batch_size 64 --learning_rate  5e-4 --epochs 10 --sz 224 --loss 'weighted_focal_loss'
```
