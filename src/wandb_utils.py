import wandb 
import os
import scipy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from collections import defaultdict


def log_wandb_table(data_dir, df, table_name, n_sample=100):
    wandb_table = wandb.Table(columns=['Image Name', 'Image', 'Target', 'Diagnosis', 'Sex'])
    sample_df   = df.sample(n=n_sample).reset_index(drop=True).copy()
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        img_path = os.path.join(data_dir, row.image_name+'.jpg')
        wandb_table.add_data(
            row.image_name, 
            wandb.Image(Image.open(img_path)), 
            row.benign_malignant, 
            row.diagnosis, 
            row.sex
        )
    wandb.run.log({table_name: wandb_table})


def log_oof_wandb(oof_df, frac=1.):
    oof_df = oof_df.sample(frac=frac).reset_index(drop=True)
    wandb_table = wandb.Table(columns=['Image Name', 'Image', 'Prediction', 'Target'])
    for idx, row in tqdm(oof_df.iterrows(), total=len(oof_df)):
        wandb_table.add_data(
            os.path.basename(row.image_path),
            wandb.Image(Image.open(row.image_path)), 
            scipy.special.expit(row.prediction), 
            row.target, 
        )
    wandb.run.log({'Predictions': wandb_table})


class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_names):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self._features = defaultdict(list)
        
        layer_dict = dict([*self.model.named_modules()])
        for layer_name in layer_names:
            layer = layer_dict[layer_name]
            layer.register_forward_hook(self.save_outputs_hook(layer_name))

    def save_outputs_hook(self, layer_name):
        def fn(_, __, output): 
            self._features[layer_name] = output
        return fn

    def forward(self, **kwargs):
        _ = self.model(**kwargs)
        return self._features


def log_features_to_wandb(epoch, model, valid_loader, args, layer_name='base_model._dropout'):
    import pdb; pdb.set_trace()
    model.eval()
    fx = FeatureExtractor(model, [layer_name])
    features=[]; labels=[]
    with torch.no_grad():
        tk0 = tqdm(valid_loader, total=len(valid_loader))
        for data in tk0:
            for key, value in data.items():
                data[key] = value.to(args.device)
            out = fx(**data, args=args)[layer_name]
            features.append(out.detach().cpu().numpy())
            labels.append(data['target'].cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels   = np.concatenate(labels, axis=0)
    # create a pandas DataFrame - easiest to log with W&B IMHO
    cols = [f"out_{i}" for i in range(features.shape[1])]
    df   = pd.DataFrame(features, columns=cols)
    df['LABEL'] = labels
    
    # Log pandas table to W&B to create Image Embedding Table
    table = wandb.Table(columns=df.columns.to_list(), data=df.values)
    wandb.run.log({f"Features at Epoch - {epoch}": table})