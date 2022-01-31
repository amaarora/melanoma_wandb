from unicodedata import name
import torch
import numpy as np
from pathlib import Path
import os
import wandb
import logging
logging.getLogger().setLevel(logging.INFO)

class EarlyStopping:
    def __init__(self, total_epochs, patience=3, mode="max", save_mode='best'):
        self.total_epochs = total_epochs
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.early_stop = False
        self.save_mode = save_mode
        # 'min' means lower is better and vice-versa for 'max' 
        self.best_score = np.Inf if self.mode == "min" else -np.Inf
        self.best_model_path = None

    def __call__(self, epoch, epoch_score, model, model_path):
        score = -1.0 * epoch_score if self.mode == "min" else epoch_score
        if score > self.best_score:
            self.counter = 0
            self.best_model_path = model_path
            logging.info("Validation score improved ({} --> {})".format(self.best_score, score))
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            if self.save_mode=='all': 
                self.save_artifact("model.pth", model_path)
        else:
            self.counter += 1
            logging.info("Early stopping counter {} of {}.".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        
        # store best model to wandb
        if epoch==self.total_epochs or self.early_stop:
            self.save_artifact("best-model.pth", self.best_model_path)

    def save_checkpoint(self, epoch_score, model, model_path):
        model_path = Path(model_path)
        parent = model_path.parent
        os.makedirs(parent, exist_ok=True)
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model saved at {model_path}")

    def save_artifact(self, name, file_path):
        artifact = wandb.Artifact(name, type='model')
        artifact.add_file(file_path)
        wandb.run.log_artifact(artifact)