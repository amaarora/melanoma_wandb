import wandb 
import os
import scipy
from tqdm import tqdm
from PIL import Image


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

