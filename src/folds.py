from sklearn import model_selection
import os
import pandas as pd 
import numpy as np


FRAC    = 0.3
DATADIR = "/home/arora/git_repos/melanoma_wandb/data"


if __name__ == '__main__':
    kf = model_selection.StratifiedKFold(n_splits=5)
    df = pd.read_csv(os.path.join(DATADIR, 'train.csv'))
    df['kfold'] = -1

    # shuffle
    df = df.sample(frac=0.3).reset_index(drop=True)
    targets = df.target.values
    for fold, (train_index, test_index) in enumerate(kf.split(X=df[['image_name']], y=targets)):
        df.loc[test_index, 'kfold'] = fold
    df.to_csv(os.path.join(DATADIR, "train_folds.csv"), index=False)
