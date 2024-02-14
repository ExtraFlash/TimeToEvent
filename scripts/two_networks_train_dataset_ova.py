import numpy as np
from SurvSet.data import SurvLoader
from utils import two_networks

if __name__ == '__main__':
    loader = SurvLoader()
    # list of available datasets
    # print(loader.df_ds.head())
    # load a dataset and print its head
    df, ref = loader.load_dataset(ds_name='ova').values()
    train_dataset, val_dataset, test_dataset = two_networks.train_val_test_split(df)

    coef1_grid = np.arange(0.1, 1.1, 0.3)
    coef2_grid = np.arange(0.1, 1.1, 0.3)
    for coef1 in coef1_grid:
        for coef2 in coef2_grid:
            # print(f'coef1: {coef1}, coef2: {coef2}')
            two_networks.train_model(train_dataset, val_dataset, 1.0, 1.0)
