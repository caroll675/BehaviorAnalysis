import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")


def rolling_variance(Dataframe, body_parts, window, pcutoff):
    rolling_df = Dataframe.rolling(window=window, center=True)
    var_list = []
    for i, rolled_df in tqdm(enumerate(rolling_df), desc="Rolling Parts", total=len(Dataframe)):
        var = get_rolling_variance(rolled_df, body_parts, pcutoff)
        var_list.append(var)
    Dataframe["var"] = pd.Series(var_list).fillna(method='ffill').fillna(method='bfill')
    Dataframe = Dataframe.replace('--', 0)
    Dataframe = Dataframe.astype(float)
    return Dataframe


def get_rolling_variance(Dataframe, body_parts, pcutoff):
    Dataframe = Dataframe.astype(float)
    var = 0
    for bp in body_parts:
        prob = Dataframe[[bp + "_likelihood"]].values.squeeze()
        mask = prob < pcutoff
        temp_x = np.ma.array(
            Dataframe[[bp + "_x"]].values.squeeze(),
            mask=mask,
        )
        temp_y = np.ma.array(
            Dataframe[[bp + "_y"]].values.squeeze(),
            mask=mask,
        )        
        var = var + np.var(temp_x) + np.var(temp_y)
    return var


def variance(Dataframe, window=90, pcutoff=0.7):
    body_parts = ['nose', 'leftear', 'rightear', 'neck', 'leftside', 'rightside', 'tailbase']
    # set new column names and reset index
    new_column_names = Dataframe.iloc[:2].apply(lambda x: '_'.join(map(str, x)), axis=0).tolist()
    Dataframe = Dataframe.set_axis(new_column_names, axis=1)
    Dataframe = Dataframe.drop([0, 1], axis=0)
    Dataframe = Dataframe.reset_index(drop=True)
    return rolling_variance(Dataframe, body_parts, window, pcutoff)

