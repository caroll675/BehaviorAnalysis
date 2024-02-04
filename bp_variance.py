import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def rolling_variance(Dataframe,window, pcutoff):
    rolling_df = Dataframe.rolling(window=window)
    var_nose = []
    var_leftear = []
    var_rightear = []
    var_neck = []
    var_leftside = []
    var_rightside = []
    var_tailbase = []
    var_list = []

    for i, rolled_df in enumerate(rolling_df):
        print(f"Rolling part {i}")
        nose = get_rolling_variance(rolled_df, 'nose', pcutoff)
        leftear = get_rolling_variance(rolled_df, 'leftear', pcutoff)
        rightear = get_rolling_variance(rolled_df, 'rightear', pcutoff)
        neck = get_rolling_variance(rolled_df, 'neck', pcutoff)
        leftside = get_rolling_variance(rolled_df, 'leftside', pcutoff)
        rightside = get_rolling_variance(rolled_df, 'rightside', pcutoff)
        tailbase = get_rolling_variance(rolled_df, 'tailbase', pcutoff)
        var = nose + leftear + rightear + neck + leftside + rightside + tailbase
        var_list.append(var)
        var_nose.append(nose)
        var_leftear.append(leftear)
        var_rightear.append(rightear)
        var_neck.append(neck)
        var_leftside.append(leftside)
        var_rightside.append(rightside)
        var_tailbase.append(tailbase)
    Dataframe["var_nose"] = pd.Series(var_nose).fillna(method='ffill').fillna(method='bfill')
    Dataframe["var_leftear"] = pd.Series(var_leftear).fillna(method='ffill').fillna(method='bfill')
    Dataframe["var_rightear"] = pd.Series(var_rightear).fillna(method='ffill').fillna(method='bfill')
    Dataframe["var_neck"] = pd.Series(var_neck).fillna(method='ffill').fillna(method='bfill')
    Dataframe["var_leftside"] = pd.Series(var_leftside).fillna(method='ffill').fillna(method='bfill')
    Dataframe["var_rightside"] = pd.Series(var_rightside).fillna(method='ffill').fillna(method='bfill')
    Dataframe["var_tailbase"] = pd.Series(var_tailbase).fillna(method='ffill').fillna(method='bfill')
    Dataframe["var"] = pd.Series(var_list).fillna(method='ffill').fillna(method='bfill')
    Dataframe = Dataframe.replace('--', 0)
    Dataframe = Dataframe.astype(float)
    return Dataframe

def get_rolling_variance(Dataframe, bp, pcutoff):
    Dataframe = Dataframe.astype(float)
    var = 0
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


def variance(Dataframe, window=120, pcutoff=0.7):
    body_parts = ['nose', 'leftear', 'rightear', 'neck', 'leftside', 'rightside', 'tailbase']
    # set new column names and reset index
    new_column_names = Dataframe.iloc[:2].apply(lambda x: '_'.join(map(str, x)), axis=0).tolist()
    Dataframe = Dataframe.set_axis(new_column_names, axis=1)
    Dataframe = Dataframe.drop([0, 1], axis=0)
    Dataframe = Dataframe.reset_index(drop=True)
    return rolling_variance(Dataframe, window, pcutoff)

