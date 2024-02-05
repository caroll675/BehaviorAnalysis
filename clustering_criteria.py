import select_roi as sr
import cutoff 
import numpy as np
from scipy.ndimage import gaussian_filter1d
from collections import Counter

def clustering(corner_arr, Dataframe, df_path):
    # FIRST CLUSTERING CRITERIA
    Dataframe['var'] = Dataframe['var'].replace('--', 1).fillna(1).astype(float)
    Dataframe['log_var'] = np.log1p(Dataframe['var'])
    cutoff1, cutoff2 = cutoff.find_cutoff(Dataframe, df_path)
    print(cutoff1, cutoff2)
    # Dataframe['var'] = gaussian_moving_average(Dataframe['var'], sigma=2)
    Dataframe["cluster"] = Dataframe["log_var"].apply(lambda x: "immobility" if x < cutoff1 else ("nonlocomotion" if x < cutoff2 else "locomotion"))
    # SECOND CLUSTERING CRITERIA
    if corner_arr:
        Dataframe['cluster2'] = ['None'] * len(Dataframe)
        for i in range(len(Dataframe)):
            nose_x = Dataframe["nose_x"][i]
            nose_y = Dataframe["nose_y"][i]
            neck_x = Dataframe["neck_x"][i]
            neck_y = Dataframe["neck_y"][i]
            if not sr.is_in_rect_roi(nose_x, nose_y, corner_arr
                ) and Dataframe["cluster"][i] == "locomotion": Dataframe["cluster2"][i] = "rearing"
            if not sr.is_in_rect_roi(neck_x, neck_y, corner_arr): Dataframe["cluster2"][i] = "rearing"
    Dataframe["cluster"] = moving_mode(Dataframe['cluster'], 15)
    if corner_arr:
        Dataframe["cluster2"] = moving_mode(Dataframe['cluster2'], 15)
    return Dataframe


def moving_mode(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        window = data[max(i-window_size+1, 0):i+1]
        mode_value = Counter(window).most_common(1)[0][0]
        smoothed_data.append(mode_value)
    return smoothed_data



def gaussian_moving_average(data, sigma):
    smoothed_data = gaussian_filter1d(data, sigma)
    return smoothed_data