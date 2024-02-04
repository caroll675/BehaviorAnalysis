import select_roi as sr

def clustering(corner_arr, Dataframe):
    # FIRST CLUSTERING CRITERIA
    Dataframe['var'] = Dataframe['var'].replace('--', 1).fillna(1).astype(float)
    Dataframe['var'] = gaussian_moving_average(Dataframe['var'], sigma=2)
    Dataframe["cluster"] = Dataframe["var"].apply(lambda x: "immobility" if x < 100 else ("nonlocomotion" if x < 10000 else "locomotion"))
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


from collections import Counter

def moving_mode(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        window = data[max(i-window_size+1, 0):i+1]
        mode_value = Counter(window).most_common(1)[0][0]
        smoothed_data.append(mode_value)
    return smoothed_data


import numpy as np
from scipy.ndimage import gaussian_filter1d

def gaussian_moving_average(data, sigma):
    smoothed_data = gaussian_filter1d(data, sigma)
    return smoothed_data