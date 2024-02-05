import os
import pandas as pd
import rolling_variance as rv
import clustering_criteria as cc


def process_spreadsheets(folder_path):
    """
    Process DLC spreadsheets located in the specified folder_path.
    
    Parameters:
    - folder_path (str): Path containing DLC spreadsheets.

    The clustered spreadsheets will be stored in the directory specified by folder_path under the 'clustered' subfolder.
    """
    items = os.listdir(folder_path)
    for item in items:
        if item.endswith('csv'):
            itempath = os.path.join(folder_path, item)

            clustered_path = os.path.join(folder_path, 'clustered')
            if not os.path.exists(clustered_path):
                os.makedirs(clustered_path)

            item_clustered_path = os.path.join(clustered_path, item[:-4]+'_clustered.csv')
            print(item_clustered_path)
            
            if not os.path.exists(item_clustered_path):
                print("Message")
                print("File does not exist")
                Dataframe = rv.variance(pd.read_csv(itempath), window=90, pcutoff=0.7, subset=False, start=0, end=4000)
                # corner_arr = sr.select_rect_roi(video_path)
                corner_arr = None
                # food_pellet = sr.select_circular_ROI(video_path)
                Dataframe = cc.clustering(corner_arr, Dataframe, item_clustered_path)
                # save the clustered dataframe
                Dataframe.to_csv(item_clustered_path, index=False)
            else:
                print("Message")
                print("File exists")
           
if __name__ == "__main__":
    process_spreadsheets(folder_path = "D:\cchen\LCstim\LCNEOPstim-cchen-2024-01-27\\videos")
