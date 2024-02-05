import pandas as pd
import numpy as np
import select_roi as sr
import rolling_variance as rv
import clustering_criteria as cc
import os
from tqdm import tqdm
import cv2

def process_video(df_path, df_clustered_path, video_path, video_clustered_path, subset=True, start=0, end=10000):
    """
    This script processes a video by applying clustering to the DLC data file and creating an edited video with cluster labels.

    Parameters:
    - df_path (str): Path to the DLC original data file in CSV format.
    - df_clustered_path (str): Path to store the clustered data file in CSV format.
    - video_path (str): Path to the original video file.
    - video_clustered_path (str): Path to store the edited video with cluster labels.
    - subset (bool, optional): Flag to indicate whether to process a subset of frames for testing purposes. Default is True.
    - start (int, optional): Starting frame index for processing. Default is 0.
    - end (int, optional): Ending frame index for processing. Default is 10000.
    """
    # Check if the clustered data file already exists
    if os.path.isfile(df_clustered_path):
        print("Message")
        print("File exists")
        clustered_dataframe = pd.read_csv(df_clustered_path)
    else:
        print("Message")
        print("File does not exist")
        original_dataframe = rv.variance(pd.read_csv(df_path), window=90, pcutoff=0.7, subset=subset, start=start, end=end)
        # corner_arr = sr.select_rect_roi(video_path)
        corner_arr = None
        # food_pellet = sr.select_circular_ROI(video_path)
        clustered_dataframe = cc.clustering(corner_arr, original_dataframe, df_clustered_path)
        # Save the clustered dataframe
        clustered_dataframe.to_csv(df_clustered_path, index=False)

    # Execute until this part

    # Map the cluster labels to the videos
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    text_size = 0.5
    text_thickness = 2

    # Save the edited video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video = cv2.VideoWriter(video_clustered_path, fourcc, fps, (width, height))

    frame_number = 0
    pbar = tqdm(total=len(clustered_dataframe))  # Create a tqdm progress bar

    while frame_number < len(clustered_dataframe):
        ret, frame = cap.read()

        if ret:
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            cv2.putText(frame, ('cluster: {}, var {}').format(
                clustered_dataframe['cluster'][frame_number-1], round(clustered_dataframe['var'][frame_number-1], 2)
            ), (10, 40), font, text_size, text_color, text_thickness)

            cv2.imshow('frame', frame)
            output_video.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            pbar.update(1)

        else:
            break

    pbar.close()
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    df_path = ...
    df_clustered_path = ...
    video_path = ...
    video_clustered_path = ...

    process_video(df_path, df_clustered_path, video_path, video_clustered_path, subset=True, start=0, end=10000)
