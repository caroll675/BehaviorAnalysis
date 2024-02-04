import pandas as pd
import select_roi as sr
import numpy as np
import rolling_variance as rv
import clustering_criteria as cc
import bp_variance as bpv
import os
from tqdm import tqdm
import cv2

df_path = ...
df_clustered_path = ...
video_path = ...
video_clustered_path = ...

if os.path.exists(df_clustered_path):
    print("Message")
    print("File exists")
    Dataframe = pd.read_csv(df_clustered_path)
else:
    print("Message")
    print("File does not exist")
    Dataframe = rv.variance(pd.read_csv(df_path), window=90, pcutoff=0.7)
    # corner_arr = sr.select_rect_roi(video_path)
    corner_arr = None
    # food_pellet = sr.select_circular_ROI(video_path)
    Dataframe = cc.clustering(corner_arr, Dataframe)
    # save the clustered dataframe
    Dataframe.to_csv(df_clustered_path, index=False)

# excute until this part 

# map the cluster labels to the videos
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (0, 0, 0)
text_size = 0.5
text_thickness = 2

# save the editted video 
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = cap.get(cv2.CAP_PROP_FPS)
output_video = cv2.VideoWriter(video_clustered_path, fourcc, fps, (width, height))



frame_number = 0
pbar = tqdm(total=len(Dataframe))  # Create a tqdm progress bar

while frame_number < len(Dataframe):
    ret, frame = cap.read()

    if ret:
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        cv2.putText(frame, ('cluster: {}, var {}').format(
             Dataframe['cluster'][frame_number-1], round(Dataframe['var'][frame_number-1], 2)
             ), (10, 40), font, text_size, text_color, text_thickness)
        #cv2.putText(frame, ('second possible behavior: {}').format(
        #        Dataframe['cluster2'][frame_number]), (10, 80), font, text_size, text_color, text_thickness)


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