import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde

def find_cutoff(df, df_path):
    # variance before rolling average
    df['var'] = np.log1p(df['var'])

    # Apply K-means clustering to segment 'var' into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=0).fit(df[['var']])
    df['cluster'] = kmeans.labels_

    # Sort the DataFrame by 'var' to find cutoffs between clusters
    df_sorted = df.sort_values(by='var')
    df['cluster'] = kmeans.labels_


    # Plot KDEs for each cluster and mark the cutoffs
    plt.figure(figsize=(10, 6))
    intersection_points = []
    for i in range(3):
        subset = df[df['cluster'] == i]['var']
        
        # Compute KDE for the current subset
        kde = gaussian_kde(subset)
        
        # Evaluate the KDE on a common range of values
        x_vals = np.linspace(subset.min(), subset.max(), 1000)
        y_vals = kde(x_vals)
        
        # Plot the KDE
        sns.lineplot(x=x_vals, y=y_vals, label=f'Cluster {i+1}')
        
        ip = np.intersect1d(x_vals, x_vals, assume_unique=True)
        intersection_points.append(ip.min())
        intersection_points.append(ip.max())

    
    intersection_points = sorted(intersection_points)
    cutoff1 = (intersection_points[1] + intersection_points[2])/2
    cutoff2 = (intersection_points[3] + intersection_points[4])/2
    plt.axvline(x=cutoff1, color='k', linestyle='--', label=f'Cutoff {np.round(cutoff1, 3)}')
    plt.axvline(x=cutoff2, color='k', linestyle='--', label=f'Cutoff {np.round(cutoff2, 3)}')

    plt.title('Distribution of Variance by Cluster')
    plt.ylabel('Density')
    plt.xlabel('Log-transformed Variance')

    # Ensure legends are not duplicated
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(df_path[:-3]+'.png')

    return cutoff1, cutoff2


# df_path = "D:\cchen\LCstim\LCNEOPstim-cchen-2024-01-27\\videos\clustered\\test\Opto-Photo_Recording_5minStim-231118-230400_M169-M170-231119-223741_Cam1DLC_resnet50_LCNEOPstimJan27shuffle1_1030000.csv"
# find_cutoff(pd.read_csv(df_path), df_path)