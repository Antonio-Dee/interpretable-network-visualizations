import pandas as pd
import numpy as np
from explainations_utils import *

import warnings
warnings.filterwarnings('ignore')

# CONSTANTS FILES AND FOLDERS
ORIGINAL_DATASET_FILE_NAME = 'dataset_processed.csv'
HEATMAP_DATASET_FILE_NAME = 'heatmaps.csv'
RESIGNS_DATASET_FILE_NAME = 'db_for_data_analysis_only_resigns.csv'
UNLABELED_DATASET_FILE_NAME = 'additional_games.csv'
WEIGHTS_FILE_NAME = 'weights.npy'
IMAGE_FOLDER = 'images/'

# LABEL TO BE STUDIED
LABELS = ['Cassette Player', 'Chainsaw', 'Church', 'Dog', 'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute', 'Fish']
LABELS_ID = [1,2,3,4,5,6,7,8,9,10]

# ENABLE TRANSLATOR
BOOL_TRANSLATE = False

#Read original datasets
original_df = pd.read_csv(ORIGINAL_DATASET_FILE_NAME)
df = original_df[~original_df.synonym.isna()]
heatmap_df = pd.read_csv(HEATMAP_DATASET_FILE_NAME)
resigns_df = pd.read_csv(RESIGNS_DATASET_FILE_NAME)
weights_list = np.load(WEIGHTS_FILE_NAME, allow_pickle=True)
unlabeled_df = pd.read_csv(UNLABELED_DATASET_FILE_NAME)
#to_translate = input('Do you want to translate? (y/n)')

def generate(df, LABEL_STRING, LABEL_ID):

    df = df[df.label_en == LABEL_STRING]
    #Print some info
    print()
    print('DATA ANALYSIS ON LABEL', LABEL_STRING.upper())
    print('\tCharacteristic count:', len(df.index))
    print('\tUnique characteristic count:', len(df['characteristic'].drop_duplicates()))
    print('\tCluster count:', len(df['cluster_id'].drop_duplicates()))
    print()

    #Compute scores for each characteristic and split dataset into cluster info and characteristics
    df_char, df_cluster = split_dataset_for_cluster_and_features(df, original_df, resigns_df, unlabeled_df, weights_list, LABEL_STRING, LABEL_ID)

    #Get best characteristics for each cluster
    df_cluster = compute_best_characteristics(df_char, df_cluster)

    #Merging clusters that are mostly seeing the same thing
    df_cluster_merged = merge_clusters_on_best_characteristics(df_cluster)

    df_cluster_fullmerge = merge_all_clusters(df_cluster)

    return df_cluster_merged, df_char, df_cluster, df_cluster_fullmerge

for i in range(len(LABELS_ID)):
    df_cluster_merged, df_char, df_cluster, df_cluster_fullmerge = generate(df, LABELS[i], LABELS_ID[i])
    
    #Compute plots for explainations and save them
    make_plots_for_explainer(df_cluster_merged, df_char, df_cluster, heatmap_df, IMAGE_FOLDER, LABELS[i])
    make_fullmege_plots_for_explainer(df_cluster_fullmerge, df_char, df_cluster, heatmap_df, IMAGE_FOLDER, LABELS[i])


#Generate file for visualizer
import os, re

plot_folder = 'plots'
labels = os.listdir(plot_folder)
json_container = dict()

for label in labels:

    label_folder = plot_folder + '/' + label
    images = os.listdir(label_folder)

    image_list = []

    for image in images:

        image_folder = label_folder + '/' + image
        layers = os.listdir(image_folder)
        layers.sort(key=int)

        layer_list = []

        for layer in layers:

            layer_folder = image_folder + '/' + layer
            plots = os.listdir(layer_folder)
            
            unmerged_list = []
            merged_list = []
            full_list = []
            for plot in plots:

                plot_path = layer_folder + '/' + plot

                if not re.match('M', plot) and not re.match('F', plot):
                    unmerged_list.append(plot_path)
                if not re.match('U', plot) and not re.match('F', plot):
                    merged_list.append(plot_path)
                if re.match('F', plot):
                    full_list.append(plot_path)

            unmerged_list = sorted(unmerged_list, key=lambda s: float(s.split('$')[1]), reverse=True) if unmerged_list is not [] else []
            merged_list = sorted(merged_list, key=lambda s: float(s.split('$')[1]), reverse=True) if merged_list is not [] else []

            layer_list.append([unmerged_list, merged_list, full_list])
        image_list.append(layer_list)
    json_container.update({label:image_list})

with open('json.js', 'w') as outfile:
    outfile.write('data = '+json_container.__str__())