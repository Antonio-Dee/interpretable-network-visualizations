import pandas as pd
import numpy as np
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import math, re
from translation_utils import *
from plot_utils import *

N_LABELS = 3

def compute_score(frequency, average_percentile, is_correct):
    score = frequency * (-0.1*average_percentile+1)
    if is_correct: return score
    else: return score/4

def split_dataset_for_cluster_and_features(df, original_df, resigns_df, unlabeled_df, weights_list, label_name, label_id):
    datalist = []
    for cluster_id in df['cluster_id'].drop_duplicates():
        df_cluster = df[df.cluster_id == cluster_id]
        values_count = df_cluster['synonym'].value_counts().to_dict()
        for characteristic in df_cluster['synonym'].drop_duplicates():
            frequency = values_count[characteristic]
            average_percentile = df_cluster[df_cluster.synonym == characteristic]['percentile'].mean()
            is_correct = not original_df[original_df.guess_id.isin(df_cluster[df_cluster.synonym == characteristic]['guess_id'])][original_df.guessed_label_id == label_id].empty
            #score = frequency**2/(1+average_percentile)
            score = compute_score(frequency, average_percentile, is_correct)
            datalist.append({
                'characteristic': characteristic,
                'score': score,
                'frequency': frequency,
                'average_percentile': average_percentile,
                'cluster_id': list(df_cluster['cluster_id'])[0],
            })
    df_char = pd.DataFrame(data=datalist)

    datalist = []
    class_df = original_df[original_df.label_en == label_name]
    for cluster_id in class_df['cluster_id'].drop_duplicates():
        df_cluster = class_df[class_df.cluster_id == cluster_id]
        cluster_id = list(df_cluster['cluster_id'])[0]
        layer = list(df_cluster['layer'])[0]
        cluster = list(df_cluster['cluster'])[0]
        image = list(df_cluster['image'])[0]
        image_id = [int(s) for s in re.findall(r'\d+', image)][1]
        weights_of_cluster = weights_list[image_id][layer][cluster]
        win_count = len(original_df[original_df.label_en == label_name][original_df.guessed_label_id == label_id][original_df.cluster_id == cluster_id]['guess_id'].value_counts())
        win_count += unlabeled_df[unlabeled_df.cluster_id == cluster_id][unlabeled_df.label == label_id][unlabeled_df.guessed_label_id == label_id].shape[0]
        loss_count = len(original_df[original_df.label_en == label_name][original_df.guessed_label_id != label_id][original_df.cluster_id == cluster_id]['guess_id'].value_counts())
        loss_count += unlabeled_df[unlabeled_df.cluster_id == cluster_id][unlabeled_df.label == label_id][unlabeled_df.guessed_label_id != label_id].shape[0]
        if not resigns_df[resigns_df.cluster_id == cluster_id].empty:
            resign_count = resigns_df[resigns_df.cluster_id == cluster_id]['number_of_resigns'].to_list()[0]
        else:
            resign_count = 0
        cluster_percentile = original_df[original_df.cluster_id == cluster_id]['percentile'].to_list()
        cluster_percentile += unlabeled_df[unlabeled_df.cluster_id == cluster_id]['percentile'].to_list()
        cluster_percentile += [5] * resign_count
        cluster_percentile = math.ceil(np.mean(cluster_percentile))
        datalist.append({
                'cluster_id': cluster_id,
                'layer': layer,
                'cluster': cluster,
                'image': image,
                'image_id': image_id,
                'weight': weights_of_cluster[0],
                'fmap_count': weights_of_cluster[2],
                'win_count': win_count,
                'loss_count': loss_count,
                'resign_count': resign_count,
                'cluster_percentile': cluster_percentile
            })
    df_cluster = pd.DataFrame(datalist)

    return df_char, df_cluster

def compute_best_characteristics(df_char, df_cluster):
    best_weighted = []
    for index, row in df_cluster.iterrows():
        cluster_id = row['cluster_id']
        df_cluster_char = df_char[df_char.cluster_id == cluster_id]
        best_char_value = np.max(df_cluster_char['score'])
        best_chars = df_cluster_char[df_cluster_char.score == best_char_value]['characteristic'].to_list()
        row['best_characteristics'] = best_chars
        best_weighted.append(row)

    df_cluster = pd.DataFrame(best_weighted)

    return df_cluster

def merge_clusters_on_best_characteristics(df_cluster):
    datalist = []
    for image_id in df_cluster['image_id'].drop_duplicates():
        df_image = df_cluster[df_cluster.image_id == image_id]
        for layer in df_image['layer'].drop_duplicates():
            df_layer = df_image[df_image.layer == layer]
            df_exploded_chars = df_layer.explode('best_characteristics')
            
            char_count = dict(df_exploded_chars['best_characteristics'].value_counts() > 1)
            must_be_merged = []
            for key, value in char_count.items():
                if value: must_be_merged.append(df_exploded_chars[df_exploded_chars.best_characteristics == key]['cluster_id'].to_list())

            changed = True
            while changed:
                changed = False
                for i in range(0, len(must_be_merged)):
                    for j in range(i+1, len(must_be_merged)):
                        for id in must_be_merged[i]:
                            if id in must_be_merged[j]:
                                temp = list(set(must_be_merged[i] + must_be_merged[j]))
                                old1 = must_be_merged[i]
                                old2 = must_be_merged[j]
                                must_be_merged.remove(old1)
                                must_be_merged.remove(old2)
                                must_be_merged.append(temp)
                                changed = True
                            if changed: break
                        if changed: break
                    if changed: break

            single_groups = []
            for cluster_id in df_layer['cluster_id'].drop_duplicates():
                present = False
                for cluster_array in must_be_merged:
                    if cluster_id in cluster_array: 
                        present = True
                        break
                if not present: single_groups.append([cluster_id])

            cluster_groups = must_be_merged + single_groups

            i = 0
            for group in cluster_groups:
                df_group = df_layer[df_layer.cluster_id.isin(group)]
                datalist.append({
                    'cluster_id': group,
                    'layer': layer,
                    'cluster': i,
                    'image': df_group['image'].to_list()[0],
                    'image_id': image_id,
                    'weight': sum(df_group['weight'].to_list()),
                    'fmap_count': sum(df_group['fmap_count'].to_list()),
                    'win_count': sum(df_group['win_count'].to_list()),
                    'loss_count': sum(df_group['loss_count'].to_list()),
                    'resign_count': sum(df_group['resign_count'].to_list())
                })
                i+=1
    
    return pd.DataFrame(datalist).sort_values(by=['image_id','layer','cluster'])

def merge_all_clusters(df_cluster: pd.DataFrame):
    datalist = []
    for image_id in df_cluster['image_id'].drop_duplicates():
        df_image = df_cluster[df_cluster.image_id == image_id]
        for layer in df_image['layer'].drop_duplicates():
            df_layer: pd.DataFrame = df_image[df_image.layer == layer]
            datalist.append({
                    'cluster_id': df_layer['cluster_id'].to_list(),
                    'layer': layer,
                    'cluster': 0,
                    'image': df_layer['image'].to_list()[0],
                    'image_id': image_id,
                    'weight': sum(df_layer['weight'].to_list()),
                    'fmap_count': sum(df_layer['fmap_count'].to_list()),
                    'win_count': sum(df_layer['win_count'].to_list()),
                    'loss_count': sum(df_layer['loss_count'].to_list()),
                    'resign_count': sum(df_layer['resign_count'].to_list())
                })

    return pd.DataFrame(datalist).sort_values(by=['image_id','layer','cluster'])

def create_and_save_plot(cluster_group, row, df_char, df_cluster, heatmap_df, image_folder, label_name, merged = None, image_index = None):
    heatmaps = heatmap_df[heatmap_df.id.isin(cluster_group)].sort_values(by='id')['heatmap'].to_list()
    heatmaps = parse_heatmaps(heatmaps)
    weights = df_cluster[df_cluster.cluster_id.isin(cluster_group)].sort_values(by='cluster_id')['weight'].to_list()
    heatmap = compute_average_heatmap(heatmaps, weights)
    
    original_image = get_image(row['image'], image_folder)
    image_height = len(np.asarray(original_image))
    image_width = len(np.asarray(original_image)[0])
    average_percentiles = [df_cluster[df_cluster.cluster_id == cluster]['cluster_percentile'].to_list()[0] for cluster in cluster_group]
    percentile = compute_group_percentile(heatmaps, average_percentiles, image_height, image_width)

    group_mask, true_percentile = compute_group_mask(heatmap, percentile, original_image, image_height, image_width)

    overlay_image = compute_overlay(original_image, heatmap, image_height, image_width)
    
    data = df_char[df_char.cluster_id.isin(cluster_group)]
    data_dict = dict()
    total_weight = sum(df_cluster[df_cluster.cluster_id.isin(cluster_group)]['weight'].values)
    for c in data['characteristic'].drop_duplicates().to_list():
        c_data = data[data.characteristic == c]
        char_score = 0
        for cluster in c_data['cluster_id'].to_list():
            char_score += df_cluster[df_cluster.cluster_id == cluster]['weight'].values[0]*c_data[c_data.cluster_id == cluster]['score'].values[0]
        data_dict.update({c: char_score/total_weight})
    sorted_data = dict(sorted(data_dict.items(), key=lambda x: (-x[1],x[0]))[:10])
    features_present = 2 if len(list(sorted_data.keys())) == 0 else 1

    fig = gridspec.GridSpec(int(2/features_present), 2)
    pl.figure(figsize=(18, int(12/features_present)))
    ax1 = pl.subplot(fig[0, 0])
    ax2 = pl.subplot(fig[0, 1])

    pl.suptitle('LAYER '+str(row['layer']+5)+' - CLUSTER: '+str(row['cluster']+1))
    ax1.imshow(group_mask)
    ax1.axis('off')
    ax1.set_title('GAMES INFO\nAverage Revealed Percentage = '+str(np.round(true_percentile,2))+'%\nWins / Resigns / Losses: '+str(row['win_count'])+' / '+str(row['resign_count'])+' / '+str(row['loss_count']))

    ax2.imshow(overlay_image)
    ax2.axis('off')
    ax2.set_title('CLUSTER INFO\nCluster Importance = '+str(np.round(row['weight'],2))+'%\nNumber of Feature Maps = '+str(row['fmap_count']))

    if features_present == 1:
        ax3 = pl.subplot(fig[1, :])
        ax3.set_title('FEATURES INFO')
        bottom = np.zeros(len(sorted_data.values()))
        i = 1
        for cluster in cluster_group:
            label = 'Subcluster '+str(i)
            temp_df = df_char[df_char.cluster_id == cluster]
            for characteristic in sorted_data.keys():
                score = temp_df[temp_df.characteristic == characteristic]['score'].to_list()[0] if not temp_df[temp_df.characteristic == characteristic].empty else 0
                score = score*df_cluster[df_cluster.cluster_id==cluster]['weight'].values[0]/total_weight
                sorted_data[characteristic] = score
            ax3.bar(sorted_data.keys(), sorted_data.values(), label=label, bottom=bottom)
            bottom += np.array(list(sorted_data.values()))
            i+=1
        if i > 2:ax3.legend()
        ax3.set_yticks([])
    else: bottom = []

    fig_file_path = get_file_name_for_plot(label_name, row['image_id'], row['layer']+5, row['cluster']+1, merged, image_index, str(np.round(row['weight'],2)))
    pl.savefig(fig_file_path + 'main.png',bbox_inches='tight')

    preview_keys = list(sorted_data.keys())[:N_LABELS]
    preview_values = list(bottom)[:3]

    if len(preview_keys) <= 0:
         preview_keys.append('')
         preview_values.append(0.0)
    if len(preview_keys) <= 1:
         preview_keys.append(' ')
         preview_values.append(0.0)
    if len(preview_keys) <= 2:
         preview_keys.append('  ')
         preview_values.append(0.0)


    fig = gridspec.GridSpec(int(2/features_present), 1)
    fig2 = pl.figure(figsize=(1.5, 3/features_present))
    ax1 = pl.subplot(fig[0, 0])
    ax1.imshow(overlay_image)
    ax1.set_xticks([])
    ax1.set_yticks([])
    preview_label = 'w = ' + str(np.round(row['weight'],1))+'%'
    ax1.set_title(label=preview_label, fontsize=10, y=-.225)
    if features_present == 1:
        ax2 = pl.subplot(fig[1, 0])
        plottemp = ax2.barh(preview_keys, preview_values, color=['#09D6B4A0', '#32ABDBA0', '#FFD15BA0'])
        ax2.set_xticks([])
        ax2.set_yticks([])
        i = 0
        for bar in ax2.patches:
                if i == N_LABELS: break
                value = preview_keys[i]
                text = ax2.text(
                    bar.get_x()+.01, bar.get_y()+bar.get_height()/1.5, value
                )
                
                i += 1
        ax2.invert_yaxis()
        ax2.set_frame_on(False)
    pl.savefig(fig_file_path + 'preview.png',bbox_inches='tight')

def make_plots_for_explainer(df_cluster_merged, df_char, df_cluster, heatmap_df, image_folder, label_name):
    print()
    print('Generating plots...')
    plot_counter = 1
    total_plots = df_cluster_merged.shape[0]
    for index, row in df_cluster_merged.iterrows():
        if len(row['cluster_id']) == 1:
            create_and_save_plot(row['cluster_id'],row,df_char, df_cluster, heatmap_df, image_folder, label_name)
            print(plot_counter, '/', total_plots, ' '*20, end='\r')
        else:
            create_and_save_plot(row['cluster_id'],row,df_char, df_cluster, heatmap_df, image_folder, label_name, merged=True)
            print(plot_counter, '( 1 /',str(len(row['cluster_id'])+1), ') /', total_plots, end='\r')
            for i in range(len(row['cluster_id'])):
                cluster_id = row['cluster_id'][i]
                row_temp = row.copy()
                row_temp['weight'] = df_cluster[df_cluster.cluster_id == cluster_id]['weight'].to_list()[0]
                row_temp['fmap_count'] = df_cluster[df_cluster.cluster_id == cluster_id]['fmap_count'].to_list()[0]
                row_temp['win_count'] = df_cluster[df_cluster.cluster_id == cluster_id]['win_count'].to_list()[0]
                row_temp['loss_count'] = df_cluster[df_cluster.cluster_id == cluster_id]['loss_count'].to_list()[0]
                row_temp['resign_count'] = df_cluster[df_cluster.cluster_id == cluster_id]['resign_count'].to_list()[0]
                create_and_save_plot([cluster_id],row_temp,df_char, df_cluster, heatmap_df, image_folder, label_name, merged=False, image_index=i+1)
                print(plot_counter, '(', str(i+2) ,'/',str(len(row['cluster_id'])+1), ') /', total_plots, end='\r')

        plot_counter+=1
    print('DONE :)', ' '*20)
    print()

def make_fullmege_plots_for_explainer(df_cluster_fullmerge, df_char, df_cluster, heatmap_df, image_folder, label_name):
    print()
    print('Generating plots...')
    plot_counter = 1
    total_plots = df_cluster_fullmerge.shape[0]
    for index, row in df_cluster_fullmerge.iterrows():
        create_and_save_fullmerge_plot(row['cluster_id'], row, df_char, df_cluster, heatmap_df, image_folder, label_name)
        print(plot_counter, '/', total_plots, ' '*20, end='\r')
        plot_counter+=1
    print('DONE :)', ' '*20)
    print()

def create_and_save_fullmerge_plot(cluster_group, row, df_char, df_cluster, heatmap_df, image_folder, label_name, merged = None, image_index = None):
    heatmaps = heatmap_df[heatmap_df.id.isin(cluster_group)].sort_values(by='id')['heatmap'].to_list()
    heatmaps = parse_heatmaps(heatmaps)
    weights = df_cluster[df_cluster.cluster_id.isin(cluster_group)].sort_values(by='cluster_id')['weight'].to_list()
    heatmap = compute_average_heatmap(heatmaps, weights)
    
    original_image = get_image(row['image'], image_folder)
    image_height = len(np.asarray(original_image))
    image_width = len(np.asarray(original_image)[0])
    average_percentiles = [df_cluster[df_cluster.cluster_id == cluster]['cluster_percentile'].to_list()[0] for cluster in cluster_group]
    percentile = compute_group_percentile(heatmaps, average_percentiles, image_height, image_width)

    group_mask, true_percentile = compute_group_mask(heatmap, percentile, original_image, image_height, image_width)

    overlay_image = compute_overlay(original_image, heatmap, image_height, image_width)
    
    data = df_char[df_char.cluster_id.isin(cluster_group)]
    data_dict = dict()
    total_weight = sum(df_cluster[df_cluster.cluster_id.isin(cluster_group)]['weight'].values)
    for c in data['characteristic'].drop_duplicates().to_list():
        c_data = data[data.characteristic == c]
        char_score = 0
        for cluster in c_data['cluster_id'].to_list():
            char_score += df_cluster[df_cluster.cluster_id == cluster]['weight'].values[0]*c_data[c_data.cluster_id == cluster]['score'].values[0]
        data_dict.update({c: char_score/total_weight})
    sorted_data = dict(sorted(data_dict.items(), key=lambda x: (-x[1],x[0]))[:10])
    features_present = 2 if len(list(sorted_data.keys())) == 0 else 1
    fig = gridspec.GridSpec(int(2/features_present), 2)
    pl.figure(figsize=(18, int(12/features_present)))
    ax1 = pl.subplot(fig[0, 0])
    ax2 = pl.subplot(fig[0, 1])

    pl.suptitle('LAYER '+str(row['layer']+5)+' - CLUSTER: '+str(row['cluster']+1))
    ax1.imshow(group_mask)
    ax1.axis('off')
    ax1.set_title('GAMES INFO\nAverage Revealed Percentage = '+str(np.round(true_percentile,2))+'%\nWins / Resigns / Losses: '+str(row['win_count'])+' / '+str(row['resign_count'])+' / '+str(row['loss_count']))

    ax2.imshow(overlay_image)
    ax2.axis('off')
    ax2.set_title('CLUSTER INFO\nCluster Importance = '+str(np.round(row['weight'],2))+'%\nNumber of Feature Maps = '+str(row['fmap_count']))

    if features_present == 1:
        ax3 = pl.subplot(fig[1, :])
        ax3.set_title('FEATURES INFO (first 10 features)')
        bottom = np.zeros(len(sorted_data.values()))
        i = 1
        for cluster in cluster_group:
            label = 'Subcluster '+str(i)
            temp_df = df_char[df_char.cluster_id == cluster]
            for characteristic in sorted_data.keys():
                score = temp_df[temp_df.characteristic == characteristic]['score'].to_list()[0] if not temp_df[temp_df.characteristic == characteristic].empty else 0
                score = score*df_cluster[df_cluster.cluster_id==cluster]['weight'].values[0]/total_weight
                sorted_data[characteristic] = score
            ax3.bar(sorted_data.keys(), sorted_data.values(), label=label, bottom=bottom)
            bottom += np.array(list(sorted_data.values()))
            i+=1
        if i > 2:ax3.legend()
        ax3.set_yticks([])
        i=0
        for bar in ax3.patches:
            if i == len(bottom): break
            height = bottom[i]
            ax3.text(
                bar.get_x() + bar.get_width() / 2, height, (str(np.round(height,2))), ha="center", va="bottom"
            )
            i += 1

    fig_file_path = get_file_name_for_plot_fullmerge(label_name, row['image_id'], row['layer']+5, row['cluster']+1, merged, image_index, str(np.round(row['weight'],2)))
    pl.savefig(fig_file_path + 'main.png',bbox_inches='tight')

    preview_keys = list(sorted_data.keys())[:6]
    preview_values = list(bottom)[:6]

    if len(preview_keys) <= 0:
         preview_keys.append('')
         preview_values.append(0.0)
    if len(preview_keys) <= 1:
         preview_keys.append(' ')
         preview_values.append(0.0)
    if len(preview_keys) <= 2:
         preview_keys.append('  ')
         preview_values.append(0.0)
    if len(preview_keys) <= 3:
         preview_keys.append('   ')
         preview_values.append(0.0)
    if len(preview_keys) <= 4:
         preview_keys.append('    ')
         preview_values.append(0.0)
    if len(preview_keys) <= 5:
         preview_keys.append('     ')
         preview_values.append(0.0)
         

    fig = gridspec.GridSpec(3, 1)
    fig2 = pl.figure(figsize=(1.5, 6))
    ax1 = pl.subplot(fig[0, 0])
    ax2 = pl.subplot(fig[1, 0])
    preview_label = 'w = ' + str(np.round(row['weight'],1))+'%'
    ax1.set_title(label=preview_label, fontsize=10, y=-.225)
    plottemp = ax2.barh(preview_keys, preview_values, color=['#09D6B4A0', '#32ABDBA0', '#FFD15BA0'])
    ax2.set_xticks([])
    ax2.set_yticks([])
    i = 0
    for bar in ax2.patches:
            if i == 6: break
            value = preview_keys[i]
            text = ax2.text(
                bar.get_x()+.01, bar.get_y()+bar.get_height()/1.5, value
            )
            
            i += 1
    ax2.invert_yaxis()
    ax2.set_frame_on(False)
    ax1.imshow(overlay_image)
    ax1.set_xticks([])
    ax1.set_yticks([])
    pl.savefig(fig_file_path + 'preview.png',bbox_inches='tight')