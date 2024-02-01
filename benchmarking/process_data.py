# This file contains functions to clean and segment the accelerometer data for the experiments in
# the accompanying manuscript.
import math 
import numpy as np 
import pandas as pd

# Change dimension of accelerometer data to be in 5s increments, truncate extra data.
# Params: accel (dict) - the combined accelerometer data for each dataset, labels (dict) - the 
# labels for each dataset, accel_start (dict) - the common start times of each dataset's 
# accelerometer data, label_start (dict) - the label start times for each dataset, datasets (list) - 
# the randomized datasets using in training and evaluation, time (int) - length of sliding window, 
# freq (int) - the frequency data was collected at, directions (int) - represents triaxial 
# accelerometer data, num_sensors - number of sensors data is used from. 
# Returns: reshaped_dfs (dict) - the truncated 5s increment accelerometer data, label_dfs (dict) - 
# the truncated labels.
def reshape_dfs(
        dfs, 
        labels, 
        accel_start, 
        label_start, 
        datasets, 
        time, 
        frequency, 
        directions, 
        num_sensors):
    reshaped_dfs = {}
    label_dfs = {}
    for ds in datasets:
        to_reshape, labels_reshape = remove_excess_data(
            dfs[ds], 
            labels[ds],
            accel_start[ds],
            label_start[ds], 
            frequency)
        size = (to_reshape.shape[0] // frequency*time) * frequency*time
        temp = to_reshape.head(size).to_numpy() # get only the first size # of data points
        reshaped = np.reshape(temp, (-1, frequency*time, directions*num_sensors))
        reshaped_dfs[ds] = reshaped
        label_dfs[ds] = labels_reshape
    return reshaped_dfs, label_dfs        

# Remove extra accelerometer data from the beginning of the dataframe if the start times of the 
# accelerometer and label data do not align. 
# Params: accel (DataFrame) - the combined accelerometer data for a dataset, labels (DataFrame) - 
# the labels for a dataset, accel_start (dict) - the common start times a dataset's 
# accelerometer data, label_start (dict) - the label start times for a dataset, freq (int) - the 
# frequency data was collected at. 
# Returns: accel - accelerometer data with extra data removed, labels - labels with extra data 
# removed. 
def remove_excess_data(accel, labels, accel_start, label_start, freq):
    dif = math.ceil((label_start - accel_start).total_seconds())
    # if the labels start after the accel, remove the excess accel 
    if dif > 0: 
        accel = accel.tail(accel.shape[0] - (dif * freq))
        return accel, labels
    # if labels start before the accel, remove the excess labels 
    else: 
        labels = labels.tail(labels.shape[0] + dif)
        return accel, labels 

# Segment labels into 5s increments of time assuming the labels accross all 5s are consistent.
# Params: labels (DataFrame) - the labels for a dataset, time (int) - length of sliding window, 
# lab_labels (list) - activities considered across experiments.
# Returns: seg_labels (array) - labels segmented into 5s incremements. 
def determine_labels(labels, time, lab_labels):
    s = 0 
    for i in range(0, labels.shape[0], time):
        if labels.shape[0] < i+time and labels.shape[0] < i:
            if (~(labels.loc[i:i+time]["LABEL_NAME"].isin(lab_labels).all())):
                s = s + 1
                labels.at[i, "LABEL_NAME"] = "BAD"
            elif((labels.loc[i]["LABEL_NAME"] != labels.loc[i+time]["LABEL_NAME"])):
                s = s + 1
                labels.at[i, "LABEL_NAME"] = "BAD"
                
    labels = labels.drop(columns=["START_TIME"])
    seg_labels = labels[::time] # creates 5s window labels 
            
    return seg_labels

# Truncate the extra data from the larger DataFrame (accel or labels, normally accel).
# Params: accel (array) - the combined accelerometer data a single dataset, labels (dict) - the 
# labels for a dataset, time (int) - length of sliding window, lab_labels (list) - activities 
# considered across experiments.
# Returns: accel (array) - acelerometer data truncated to the size of the labels, labels (DataFrame)
# - labels truncated to reflect the size of the accelerometer data (in the rare case it is larger).
def resize_dfs_for_cleaning(accel, labels, time, lab_labels):
    labels = determine_labels(labels, time, lab_labels)
    
    desired_size = np.amin(np.array([accel.shape[0], labels.shape[0]]))
    if accel.shape[0] == desired_size:
        labels = labels.head(desired_size)
    else:
        accel = accel[:desired_size]

    return accel, labels

# Remove accelerometer data that does not align with desired activities, segment data into 5s 
# windows.  
# Params: accel (array) - the combined accelerometer data a single dataset, labels (dict) - the 
# labels for a dataset, time (int) - length of sliding window, lab_labels (list) - activities 
# considered across experiments.
# Returns: clean_accel (array) - accelerometer data segmented into 5s activities, renumbered_labels 
# (array) - only labels that are in our expereiment.
def clean_data(accel, labels, time, lab_labels):
    # add labels to accel df
    clean_accel, labels_new = resize_dfs_for_cleaning(accel, labels, time, lab_labels)
    renumbered_labels = pd.DataFrame(labels_new.to_numpy(), columns=["label"])

    index_unlabeled = renumbered_labels.index[~(renumbered_labels.isin(lab_labels).any(axis=1))] 
    renumbered_labels.drop(index_unlabeled, inplace=True)
    clean_accel = np.delete(clean_accel, index_unlabeled, axis=0)
    
    renumbered_labels = renumbered_labels.to_numpy()

    return clean_accel, renumbered_labels

# Perform all preprocessing steps (e.g., align data stream start times, seperate out the activities 
# we are using, and segment data into 5s windows).
# Params: accel (dict) - the combined accelerometer data for each dataset, labels (dict) - the 
# labels for each dataset, accel_start (dict) - the common start times of each dataset's 
# accelerometer data, label_start (dict) - the label start times for each dataset, datasets (list) - 
# the randomized datasets using in training and evaluation, time (int) - length of sliding window, 
# freq (int) - the frequency data was collected at, directions (int) - represents triaxial 
# accelerometer data, num_sensors (int) - number of sensors data is used from, lab_labels (list) - 
# activities considered across experiments.
# Returns: clean_accel (dict) - preprocessed (5s windows of) accelerometer data, clean_labels (dict) 
# - a dictionary of preprocessed (5s windows of) label data.
def clean_all_data(
        accel, 
        labels, 
        accel_start, 
        label_start, 
        datasets, 
        time, 
        freq, 
        directions, 
        num_sensors,
        lab_labels):
    
    clean_accel = {}
    clean_labels = {}

    reshaped_dfs, reshaped_labels = reshape_dfs(
        accel, 
        labels, 
        accel_start, 
        label_start, 
        datasets, 
        time, 
        freq, 
        directions, 
        num_sensors)
    
    for ds in datasets:
        clean_accel[ds], clean_labels[ds] = clean_data(
            reshaped_dfs[ds],
            reshaped_labels[ds], 
            time, 
            freq,
            lab_labels)
                
    return clean_accel, clean_labels