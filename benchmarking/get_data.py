# This file contains functions to obtain accelerometer and labeled data from its original source.
from datetime import datetime
import math 
import numpy as np 
import os 
import pandas as pd

# Obtain the labels and the start time of the first label for all datasets.
# Params: datasets (list) - the randomized datasets using in training and evaluation.
# Retuns: labels_dict (dict) - label data, label_starts (dict) - label start times.
def get_labels(datasets):
    labels_dict = {}
    label_set = "PA TYPE_corr"
    for ds in datasets:
        print(ds)
        path = #TODO: add path name
        for r, _, f in os.walk(path):
            if ds < 113 and "newScheme" in r:
                    for name in f:
                        if label_set in name:
                            csv = os.path.join(r, name)
                            if os.path.exists(csv):
                                df = pd.read_csv(
                                    csv,
                                    parse_dates=['START_TIME','STOP_TIME'], 
                                    infer_datetime_format=True)
            elif ds >= 113 or ds == 29 or ds == 58 or (ds >= 10 and ds <=18): 
                for name in f:
                        if label_set in name:
                            csv = os.path.join(r, name)
                            if os.path.exists(csv):
                               df = pd.read_csv(
                                    csv,
                                    parse_dates=['START_TIME','STOP_TIME'], 
                                    infer_datetime_format=True)
        
        labels_dict[ds] = resample_labels(df, 1)

    label_starts = get_label_start_times(datasets, labels_dict)
    print(labels_dict.keys())
    return labels_dict, label_starts

# Resample labels to be in one second incremements. If an acitivity lasted n seconds, this activity
# will have n 1s labels.
# Params: og_labels_df (df) - labels where start/stop_time are in HH:MM:SS, s (int) - the number of 
# seconds we want per label (default 1).
# Returns: sec_labels_df (df) - a resamapled dataframe of the original labels.
def resample_labels(og_labels_df, s = 1):
    og_labels_df['START_TIME'] = og_labels_df['START_TIME'].dt.round('1s')
    og_labels_df['STOP_TIME'] = og_labels_df['STOP_TIME'].dt.round('1s')
    
    temp_times = [(pd.date_range(
        e.START_TIME, 
        e.STOP_TIME, 
        freq='1S'), e.PREDICTION) for e in og_labels_df.itertuples()]    
    res = [pd.DataFrame(data={'START_TIME': line[0], 'LABEL_NAME': line[1]}) for line in temp_times]
    
    sec_labels = pd.concat(res, axis=0, ignore_index=True)
    sec_labels.sort_values(by=['START_TIME'],inplace=True)
    sec_labels.set_index('START_TIME',inplace=True)

    g = lambda x: x.iat[0] if len(set(x))==1 else pd.NA
    
    sec_labels_df = sec_labels.resample(str(s)+'s').agg({'LABEL_NAME':g})    
    sec_labels_df.reset_index(inplace=True)
    
    return sec_labels_df

# Obtain the overall start time of a participants labels.  
# Params: datasets (list) - the randomized datasets using in training and evaluation, labels (list)- 
# label data (from get_labels()).
# Returns: start_times (dict) - label start times for each dataset. 
def get_label_start_times(datasets, labels):
    start_times = {}
    for ds in datasets:
        start_times[ds] = labels[ds].loc[0]['START_TIME']
    return start_times

# Get the datetime of when the accelerometer data collection began.
# Params: file (string) - DS accelerometer filename.
# Returns: dt (datetime) - a datetime when that accel file started.
def get_accel_start_time(file):
    odf = pd.read_csv(file,sep=' ',header=None,skiprows=2,nrows=2)
    odf = odf.loc[:,2]
    dt_str = odf[1] +' '+odf[0]
    dt = datetime.strptime(dt_str, "%m/%d/%Y %H:%M:%S")
    print("accel start is:", type(dt))
    return dt  

# Get sensor data for specific sensors (wrist, ankle, thigh).
# Params: datasets (list) - the randomized datasets using in training and evaluation, sensors (list) 
# - the sensors we are using data accelerometer from.
# Returns: accel_dfs (dict) - the accelerometer data, accel_start_times (dict) - a dictionary of 
# the accelerometer start times.
def get_accel_data(datasets, sensors):
    accel_dfs = {}
    accel_start_times = {}
    for ds in datasets:
        print(ds)
        accel_path = #TODO: enter path name
        for r, _, f in os.walk(accel_path):
            for name in f: 
                if ".csv" in name and name != "sync_sensor.csv" and name != "sync_video.csv" \
                    and "IMU" not in name and "1sec" not in name:
                    for sens in sensors: # sensors are passed in as "Thigh", "Wrist", or "Ankle"
                        if f"Right{sens}" in name or f"Right{sens.lower()}" in name:
                            print(name)
                            f = os.path.join(r, name)
                            df = pd.read_csv(f, skiprows=10)
                            if len(sensors) > 1:
                                accel_start_times[f'{ds} {sens}'] = get_accel_start_time(f)
                                accel_dfs[f'{ds} {sens}'] = df
                            else:
                                accel_start_times[ds] = get_accel_start_time(f)
                                print(type(accel_start_times[ds]))
                                accel_dfs[ds] = df
    return accel_dfs, accel_start_times   

# It is possible accelerometers begin at differet time, this function removes rows of data from the 
# beginning of the accelerometer data until all sensors' start times match. 
# Params: accel (dict) - accel data, start_times (dict) - accel start times, datasets (list) - the 
# randomized datasets using in training and evaluation, sensors (list) - the sensors we are using 
# accelerometer data from, freq (int) - the frequency data was collected at.
# Returns: dfs (dict) - updated accels dataframes, times (dict) - updated start_times accel.
def remove_unlabelled_start_data_from_accel(accel, start_times, datasets, sensors, freq):
    times = {}
    for ds in datasets:
            m = []
            for s in sensors:
                m.append(start_times[f'{ds} {s}'])
            m = np.max(m) # get the latest start time.
            times[ds] = m # update the universal start time of the accel data to be the latest time

            for s in sensors:
                accel[f'{ds} {s}'] = pd.DataFrame(
                    accel[f'{ds} {s}'].tail(
                        accel[f'{ds} {s}'].shape[0] - (math.ceil(
                            (m - start_times[f'{ds} {s}']).total_seconds()) * freq)).to_numpy(), 
                    columns=["Accelerometer X", "Accelerometer Y", "Accelerometer Z"])
    return accel, times

# Merge accelerometer data from multiple sensors into a single dataframe.
# Params: datasets (list) - the randomized datasets using in training and evaluation, indv_sensors 
# (dict) - a dictionary containing the individual sensor data per dataset, sensors (list) - the 
# sensors we are using accelerometer data from.
# Returns: accel_dfs (dict) - the combined (all sensor) accelerometer data for each dataset.
def combine_sensors(datasets, indv_sensors, sensors):
    accel_dfs = {}
    for ds in datasets:
        m = []
        for s in sensors:
            m.append(indv_sensors[f'{ds} {s}'].shape[0])
        m = np.amin(m)

        # create a dataframe of all the accelerometer data from all sensors.
        df = pd.DataFrame()
        for s in sensors:
            df[f"Accelerometer X {s}"] = indv_sensors[f'{ds} {s}']["Accelerometer X"]
            df[f"Accelerometer Y {s}"] = indv_sensors[f'{ds} {s}']["Accelerometer Y"]
            df[f"Accelerometer Z {s}"] = indv_sensors[f'{ds} {s}']["Accelerometer Z"]

        accel_dfs[ds] = df.head(m)

    return accel_dfs

# Get and merge all acceleromter data from the neccesary sensors for all datasets.
# Params: datasets  (list) - the randomized datasets using in training and evaluation, sensors (list) 
# - the sensors we are using accelerometer data from, freq (int) - the frequency data was collected 
# at. 
# Returns: a tuple of dictionaries, one containing the combined accelerometer data for each dataset
# and the other containing the common start times of each dataset's accelerometer data. 
def create_accel_dfs(datasets, sensors, freq):
    unsync_accel, unsync_start_times = get_accel_data(datasets, sensors)
    
    # accel is only combined (and made the same shape) when considering multiple sensors
    if len(sensors) > 1:
        indv_accel_data, start_times = remove_unlabelled_start_data_from_accel(
            unsync_accel, 
            unsync_start_times, 
            datasets, 
            sensors,
            freq)
        accel_dfs = combine_sensors(datasets, indv_accel_data, sensors)
    else:
        return unsync_accel, unsync_start_times

    return accel_dfs, start_times

# Gets all labels and accelerometer data for the neccesary datasets and sensors.
# Params: datasets (list) - the randomized datasets using in training and evaluation, sensors (list) 
# - the sensors we are using accelerometer data from, freq (int) - the frequency data was collected 
# at.
# Returns: accel (dict) - the combined accelerometer data for each dataset, accel_starts (dict) - 
# the common start times of each dataset's accelerometer data, labels (dict) - the labels for each 
# dataset, label_starts (dict) - the label start times for each dataset. 
def get_data(datasets, sensors, freq):
    labels, label_starts = get_labels(datasets)
    accel, accel_starts = create_accel_dfs(datasets, sensors, freq)
    return accel, accel_starts, labels, label_starts