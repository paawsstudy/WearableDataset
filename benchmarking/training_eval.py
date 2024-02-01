# This file contains functions for the benchmarking in the the accompanying manuscript.

import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import find_peaks 
from scipy import stats
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
import time 
from tensorflow.keras.callbacks import CSVLogger
from models import *
from make_training_sets import * 

# evaluation methods:

# Make a confusion matrix. 
# Params: lab_pool (list) - activities considered across experiments, true (array) - the true labels 
# of an evaluation set, preds (arrary) - the model's predictions across an evaluation set. 
# Returns: mat_df (DataFrame) - the confusion matrix (rows = true, cols = preds).
def confusion_matrix(lab_pool, true, preds, exp=""):
    #act_labels = map_labels(lab_pool, lab_pool)
    mat_df = pd.DataFrame(columns=[lab_pool], index=lab_pool)

    unmapped_true = unmap_labels(true, lab_pool)
    unmapped_preds = unmap_labels(preds, lab_pool)

    for lab in lab_pool:
        mat_df[lab] = 0

    if "SVM" in exp:
        for i in range(len(unmapped_true)):
            mat_df.loc[unmapped_true[i][0]][unmapped_preds[i]] = mat_df.loc[unmapped_true[i][0]][unmapped_preds[i]] + 1
    else:
        for i in range(len(unmapped_true)):
            mat_df.loc[unmapped_true[i]][unmapped_preds[i]] = mat_df.loc[unmapped_true[i]][unmapped_preds[i]] + 1
    return mat_df

# Map labels from a string to a one-hot-vector.
# Params: labels (array) - labels for training/evalutation sets, lab_pool (list) - activities 
# considered across experiments.
# Returns: mapped_labels (array) - all labels as one-hot-vectors. 
def map_labels(labels, lab_pool):
    mapped_labels = []
    for label in labels: 
        map_l = np.zeros(len(lab_pool))
        i = lab_pool.index(label)
        map_l[i] = 1
        mapped_labels.append(map_l)

    return np.array(mapped_labels)

# Map labels from a one-hot-vector to a string.
# Params: labels (array) - labels for training/evalutation sets (as one-hot-vectors), lab_pool 
# (list) - activities considered across experiments.
# Returns: unmapped (array) - all labels as their original strings. 
def unmap_labels(labels, lab_pool):
    unmapped = []
    for lab in labels:
        if lab not in lab_pool:
            unmapped.append(unmap_label(lab, lab_pool))
        else:
            unmapped.append(lab)
    return unmapped

# Map a label from a one-hot-vector to a string.
# Params: label (string) - a label (as a one-hot-vector), lab_pool (list) - activities considered 
# across experiments.
# Returns: unmapped (string) - the labels as its original string. 
def unmap_label(label, lab_pool):
    ind = np.argmax(label)
    if ind <= len(lab_pool):
        return lab_pool[ind]
    else:
        return "catch all"

# All functions named "compute_" compute a certain feature from the raw accelerometer data.
def compute_mean(df, data, sensor, fft = ""):
    df[f'{fft}x_mean_{sensor}'] = pd.DataFrame(data[:, :, 0].T).apply(lambda x: x.mean())
    df[f'{fft}y_mean_{sensor}'] = pd.DataFrame(data[:, :, 1].T).apply(lambda x: x.mean())
    df[f'{fft}z_mean_{sensor}'] = pd.DataFrame(data[:, :, 2].T).apply(lambda x: x.mean())
    return df 

def compute_std(df, data, sensor, fft = ""):
    df[f'{fft}x_std_{sensor}'] = pd.DataFrame(data[:, :, 0].T).apply(lambda x: x.std())
    df[f'{fft}y_std_{sensor}'] = pd.DataFrame(data[:, :, 1].T).apply(lambda x: x.std())
    df[f'{fft}z_std_{sensor}'] = pd.DataFrame(data[:, :, 2].T).apply(lambda x: x.std())
    return df 

def compute_aad(df, data, sensor, fft = ""):
    df[f'{fft}x_aad_{sensor}'] = pd.DataFrame(data[:, :, 0].T).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    df[f'{fft}y_aad_{sensor}'] = pd.DataFrame(data[:, :, 1].T).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    df[f'{fft}z_aad_{sensor}'] = pd.DataFrame(data[:, :, 2].T).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    return df 

def compute_median(df, data, sensor, fft = ""):
    df[f'{fft}x_median_{sensor}'] = pd.DataFrame(data[:, :, 0].T).apply(lambda x: np.median(x))
    df[f'{fft}y_median_{sensor}'] = pd.DataFrame(data[:, :, 1].T).apply(lambda x: np.median(x))
    df[f'{fft}z_median_{sensor}'] = pd.DataFrame(data[:, :, 2].T).apply(lambda x: np.median(x))
    return df 

def compute_min(df, data, sensor, fft = ""):
    df[f'{fft}x_min_{sensor}'] = pd.DataFrame(data[:, :, 0].T).apply(lambda x: x.min())
    df[f'{fft}y_min_{sensor}'] = pd.DataFrame(data[:, :, 1].T).apply(lambda x: x.min())
    df[f'{fft}z_min_{sensor}'] = pd.DataFrame(data[:, :, 2].T).apply(lambda x: x.min())
    return df 

def compute_max(df, data, sensor, fft = ""):
    df[f'{fft}x_max_{sensor}'] = pd.DataFrame(data[:, :, 0].T).apply(lambda x: x.max())
    df[f'{fft}y_max_{sensor}'] = pd.DataFrame(data[:, :, 1].T).apply(lambda x: x.max())
    df[f'{fft}z_max_{sensor}'] = pd.DataFrame(data[:, :, 2].T).apply(lambda x: x.max())
    return df 

# TODO: these two need to be checked
def compute_median_jerk(df, data, sensor, fft = ""):
    df[f'{fft}x_median_{sensor}'] = pd.DataFrame(np.gradient(data[:, :, 0], axis=1).T).apply(lambda x: np.median(x))
    df[f'{fft}y_median_{sensor}'] = pd.DataFrame(np.gradient(data[:, :, 1], axis=1).T).apply(lambda x: np.median(x))
    df[f'{fft}z_median_{sensor}'] = pd.DataFrame(np.gradient(data[:, :, 2], axis=1).T).apply(lambda x: np.median(x))
    return df 

def compute_max_jerk(df, data, sensor, fft = ""):
    df[f'{fft}x_max_{sensor}'] = pd.DataFrame(np.gradient(data[:, :, 0], axis=1).T).apply(lambda x: x.max())
    df[f'{fft}y_max_{sensor}'] = pd.DataFrame(np.gradient(data[:, :, 1], axis=1).T).apply(lambda x: x.max())
    df[f'{fft}z_max_{sensor}'] = pd.DataFrame(np.gradient(data[:, :, 2], axis=1).T).apply(lambda x: x.max())
    return df 

def compute_avg(df, data, sensor, fft = ""):
    df[f'{fft}avg_result_accl_{sensor}'] = pd.DataFrame((((data[:, :, 0])**2 + (data[:, :, 1])**2 + (data[:, :, 2])**2)**0.5).T).apply(lambda x: np.mean(x))
    return df

def compute_num_peaks(df, data, sensor, fft = ""):
    df[f'{fft}x_num_peaks_{sensor}'] = pd.DataFrame(data[:, :, 0].T).apply(lambda x: len(find_peaks(x)[0]))
    df[f'{fft}y_num_peaks_{sensor}'] = pd.DataFrame(data[:, :, 1].T).apply(lambda x: len(find_peaks(x)[0]))
    df[f'{fft}z_num_peaks_{sensor}'] = pd.DataFrame(data[:, :, 2].T).apply(lambda x: len(find_peaks(x)[0]))
    return df

def compute_kurtosis(df, data, sensor, fft = ""):
    df[f'{fft}x_kurtosis_{sensor}'] = pd.DataFrame(data[:, :, 0].T).apply(lambda x: stats.kurtosis(x))
    df[f'{fft}y_kurtosis_{sensor}'] = pd.DataFrame(data[:, :, 1].T).apply(lambda x: stats.kurtosis(x))
    df[f'{fft}z_kurtosis_{sensor}'] = pd.DataFrame(data[:, :, 2].T).apply(lambda x: stats.kurtosis(x))
    return df

def compute_skewness(df, data, sensor, fft = ""):
    df[f'{fft}x_skewness_{sensor}'] = pd.DataFrame(data[:, :, 0].T).apply(lambda x: stats.skew(x))
    df[f'{fft}y_skewness_{sensor}'] = pd.DataFrame(data[:, :, 1].T).apply(lambda x: stats.skew(x))
    df[f'{fft}z_skewness_{sensor}'] = pd.DataFrame(data[:, :, 2].T).apply(lambda x: stats.skew(x))
    return df

def compute_IQR(df, data, sensor, fft = ""):
    df[f'{fft}x_IQR_{sensor}'] = pd.DataFrame(data[:, :, 0].T).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    df[f'{fft}y_IQR_{sensor}'] = pd.DataFrame(data[:, :, 1].T).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    df[f'{fft}z_IQR_{sensor}'] = pd.DataFrame(data[:, :, 2].T).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    return df 

def compute_sma(df, data, sensor, fft = ""):
    if fft == "":
        series = pd.DataFrame(data[:, :, 0].T).apply(lambda x: np.sum(abs(x)/400)) + pd.DataFrame(data[:, :, 1].T).apply(lambda x: np.sum(abs(x)/400)) \
                  + pd.DataFrame(data[:, :, 2].T).apply(lambda x: np.sum(abs(x)/400))
    else:
        series = pd.DataFrame(data[:, :, 0].T).apply(lambda x: np.sum(abs(x)/200)) + pd.DataFrame(data[:, :, 1].T).apply(lambda x: np.sum(abs(x)/200)) \
                  + pd.DataFrame(data[:, :, 2].T).apply(lambda x: np.sum(abs(x)/200))
    df[f'{fft}sma_{sensor}'] = series
    return df

def compute_energy(df, data, sensor, fft = ""):
    df[f'{fft}x_energy_{sensor}'] = pd.DataFrame(data[:, :, 0].T).apply(lambda x: np.sum(x**2)/400)
    df[f'{fft}y_energy_{sensor}'] = pd.DataFrame(data[:, :, 1].T).apply(lambda x: np.sum(x**2)/400)
    df[f'{fft}z_energy_{sensor}'] = pd.DataFrame(data[:, :, 2].T).apply(lambda x: np.sum(x**2)/400)
    return df

# Compute all features for the SVM and KNN.
# Params: data (array) - a colelction of 5s windows of raw accel signal from a single sensor.
# Returns: accel (array) - the features corresponding to the windows of raw accel data. 
def make_features(data):
    accel = pd.DataFrame()

    data_dict = {
        "sens": data[:, :, 0:3],
        "fft_sens": np.abs(np.fft.fft(data, axis=1)[:, 1:201, 0:3]),
    }

    for sensor in ["sens"]:
        for fft in ["", "fft_"]:
            accel = compute_mean(accel, data_dict[fft+sensor], sensor, fft)
            accel = compute_std(accel, data_dict[fft+sensor], sensor, fft)
            accel = compute_aad(accel, data_dict[fft+sensor], sensor, fft)
            accel = compute_min(accel, data_dict[fft+sensor], sensor, fft)
            accel = compute_max(accel, data_dict[fft+sensor], sensor, fft)
            accel = compute_median(accel, data_dict[fft+sensor], sensor, fft)
            accel = compute_IQR(accel, data_dict[fft+sensor], sensor, fft)
            accel = compute_num_peaks(accel, data_dict[fft+sensor], sensor, fft)
            accel = compute_kurtosis(accel, data_dict[fft+sensor], sensor, fft)
            accel = compute_energy(accel, data_dict[fft+sensor], sensor, fft)
            accel = compute_avg(accel, data_dict[fft+sensor], sensor, fft)
            accel = compute_sma(accel, data_dict[fft+sensor], sensor, fft)
    
    return accel

# Create a csv containing reported accuracies across all evaluation sets and seeds. 
# Params: acc (dict) - evaluated accuracies across evaluation sets for all seeds, file (string) - 
# path to where accuracies are reported, exp (string) - the experiment number.
# Returns: nothing. 
def make_acc_csv(acc, file, exp):
    if "SVM" in exp:
        cols = ["DS_LO", "SVM_Acc", "KNN_Acc", "Time"]
    else:
        cols = [
            'DS Left Out', 
            'Training 1 Acc', 
            'Training 2 Acc', 
            'Training 3 Acc', 
            'Training Time (all models)']
        
    np_acc = np.array(list(acc.values()))
    df = pd.DataFrame(np_acc, columns=cols)
    df.to_csv(file)
    
# Create a csv containing reported f1s across all evaluation sets and seeds. 
# Params: f1s (dict) - evaluated f1s across evaluation sets for all seeds, labs (list) - 
# activities considered across experiments, file (string) - path to where f1s are reported.
# Returns: nothing. 
def make_f1_csv(f1s, labs, file):
    cols = [x for x in labs]
    cols.insert(0, 'Training #')
    cols.insert(0, 'DS Left Out')
    cols.append("F1 (avg)")
    np_f1 = np.array(list(f1s.values()))
    df = pd.DataFrame(np_f1, columns=cols)
    df.to_csv(file)

# Create a csv containing the confusion matrix across all evaluation sets and seeds. 
# Params: conf (DataFrame) - the confusion matrix for an evaluation set, file (string) - 
# path to where confusion matrices are reported.
# Returns: nothing. 
def make_conf_matrix(conf, file):
    print(conf.keys())
    for key in conf.keys():
        path = file + f'{key}.csv'
        conf[key].to_csv(path)

# Train and evaluate the SVM + KNN models in the benchmarking of the accompanying manuscript.
# Params: training_accel_sets (dict) - collection of accelerometer data used for training, 
# training_label_sets (dict) - labels corresponding to the accel training data, testing_accel_sets 
# (dict) - collection of accelerometer data used for testing, testing_label_sets (dict) - labels 
# corresponding to the accel testing data, label_set (dict) - activities considered across 
# experiments, ds_lo (int) - the ID of the left out participant
# Returns: accuracies (dict) - evaluated accuracies across evaluation sets for all seeds, f1_scores 
# (dict) - evaluated f1 scores across evaluation sets for all seeds, confusion_matrices (dict) - 
# confusion matrices across evaluation sets for all seeds. 
def SVM_train_and_test(
    training_accel, 
    training_label, 
    testing_accel, 
    testing_label, 
    label_set,
    ds):

    accuracies = {}
    f1_scores = {}
    confusion_matrices = {}

    training_accel_set = make_features(training_accel[ds])
    testing_accel_set = make_features(testing_accel[ds])

    accuracies[ds] = [ds]
    t1 = time.perf_counter()
    print(training_label[ds].shape)

    # define the single training number
    i = 1
    print(f'DS {ds} training num {i}')
    # define model 
                    
    model = svm.SVC(kernel='poly', degree=3, C=1)
            
    # norm features
    scaler = StandardScaler()
    scaler.fit(training_accel_set)
    train_reg = scaler.transform(training_accel_set)
    test_reg = scaler.transform(testing_accel_set)

    # train the SVM
    model.fit(
        train_reg, 
        training_label[ds])
        
    # evaluate model on the left out ds 
    preds = model.predict(test_reg)
    acc = round(accuracy_score(testing_label[ds], preds)*100, 2)
    accuracies[ds].append(acc)
    f1 = list(f1_score(testing_label[ds], preds, average=None, labels=label_set))
    f1.append(np.average(f1)) # add the average to the very end 

    f1.insert(0, i) # add training num
    f1.insert(0, ds) # add DS left out 
    f1_scores[f'{ds} training {i}'] = f1 

    confusion_matrices[f'{ds} SVM training {i}'] = confusion_matrix(label_set, testing_label[ds], preds, "SVM")

    # train the KNN
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(
        train_reg, 
        training_label[ds], )

    # evaluate model on the left out ds 
    preds = neigh.predict(test_reg)
    acc = round(accuracy_score(testing_label[ds], preds)*100, 2)
    accuracies[ds].append(acc)
    f1 = list(f1_score(testing_label[ds], preds, average=None, labels=label_set))
    f1.append(np.average(f1)) # add the average to the very end 
    f1.insert(0, i) # add training num
    f1.insert(0, ds) # add DS left out 
    f1_scores[f'{ds} training {i}'] = f1

    confusion_matrices[f'{ds} KNN training {i}'] = confusion_matrix(label_set, testing_label[ds], preds, "SVM")
                

    t2 = time.perf_counter()
    accuracies[ds].append(round(t2 - t1, 2)) # add training time of both models
                 
    return accuracies, f1_scores, confusion_matrices        

# Train and evaluate the CNN model in the benchmarking of the accompanying manuscript.
# Params: training_accel_sets (dict) - collection of accelerometer data used for training, 
# training_label_sets (dict) - labels corresponding to the accel training data, testing_accel_sets 
# (dict) - collection of accelerometer data used for testing, testing_label_sets (dict) - labels 
# corresponding to the accel testing data, label_set (dict) - activities considered across 
# experiments, ds_lo (int) - the ID of the left out participant, training_aug (string) - the 
# orientations to be used in the training set.
# Returns: accuracies (dict) - evaluated accuracies across evaluation sets for all seeds, f1_scores 
# (dict) - evaluated f1 scores across evaluation sets for all seeds, confusion_matrices (dict) - 
# confusion matrices across evaluation sets for all seeds. 
def exp_1_train_and_test(
    training_accel_sets, 
    training_label_sets, 
    testing_accel_sets, 
    testing_label_sets,  
    label_set,
    ds_lo,
    path,
    training_aug):

    accuracies = {}
    f1_scores = {}
    confusion_matrices = {}

    # these dicts use the keys of the dataset aug (e.g., na, rots, flips, rfs). Training with rfs means train on 
    eval_labels_aug = make_labels_for_aug_data(testing_label_sets[ds_lo])
    eval_accel_aug = create_augmented_datasets(testing_accel_sets[ds_lo], 0)
    train_labels_aug = make_labels_for_aug_data(training_label_sets[ds_lo])
    train_accel_aug = create_augmented_datasets(training_accel_sets[ds_lo], 0)

    if training_aug != "na":
        train_accel = np.concatenate((train_accel_aug["na"], train_accel_aug[training_aug]))
        train_labels = map_labels(np.concatenate((train_labels_aug["na"], train_labels_aug[training_aug])), label_set)
    else:
        train_accel = train_accel_aug["na"]
        train_labels = map_labels(train_labels_aug["na"], label_set)

    accuracies[ds_lo] = [ds_lo]
    t1 = time.perf_counter()
    input_shape = train_accel.shape[1:]
    output_shape = train_labels.shape[1]

    # train and eval three random seeds with the same training set 
    for i in range(1, 4): 
            tf.keras.utils.disable_interactive_logging()
            
            # define, compile, and train model 
            cnn = single_sensor_CNN(input_shape, output_shape)
            cnn.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy'])
            csv_logger = CSVLogger(f'{path}/DS_{ds_lo}_Loss_Training_{i}.csv', append=True, separator=',') # 
            cnn.fit(
                train_accel, 
                train_labels, 
                batch_size=100, 
                epochs=2,
                callbacks=[csv_logger])
        
            # evaluate model on all orientations of the left out ds 
            results = cnn.evaluate(eval_accel_aug["na"], map_labels(eval_labels_aug["na"], label_set))
            acc = round(results[1]*100, 2)
            accuracies[ds_lo].append(acc)
                
            # generate f1 scores for each orientation of the left out ds
            preds = cnn.predict(eval_accel_aug["na"])
            y_pred = np.argmax(preds, axis=1)
            y_true = np.argmax(map_labels(eval_labels_aug["na"], label_set), axis=1)
            f1 = list(f1_score(y_true, y_pred, average=None, labels=np.arange(output_shape)))
            f1.append(np.average(f1)) # add the average to the very end 
            f1.insert(0, i) # add training num
            f1.insert(0, ds_lo) # add DS left out 
            f1_scores[f'{ds_lo} training {i}'] = f1
            
            confusion_matrices[f'{ds_lo}_training_{i}_na'] = confusion_matrix(
                label_set, 
                map_labels(eval_labels_aug["na"], label_set), preds)

    t2 = time.perf_counter()
    accuracies[ds_lo].append(round((t2 - t1)/3, 2))
                 
    return accuracies, f1_scores, confusion_matrices
