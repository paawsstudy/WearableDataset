# This file contains functions to augmented the accelerometer data and simulate multiple orientation
# and to create non-augmented training sets as defined in the accompanying manuscript. 
import numpy as np 

# Augment the the original accel data to be rotated 180 degrees. 
# Params: accel (array) - original, unaugmented accelerometer data, sensor (int) - specifies which 
# sensor is being augmented (0 indexed, assumed sensors are combined in a specified order across 
# all datasets). 
# Returns: acc (array) - a augemnted copy of the original accel data. 
def aug_sensor_180_rot(accel, sensor):
    acc = np.copy(accel)
    acc[:, :, (sensor*3)+0] = -1 * acc[:, :, (sensor*3)+0]
    acc[:, :, (sensor*3)+1] = -1 * acc[:, :, (sensor*3)+1]
    return acc

# Augment the the original accel data to be rotated 90 degrees. 
# Params: accel (array) - original, unaugmented accelerometer data, sensor (int) - specifies which 
# sensor is being augmented (0 indexed, assumed sensors are combined in a specified order across 
# all datasets). 
# Returns: acc (array) - a augemnted copy of the original accel data. 
def aug_sensor_90_rot(accel, sensor):
    acc = np.copy(accel)
    temp = np.copy(acc[:, :, (sensor*3)+0])
    acc[:, :, (sensor*3)+0] = -1 * acc[:, :, (sensor*3)+1]
    acc[:, :, (sensor*3)+1] = temp
    return acc

# Augment the the original accel data to be flipped 180 degrees. 
# Params: accel (array) - original, unaugmented accelerometer data, sensor (int) - specifies which 
# sensor is being augmented (0 indexed, assumed sensors are combined in a specified order across 
# all datasets). 
# Returns: acc (array) - a augemnted copy of the original accel data. 
def aug_sensor_flip(accel, sensor):
    acc = np.copy(accel)
    acc[:, :, (sensor*3)+0] = -1 * acc[:, :, (sensor*3)+0]
    acc[:, :, (sensor*3)+2] = -1 * acc[:, :, (sensor*3)+2]
    return acc

# Create a set of simulated rotations (R) as specified in the accompanying manuscript.
# Params: accel (array) - original, unaugmented accelerometer data, sensor (int) - specifies which 
# sensor is being augmented (0 indexed, assumed sensors are combined in a specified order across 
# all datasets). 
# Returns: aug_90 (array) - a copy of 90 degree rotations of acc, aug_180 (array) - a copy of 180 
# degree rotations of acc, aug_270 (array) - a copy of 270 degree rotations of acc. 
def create_one_sensor_rot_aug_dataset(acc, sensor):
    aug_90 = aug_sensor_90_rot(acc, sensor)
    aug_180 = aug_sensor_180_rot(acc, sensor)
    aug_270 = aug_sensor_90_rot(aug_180, sensor)
    return aug_90, aug_180, aug_270

# Create a set of simulated flips (F) as specified in the accompanying manuscript.
# Params: accel (array) - original, unaugmented accelerometer data, sensor (int) - specifies which 
# sensor is being augmented (0 indexed, assumed sensors are combined in a specified order across 
# all datasets). 
# Returns: aug_flip (array) - a copy of 180 degree flips of acc. 
def create_one_sensor_flip_aug_dataset(acc, sensor):
    aug_flip = aug_sensor_flip(acc, sensor)
    return aug_flip

# Create a set of simulated rotations and flips (RF) as specified in the accompanying manuscript.
# Params: accel (array) - original, unaugmented accelerometer data, sensor (int) - specifies which 
# sensor is being augmented (0 indexed, assumed sensors are combined in a specified order across 
# all datasets). 
# Returns: aug_90 (array) - a copy of 90 degree rotations of acc, aug_180 (array) - a copy of 180 
# degree rotations of acc, aug_270 (array) - a copy of 270 degree rotations of acc, aug_flip (array) 
# - a copy of 180 degree flips of acc, aug_flip_90 (array) - a copy of 90 degree rotations that have 
# been flipped by 180 degrees of acc, aug_flip_180 (array) a copy of 180 degree rotations that have 
# been flipped by 180 degrees of acc, aug_flip_270 (array) - a copy of 270 degree rotations that 
# have been flipped by 180 degrees of acc.
def create_one_sensor_flip_and_rot_aug_dataset(acc, sensor):
    aug_90, aug_180, aug_270 = create_one_sensor_rot_aug_dataset(acc, sensor)
    aug_flip = create_one_sensor_flip_aug_dataset(acc, sensor)
    aug_flip_90, aug_flip_180, aug_flip_270 = create_one_sensor_rot_aug_dataset(aug_flip, sensor)
    return aug_90, aug_180, aug_270, aug_flip, aug_flip_90, aug_flip_180, aug_flip_270

# Create sets of augmeneted data, specifically NA (the original accelerometer data), R, F, and RFS. 
# Params: accel (array) - original, unaugmented accelerometer data, sensor (int) - specifies which 
# sensor is being augmented (0 indexed, assumed sensors are combined in a specified order across 
# all datasets).
# Returns: accel_dic (dictionary) - the accelerometer data for each sensor orientation. 
def create_augmented_datasets(acc, sensor):
    rots = np.concatenate(create_one_sensor_rot_aug_dataset(acc, sensor))
    flips = create_one_sensor_flip_aug_dataset(acc, sensor)
    rots_and_flips = np.concatenate(create_one_sensor_flip_and_rot_aug_dataset(acc, sensor))
    accel_dic = {
        "na": acc,
        "rots": rots,
        "flips": flips,
        "rfs": rots_and_flips,
    }
    return accel_dic

# Create label sets for the augmeneted datasets; the augmented datasets are (oftentimes) larger than
# the original acceleromter data. 
# Params: labels (array) - the original labelset for the non-augmented accelerometer data.
# Returns: lab_dic (dictionary) - the labels for each augmented dataset. 
def make_labels_for_aug_data(labels):
    print(labels.shape)
    rot_labels = np.concatenate((labels, labels, labels))
    flip_labels = labels
    rot_and_flip_labels = np.concatenate((labels, labels, labels, labels, labels, labels, labels))
    lab_dic = {
        "na": labels,
        "rots": rot_labels,
        "flips": flip_labels,
        "rfs": rot_and_flip_labels,
    }
    return lab_dic

# Create the training and evaluation sets; this method creates a training/testing set as though each 
# dataset is being reserved for evalation data. 
# Params: clean_accel (dict) - preprocessed (5s windows of) accelerometer data, clean_label (dict) 
# - preprocessed (5s windows of) label data, datasets (list) - the randomized datasets using in 
# training and evaluation, time (int) - length of sliding window, freq (int) - the frequency data 
# was collected at, directions (int) - represents triaxial accelerometer data, num_sensors (int) - 
# number of sensors data is used from. 
# Returns: training_accel_sets (dict) - collection of accelerometer data used for training, 
# training_label_sets (dict) - labels corresponding to the accel training data, testing_accel_sets 
# (dict) - collection of accelerometer data used for testing, testing_label_sets (dict) - labels 
# corresponding to the accel testing data. 
def create_LOO_train_test(clean_accel, clean_label, datasets, time, freq, directions, num_sensors):
    training_accel_sets = {}
    testing_accel_sets = {}
    training_label_sets = {}
    testing_label_sets = {}
    
    for ds in datasets:
        train_accel_tuple = []
        train_label_tuple = []
        testing_accel_set = None
        testing_label_set = None
        for i in datasets:
            if i == ds: 
                testing_accel_set = clean_accel[i]
                testing_label_set = clean_label[i]
            else: 
                train_accel_tuple.append(clean_accel[i])
                train_label_tuple.append(clean_label[i])     

        # combine the training/testing accelerometer data into one array 
        training_accel_sets[ds] = np.reshape(
            np.vstack(train_accel_tuple), 
            (-1, freq*time, directions*num_sensors))
        testing_accel_sets[ds] = np.reshape(
            testing_accel_set, 
            (-1, freq*time, directions*num_sensors))
        
        testing_label_sets[ds] = testing_label_set
        training_label_sets[ds] = np.vstack(train_label_tuple)
        
    return training_accel_sets, training_label_sets, testing_accel_sets, testing_label_sets

