# exp = single sensor 
# to run: python3 run_exp_CNN.py ds_lo sensor1 "" exp_name labelset datasets

from get_data import get_data
from make_training_sets import create_LOO_train_test
import os 
from process_data import clean_all_data
from sys import argv
from training_eval import * 

# set global vars 
f = 80 # data is collected at a frequence of 80Hz
t = 5 # each activity is segmented into 5s increments 
d = 3 # for triaxial accelerometer data x, y, z

# aquire global vars from passed in args 
ds_lo = int(argv[1])
sensors = [argv[2]]
training_aug = argv[3] # na, rot, flip, rf
experiment_num = argv[4]
lab = int(argv[5])
datasets = list(argv[6:])
datasets = [int(x) for x in datasets]

print("ds_lo:", ds_lo)
print("sensors:", sensors)
print("training_aug:", training_aug)
print("exp num:", experiment_num)
print("datasets:", datasets)

labels = [
["Standing_Still", "Sitting_Still","Walking","Lying On Back Lab", "Walking_Down_Stairs","Walking_Up_Stairs", 
 "Cycling_Active_Pedaling_Regular_Bicycle"],
["Standing_Still", "Sitting_Still","Walking","Lying On Back Lab", "Walking_Down_Stairs","Walking_Up_Stairs", 
 "Cycling_Active_Pedaling_Regular_Bicycle", "Lying on Right Side Lab", "Treadmill 3mph Free Walk Lab", 
 "Treadmill 5.5mph Lab", "Treadmill 2mph Lab", "Treadmill 4mph Lab", "Stationary Biking 300 Lab",  "Lying on Stomach Lab",
   "Lying on Left Side Lab" ],
["Standing_Still","Treadmill 3mph Hands Pockets Lab","Lying on Left Side Lab","Stand Conversation Lab",
 "Treadmill 3mph Conversation Lab","Chopping Food Lab","Sitting_Still","Walking","Ab Crunches Lab",
 "Lying_Still","Treadmill 5.5mph Lab","Machine Leg Press Lab","Walking_Treadmill","Sit Recline Web Browse Lab",
 "Standing_With_Movement","Treadmill 2mph Lab","Vacuuming","Sitting_With_Movement","Sit Writing Lab",
 "Treadmill 3mph Phone Lab","Sit Typing Lab","Folding_Clothes",
 "Machine Chest Press Lab","Push Up - Male Lab","Walking_Down_Stairs","Walking_Up_Stairs","Stationary Biking 300 Lab",
 "Sweeping","Lying on Right Side Lab","Lying_With_Movement","Sit Recline Talk Lab","Stand Shelf Unload Lab",
 "Organizing_Shelf/Cabinet","Push Up - Female Lab","Arm Curls Lab","Treadmill 4mph Lab","Treadmill 3mph Briefcase Lab",
 "Lying on Stomach Lab","Cycling_Active_Pedaling_Regular_Bicycle","Washing Dishes Lab","Treadmill 3mph Free Walk Lab",
 "Stand Shelf Load Lab","Playing_Frisbee","Lying On Back Lab","Treadmill 3mph Drink Lab"]]
# activity types we are considering for these experiments
labelset = labels[lab]
print("labels:", labelset)

print(f"***** {datasets} getting data.\n")
accel, accel_start, labels, label_start = get_data(datasets, sensors, f)
print(accel.keys())
print(labels.keys())

print('checking in main:', type(accel_start[datasets[0]]))

print(f"***** {datasets} cleaning data.\n")
clean_accel, clean_label = clean_all_data(
    accel,
    labels, 
    accel_start, 
    label_start, 
    datasets, 
    t, 
    f, 
    d, 
    len(sensors), 
    labelset)

print(f"***** {datasets} making LOO sets.\n")

training_accel_sets, \
    training_label_sets, \
    testing_accel_sets, \
    testing_label_sets = create_LOO_train_test(
        clean_accel, 
        clean_label, 
        datasets, 
        t, 
        f, 
        d, 
        len(sensors))

for ds in datasets:
    print("breakdown:", ds)
    u, c = np.unique(clean_label[ds], return_counts=True)
    print(np.asarray((u, c)).T)

print("training accel keys", training_accel_sets.keys())
print("testing accel keys", testing_accel_sets.keys())

loss_path = f'RESULTS_Loss/Exp_{experiment_num}/'

try:
    os.mkdir(loss_path)
except FileExistsError:
    pass

loss_path += f'{datasets}'

try:
    os.mkdir(loss_path)
except FileExistsError:
    pass

print(f"***** {datasets} training.\n")

tic = time.perf_counter()
acc, f1s, conf = exp_1_train_and_test(
    training_accel_sets, 
    training_label_sets, 
    testing_accel_sets, 
    testing_label_sets,  
    labelset,
    ds_lo, 
    loss_path,
    "na")

acc_path = f'RESULTS_Accuracies/Exp_{experiment_num}/' 
f1_path = f'RESULTS_f1s/Exp_{experiment_num}/' 
conf_path = file = f'CONFUSION_MATRICES/Exp_{experiment_num}/'

try:
    os.mkdir(acc_path)
except FileExistsError:
    pass

try:
    os.mkdir(f1_path)
except FileExistsError:
    pass

try:
    os.mkdir(conf_path)
except FileExistsError:
    pass

acc_path += f'{datasets}'
f1_path += f'{datasets}' 
conf_path += f'{datasets}'

try:
    os.mkdir(acc_path)
except FileExistsError:
    pass

try:
    os.mkdir(f1_path)
except FileExistsError:
    pass

try:
    os.mkdir(conf_path)
except FileExistsError:
    pass

acc_path += f"/DS_{ds_lo}.csv"
f1_path += f"/DS_{ds_lo}.csv"
conf_path += f"/DS_"

make_acc_csv(acc, acc_path, experiment_num)
make_f1_csv(f1s, labelset, f1_path)
make_conf_matrix(conf, conf_path)

toc = time.perf_counter()

print(f'***** {datasets} total training took: {toc-tic} seconds.\n')



