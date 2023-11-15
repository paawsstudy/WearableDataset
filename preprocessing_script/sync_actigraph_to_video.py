import numpy as np
import sys, json, os, datetime, glob
import pandas as pd

annotator = 'annotator'
header_lis = ['START_TIME','STOP_TIME','PREDICTION','SOURCE','LABELSET']
file_types = ['Behavioral Parameters','Experimental situation','HIGH LEVEL BEHAVIOR','PA TYPE','POSTURE', 'Lab Live Annotation']

def main(inPath):
    main_df = pd.read_csv(inPath)
    for index,row in main_df.iterrows():
        frame_per_sec = row['FRAME_PER_SEC']
        frame_ev1 = row['START_FRAME']
        frame_ev2 = row['STOP_FRAME']
        data_per_sec = row['DATA_PER_SEC']
        sample_ev1 = row['START_DATA']
        sample_ev2 = row['STOP_DATA']
        # for compatibility with older file
        if 'LABEL_FILE_PATH' in main_df.columns:
            label_files = [os.path.join(row['LABEL_FILE_PATH'],'labels.json')]
        elif 'LABEL_FOLDER_PATH' in main_df.columns:
            label_files = os.path.join(row['LABEL_FOLDER_PATH'], '*', 'labels.json') # the folder will contain the folder with a json file 
            label_files = glob.glob(label_files)
            # Allowing more than 1 folder
            #if len(label_file) != 1:
            #    print("annotation folder is missing or more than 1 annotation folder is present in the parent folder")
            #    continue 
            #label_file = label_file[0]
        
        actigraph_file_path = row['ACTIGRAPH_CSV_FILE_PATH']

        for label_file in label_files:
            if not os.path.exists(label_file):
                print('json file does not exists')
                continue

            if not os.path.exists(actigraph_file_path):
                print('Actigraph CSV file does not exist')
                continue
            print(label_file)
            label_file_path = os.path.dirname(label_file)

            with open(actigraph_file_path) as fp:
                for li, line in enumerate(fp):
                    if (li == 2):
                       ti = line.split(" ")[-1].strip()
                    elif (li == 3):
                        da = line.split(" ")[-1].strip()
                    elif (li > 3):
                        break
            data_start_time = da + ' ' + ti
            data_start_time_object = datetime.datetime.strptime(data_start_time, '%m/%d/%Y %H:%M:%S')

            A = np.array([[frame_ev1,1],[frame_ev2,1]])
            b = np.array([sample_ev1,sample_ev2])
            z = np.linalg.solve(A,b)

            with open(label_file, 'r') as myfile:
                data=myfile.read()
            # parse file
            obj = json.loads(data)
            annot= obj['annotations']
            final_df = pd.DataFrame(columns=header_lis)
            row_num = 0
            for i in range(len(annot)):
                dat = annot[i]
                sta = frame_per_sec*dat['range']['start']
                sto = frame_per_sec*dat['range']['end']
                sta_data = int(sta * z[0] + z[1])
                sto_data = int(sto * z[0] + z[1])

                sta_sec = sta_data / data_per_sec
                sto_sec = sto_data / data_per_sec

                sta_time = data_start_time_object + datetime.timedelta(0,sta_sec)
                sto_time = data_start_time_object + datetime.timedelta(0, sto_sec)

                sta_st = sta_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                sto_st = sto_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                comment = dat['comments'][0]['body']
                
                # use json to read comment instead
                comm_lis = json.loads(comment)
                for j in comm_lis:
                    category = j['category']
                    value = j.get('selectedValue', None)
                    if isinstance(value, list):
                        # remove unlabelled label
                        value = [x for x in value if 'Unlabeled' not in x]
                        if len(value) == 0:
                            value = ""
                        else:
                            value = "|".join(value)
                    if value is None or 'Unlabeled' in value:
                        value = ""
                    if value == "":
                        continue
                    this_lis = [sta_st,sto_st,value,annotator,category]
                    final_df.loc[row_num,:] = this_lis
                    row_num += 1

            for f_type in file_types:
                this_df = final_df[final_df['LABELSET']==f_type]
                this_df.to_csv(os.path.join(label_file_path,f_type+'_corr.csv'),index=False)

            print("Done.")

if __name__ == "__main__":

    infile = sys.argv[1]
    main(infile)
