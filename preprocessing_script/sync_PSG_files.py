import pandas as pd
import numpy as np
import os, sys
from itertools import islice
from datetime import datetime,timedelta

relevant_events = ['Wake', 'N1', 'N2', 'N3', 'REM']


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line + content)
    f.close()

def compute_sync_points(infile, reffile, sensor_shake_start, sensor_shake_end):
    # only need to look at header
    print("Computing sync points...")

    start_ref_time = get_start_time(reffile)
    start_sleep_time = get_start_time(infile)
    delta = start_sleep_time - start_ref_time
    delta = delta.total_seconds() * 80 # convert difference in time into difference in sample
    return sensor_shake_start - delta, sensor_shake_end - delta


def main(inPath, st_shake, en_shake, st_offset, en_offset, outPath, inPath_label, outPath_label):
    st_shake = int(st_shake)
    en_shake = int(en_shake)
    offset_begin = int(st_offset)
    offset_end = int(en_offset)

    with open(inPath) as myfile:
        head = list(islice(myfile, 4))

    line_as_pre = ''.join(head)
    resultant_offset = offset_begin - offset_end

    df_ori = pd.read_csv(inPath, skiprows=4, header=0)
    df = df_ori.shift(periods=-1 * offset_begin, fill_value=0)
    df.reset_index(inplace=True, drop=True)

    odf_label = pd.read_excel(inPath_label, parse_dates=True)

    df_label = odf_label.iloc[1:]

    df_label.set_index('Start Time', inplace=True)
    df_label.sort_index(inplace=True)
    df_label.reset_index(drop=False, inplace=True)

    sdf = df_label
    # keep all event - uncomment to only keep sleep stages
    #sdf = df_label.loc[df_label['Event'].isin(relevant_events)]
    fdf = sdf[['Start Time', 'End Time', 'Event']]
    fdf.rename(columns={'Start Time': 'START_DATETIME_ORI', 'End Time': 'STOP_DATETIME_ORI', 'Event': 'PREDICTION'},
               inplace=True)

    dateStr = head[2].strip().split(" ")[-1]
    ori_datetime_object = fdf.iloc[0]['START_DATETIME_ORI']
    timeStr = ori_datetime_object.strftime("%H:%M:%S")
    dateTimeStr = dateStr + " " + timeStr
    datetime_object = datetime.strptime(dateTimeStr, '%m/%d/%Y %H:%M:%S')
    sec = (datetime_object - ori_datetime_object).total_seconds()
    fdf['START_TIME'] = fdf['START_DATETIME_ORI'] + pd.Timedelta(seconds=sec)
    fdf['STOP_TIME'] = fdf['STOP_DATETIME_ORI'] + pd.Timedelta(seconds=sec)

    sampRate = int(head[0].strip().split(" ")[-2])
    average_shift_sample = (offset_begin + offset_end) / 2
    ori_average_shift_sec = average_shift_sample / sampRate
    average_shift_sec = round(ori_average_shift_sec, 3)

    fdf['START_TIME'] = fdf['START_TIME'] - pd.Timedelta(seconds=average_shift_sec)
    fdf['STOP_TIME'] = fdf['STOP_TIME'] - pd.Timedelta(seconds=average_shift_sec)

    fdf['SOURCE'] = 'Expert'
    fdf['LABELSET'] = 'PSG_SLEEP'

    fdf['START_TIME'] = pd.to_datetime(fdf['START_TIME']) - pd.Timedelta(seconds=average_shift_sec)
    fdf['STOP_TIME'] = pd.to_datetime(fdf['STOP_TIME']) - pd.Timedelta(seconds=average_shift_sec)

    # raw csv time
    startTime_PSG = get_start_time(inPath)
    # start time from label file
    startTime_label = fdf['START_TIME'].iloc[0]
    # compare the time whether label time is before PSG time, then add one day to label time
    if startTime_label < startTime_PSG:
        fdf['START_TIME'] = fdf['START_TIME'] + timedelta(days = 1)
        fdf['STOP_TIME'] = fdf['STOP_TIME'] + timedelta(days = 1)

    fdf = fdf[['START_TIME','STOP_TIME','PREDICTION','SOURCE','LABELSET']]
    fdf.to_csv(outPath_label, index=False)
    print("Finished writing synced PSG label file")

    if resultant_offset == 0:
        print(datetime.now().time())
        df.to_csv(outPath, index=False, float_format='%.3f')
        print(datetime.now().time())
        line_prepender(outPath, line_as_pre)
        print("Finished syncing file.")

    else:
        # top = int(0.1*df_size)
        # bottom = int(0.9*df_size)
        top = st_shake - offset_begin
        bottom = en_shake - offset_begin

        top_df = df.loc[:top, :]
        bottom_df = df.loc[bottom:, :]
        bottom_df.reset_index(inplace=True, drop=True)
        bottom_df_len = len(bottom_df.index)

        mid_df = df.loc[top:bottom, :]
        mid_df.reset_index(inplace=True, drop=True)
        mid_df_len = len(mid_df.index)

        if resultant_offset < 0:
            resultant_offset = abs(resultant_offset)
            n = int(mid_df_len / resultant_offset)
            li = [i for i in range(0, mid_df_len, n)]
            mid_df.loc[li, :] = np.NaN
            mid_df.dropna(inplace=True)
            mid_df.reset_index(inplace=True, drop=True)

            li_b = [i for i in range(0, bottom_df_len, n)]
            bottom_df.loc[li_b, :] = np.NaN
            bottom_df.dropna(inplace=True)
            bottom_df.reset_index(inplace=True, drop=True)

        elif resultant_offset > 0:
            n = int(mid_df_len / resultant_offset)
            li = [i for i in range(0, mid_df_len, n)]
            li.pop(0)
            count = 0
            for r in li:
                r = r + count
                line = pd.DataFrame(data={'Accelerometer X': [mid_df.loc[r, 'Accelerometer X']],
                                          'Accelerometer Y': [mid_df.loc[r, 'Accelerometer Y']],
                                          'Accelerometer Z': [mid_df.loc[r, 'Accelerometer Z']]})
                mid_df = pd.concat([mid_df.iloc[:r, :], line, mid_df.iloc[r:, :]], ignore_index=True)
                mid_df.reset_index(inplace=True, drop=True)
                count += 1

            li_b = [i for i in range(0, bottom_df_len, n)]
            li_b.pop(0)
            count_b = 0
            for r in li_b:
                r = r + count_b
                line_b = pd.DataFrame(data={'Accelerometer X': [bottom_df.loc[r, 'Accelerometer X']],
                                            'Accelerometer Y': [bottom_df.loc[r, 'Accelerometer Y']],
                                            'Accelerometer Z': [bottom_df.loc[r, 'Accelerometer Z']]})
                bottom_df = pd.concat([bottom_df.iloc[:r, :], line_b, bottom_df.iloc[r:, :]], ignore_index=True)
                bottom_df.reset_index(inplace=True, drop=True)
                count_b += 1

        final_df = pd.concat([top_df, mid_df, bottom_df], ignore_index=True)
        final_df.reset_index(inplace=True, drop=True)

        print(datetime.now().time())
        final_df.to_csv(outPath, index=False, float_format='%.3f')
        print(datetime.now().time())
        line_prepender(outPath, line_as_pre)
        print("Finished syncing file.")

# only need to look at header
def get_start_time(filename):
    start_date = None
    start_time = None
    with open(filename) as f:
        # skip first header
        f.readline()
        line  = f.readline()
        while ('---' not in line): # not end of header
            if ("Start Date" in line):
                start_date = line.split()[-1]
            if ("Start Time" in line):
                start_time = line.split()[-1]
            line = f.readline()
    return datetime.strptime(start_date + " " + start_time, "%m/%d/%Y %H:%M:%S")


def compute_sync_points(infile, reffile, sensor_shake_start, sensor_shake_end):
    print("Compute sync points")
    start_ref_time = get_start_time(reffile)
    start_sleep_time = get_start_time(infile)
    delta = start_sleep_time - start_ref_time
    delta = delta.total_seconds() * 80 # convert difference in time into difference in sample
    print("Sleep file ahead of reference by", delta, "sample")
    return sensor_shake_start - delta, sensor_shake_end - delta



if __name__ == "__main__":

    print(len(sys.argv))

    if len(sys.argv) == 2:
        infile = sys.argv[1]
        if (os.path.exists(infile)):
            main_df = pd.read_csv(infile)
            for index, row in main_df.iterrows():
                infile = row['SLEEP_ACTIGRAPH_INFILE']
                outfile = row['SLEEP_ACTIGRAPH_OUTFILE']
                reffile = row['ACTIGRAPH_REFERENCE'] # the sensor shake start and end in reference to this file
                sensor_shake_start = row['START']
                sensor_shake_end = row['END']
                sensor_shake_start, sensor_shake_end = compute_sync_points(infile, reffile, sensor_shake_start, sensor_shake_end)
                print("Done computing sync points")
                offset_start = row['OFFSET_START']
                offset_end = row['OFFSET_END']
                infile_label = row['SLEEP_LABEL_INFILE']
                outfile_label = row['SLEEP_LABEL_OUTFILE']
                if not os.path.exists(outfile):
                    main(infile, sensor_shake_start, sensor_shake_end, offset_start, offset_end, outfile,infile_label,outfile_label)
                else:
                    print("Synced file already exists : " + outfile)

    elif len(sys.argv) == 7:
        infile = sys.argv[1]
        outfile = sys.argv[2]
        start = sys.argv[3]
        end = sys.argv[4]
        offset_start = sys.argv[5]
        offset_end = sys.argv[6]
        infile_label = sys.argv[7]
        outfile_label = sys.argv[8]

        if not os.path.exists(outfile):
            main(infile, start, end, offset_start, offset_end, outfile)
        else:
            print("Synced file already exists.")

    else:
        print("Number of input arguments does not match the expected number")











