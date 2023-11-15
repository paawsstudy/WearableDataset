import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
from itertools import islice

col_list = ['Axis1','Axis2','Axis3','Steps','HR','Lux','Inclinometer Off','Inclinometer Standing','Inclinometer Sitting','Inclinometer Lying']
def line_prepender(filename, line):
  with open(filename, 'r+') as f:
      content = f.read()
      f.seek(0, 0)
      f.write(line + content)
  f.close()

def read_data(filename):
    start_date = None
    start_time = None
    with open(filename) as f:
        f.readline()
        f.readline()
        start_time = f.readline().split()[-1]
        start_date = f.readline().split()[-1]
    start = datetime.strptime(start_date + ' ' + start_time, '%m/%d/%Y %H:%M:%S')
    return start

def get_header(filename):
    with open(filename) as myfile:
        head = list(islice(myfile, 10))
    return head

def main(inPath,st_shake,en_shake,st_offset,en_offset,outPath):
  st_shake = int(st_shake)
  en_shake = int(en_shake)
  offset_begin = int(st_offset)
  offset_end = int(en_offset)
  
  head = get_header(inPath)
  
  line_as_pre = ''.join(head)
  resultant_offset = offset_begin - offset_end
  
  df_ori = pd.read_csv(inPath,skiprows=10,header=0)
  df = df_ori.shift(periods= -1*offset_begin,fill_value=0)
  df.reset_index(inplace=True,drop=True)
  
  # df_size = len(df.index)
  
  if resultant_offset == 0:
    print(datetime.now().time()) 
    df.to_csv(outPath,index=False,float_format='%.3f')
    print(datetime.now().time()) 
    line_prepender(outPath,line_as_pre)   
    print("Finished syncing file.")
  
  else:
    # top = int(0.1*df_size)
    # bottom = int(0.9*df_size)
    top = st_shake - offset_begin
    bottom = en_shake - offset_begin

    top_df = df.loc[:top,:]
    bottom_df = df.loc[bottom:,:]
    bottom_df.reset_index(inplace=True,drop=True)
    bottom_df_len = len(bottom_df.index)

    mid_df = df.loc[top:bottom,:]
    mid_df.reset_index(inplace=True,drop=True)
    mid_df_len = len(mid_df.index) 
    
    if resultant_offset < 0:
      resultant_offset = abs(resultant_offset)
      n = int(mid_df_len/resultant_offset)
      li = [i for i in range(0,mid_df_len,n)]
      mid_df.loc[li,:] = np.NaN
      mid_df.dropna(inplace=True)
      mid_df.reset_index(inplace=True,drop=True)

      li_b = [i for i in range(0, bottom_df_len, n)]
      bottom_df.loc[li_b, :] = np.NaN
      bottom_df.dropna(inplace=True)
      bottom_df.reset_index(inplace=True, drop=True)

    elif resultant_offset > 0:
      n = int(mid_df_len/resultant_offset)
      li = [i for i in range(0,mid_df_len,n)]
      li.pop(0)
      count = 0
      for r in li:  
        r = r + count 
        
        # have to construct this following line with IMU data        
        #line = pd.DataFrame(data={'Accelerometer X':[mid_df.loc[r,'Accelerometer X']],
        #'Accelerometer Y':[mid_df.loc[r,'Accelerometer Y']],'Accelerometer Z':[mid_df.loc[r,'Accelerometer Z']]})
        
        line = pd.DataFrame(data = {
            colname:[mid_df.loc[r,colname]] for colname in col_list
        })
        
        mid_df = pd.concat([mid_df.iloc[:r,:], line, mid_df.iloc[r:,:]], ignore_index=True)
        mid_df.reset_index(inplace=True,drop=True)
        count += 1

      li_b = [i for i in range(0, bottom_df_len, n)]
      print(bottom_df_len,n, mid_df_len, resultant_offset)
      if len(li_b):
        li_b.pop(0)
      count_b = 0
      for r in li_b:
        r = r + count_b
        #line_b = pd.DataFrame(data={'Accelerometer X': [bottom_df.loc[r, 'Accelerometer X']],
        #                          'Accelerometer Y': [bottom_df.loc[r, 'Accelerometer Y']],
        #                          'Accelerometer Z': [bottom_df.loc[r, 'Accelerometer Z']]})
        line_b = pd.DataFrame(data = {
            colname:[bottom_df.loc[r,colname]] for colname in col_list
        })
        bottom_df = pd.concat([bottom_df.iloc[:r, :], line_b, bottom_df.iloc[r:, :]], ignore_index=True)
        bottom_df.reset_index(inplace=True, drop=True)
        count_b += 1
    
    final_df = pd.concat([top_df, mid_df, bottom_df], ignore_index=True)
    final_df.reset_index(inplace=True,drop=True)  
    
    print(datetime.now().time())    
    final_df.to_csv(outPath,index=False,float_format='%.3f')
    print(datetime.now().time()) 
    line_prepender(outPath,line_as_pre)   
    print("Finished syncing file.")
  
  
# this function takes 80Hz samples and map that to 1Hz samples
def map_to_1_hz(x):
    return int(x/80) + int((x - int(x/80) * 80)/80)

if __name__ == "__main__":

  if len(sys.argv) == 2:
    infile = sys.argv[1]
    if(os.path.exists(infile)):

      main_df = pd.read_csv(infile)
      for index, row in main_df.iterrows():
        infile = row['INFILE']
        outfile = row['OUTFILE']
        start_time_og = read_data(outfile)
        header_og = get_header(outfile)
        sensor_shake_start = row['SENSOR_SHAKE_START']
        sensor_shake_end = row['SENSOR_SHAKE_END']
        offset_to_start = row['OFFSET_TO_START']
        offset_to_end = row['OFFSET_TO_END']
        # edit these fields to match 1Hz sampling rate of IMU compared to 80Hz of accel
        # also need to edit filename to match the IMU format
        folder = os.path.commonpath([infile, outfile])
        sensor_name = os.path.basename(infile)[:-7] + "1sec.csv"
        infile = os.path.join(folder, sensor_name)
        outfile = os.path.join(folder, sensor_name[:-4] + "_synced.csv")
        sensor_shake_start = map_to_1_hz(sensor_shake_start)
        sensor_shake_end = map_to_1_hz(sensor_shake_end)
        offset_to_start = map_to_1_hz(offset_to_start)
        offset_to_end = map_to_1_hz(offset_to_end)
        if not os.path.exists(infile):
            continue
            
        start_time_agd = read_data(infile)
        if start_time_agd > start_time_og: #mismatch header, need to prepend data to align
            # need to copy new start time
            header_agd = get_header(infile)
            header_agd[2] = header_og[2]
            header_agd[3] = header_og[3]
            new_header = ''.join(header_agd)
            # add in rows of 0s
            df = pd.read_csv(infile,skiprows=10,header=0)
            prepend_df = pd.DataFrame(np.zeros((padding, len(df.columns))), columns = df.columns)
            df = pd.concat((prepend_df, df), axis = 0)
            df.to_csv(infile,index=False)
            line_prepender(sensor_name,new_header)   
        print(infile, outfile)  
        if not os.path.exists(outfile):
          main(infile, sensor_shake_start, sensor_shake_end, offset_to_start, offset_to_end, outfile)
        else:
          print("Synced file already exists : " + outfile)

  elif len(sys.argv) == 7:
    infile = sys.argv[1]
    outfile = sys.argv[2]
    sensor_shake_start = sys.argv[3]
    sensor_shake_end = sys.argv[4]
    offset_to_start = sys.argv[5]
    offset_to_end = sys.argv[6]
    
    if not os.path.exists(outfile):
      main(infile,sensor_shake_start,sensor_shake_end,offset_to_start,offset_to_end,outfile)
    else:
      print("Synced file already exists.")

  else:
    print("Number of input arguments does not match the expected number")


  
  
      
  
  
  
  


