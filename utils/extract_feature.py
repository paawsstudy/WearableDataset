

# each window is num_sample * 3 channel
# return an array of whatever size represent the feature extracted from this window
# might return feature name if needed
def extract_feature_muss(window, sampling_rate = 80, return_name = False):
    import numpy as np
    from scipy.stats import skew, kurtosis
    from scipy.fft import rfft,rfftfreq
    from math import atan
    import sys
    epsilon = sys.float_info.epsilon
    feature_list = []
    feature_name = []
    # compute the magnitude of the accelerometer data
    magnitude = np.sqrt(np.sum(window**2, axis = -1))
    # simple time-domain features
    feature_list.append(np.nanmean(magnitude))
    feature_list.append(np.nanvar(magnitude))
    feature_name.append('mean')
    feature_name.append('variance')
    feature_list.append(np.amax(magnitude) - np.amin(magnitude))
    feature_list.append(np.nanmedian(magnitude))
    feature_name.append('acceleration_range')
    feature_name.append('acceleration_median')
    feature_list.append(np.nanmean(np.sum(window**2, axis = 0))) # power_x + power_y + power_z with power = acc**2
    feature_name.append('acceleration_tot_pow')
    feature_list.append(skew(magnitude))
    feature_list.append(kurtosis(magnitude))
    feature_name.append('skew')
    feature_name.append('kurtosis')
    
    feature_list.append(np.nanmean(window[:,0]))
    feature_list.append(np.nanvar(window[:,0]))
    feature_name.append('mean_X')
    feature_name.append('var_X')
    feature_list.append(np.nanmean(window[:,1]))
    feature_list.append(np.nanvar(window[:,1]))
    feature_name.append('mean_Y')
    feature_name.append('var_Y')
    feature_list.append(np.nanmean(window[:,2]))
    feature_list.append(np.nanvar(window[:,2]))
    feature_name.append('mean_Z')
    feature_name.append('var_Z')
    # frequency features
    fft_magnitude = np.abs(rfft(magnitude - np.mean(magnitude))) # remove the 0 freq then do fft
    frequency = rfftfreq(magnitude.shape[0], d=1/sampling_rate)
    dom_ind = np.argmax(fft_magnitude)
    feature_list.append(frequency[dom_ind])
    feature_list.append(fft_magnitude[dom_ind]**2)
    feature_name.append('dominant_freq')
    feature_name.append('dominant_freq_power')

    # orientation features - extract from https://bitbucket.org/mhealthresearchgroup/mdcas-python/src/master/SWaN_pack/orientation.py - special thanks to Qu
    sub_win_num = 4
    sub_win_sample = int(window.shape[0]/sub_win_num)
    sub_win_list = [window[i* sub_win_sample: (i+1)*sub_win_sample] for i in range(sub_win_num)]
    orientation_xyz = []
    for sub_win in sub_win_list:
        gravity = np.nanmedian(sub_win, axis=0)
        theta = atan(gravity[0] / (np.sqrt(np.sum(np.square(np.array(gravity[[1,2]])))) +epsilon)) * (180 / pi)
        trident = atan(gravity[1] / (np.sqrt(np.sum(np.square(np.array(gravity[[0,2]])))) +epsilon)) * (180 / pi)
        phi= atan( np.sqrt(np.sum(np.square(np.array(gravity[[0,1]])))) / (gravity[2]+epsilon) ) * (180 / pi) 
        orientation_xyz.append(([theta,trident,phi]))
    orientation_xyz = np.asarray(orientation_xyz)
    
    feature_list.append(np.nanmedian(orientation_xyz))
    feature_list.append(np.nanmedian(orientation_xyz[:,0]))
    feature_list.append(np.nanmedian(orientation_xyz[:,1]))
    feature_list.append(np.nanmedian(orientation_xyz[:,2]))
    feature_name.append('median_angle')
    feature_name.append('median_X_roll')
    feature_name.append('median_Y_yaw')
    feature_name.append('median_Z_pitch')
    
    feature_list.append(np.amax(orientation_xyz[:,0]) - np.amin(orientation_xyz[:,0]))
    feature_list.append(np.amax(orientation_xyz[:,1]) - np.amin(orientation_xyz[:,1]))
    feature_list.append(np.amax(orientation_xyz[:,2]) - np.amin(orientation_xyz[:,2]))
    feature_name.append('range_roll')
    feature_name.append('range_yaw')
    feature_name.append('range_pitch')
    
    feature_list.append(np.nanvar(orientation_xyz[:,0]))
    feature_list.append(np.nanvar(orientation_xyz[:,1]))
    feature_list.append(np.nanvar(orientation_xyz[:,2]))
    feature_name.append('var_roll')
    feature_name.append('var_yaw')
    feature_name.append('var_pitch')
    
    if return_name:
        return feature_list, feature_name
    return feature_list