import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Functional
from keras.layers import *
from keras.optimizers import *
from tensorflow import keras
import scipy
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.fftpack import fft, ifft
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import preproc
import model

import time


######### PREAMBLE #########

### Path to data ###
# Subject 1
hum_data_path = 'C:/Users/PATH_TO_SUBJECT1_HUM_DATA'
bkg_data_path = 'C:/Users/PATH_TO_SUBJECT1_BKG_DATA'

# Subject 2
bkg_data_path2 = 'C:/Users/PATH_TO_SUBJECT2_BKG_DATA'
hum_data_path2 = 'C:/Users/PATH_TO_SUBJECT2_HUM_DATA'


save_data_path = "C:/Users/SAVE_DATA_PATH"


fs = 44100
t_dur = 3
f_cutoff = 20
nperseg_fct = 3
f_min = 1/3
feat_len = fs*t_dur
t_div = 1/fs
f_c_idx = f_cutoff*nperseg_fct
f_0_idx = 1



######
subjectNum = 1
######


bkg_data_df = pd.read_csv(bkg_data_path, skiprows=3, index_col=0, header=0)
hum_data_df = pd.read_csv(hum_data_path, skiprows=3, index_col=0, header=0)

bkg_data = bkg_data_df.to_numpy()
hum_data = hum_data_df.to_numpy()

print(bkg_data_df.isnull().values.any())
print(hum_data_df.isnull().values.any())

print(bkg_data_df.isnull().sum().sum())
print(hum_data_df.isnull().sum().sum())

print(len(bkg_data), len(hum_data))
print(np.isnan(bkg_data).any(), np.isnan(hum_data).any())
bkg_len = len(bkg_data)
hum_len = len(hum_data)
bkg_nan = np.isnan(bkg_data).any()
hum_nan = np.isnan(hum_data).any()

if bkg_nan == True and hum_nan == True:
    print('Both NaNs')
    bkg_nan_idx = np.argwhere(np.isnan(bkg_data))
    hum_nan_idx = np.argwhere(np.isnan(hum_data))
    print(bkg_data_df.isnull().sum().sum())
    print(hum_data_df.isnull().sum().sum())
    lowest_idx_b = np.min(bkg_nan_idx)
    lowest_idx_h = np.min(hum_nan_idx)

    print('Lowest indices: ', lowest_idx_b, lowest_idx_h)
    print(np.argwhere(np.isnan(bkg_data)), np.argwhere(np.isnan(hum_data)))

    if lowest_idx_b > lowest_idx_h:
        bkg_data = bkg_data[0:lowest_idx_h]
        hum_data = hum_data[0:lowest_idx_h]
    elif lowest_idx_h > lowest_idx_b:
        bkg_data = bkg_data[0:lowest_idx_b]
        hum_data = hum_data[0:lowest_idx_b]

elif bkg_nan == False and hum_nan == False:
    print('No NaNs')
    print(len(bkg_data), len(hum_data))
    if len(bkg_data) > len(hum_data):
        bkg_data = bkg_data[0:len(hum_data)]
        hum_data = hum_data[0:len(hum_data)]
    elif len(hum_data) > len(bkg_data):
        bkg_data = bkg_data[0:len(bkg_data)]
        hum_data = hum_data[0:len(bkg_data)]

elif bkg_nan == True and hum_nan == False:
    print('Bkg Nan, Hum not NaN')
    bkg_nan_idx = np.argwhere(np.isnan(bkg_data))
    lowest_idx_b = np.min(bkg_nan_idx)
    if bkg_nan_idx > hum_len:
        bkg_data = bkg_data[0:hum_len]
        hum_data = hum_data[0:hum_len]
    elif bkg_nan_idx < hum_len:
        bkg_data = bkg_data[0:lowest_idx_b]
        hum_data = hum_data[0:lowest_idx_b]

elif bkg_nan == False and hum_nan == True:
    print('Hum NaN, Bkg not NaN')
    hum_nan_idx = np.argwhere(np.isnan(hum_data))
    lowest_idx_h = np.min(hum_nan_idx)
    if hum_nan_idx > bkg_len:
        bkg_data = bkg_data[0:bkg_len]
        hum_data = hum_data[0:bkg_len]
    elif hum_nan_idx < bkg_len:
        bkg_data = bkg_data[0:lowest_idx_h]
        hum_data = hum_data[0:lowest_idx_h]

print('Selected data')
print(np.shape(bkg_data), np.shape(hum_data))

bkg_data = np.abs(bkg_data)
hum_data = np.abs(hum_data)

bkg_data = np.reshape(bkg_data, (len(bkg_data),))
hum_data = np.reshape(hum_data, (len(hum_data),))
print('Reshaped Data')
print('bkg')
print(np.shape(bkg_data))

print('hum')
print(np.shape(hum_data))


bkg_data, hum_data = preproc.split_data(bkg_data, hum_data, feed_len=feat_len)
print('SPLIT')
print(np.shape(bkg_data), np.shape(hum_data))


bkg_psd_data = []
bkg_psd_log = []
bkg_psd_freqs = []
bkg_psd_labels = []

plot_once = False
for i, seq in enumerate(bkg_data):

    f, Pxx = preproc.signal_psd(seq, fs=fs, n_per_seg=fs*nperseg_fct, plot=plot_once)

    Pxx = np.reshape(Pxx, (len(Pxx)))
    Pxx_log = np.log10(Pxx)
    bkg_psd_data.append(Pxx[f_0_idx:f_c_idx])
    bkg_psd_log.append(Pxx_log[f_0_idx:f_c_idx])
    bkg_psd_labels.append(int(0))
    plot_once = False

print(np.shape(bkg_psd_data))

print('f slide') 
print(f[f_0_idx], f[f_c_idx])
print(len(bkg_psd_data[0]))

hum_psd_data = []
hum_psd_log = []
hum_psd_labels = []

for i, seq in enumerate(hum_data):

    f, Pxx = preproc.signal_psd(seq, fs=fs, n_per_seg=fs*nperseg_fct, plot=False)

    Pxx = np.reshape(Pxx, (len(Pxx)))
    Pxx_log = np.log10(Pxx)
    hum_psd_data.append(Pxx[f_0_idx:f_c_idx])
    hum_psd_log.append(Pxx_log[f_0_idx:f_c_idx])
    #psd.append(Pxx)
    hum_psd_labels.append(int(subjectNum))

print(np.shape(hum_psd_log))


print('Data shapes')
print(np.shape(bkg_psd_log), np.shape(hum_psd_log))
print(np.shape(bkg_psd_data), np.shape(hum_psd_data))
print('Labels')
print(np.shape(bkg_psd_labels), np.shape(hum_psd_labels))
print('Saving to %s' %save_data_path)


np.save(save_data_path + 'bkg_psd_data_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), bkg_psd_data)
np.save(save_data_path + 'hum_psd_data_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), hum_psd_data)
np.save(save_data_path + 'bkg_psd_log_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), bkg_psd_log)
np.save(save_data_path + 'hum_psd_log_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), hum_psd_log)
np.save(save_data_path + 'bkg_psd_labels_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), bkg_psd_labels)
np.save(save_data_path + 'hum_psd_labels_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), hum_psd_labels)


####################################


######
subjectNum = 2
######


bkg_data_df = pd.read_csv(bkg_data_path2, skiprows=3, index_col=0, header=0)
hum_data_df = pd.read_csv(hum_data_path2, skiprows=3, index_col=0, header=0)


bkg_data = bkg_data_df.to_numpy()
hum_data = hum_data_df.to_numpy()

print(bkg_data_df.isnull().values.any())
print(hum_data_df.isnull().values.any())

print(bkg_data_df.isnull().sum().sum())
print(hum_data_df.isnull().sum().sum())

print(len(bkg_data), len(hum_data))
print(np.isnan(bkg_data).any(), np.isnan(hum_data).any())
bkg_len = len(bkg_data)
hum_len = len(hum_data)
bkg_nan = np.isnan(bkg_data).any()
hum_nan = np.isnan(hum_data).any()

if bkg_nan == True and hum_nan == True:
    print('Both NaNs')
    bkg_nan_idx = np.argwhere(np.isnan(bkg_data))
    hum_nan_idx = np.argwhere(np.isnan(hum_data))
    print(bkg_data_df.isnull().sum().sum())
    print(hum_data_df.isnull().sum().sum())
    lowest_idx_b = np.min(bkg_nan_idx)
    lowest_idx_h = np.min(hum_nan_idx)

    print('Lowest indices: ', lowest_idx_b, lowest_idx_h)
    print(np.argwhere(np.isnan(bkg_data)), np.argwhere(np.isnan(hum_data)))

    if lowest_idx_b > lowest_idx_h:
        bkg_data = bkg_data[0:lowest_idx_h]
        hum_data = hum_data[0:lowest_idx_h]
    elif lowest_idx_h > lowest_idx_b:
        bkg_data = bkg_data[0:lowest_idx_b]
        hum_data = hum_data[0:lowest_idx_b]

elif bkg_nan == False and hum_nan == False:
    print('No NaNs')
    print(len(bkg_data), len(hum_data))
    if len(bkg_data) > len(hum_data):
        bkg_data = bkg_data[0:len(hum_data)]
        hum_data = hum_data[0:len(hum_data)]
    elif len(hum_data) > len(bkg_data):
        bkg_data = bkg_data[0:len(bkg_data)]
        hum_data = hum_data[0:len(bkg_data)]

elif bkg_nan == True and hum_nan == False:
    print('Bkg Nan, Hum not NaN')
    bkg_nan_idx = np.argwhere(np.isnan(bkg_data))
    lowest_idx_b = np.min(bkg_nan_idx)
    if bkg_nan_idx > hum_len:
        bkg_data = bkg_data[0:hum_len]
        hum_data = hum_data[0:hum_len]
    elif bkg_nan_idx < hum_len:
        bkg_data = bkg_data[0:lowest_idx_b]
        hum_data = hum_data[0:lowest_idx_b]

elif bkg_nan == False and hum_nan == True:
    print('Hum NaN, Bkg not NaN')
    hum_nan_idx = np.argwhere(np.isnan(hum_data))
    lowest_idx_h = np.min(hum_nan_idx)
    if hum_nan_idx > bkg_len:
        bkg_data = bkg_data[0:bkg_len]
        hum_data = hum_data[0:bkg_len]
    elif hum_nan_idx < bkg_len:
        bkg_data = bkg_data[0:lowest_idx_h]
        hum_data = hum_data[0:lowest_idx_h]

print('Selected data')
print(np.shape(bkg_data), np.shape(hum_data))


if len(bkg_data) > len(hum_data):
    bkg_data = bkg_data[0:len(hum_data)]
    hum_data = hum_data[0:len(hum_data)]
else:
    bkg_data = bkg_data[0:len(bkg_data)]
    hum_data = hum_data[0:len(bkg_data)]

print('Selected data')
print(np.shape(bkg_data), np.shape(hum_data))

bkg_data = np.abs(bkg_data)
hum_data = np.abs(hum_data)


bkg_data = np.reshape(bkg_data, (len(bkg_data),))
hum_data = np.reshape(hum_data, (len(hum_data),))
print('Reshaped Data')
print('bkg')
print(np.shape(bkg_data))
print('hum')
print(np.shape(hum_data))
t_dur = 3
feat_len = fs*t_dur

bkg_data, hum_data = preproc.split_data(bkg_data, hum_data, feed_len=feat_len)
print('SPLIT')
print(np.shape(bkg_data), np.shape(hum_data))

bkg_psd_data = []
bkg_psd_log = []
bkg_psd_freqs = []
bkg_psd_labels = []

plot_once = False
for i, seq in enumerate(bkg_data):

    f, Pxx = preproc.signal_psd(seq, fs=fs, n_per_seg=fs*nperseg_fct, plot=plot_once)
    Pxx = np.reshape(Pxx, (len(Pxx)))
    Pxx_log = np.log10(Pxx)
    bkg_psd_data.append(Pxx[f_0_idx:f_c_idx])
    bkg_psd_log.append(Pxx_log[f_0_idx:f_c_idx])
    bkg_psd_labels.append(int(0))
    plot_once = False

print(np.shape(bkg_psd_data))

print('f slide') 
print(f[f_0_idx], f[f_c_idx])
print(len(bkg_psd_data[0]))

hum_psd_data = []
hum_psd_log = []
hum_psd_labels = []

for i, seq in enumerate(hum_data):

    f, Pxx = preproc.signal_psd(seq, fs=fs, n_per_seg=fs*nperseg_fct, plot=False)

    Pxx = np.reshape(Pxx, (len(Pxx)))
    Pxx_log = np.log10(Pxx)
    hum_psd_data.append(Pxx[f_0_idx:f_c_idx])
    hum_psd_log.append(Pxx_log[f_0_idx:f_c_idx])
    hum_psd_labels.append(int(subjectNum))

print(np.shape(hum_psd_log))


print('Data shapes')
print(np.shape(bkg_psd_log), np.shape(hum_psd_log))
print(np.shape(bkg_psd_data), np.shape(hum_psd_data))
print('Labels')
print(np.shape(bkg_psd_labels), np.shape(hum_psd_labels))
print('Saving to %s' %save_data_path)
print('Example save name:\n' + 'bkg_psd6_data_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff) )


np.save(save_data_path + 'bkg_psd_data_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), bkg_psd_data)
np.save(save_data_path + 'hum_psd_data_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), hum_psd_data)
np.save(save_data_path + 'bkg_psd_log_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), bkg_psd_log)
np.save(save_data_path + 'hum_psd_log_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), hum_psd_log)
np.save(save_data_path + 'bkg_psd_labels_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), bkg_psd_labels)
np.save(save_data_path + 'hum_psd_labels_%inps_subject%i_%ss_fc%i' %(nperseg_fct, subjectNum, t_dur, f_cutoff), hum_psd_labels)
