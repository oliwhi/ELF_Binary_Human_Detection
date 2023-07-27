import os
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy
from scipy import signal
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import find_peaks
from scipy.fftpack import fft, ifft
from time import sleep
from datetime import datetime
from collections import Iterable
from sklearn.preprocessing import LabelEncoder


def normalize_df(df):
    min_val = min(df['V'])
    max_val = max(df['V'])
    #print(max_val, min_val)
    nrm_df = df[['Unit']].copy()
    #nrm_df['Unit'] = df['Unit']
    nrm_df['V'] = df['V'].map(lambda V: (V - min_val) / (max_val - min_val))
    return nrm_df



def make_window(signal, fs, overlap, window_size_sec):
    """
    perform cropped signals of window_size seconds for the whole signal
    overlap input is in percentage of window_size
    window_size is in seconds """

    window_size = fs * window_size_sec
    overlap     = int(window_size * (overlap / 100))
    start       = 0
    segmented   = np.zeros((1, window_size), dtype = int)
    while(start+window_size <= len(signal)):
        segment     = signal[start:start+window_size]
        segment     = segment.reshape(1, len(segment))
        segmented   = np.append(segmented, segment, axis =0)
        start       = start + window_size - overlap
    return segmented[1:]


def signal_psd(data, fs, n_per_seg, plot=False, scale='density', overlap=None, average='mean'):
    N = len(data)
    (f, Pxx) = scipy.signal.welch(data, fs, nperseg=n_per_seg, scaling=scale, noverlap=overlap, average=average)
    #print(f)
    if plot:
        plt.plot(f, Pxx)#, vmin=0, vmax=0.5)
        plt.title('PSD')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.yscale('log')
        plt.xlim(0, 70)
        plt.grid()
        plt.show()
    return f, Pxx

def signal_fft(data, fs=44100, plot=False):
    N = len(data)
    fft_data = fft(data)#, axis=0)#, noverlap=0)
    f = fftfreq(N, d=(1/fs))
    if plot:
        #f = fftfreq(N, d=(1/fs))
        plt.plot(f, fft_data)#, vmin=0, vmax=0.5)
        plt.title('FFT')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.yscale('log')
        plt.xlim(0, 70)
        plt.grid()
        plt.show()
    return fft_data, f



def split_data(bkg_data, hum_data, feed_len):
    # Splits data into windows of size `feed_len`. Turns 1D array into 2D array
    bkg_split = []
    hum_split = []
    bkg_len = len(bkg_data[:])
    hum_len = len(hum_data[:])

    if bkg_len < hum_len:
        split_num = bkg_len // feed_len
    else:
        split_num = hum_len // feed_len

    #feed_len = fs*3

    split_num = hum_len // feed_len
    assert split_num*feed_len <= hum_len, "Feed len too big for hum data"

    for sec in np.arange(split_num):
        bkg_split.append(bkg_data[sec*feed_len:(sec+1)*feed_len])
        hum_split.append(hum_data[sec*feed_len:(sec+1)*feed_len])
    
    return bkg_split, hum_split

def split_scale_data(bkg_data, hum_data, scaler):
    bkg_split = []
    hum_split = []
    bkg_len = len(bkg_data[:])
    hum_len = len(hum_data[:])

    feed_len = 720*3

    split_num = hum_len // feed_len
    assert split_num*feed_len <= hum_len, "Feed len too big for hum data"

    for sec in np.arange(split_num):
        bkg_split.append(bkg_data[sec*feed_len:(sec+1)*feed_len])
        hum_split.append(hum_data[sec*feed_len:(sec+1)*feed_len])
    
    return bkg_split, hum_split


def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:        
             yield item


def encode_labels(encoder, labels):
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)
    return encoded_Y
