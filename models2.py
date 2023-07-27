import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Functional, Model
from keras.layers import *
from keras.optimizers import *



def LSTM_Model1(input_shape, nodes=300, output_shape=1):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(input_shape[1], 1), activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
    model.add(Flatten())
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
    return model

def LSTM_Model2(input_shape, nodes=300, output_shape=1):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(input_shape[1], 1), activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(nodes, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
    return model


def LSTM_Model3(input_shape, nodes=300, output_shape=1):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(input_shape[1], 1), activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(nodes, activation=keras.layers.LeakyReLU(alpha=0.01)))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
    return model


def GRU_Model1(input_shape, nodes=300, output_shape=1):
    model = Sequential()
    model.add(GRU(nodes, input_shape=(input_shape[1], 1), activation='tanh', return_sequences=False))
    model.add(Flatten())
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
    return model


def GRU_Model2(input_shape, nodes=300, output_shape=1):
    model = Sequential()
    model.add(GRU(nodes, input_shape=(input_shape[1], 1), activation='tanh', return_sequences=False))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(nodes, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
    return model


def GRU_Model3(input_shape, nodes=300, output_shape=1):
    model = Sequential()
    model.add(GRU(nodes, input_shape=(input_shape[1], 1), activation='tanh', return_sequences=False))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(nodes, activation=keras.layers.LeakyReLU(alpha=0.01)))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
    return model

####################################
def Dense_Model1(input_shape, nodes=89, output_shape=1):
    model = Sequential()
    model.add(Dense(nodes, input_shape=(input_shape[1],), activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
    return model


def Dense_Model2(input_shape, nodes=89, output_shape=1):
    model = Sequential()
    model.add(Dense(nodes, input_shape=(input_shape[1],), activation=keras.layers.LeakyReLU(alpha=0.01)))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
    return model
