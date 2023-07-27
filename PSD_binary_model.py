import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from keras.models import Sequential, Functional
from keras.layers import *
from keras.optimizers import *
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
import scipy
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
from scipy.fftpack import fft, ifft
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import preproc
import model
import models2 as models
import utilities as ut
import time
print(len(tf.config.list_physical_devices('GPU')))

### Identifying the two subject from each other ###

######### END PREAMBLE #########

# Sampling frequency
fs = 44100

### Path to data ###

data_proc_path = "C:/Users/PATH_TO_DATA"
save_data_path = "C:/Users/PATH_TO_MODEL_DATA" # Save directory
save_fig_path = "C:/Users/PATH_TO_FIGS_FOLDER"

### PSD COMPARISON FILE ###
######### RUN NUMBER #########
#################################
# 0 - 

run_number = 0
model_type = 'GRU' #options: ('GRU','LSTM', 'Dense')
model_num = 3 

# PSD parameters
t_dur = 3
nperseg_fct = 3
f_cutoff = 30


# Load in data

bkg1_psd_log = np.load(data_proc_path + 'bkg_psd_log_%inps_subject1_%is_fc%i.npy' %(nperseg_fct, t_dur, f_cutoff))
bkg2_psd_log = np.load(data_proc_path + 'bkg_psd_log_%inps_subject2_%is_fc%i.npy' %(nperseg_fct, t_dur, f_cutoff))

bkg1_labels = np.load(data_proc_path + 'bkg_psd_labels_%inps_subject1_%ss_fc%i.npy' %(nperseg_fct, t_dur, f_cutoff))
bkg2_labels = np.load(data_proc_path + 'bkg_psd_labels_%inps_subject2_%ss_fc%i.npy' %(nperseg_fct, t_dur, f_cutoff))

hum1_psd_log = np.load(data_proc_path + 'hum_psd_log_%inps_subject1_%ss_fc%i.npy' %(nperseg_fct, t_dur, f_cutoff))
hum2_psd_log = np.load(data_proc_path + 'hum_psd_log_%inps_subject2_%ss_fc%i.npy' %(nperseg_fct, t_dur, f_cutoff))

hum1_labels = np.load(data_proc_path + 'hum_psd_labels_%inps_subject1_%ss_fc%i.npy' %(nperseg_fct, t_dur, f_cutoff))
hum2_labels = np.load(data_proc_path + 'hum_psd_labels_%inps_subject2_%ss_fc%i.npy' %(nperseg_fct, t_dur, f_cutoff))



print('Data shapes (Sliced)')


min_idx = min(len(bkg1_psd_log), len(bkg2_psd_log))

print('BKG DATA')
print(np.shape(bkg1_psd_log[:min_idx]), np.shape(bkg1_labels[:min_idx]))
print(np.shape(bkg2_psd_log[:min_idx]), np.shape(bkg2_labels[:min_idx]))


print('HUM DATA')
print(np.shape(hum1_psd_log[:min_idx]), np.shape(hum1_labels[:min_idx]))
print(np.shape(hum2_psd_log[:min_idx]), np.shape(hum2_labels[:min_idx]))

hum_tot_labs = []
hum_tot = np.concatenate((hum1_psd_log[:min_idx], hum2_psd_log[:min_idx]))

for i in hum_tot:
    hum_tot_labs.append(int(1))

bkg_tot = np.concatenate((bkg1_psd_log[:min_idx], bkg2_psd_log[:min_idx]))#, bkg3_psd))#, bkg4_psd))
bkg_tot_labs = np.concatenate((bkg1_labels[:min_idx], bkg2_labels[:min_idx]))#, bkg3_labels))#, bkg4_labels))


print('Bkg and Hum Data Tot')
print('Min/Max of raw bkg and hum data')
print(np.min(bkg_tot), np.max(bkg_tot))
print(np.min(hum_tot), np.max(hum_tot))

print('Total shapes')
print(np.shape(bkg_tot), np.shape(bkg_tot_labs))
print(np.shape(hum_tot), np.shape(hum_tot_labs))

# Concantenate data into single set
x_data = np.concatenate((bkg_tot, hum_tot))
y_data = np.concatenate((bkg_tot_labs, hum_tot_labs))
print(np.shape(x_data))

print('x_data, y_data shapes')
print(np.shape(x_data), np.shape(y_data))
print(y_data[0], y_data[-1])

# Scale data to bounds `feature_range`
scaler = MinMaxScaler(feature_range=(-1,1))
scaler = scaler.fit(x_data)
x_data = scaler.transform(x_data)

print('X data Min/Max')
print(np.min(x_data), np.max(x_data))
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(y_data)
print('Y encoder classes', encoder.classes_)
print('Encoded Y Shape')
print(np.shape(encoded_Y))

X_train, X_test, Y_train, Y_test = train_test_split(x_data, encoded_Y, random_state = 123, test_size=0.2, stratify=encoded_Y)

print('TRAIN INPUT')
print(np.shape(X_train), np.shape(Y_train))
print("TEST INPUT")
print(np.shape(X_test), np.shape(Y_test))

nclasses = len(encoder.classes_)
print(nclasses)
batch_size = 32
epochs = 1000 #100 or 50-100 w/ batch=10
print('Input shapes:') 
input_shape=(X_train.shape)
print(input_shape)
nodes = input_shape[1]
model_name = 'psd_bin2_model%i_bs%i_ep%i_nds%i_%is_fc%i.h5' %(run_number, batch_size, epochs, nodes, t_dur, f_cutoff)
history_name = 'psd_bin2_history%i_bs%i_ep%i_nds%i_%is_fc%i.npy' %(run_number, batch_size, epochs, nodes, t_dur, f_cutoff)
model_path = save_data_path + model_name 
history_path = save_data_path + history_name
print('Batch size: %i\nEpochs: %i\nNodes: %i' %(batch_size, epochs, nodes))
plot = True

recreate_model = True

####################################################################
###### CHANGE DEPENDING ON WHICH MODEL IS DEFINED IN PREAMBLE ######

model1 = models.GRU_Model3(input_shape, nodes=nodes, output_shape=1)

####################################################################

if (os.path.exists(model_path)) and (os.path.exists(history_path)) and recreate_model == False:
    print('Model exists.. loading model: %s' %model_name)
    print(model1.summary())
    model1.load_weights(model_path)
    history=np.load(history_path, allow_pickle='TRUE').item()
    #print(history)
    if plot == True:
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        plot = False
    save = False

else:
    print('model does not exist.. creating model: %s' %model_name)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
    print(model1.summary())
    history = model1.fit(X_train, Y_train, epochs=epochs, validation_split = 0.2, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[es])
    if plot == True:
        sns.set_theme(style='whitegrid', font_scale=2, rc={'lines.linewidth': 2})
        plt.figure(figsize=(12, 8))
        plt.plot(history.history['accuracy'], label='Acc. Train')
        plt.plot(history.history['val_accuracy'], label='Acc. Val')
        plt.plot(history.history['loss'], label='Loss Train')
        plt.plot(history.history['val_loss'], label='Loss Val')
        plt.title('%s Model %i Accuracy and Loss' %(model_type, model_num))
        plt.ylabel('Accuracy/Loss')
        plt.xlabel('Epoch')
        plt.ylim(0, 1)
        plt.legend()#['train', 'test'], loc='upper left')
        plt.show()
        plot = False
    np.save(history_path, history.history)
    print('History saved to: ', history_path)
    model1.save(model_path) 
    print('model saved to: ', model_path)

te_history = model1.evaluate(X_test, Y_test, batch_size=batch_size, return_dict=True)
print('Evaluate history: ', te_history)

test_pred = model1.predict(X_test)
train_pred = model1.predict(X_train)


threshold = 0.5
save_figs = True
test_pred = model1.predict(X_test)

test_pred_th = np.where(test_pred > threshold, int(1), int(0))
#train_pred_th = np.where(train_pred > threshold, int(1), int(0))

test_pred_max = np.argmax(test_pred, axis=1)


matrix = confusion_matrix(Y_test, test_pred_th)
print('Matrix')
print(matrix)


fname_wo_ext = os.path.splitext(model_name)[0]
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, test_pred_th)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, test_pred_th)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, test_pred_th)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, test_pred_th)
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(Y_test, test_pred_th)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(Y_test, test_pred)
print('ROC AUC: %f' % auc)



classNames = ['Background', 'Human']#, 'Subject 4']#, 'Subject 4']

plot_title = 'Binary %s Model %i ($f_c = %i$ Hz)' %(model_type, model_num, f_cutoff)
img_name =  fname_wo_ext + '_test.png'
ut.plot_confusion_matrix3(Y_test, test_pred_th, classNames, save_fig_path, img_name, plot_title, binary=True, save=save_figs, dark_bkg=True)


metric_dict = {'accuracy' : accuracy, 'precision' : precision, 'recall':recall, 'f1':f1, 'kappa':kappa, 'auc':auc}
df_metric_dict = pd.DataFrame.from_dict(metric_dict, orient='index')
metric_name = fname_wo_ext + '_te_metrics'
if os.path.exists(save_data_path + metric_name) == False:
    print('Saving metrics...', save_data_path, metric_name)
    df_metric_dict.to_csv(save_data_path + metric_name, index=True)

elif recreate_model == True:
    print('Saving metrics...', save_data_path, metric_name)
    df_metric_dict.to_csv(save_data_path + metric_name, index=True)

else:
    print('Fig exists')

print("**** Classification Report ****")
print(classification_report(Y_test, test_pred_th, target_names=classNames))


print('Done')
