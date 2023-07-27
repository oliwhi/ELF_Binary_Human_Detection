import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_validate
from matplotlib import pyplot as plt
import seaborn as sns
#from tqdm.auto import tqdm, trange
#from nptdms import TdmsFile
#from skimage.measure import block_reduce
import scipy
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import math
import pandas as pd



def plot_confusion_matrix2(y_test, y_scores, classNames, save_dir, img_name, plot_title, binary=False, save=True):
   if binary == False:
      y_test=np.argmax(y_test, axis=1)
      y_scores=np.argmax(y_scores, axis=1)
   classes = len(classNames)
   cm = confusion_matrix(y_test, y_scores)
   print("**** Confusion Matrix ****")
   print(cm)
   print("**** Classification Report ****")
   print(classification_report(y_test, y_scores, target_names=classNames))

   con = np.zeros((classes, classes))
   names = np.zeros((classes, classes))
   for x in range(classes):
      for y in range(classes):
         prob = cm[x,y]/np.sum(cm[x,:])
         con[x,y] = prob*100
         names[x,y] = str(cm[x,y])
   plt.figure(figsize=(40,40))
   sns.set(font_scale=3.0) # for label size
   off_diag_mask = np.eye(*cm.shape, dtype=bool)
   df = sns.heatmap(con, annot=names, mask=~off_diag_mask, fmt='.0f', cmap='Blues', vmin=0, vmax=100, xticklabels= classNames, yticklabels=classNames, cbar_kws={'label':'% Success', 'pad':0.01})
   df = sns.heatmap(con, annot=names, mask=off_diag_mask, fmt='.0f', cmap='Reds', vmin=0, vmax=100, xticklabels= classNames, yticklabels=classNames, cbar_kws={'label':'% Error'})
   #sns.heatmap(cm, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax)
   #sns.heatmap(cm, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]))
   df.set_title(plot_title)
   df.set(xlabel="Predicted Label", ylabel="True Label")
   df.figure.show()
   if save == True:
      df.figure.savefig(save_dir + img_name)
   return df


def plot_confusion_matrix3(y_test, y_scores, classNames, save_dir, img_name, plot_title, binary=False, save=True, dark_bkg = False):
   if binary == False:
      y_test=np.argmax(y_test, axis=1)
      y_scores=np.argmax(y_scores, axis=1)
   classes = len(classNames)
   cm = confusion_matrix(y_test, y_scores)
   print("**** Confusion Matrix ****")
   print(cm)
   print("**** Classification Report ****")
   print(classification_report(y_test, y_scores, target_names=classNames))

   con = np.zeros((classes, classes))
   names = np.zeros((classes, classes), dtype='object')
   for x in range(classes):
      for y in range(classes):
         prob = cm[x,y]/np.sum(cm[x,:])
         con[x,y] = prob*100
         names[x,y] = '%i \n%.1f%%' %(cm[x,y], con[x,y])
         #names[x,y] = str(cm[x,y])
         #names[x,y] = names[x,y] + '\n%.1f%%' %(con[x,y])
   plt.figure(figsize=(12,12))
   sns.set(font_scale=3.0) # for label size
   off_diag_mask = np.eye(*cm.shape, dtype=bool)
   if dark_bkg == True:
      plt.style.use("dark_background")
      df = sns.heatmap(con, annot=names, mask=~off_diag_mask, fmt='', cmap='Greens', vmin=0, vmax=100, xticklabels= classNames, yticklabels=classNames, cbar=False) #cbar_kws={'label':'% Success', 'pad':0.01})
      df = sns.heatmap(con, annot=names, mask=off_diag_mask, fmt='', cmap='Reds', vmin=0, vmax=100, xticklabels= classNames, yticklabels=classNames, cbar=False) #cbar_kws={'label':'% Error'})
   else:
      df = sns.heatmap(con, annot=names, mask=~off_diag_mask, fmt='', cmap='Blues', vmin=0, vmax=100, xticklabels= classNames, yticklabels=classNames, cbar=False) #cbar_kws={'label':'% Success', 'pad':0.01})
      df = sns.heatmap(con, annot=names, mask=off_diag_mask, fmt='', cmap='Reds', vmin=0, vmax=100, xticklabels= classNames, yticklabels=classNames, cbar=False) #cbar_kws={'label':'% Error'})
   #sns.heatmap(cm, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax)
   #sns.heatmap(cm, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]))
   df.set_title(plot_title)
   df.set(xlabel="Predicted Label", ylabel="True Label")
   df.figure.show()
   if save == True:
      df.figure.savefig(save_dir + img_name, bbox_inches='tight')
   return df

def plot_confusion_matrix3_og(y_test, y_scores, classNames, save_dir, img_name, plot_title, binary=False, save=True):
   if binary == False:
      y_test=np.argmax(y_test, axis=1)
      y_scores=np.argmax(y_scores, axis=1)
   classes = len(classNames)
   cm = confusion_matrix(y_test, y_scores)
   print("**** Confusion Matrix ****")
   print(cm)
   print("**** Classification Report ****")
   print(classification_report(y_test, y_scores, target_names=classNames))

   con = np.zeros((classes, classes))
   names = np.zeros((classes, classes), dtype='object')
   for x in range(classes):
      for y in range(classes):
         prob = cm[x,y]/np.sum(cm[x,:])
         con[x,y] = prob*100
         names[x,y] = '%i \n%.1f%%' %(cm[x,y], con[x,y])
         #names[x,y] = str(cm[x,y])
         #names[x,y] = names[x,y] + '\n%.1f%%' %(con[x,y])
   plt.figure(figsize=(12,12))
   sns.set(font_scale=3.0) # for label size
   off_diag_mask = np.eye(*cm.shape, dtype=bool)
   df = sns.heatmap(con, annot=names, mask=~off_diag_mask, fmt='', cmap='Blues', vmin=0, vmax=100, xticklabels= classNames, yticklabels=classNames, cbar=False) #cbar_kws={'label':'% Success', 'pad':0.01})
   df = sns.heatmap(con, annot=names, mask=off_diag_mask, fmt='', cmap='Reds', vmin=0, vmax=100, xticklabels= classNames, yticklabels=classNames, cbar=False) #cbar_kws={'label':'% Error'})
   #sns.heatmap(cm, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax)
   #sns.heatmap(cm, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]))
   df.set_title(plot_title)
   df.set(xlabel="Predicted Label", ylabel="True Label")
   df.figure.show()
   if save == True:
      df.figure.savefig(save_dir + img_name, bbox_inches='tight')
   return df

def cross_validation(model, _X, _y, _cv=5):
      '''Function to perform 5 Folds Cross-Validation
       Parameters
       ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      _X: array
           This is the matrix of features.
      _y: array
           This is the target variable.
      _cv: int, default=5
          Determines the number of folds for cross-validation.
       Returns
       -------
       The function returns a dictionary containing the metrics 'accuracy', 'precision',
       'recall', 'f1' for both training set and validation set.
      '''
      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }


def plot_result(x_label, y_label, plot_title, train_data, val_data):
        '''Function to plot a grouped bar chart showing the training and validation
          results of the ML model in each fold after applying K-fold cross-validation.
         Parameters
         ----------
         x_label: str, 
            Name of the algorithm used for training e.g 'Decision Tree'
          
         y_label: str, 
            Name of metric being visualized e.g 'Accuracy'
         plot_title: str, 
            This is the title of the plot e.g 'Accuracy Plot'
         
         train_result: list, array
            This is the list containing either training precision, accuracy, or f1 score.
        
         val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
         Returns
         -------
         The function returns a Grouped Barchart showing the training and validation result
         in each fold.
        '''
        
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()



def bkgrnd_subtraction(hum_signal, bkg_signal):
	norm_data = []
	assert len(hum_signal) == len(bkg_signal), 'Signal lengths are different'

	for i in np.arange(len(hum_signal)):
		norm_data.append(hum_signal[i]/bkg_signal[i])
	return norm_data

