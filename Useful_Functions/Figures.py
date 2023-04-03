# Aneja Lab Common Code
# Figures
# Aneja Lab | Yale School of Medicine
# Sanjay Aneja, MD
# Created (3/6/20)
# Updated (11/25/20)

import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

# Plot Training and Validation
# Employed (11-7-19)
# Debugged (11-28-20)
# Created By (Sanjay Aneja, MD)
# Debugged By (Guneet Janda)
# Plots training accuracy and validation
# Plot Training and Validation
# Employed (11-7-19)
# Debugged (11-25-20)
# Created By (Sanjay Aneja, MD)
# Plots training accuracy and validation
def plot_tv(
        fit_model,
        path,
        model_title,
        filename=None
):
    """
    Description: This function plots training and validation curves for your DL model
    Param:
        fit_model = compiled model in keras
        path = folder you would like to save pictures
        model_title = keyword (string) to define model
        metrics = metrics list from keras model
        filename= string for date/time
    Return:
        Training and validation curves (loss and accuracy) for your DL model
        Figure will be saved with title [Timestamp_keyword]
    """
    # Plot the results
    if filename == None: filename = datetime.datetime.now().strftime("%m-%d-%Y_%H%M")
    for i in fit_model.history.keys():
        train_acc = fit_model.history[i]
        try: val_acc = fit_model.history['val_' + (i)]
        except: continue
        fig1, (fig_acc) = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        xa = np.arange(len(val_acc))
        fig_acc.plot(xa, train_acc)
        fig_acc.plot(xa, val_acc)
        fig_acc.set(xlabel='Epochs')
        fig_acc.set(ylabel=i)
        fig_acc.set(title='Training ' + i + ' vs Validation ' + i)
        fig1.legend(['Train', 'Validation'], loc=1, borderaxespad=1)
        # fig.grid('True')
        fig1.savefig((os.path.join(path, (filename + '_' + i + '_' + model_title + '.png'))))
        print(i + ' Figure Saved')


# CSV for Hyperparameters + Training
# Employed (4-7-18)
# Debugged (8-3-20)
# Created By (Sanjay Aneja, MD)
def csv_hyp(
        label_list,
        train_freq,
        val_freq,
        hyperparameter_list,
        hyperparameter_val,
        model_title,
        path,
        metrics
):
    """
       Description: This function allows CSV logging for deep learning model
       Param:
        label_list= list of class labels
        train_freq= list of frequencies of class labels in training data
        val_freq= list of frequencies of class labels in validation data
        hyperparameter_list= list of hyperparameters wished to be logged
        hyperparameter_val= list of hyperparamters values wished to be logged
        model_title= model keyword
        path= location where csv will be stored
        metrics=metrics list from keras model
       Return:
           csv_file = path to csv file
           To log epochs please insert into CSV logger callbacks using following command
           'CSVLogger(((csv_file)), separator=',', append=True'
           CSV will be saved with title [Epoch_log_keyword_timestamp].csv
       """
    timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H%M")
    csv_file= os.path.join(path, ('Epoch_log_' + str(model_title) + timestamp + '.csv'))
    c=(open(csv_file, 'w'))
    writer=csv.writer(c, delimiter=',', quotechar=' ')
    writer.writerow('Epoch')
    writer.writerow([hyperparameter_list])
    writer.writerow([hyperparameter_val])
    writer.writerow([' '])
    writer.writerow(['Train Freq', label_list])
    writer.writerow([' ', train_freq])
    writer.writerow(['Val Freq', label_list])
    writer.writerow([' ', val_freq])
    writer.writerow([' '])
    met_list=[]
    for i in metrics:
        x=i.name
        met_list.append(x)
    met_list.sort()
    met_list.append("Loss")
    val_met=[s +"_Val" for s in met_list]
    metric_list=met_list+val_met
    metric_list.insert(0,"Epoch")
    writer.writerow([metric_list])
    c.close()
    return csv_file

# </editor-fold>