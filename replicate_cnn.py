# Heart Age Model
# 3D CNN + Feature Extraction
# Aneja Lab | Yale School of Medicine
# Crystal Cheung
# Created (01/27/23)
# Updated (02/23/23)
#Import
import tensorflow as tf
# importing tensorflow software lib as variable tf (for machine learning)
# import numpy as np
# importing numpy software lib as variable np (for math fx)
#from Useful_Functions.Misc_Functions import listdir_nohidden
#from Useful_Functions.Misc_Functions import listdir_dicom
#from Keras_Miscellaneous.Keras_Callbacks import callbacks_model as cb
#import Keras_Callbacks
#from Keras_Callbacks import callbacks_model as cb
import matplotlib.pyplot as plt
#from pydotplus import graphviz
#from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from volumentations import *
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pandas as pd
import numpy as np


#project_root = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/'
#image_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Whole_CT/'
#mask_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Heart_segmentations/'

project_root = '/home/crystal_cheung/'
detail = 'apr6_mae_waug'
def callbacks_model(model_save_path,
                    csv_log_file,
                    patience,
                    min_lr,
):
    """
    Description:
        This function provides callbacks for keras model
    Param:
        sample=
        model_save_path = directory to save model and h5
        csv_log_file = file path to save CSV of training
        patience
        minimum_lrate = minimum learning rate
        lrate_schedule
    Return:
        callbacks variable to be used in model.fit statement
    """

    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch', options=None),
        ## Save model after every epoch, save_best_only = only saves if improved, save_freq = number of batches per save (epoch = after each epoch)
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto',
            baseline=None, restore_best_weights=False),
        ## Stop training when quality hasn't improved, patients = number of epochs without improvement
        tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0),
        ## Reduce learning rate per function, schedule = function(epoch index) that defines
        #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience, verbose=0, mode='auto', min_delta=0.0001, cooldown=0,
                          #min_lr=minimum_lrate),
        ## Reduces LR when metric hasn't improved (usually by factor 2-10), patience is number of epochs without improvement
        tf.keras.callbacks.CSVLogger(csv_log_file, separator=',', append=True),
        ## Streams epoch results to CSV
        #tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False,
        #             embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
        #             update_freq='epoch'),
        ## Tensorboard basic visualizations (dynamic training graphs)
        ]
    return callbacks

def scheduler(epoch):

    initial_lr = 0.001
    # Set the number of epochs after which the learning rate should drop
    drop_every = 5

    # Set the factor by which the learning rate should drop
    drop_factor = 0.8

    # Calculate the new learning rate based on the current epoch
    lr = initial_lr * drop_factor ** (epoch // drop_every)

    return lr

#3D CNN

# input sizes are (z x h x w)
# raw data: 182 x 218 x 182
# registered data: 121 x 145 x 121

#Folders
fig_accuracy = project_root + 'results/accuracy_graph_' + detail +'.png'  # change to local folder
fig_loss = project_root + 'results/loss_graph_' + detail +'.png'  # change to local folder
fig_mae = project_root + 'results/mae_graph_' + detail +'.png'
fig_mse = project_root + 'results/mse_graph_' + detail +'.png'  # change to local folder
#fig_loss_accuracy = project_root + 'results/loss_acc_graph_mar31.png'  # change to local folder
fig_prediction = project_root + 'results/prediction_graph_' + detail +'.png'
#fig_AUC = project_root + 'results/AUC_graph_mar27.png'  # change to local folder
model_save_path = project_root + 'results/saved-model_' + detail +'.hdf5'  # change to local folder
csv_log_file = project_root + 'results/model_log_' + detail +'.csv' # change to local folder

#Hyperparameters
batch_size = 2
# start small on batch (2-3)
# size of batch is 10 samples before updating parameters
epochs = 200
# number of times training set is run for algorithm to learn
patience = 15
# how many epochs that it doesn't improve and then stops
min_lr = .0001
# minimum learning rate
callbacks_model = callbacks_model(model_save_path, csv_log_file, patience, min_lr)
# runs cb from above using previous parameters using callback model from Keras
optimizer = tf.keras.optimizers.Adam(use_ema=True)
# sets optimization for loss

#regression
#loss= tf.keras.losses.CategoricalCrossentropy(name='loss')
loss = tf.keras.losses.MeanAbsoluteError(name='loss')
# mean squared error (regression)
# uses tf.keras... function to be the loss
#met = [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
met = [tf.keras.metrics.RootMeanSquaredError(name='rmse'),tf.keras.metrics.MeanAbsoluteError(name='mae'),tf.keras.metrics.MeanSquaredError(name='mse')]
# met = metrics, set as matrix of accuracy, AUC and false negatives from the tf.keras functions
# mean squared error

# Constants
final_img_length = 60
final_img_slice = 47
input_shape = (final_img_length,final_img_length,final_img_slice,1) # need to fill this in (x, y, z, channel)
#num_class = 4  # need to fill this in (outcomes:age ranges)
#age_labels = list(np.load(project_root + 'data/age_class.npy'))
age_labels = list(np.load(project_root + 'data/ages.npy'))
images = list(np.load(project_root + 'data/final_images.npy'))
patient_IDs = list(np.load(project_root + 'data/corrected_NLST.npy'))
age_class = list(np.load(project_root + 'data/age_class.npy'))

for j,patient in enumerate(images):
    # Zero center
    mean_intensity = np.mean(images[j])
    zeroed = images[j] - mean_intensity

    # Normalize
    std_intensity = np.std(zeroed)
    images[j] = zeroed / std_intensity

indices = np.arange(len(images))
x_train,x_valtest,y_train_label,y_valtest_label,idx1,idx2 = train_test_split(images,age_labels,indices, test_size = 0.3, stratify =age_class, random_state = 42)

#x_train_pre,x_val,y_train_label_pre,y_val_label,idx1,idx2 = train_test_split(images,age_labels,indices, test_size = 0.2, random_state = 42)

#indices2 = np.arange(len(x_train_pre))
#x_train,x_test,y_train_label,y_test_label,idx3,idx4 = train_test_split(x_train_pre,y_train_label_pre,indices2,test_size = 0.25, random_state = 42)
#x_train = np.asarray(x_train)
#x_test = np.asarray(x_test)
#y_train_label = np.asarray(y_train_label)
#y_test_label = np.asarray(y_test_label)


#test_ID = []
#val_ID = []
#train_age_new = []
#for id in range(len(idx4)):
    #image_index = idx1[idx4[id]]
    #test_ID.append(patient_IDs[image_index])
#for id in range(len(idx3)):
    #image_index = idx1[idx3[id]]
    #train_ID.append(patient_IDs[image_index])
    #train_age_new.append(y_train_label[id])
#for id in range(len(idx2)):
    #val_ID.append(patient_IDs[idx2[id]])

#train_ID = []
#valtest_ID = []
#for patient in range(len(idx2)):
    #valtest_ID.append(patient_IDs[idx2[patient]])
#for patient in range(len(idx1)):
    #train_ID.append(patient_IDs[idx1[patient]])

y_expected = np.asarray(y_valtest_label)
x_test_plug = np.asarray(x_valtest)

#np.save(project_root + '/results/test_ID_' + detail + '.npy', test_ID)
#np.save(project_root + '/results/train_ID_' + detail + '.npy', train_ID)
#np.save(project_root + '/results/val_ID_' + detail + '.npy', val_ID)
#np.save(project_root + '/results/valtest_ID_' + detail + '.npy', valtest_ID)
#np.savetxt(project_root + "/results/valtest_age.csv", valtest_age, delimiter=",",fmt='%i')
#np.savetxt(project_root + "/results/train_age_new.csv", train_age_new, delimiter=",",fmt='%i')

#Data Augmentation
x_augmented = x_train
y_augmented = y_train_label
x_val_augmented = x_valtest
y_val_augmented = y_valtest_label
def vol_flip():
    return Compose([Flip(p=1)],p=1)
def vol_rotate():
    return Compose([Rotate(x_limit=(-40, 40), y_limit=(0, 0), z_limit=(0, 0), p=1)],p=1)
def vol_blur():
    return Compose([GlassBlur(sigma=0.2,max_delta=2,p=1)],p=1)
def vol_noise():
    return Compose([GaussianNoise(p=1)],p=1)
def vol_bright():
    return Compose([RandomBrightnessContrast(p=1)],p=1)
def combo1():
    return Compose([Rotate(x_limit=(-40, 40), y_limit=(0, 0), z_limit=(0, 0), p=0.75),
                    GaussianNoise(p=0.25),
                    GlassBlur(sigma=0.2,max_delta=2,p=0.25),
                    Flip(p=0.2)],p=1)
def combo2():
    return Compose([Rotate(x_limit=(-40, 40), y_limit=(0, 0), z_limit=(0, 0), p=0.75),
                    GaussianNoise(p=0.25),
                    GlassBlur(sigma=0.2,max_delta=2,p=0.25),
                    Flip(p=0.2)],p=1)
def combo3():
    return Compose([Rotate(x_limit=(-40, 40), y_limit=(0, 0), z_limit=(0, 0), p=0.75),
                    GaussianNoise(p=0.25),
                    GlassBlur(sigma=0.2,max_delta=2,p=0.25),
                    Flip(p=0.2)],p=1)

#aug = vol_gamma()
#img = x_train[0]
#data = {'image':img}
#aug_data = aug(**data)
#img = aug_data['image']

for i in range(len(x_train)):

    flip = vol_flip()
    rotate = vol_rotate()
    blur = vol_blur()
    gauss = vol_noise()
    bright = vol_bright()
    #combo_1 = combo1()
    #combo_2 = combo2()
    #combo_3 = combo3()

    data = {'image':x_train[i]}

    aug_flip = flip(**data)
    aug_rotate = rotate(**data)
    aug_blur = blur(**data)
    aug_gauss = gauss(**data)
    aug_bright = bright(**data)
    #aug_combo1 = combo_1(**data)
    #aug_combo2 = combo_2(**data)
    #aug_combo3 = combo_3(**data)

    image_flip = aug_flip['image']
    image_rotate = aug_rotate['image']
    image_rotate = np.reshape(image_rotate,(final_img_length,final_img_length,final_img_slice))
    image_blur = aug_blur['image']
    image_gauss = aug_gauss['image']
    image_bright = aug_bright['image']
    #image_combo1 = aug_combo1['image']
    #image_combo1 = np.reshape(image_combo1,(final_img_length,final_img_length,final_img_slice))
    #image_combo2 = aug_combo1['image']
    #image_combo2 = np.reshape(image_combo2, (final_img_length, final_img_length, final_img_slice))
    #image_combo3 = aug_combo1['image']
    #image_combo3 = np.reshape(image_combo3, (final_img_length, final_img_length, final_img_slice))

    x_augmented.append(image_flip)
    x_augmented.append(image_rotate)
    x_augmented.append(image_blur)
    x_augmented.append(image_gauss)
    x_augmented.append(image_bright)
    #x_augmented.append(image_combo1)
    #x_augmented.append(image_combo2)
    #x_augmented.append(image_combo3)

    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    #y_augmented.append(y_train_label[i])
    #y_augmented.append(y_train_label[i])
    #y_augmented.append(y_train_label[i])

x_augmented = np.asarray(x_augmented)
y_augmented = np.asarray(y_augmented)
#y_train = tf.keras.utils.to_categorical(y_augmented,num_classes=num_class)

for i in range(len(x_valtest)):

    flip = vol_flip()
    rotate = vol_rotate()
    blur = vol_blur()
    gauss = vol_noise()
    bright = vol_bright()
    #combo_1 = combo1()
    #combo_2 = combo2()
    #combo_3 = combo3()

    data = {'image':x_valtest[i]}

    aug_flip = flip(**data)
    aug_rotate = rotate(**data)
    aug_blur = blur(**data)
    aug_gauss = gauss(**data)
    aug_bright = bright(**data)
    #aug_combo1 = combo_1(**data)
    #aug_combo2 = combo_2(**data)
    #aug_combo3 = combo_3(**data)

    image_flip = aug_flip['image']
    image_rotate = aug_rotate['image']
    image_rotate = np.reshape(image_rotate,(final_img_length,final_img_length,final_img_slice))
    image_blur = aug_blur['image']
    image_gauss = aug_gauss['image']
    image_bright = aug_bright['image']
    #image_combo1 = aug_combo1['image']
    #image_combo1 = np.reshape(image_combo1, (final_img_length, final_img_length, final_img_slice))
    #image_combo2 = aug_combo1['image']
    #image_combo2 = np.reshape(image_combo2, (final_img_length, final_img_length, final_img_slice))
    #image_combo3 = aug_combo1['image']
    #image_combo3 = np.reshape(image_combo3, (final_img_length, final_img_length, final_img_slice))

    x_val_augmented.append(image_flip)
    x_val_augmented.append(image_rotate)
    x_val_augmented.append(image_blur)
    x_val_augmented.append(image_gauss)
    x_val_augmented.append(image_bright)
    #x_val_augmented.append(image_combo1)
    #x_val_augmented.append(image_combo2)
    #x_val_augmented.append(image_combo3)

    y_val_augmented.append(y_valtest_label[i])
    y_val_augmented.append(y_valtest_label[i])
    y_val_augmented.append(y_valtest_label[i])
    y_val_augmented.append(y_valtest_label[i])
    y_val_augmented.append(y_valtest_label[i])
    #y_val_augmented.append(y_val_label[i])
    #y_val_augmented.append(y_val_label[i])
    #y_val_augmented.append(y_val_label[i])

x_val_augmented = np.asarray(x_val_augmented)
y_val_augmented = np.asarray(y_val_augmented)

#y_train = tf.keras.utils.to_categorical(y_augmented,num_classes=num_class)
#y_val = tf.keras.utils.to_categorical(y_val_augmented,num_classes=num_class)
#y_test = tf.keras.utils.to_categorical(y_test_label,num_classes=num_class)


#plt.imshow(x_augmented[74][:,:,60],cmap='gray')

# repeated 5 blocks of --
# 3x3x3 layer, stride 1
# relu
# 3x3x3 layer,stride 1
# 3d batch normalization layer
# relu
# 2x2x2 max pooling layer, stride 2

# CNN Block 1
input = tf.keras.Input(shape = input_shape, batch_size = batch_size)
x = tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1),name="conv1")(input)
# find filter integer
x = tf.keras.layers.BatchNormalization(name='bn1')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),name="maxpool1")(x)
#x = tf.keras.layers.Dropout(0.1,name='dropout1')(x)
#x = tf.keras.layers.Conv3D(16, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1),name="conv2")(x)
# find filter integer
#x = tf.keras.layers.BatchNormalization(name='bn2')(x)
#x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),name="maxpool2")(x)
#x = tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1),name="conv3")(x)
# find filter integer
#x = tf.keras.layers.BatchNormalization(name='bn3')(x)
#x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),name="maxpool3")(x)

#CNN Block 2-5
#for j in range(1): #change to 3 blocks
    #x = tf.keras.layers.Conv3D(32, kernel_size=(3,3,3), activation ='relu', strides=(1, 1, 1), name="conv"+str(3+2*j))(x)
    # find filter integer
    #x = tf.keras.layers.BatchNormalization(name='bn' + str(3+2*j))(x)
    #x = tf.keras.layers.Conv3D(32, kernel_size=(3,3,3), activation ='relu',strides=(1, 1, 1), name="conv"+str(4+2*j))(x)
    # find filter integer
    #x = tf.keras.layers.BatchNormalization(name='bn'+str(4+2*j))(x)
    #x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2), strides=(2, 2, 2),name='maxpool'+str(1+2+j))(x)

#CNN output
x1 = tf.keras.layers.Flatten(name='output')(x)
x2 = tf.keras.layers.Dropout(0.5,name='dropoutdense')(x1)
# fraction of the input units to drop
x2 = tf.keras.layers.Dense(16, activation = 'relu', kernel_regularizer="l2")(x2)
#x2 = tf.keras.layers.Dropout(0.1,name='dropoutdense2')(x2)
x2 = tf.keras.layers.Dense(8, activation = 'relu', kernel_regularizer="l2")(x2)
# fraction of the input units to drop
output = tf.keras.layers.Dense(1, activation="linear",kernel_regularizer="l2")(x2)
#output = tf.keras.layers.Dense(1, activation="linear",kernel_regularizer="l2")(x2)
#positive integer, dimensionality of the output space

#Model
model = tf.keras.models.Model(inputs=[input], outputs=[output])
model.summary()
#pic=tf.keras.utils.plot_model(model, to_file=fig_print, show_shapes=True)
#pic = plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB', expand_nested=False, dpi=96)

model.compile(loss=loss,
              optimizer=optimizer,
              metrics = met
              )

history = model.fit(x_augmented, y_augmented,
          validation_data=(x_val_augmented, y_val_augmented),
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks_model
          )


y_predicted = model.predict(x_test_plug, batch_size=batch_size)

results = model.evaluate(x_test_plug,y_expected)
model_metrics = model.metrics_names
r2 = r2_score(y_expected, y_predicted)
model_metrics.append('r2')
results.append(r2)

corr, _ = pearsonr(y_expected, y_predicted)
model_metrics.append('r')
results.append(corr)

dict = {'Metric': model_metrics, 'Value': results}
df = pd.DataFrame(dict)
df.to_csv(project_root + '/results/test_evaluation_' + detail + '.csv',index=False)

#np.savetxt(project_root +'results/scores.csv',evaluation)
#a=np.array(y_predicted)
#y_predicted_label = np.where(a)[2]
np.savetxt(project_root + 'results/age_predictions_reg_' + detail +'.csv', y_predicted, delimiter=",",fmt='%i')

# summarize history for accuracy
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['Train', 'Validation'], loc='upper left')
#plt.show()
#plt.savefig(fig_accuracy)

# summarize history for loss + accuracy
#plt.figure(figsize=(10,8))
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['Train', 'Validation'], loc='upper left')
#plt.show()
#plt.savefig(fig_loss)

plt.figure(figsize=(10,8))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('MAE Loss '+ detail)
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.ylim([0, 50])
plt.legend()
#plt.show()
plt.savefig(fig_loss)

plt.figure(figsize=(10,8))
plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'], label='val')
plt.title('MAE '+ detail)
plt.xlabel('Epoch')
plt.ylabel('MAE')
#plt.ylim([0, 50])
plt.legend()
#plt.show()
plt.savefig(fig_mae)

#corr_str = round(corr[0], 2)
#r2_str = round(r2, 2)

#compare predicted and true age
plt.figure(figsize=(10,8))
plt.scatter(y_expected,y_predicted)
plt.plot([min(y_expected), max(y_expected)], [min(y_expected), max(y_expected)], 'k--', lw=4)
#plt.annotate('Pearson correlation coefficient = ' + corr_str,xy=(0.1,0.9), xycoords='axes fraction')
#plt.annotate(f'R-squared = ' + r2_str, xy=(0.1,0.8), xycoords='axes fraction')
plt.title('comparison '+ detail)
plt.ylabel('predicted age')
plt.xlabel('true age')
plt.savefig(fig_prediction)

#r = np.corrcoef(x_test, y_expected)[0, 1]

#loss, mae = model.evaluate(x_test, y_expected)
#evaluation = np.array([loss,mae,r2, r])

#np.savetxt(project_root+'/results/evaluation.csv',evaluation)
#Feature extraction
#layer_name='x6'
#layer_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#feature_extraction=layer_extractor.predict(x_test)
#print(feature_extraction.shape)