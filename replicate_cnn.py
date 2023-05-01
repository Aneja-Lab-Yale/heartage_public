# Heart Age Model
# 3D CNN + Feature Extraction
# Aneja Lab | Yale School of Medicine
# Crystal Cheung
# Created (01/27/23)
# Updated (05/01/23)

#Import
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from volumentations import *
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

# local paths
# project_root = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/'
#image_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Whole_CT/'
#mask_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Heart_segmentations/'

# server paths
project_root = '/home/crystal_cheung/'
detail = 'apr16_mae_waug_10yr'

# defines callbacks for keras model
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

# defines learning rate scheduler
def scheduler(epoch):

    initial_lr = 0.001
    # Set the number of epochs after which the learning rate should drop
    drop_every = 5

    # Set the factor by which the learning rate should drop
    drop_factor = 0.8

    # Calculate the new learning rate based on the current epoch
    lr = initial_lr * drop_factor ** (epoch // drop_every)

    return lr

#Folders
fig_accuracy = project_root + 'results/accuracy_graph_' + detail +'.png'  # change to local folder
fig_loss = project_root + 'results/loss_graph_' + detail +'.png'  # change to local folder
fig_mae = project_root + 'results/mae_graph_' + detail +'.png'
fig_mse = project_root + 'results/mse_graph_' + detail +'.png'  # change to local folder
fig_confusion = project_root + 'results/confusion_matrix_' + detail +'.png'  # change to local folder
fig_prediction = project_root + 'results/prediction_graph_' + detail +'.png'
model_save_path = project_root + 'results/saved-model_' + detail +'.hdf5'  # change to local folder
csv_log_file = project_root + 'results/model_log_' + detail +'.csv' # change to local folder

#Hyperparameters
batch_size = 2
# start small on batch (2-3)
# size of batch is 10 samples before updating parameters
epochs = 80
# number of times training set is run for algorithm to learn
patience = 15
# how many epochs that it doesn't improve and then stops
min_lr = .0001
# minimum learning rate
callbacks_model = callbacks_model(model_save_path, csv_log_file, patience, min_lr)
# runs cb from above using previous parameters using callback model from Keras
optimizer = tf.keras.optimizers.Adam(use_ema=True)
# sets optimization for loss

#Regression
#loss= tf.keras.losses.CategoricalCrossentropy(name='loss')
loss = tf.keras.losses.MeanAbsoluteError(name='loss')
# mean absolute error (regression)
# uses tf.keras... function to be the loss
met = [tf.keras.metrics.RootMeanSquaredError(name='rmse'),tf.keras.metrics.MeanAbsoluteError(name='mae'),tf.keras.metrics.MeanSquaredError(name='mse')]
# met = metrics

# Constants
final_img_length = 60
final_img_slice = 47
input_shape = (final_img_length,final_img_length,final_img_slice,1) # need to fill this in (x, y, z, channel)
age_labels = list(np.load(project_root + 'data/ages.npy')) # patient ages
images = list(np.load(project_root + 'data/final_images.npy')) # patient images
patient_IDs = list(np.load(project_root + 'data/corrected_NLST.npy')) # patient IDs
age_class = list(np.load(project_root + 'data/age_class.npy')) # patient age classes

for j,patient in enumerate(images):
    # Zero center
    mean_intensity = np.mean(images[j])
    zeroed = images[j] - mean_intensity

    # Normalize
    std_intensity = np.std(zeroed)
    images[j] = zeroed / std_intensity

# split data into training and validation/test sets
indices = np.arange(len(images))
x_train,x_valtest,y_train_label,y_valtest_label,idx1,idx2 = train_test_split(images,age_labels,indices, test_size = 0.3, stratify =age_class, random_state = 42)

# pulls age and ID of patients in test and train set
#test_ID = []
#val_ID = []
#train_age_new = []
#test_age_new =[]
#for id in range(len(idx1)):
    #train_age_new.append(y_train_label[id])
#for id in range(len(idx2)):
    #test_age_new.append(y_valtest_label[id])

#train_ID = []
#valtest_ID = []
#for patient in range(len(idx2)):
    #valtest_ID.append(patient_IDs[idx2[patient]])
#for patient in range(len(idx1)):
    #train_ID.append(patient_IDs[idx1[patient]])

#np.savetxt(project_root + "/results/valtest_ID.csv", valtest_ID, delimiter=",",fmt='%s')
#train_age_new = np.asarray(train_age_new)
#test_age_new = np.asarray(test_age_new)
y_expected = np.asarray(y_valtest_label)
x_test_plug = np.asarray(x_valtest)

#np.save(project_root + '/results/test_ID_' + detail + '.npy', test_ID)
#np.save(project_root + '/results/train_ID_' + detail + '.npy', train_ID)
#np.save(project_root + '/results/val_ID_' + detail + '.npy', val_ID)
#np.save(project_root + '/results/valtest_ID_' + detail + '.npy', valtest_ID)
#np.savetxt(project_root + "/results/valtest_age.csv", y_valtest_label, delimiter=",",fmt='%i')
#np.savetxt(project_root + "/results/train_age_new.csv", train_age_new, delimiter=",",fmt='%i')
#np.savetxt(project_root + "/results/test_age_new.csv", test_age_new, delimiter=",",fmt='%i')

#Data Augmentation
x_augmented = x_train
y_augmented = y_train_label
x_val_augmented = x_valtest
y_val_augmented = y_valtest_label

def vol_flip():
    return Compose([Flip(p=1)],p=1) # flips image across vertical
def vol_rotate():
    return Compose([Rotate(x_limit=(-40, 40), y_limit=(0, 0), z_limit=(0, 0), p=1)],p=1) # rotates image
def vol_blur():
    return Compose([GlassBlur(sigma=0.2,max_delta=2,p=1)],p=1) # blurs image
def vol_noise():
    return Compose([GaussianNoise(p=1)],p=1) # introduces Gaussian noise to image
def vol_bright():
    return Compose([RandomBrightnessContrast(p=1)],p=1) # changes brightness/contrast of image

# applying augmentations
for i in range(len(x_train)):

    flip = vol_flip()
    rotate = vol_rotate()
    blur = vol_blur()
    gauss = vol_noise()
    bright = vol_bright()

    # setting training as data set
    data = {'image':x_train[i]}

    # applying each augment to data
    aug_flip = flip(**data)
    aug_rotate = rotate(**data)
    aug_blur = blur(**data)
    aug_gauss = gauss(**data)
    aug_bright = bright(**data)

    image_flip = aug_flip['image']
    image_rotate = aug_rotate['image']
    image_rotate = np.reshape(image_rotate,(final_img_length,final_img_length,final_img_slice))
    image_blur = aug_blur['image']
    image_gauss = aug_gauss['image']
    image_bright = aug_bright['image']

    # adding augmented images to a new list with original training images
    x_augmented.append(image_flip)
    x_augmented.append(image_rotate)
    x_augmented.append(image_blur)
    x_augmented.append(image_gauss)
    x_augmented.append(image_bright)

    # adding ages to a new list with original training ages
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])

x_augmented = np.asarray(x_augmented)
y_augmented = np.asarray(y_augmented)

# performing the same augmentation as above for test set
for i in range(len(x_valtest)):

    flip = vol_flip()
    rotate = vol_rotate()
    blur = vol_blur()
    gauss = vol_noise()
    bright = vol_bright()

    data = {'image':x_valtest[i]}

    aug_flip = flip(**data)
    aug_rotate = rotate(**data)
    aug_blur = blur(**data)
    aug_gauss = gauss(**data)
    aug_bright = bright(**data)

    image_flip = aug_flip['image']
    image_rotate = aug_rotate['image']
    image_rotate = np.reshape(image_rotate,(final_img_length,final_img_length,final_img_slice))
    image_blur = aug_blur['image']
    image_gauss = aug_gauss['image']
    image_bright = aug_bright['image']

    x_val_augmented.append(image_flip)
    x_val_augmented.append(image_rotate)
    x_val_augmented.append(image_blur)
    x_val_augmented.append(image_gauss)
    x_val_augmented.append(image_bright)

    y_val_augmented.append(y_valtest_label[i])
    y_val_augmented.append(y_valtest_label[i])
    y_val_augmented.append(y_valtest_label[i])
    y_val_augmented.append(y_valtest_label[i])
    y_val_augmented.append(y_valtest_label[i])

x_val_augmented = np.asarray(x_val_augmented)
y_val_augmented = np.asarray(y_val_augmented)

# Cole paper CNN for reference
# repeated 5 blocks of --
# 3x3x3 layer, stride 1
# relu
# 3x3x3 layer,stride 1
# 3d batch normalization layer
# relu
# 2x2x2 max pooling layer, stride 2

# CNN Block 1
input = tf.keras.Input(shape = input_shape, batch_size = batch_size)
x = tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1),name="conv1",kernel_regularizer='l1_l2')(input)
# find filter integer
x = tf.keras.layers.BatchNormalization(name='bn1')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),name="maxpool1")(x)

#CNN output
x1 = tf.keras.layers.Flatten(name='output')(x)
x2 = tf.keras.layers.Dropout(0.5,name='dropoutdense')(x1)
# fraction of the input units to drop
x2 = tf.keras.layers.Dense(16, activation = 'relu', kernel_regularizer="l1_l2")(x2)
x2 = tf.keras.layers.Dense(8, activation = 'relu', kernel_regularizer="l1_l2")(x2)
# fraction of the input units to drop
output = tf.keras.layers.Dense(1, activation="linear",kernel_regularizer="l1_l2")(x2)
# positive integer, dimensionality of the output space

#Model
model = tf.keras.models.Model(inputs=[input], outputs=[output])
model.summary()

model.compile(loss=loss,
              optimizer=optimizer,
              metrics = met
              )

# training model
history = model.fit(x_augmented, y_augmented,
          validation_data=(x_val_augmented, y_val_augmented),
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks_model
          )

# predicting on test set using model
y_predicted = model.predict(x_test_plug, batch_size=batch_size)

# putting expected ages into bins
age_class_ex = []
for i in range(len(y_expected)):
    if y_expected[i] <= 60:
        age_bin = 1
    elif 60 < y_expected[i] <= 70:
        age_bin = 2
    elif 70 < y_expected[i]:
        age_bin = 3

    age_class_ex.append(age_bin)

# putting predicted ages into bins
age_class_pred = []
for i in range(len(y_predicted)):
    if y_predicted[i] <= 60:
        age_bin = 1
    elif 60 < y_predicted[i] <= 70:
        age_bin = 2
    elif 70 < y_predicted[i]:
        age_bin = 3

    age_class_pred.append(age_bin)

# saving list of predicted ages
np.savetxt(project_root + 'results/age_predictions_reg_' + detail +'.csv', y_predicted, delimiter=",",fmt='%i')

# saving list of predicted age bins and expected age bins
np.savetxt(project_root + 'results/age_expected_bin_' + detail +'.csv', age_class_ex, delimiter=",",fmt='%i')
np.savetxt(project_root + 'results/age_predictions_bin_' + detail +'.csv', age_class_pred, delimiter=",",fmt='%i')

# saving model plot of loss
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('MAE Loss '+ detail)
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.legend()
#plt.show()
plt.savefig(fig_loss)

# saving model MAE
plt.figure(figsize=(10,8))
plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'], label='val')
plt.title('MAE '+ detail)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
#plt.show()
plt.savefig(fig_mae)

#compare predicted and true age by bins
plt.figure(figsize=(10,8))
plt.scatter(age_class_ex,age_class_pred)
plt.plot([min(age_class_ex), max(age_class_ex)], [min(age_class_ex), max(age_class_ex)], 'k--', lw=4)
plt.title('comparison '+ detail)
plt.ylabel('predicted age')
plt.xlabel('true age')
plt.savefig(fig_prediction)