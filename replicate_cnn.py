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
# from sklearn.metrics import r2_score


#project_root = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/'
#image_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Whole_CT/'
#mask_path = '/Users/Crystal/Desktop/College/PMAE/Thesis/Code/Heart_segmentations/'

project_root = '/home/crystal_cheung/'
def r_squared(y_expected, y_predicted):
    """Custom metric to compute R² from mean squared error and TSS"""
    mse = tf.reduce_mean(tf.square(y_expected - y_predicted))
    tss = tf.reduce_mean(tf.square(y_expected - tf.reduce_mean(y_expected)))
    return 1 - mse / tss

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

def scheduler(epoch, lr):
  if epoch < 15:
    return lr
  else:
    return lr * tf.math.exp(-0.01)

#3D CNN

# input sizes are (z x h x w)
# raw data: 182 x 218 x 182
# registered data: 121 x 145 x 121

#Folders
fig_accuracy = project_root + 'results/accuracy_graph_apr3_2.png'  # change to local folder
fig_loss = project_root + 'results/loss_graph_apr3_2.png'  # change to local folder
#fig_loss_accuracy = project_root + 'results/loss_acc_graph_mar31.png'  # change to local folder
fig_prediction = project_root + 'results/prediction_graph_apr3_2.png'
#fig_AUC = project_root + 'results/AUC_graph_mar27.png'  # change to local folder
model_save_path = project_root + 'results/saved-model_apr3_2.hdf5'  # change to local folder
csv_log_file = project_root + 'results/model_log_apr3_2.csv' # change to local folder

#Hyperparameters
batch_size = 2
# start small on batch (2-3)
# size of batch is 10 samples before updating parameters
epochs = 50
# number of times training set is run for algorithm to learn
patience = 25
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
met = [tf.keras.metrics.MeanSquaredError(name='mse'),tf.keras.metrics.MeanAbsoluteError(name='mae')]
# met = metrics, set as matrix of accuracy, AUC and false negatives from the tf.keras functions
# mean squared error

# Constants
final_img_length = 60
final_img_slice = 46
input_shape = (final_img_length,final_img_length,final_img_slice,1) # need to fill this in (x, y, z, channel)
num_class = 4  # need to fill this in (outcomes:age ranges)
#age_labels = list(np.load(project_root + 'data/age_class.npy'))
age_labels = list(np.load(project_root + 'data/ages.npy'))
images = list(np.load(project_root + 'data/final_images.npy'))
patient_IDs = list(np.load(project_root + 'data/corrected_NLST.npy'))

for j in range(len(images)):
    mean = images[j].mean()
    std = images[j].std()

    # Normalize the pixel values to have zero mean and unit variance
    images[j] = (images[j] - mean) / std

indices = np.arange(len(images))
x_train_pre,x_val,y_train_label_pre,y_val_label,idx1,idx2 = train_test_split(images,age_labels,indices, test_size = 0.2, random_state = 42)

indices2 = np.arange(len(x_train_pre))
x_train,x_test,y_train_label,y_test_label,idx3,idx4 = train_test_split(x_train_pre,y_train_label_pre,indices2,test_size = 0.25, random_state = 42)
#x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
#y_train_label = np.asarray(y_train_label)
y_test_label = np.asarray(y_test_label)
y_expected = y_test_label

test_ID = []
train_ID = []
val_ID = []
for id in range(len(idx4)):
    image_index = idx1[idx4[id]]
    test_ID.append(patient_IDs[image_index])
for id in range(len(idx3)):
    image_index = idx1[idx3[id]]
    train_ID.append(patient_IDs[image_index])
for id in range(len(idx2)):
    val_ID.append(patient_IDs[idx2[id]])

np.save(project_root + '/results/test_ID_apr3_2.npy', test_ID)
np.save(project_root + '/results/train_ID_apr3_2.npy', train_ID)
np.save(project_root + '/results/val_ID_apr3_2.npy', val_ID)

#Data Augmentation
x_augmented = x_train
y_augmented = y_train_label
x_val_augmented = x_val
y_val_augmented = y_val_label
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
    combo_1 = combo1()
    combo_2 = combo2()
    combo_3 = combo3()

    data = {'image':x_train[i]}

    aug_flip = flip(**data)
    aug_rotate = rotate(**data)
    aug_blur = blur(**data)
    aug_gauss = gauss(**data)
    aug_bright = bright(**data)
    aug_combo1 = combo_1(**data)
    aug_combo2 = combo_2(**data)
    aug_combo3 = combo_3(**data)

    image_flip = aug_flip['image']
    image_rotate = aug_rotate['image']
    image_rotate = np.reshape(image_rotate,(final_img_length,final_img_length,final_img_slice))
    image_blur = aug_blur['image']
    image_gauss = aug_gauss['image']
    image_bright = aug_bright['image']
    image_combo1 = aug_combo1['image']
    image_combo1 = np.reshape(image_combo1,(final_img_length,final_img_length,final_img_slice))
    image_combo2 = aug_combo1['image']
    image_combo2 = np.reshape(image_combo2, (final_img_length, final_img_length, final_img_slice))
    image_combo3 = aug_combo1['image']
    image_combo3 = np.reshape(image_combo3, (final_img_length, final_img_length, final_img_slice))

    x_augmented.append(image_flip)
    x_augmented.append(image_rotate)
    x_augmented.append(image_blur)
    x_augmented.append(image_gauss)
    x_augmented.append(image_bright)
    x_augmented.append(image_combo1)
    x_augmented.append(image_combo2)
    x_augmented.append(image_combo3)

    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])
    y_augmented.append(y_train_label[i])

x_augmented = np.asarray(x_augmented)
y_augmented = np.asarray(y_augmented)
#y_train = tf.keras.utils.to_categorical(y_augmented,num_classes=num_class)

x_val_augmented = x_val
y_val_augmented = y_val_label

for i in range(len(x_val)):

    flip = vol_flip()
    rotate = vol_rotate()
    blur = vol_blur()
    gauss = vol_noise()
    bright = vol_bright()
    combo_1 = combo1()
    combo_2 = combo2()
    combo_3 = combo3()

    data = {'image':x_val[i]}

    aug_flip = flip(**data)
    aug_rotate = rotate(**data)
    aug_blur = blur(**data)
    aug_gauss = gauss(**data)
    aug_bright = bright(**data)
    aug_combo1 = combo_1(**data)
    aug_combo2 = combo_2(**data)
    aug_combo3 = combo_3(**data)

    image_flip = aug_flip['image']
    image_rotate = aug_rotate['image']
    image_rotate = np.reshape(image_rotate,(final_img_length,final_img_length,final_img_slice))
    image_blur = aug_blur['image']
    image_gauss = aug_gauss['image']
    image_bright = aug_bright['image']
    image_combo1 = aug_combo1['image']
    image_combo1 = np.reshape(image_combo1, (final_img_length, final_img_length, final_img_slice))
    image_combo2 = aug_combo1['image']
    image_combo2 = np.reshape(image_combo2, (final_img_length, final_img_length, final_img_slice))
    image_combo3 = aug_combo1['image']
    image_combo3 = np.reshape(image_combo3, (final_img_length, final_img_length, final_img_slice))

    x_val_augmented.append(image_flip)
    x_val_augmented.append(image_rotate)
    x_val_augmented.append(image_blur)
    x_val_augmented.append(image_gauss)
    x_val_augmented.append(image_bright)
    x_val_augmented.append(image_combo1)
    x_val_augmented.append(image_combo2)
    x_val_augmented.append(image_combo3)

    y_val_augmented.append(y_val_label[i])
    y_val_augmented.append(y_val_label[i])
    y_val_augmented.append(y_val_label[i])
    y_val_augmented.append(y_val_label[i])
    y_val_augmented.append(y_val_label[i])
    y_val_augmented.append(y_train_label[i])
    y_val_augmented.append(y_train_label[i])
    y_val_augmented.append(y_train_label[i])

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
x = tf.keras.layers.Conv3D(8, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1),name="conv1")(input)
# find filter integer
x = tf.keras.layers.BatchNormalization(name='bn1')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),name="maxpool1")(x)
x = tf.keras.layers.Dropout(0.4,name='dropout1')(x)
x = tf.keras.layers.Conv3D(16, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1),name="conv2")(x)
# find filter integer
x = tf.keras.layers.BatchNormalization(name='bn2')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),name="maxpool2")(x)
x = tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1),name="conv3")(x)
# find filter integer
x = tf.keras.layers.BatchNormalization(name='bn3')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),name="maxpool3")(x)

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
x2 = tf.keras.layers.Dropout(0.2,name='dropoutdense')(x1)
# fraction of the input units to drop
output = tf.keras.layers.Dense(1, activation="linear",kernel_regularizer="l2")(x2)
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

history = model.fit(x_augmented[0:656], y_augmented[0:656], #only 300 samples for time
          validation_data=(x_val_augmented[0:224], y_val_augmented[0:224]),
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks_model
          )


y_predicted = model.predict(x_test, batch_size=batch_size)

#np.savetxt(project_root +'results/scores.csv',evaluation)

#r2 = r2_score(y_expected, y_predicted)
#a=np.array(y_predicted)
#y_predicted_label = np.where(a)[2]
np.savetxt(project_root + "results/age_predictions_reg_apr3_2.csv", y_predicted, delimiter=",",fmt='%i')

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
plt.title('MAE Loss')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.legend()
#plt.show()
plt.savefig(fig_loss)

#compare predicted and true age
plt.figure(figsize=(10,8))
plt.scatter(y_expected,y_predicted)
plt.plot([min(y_expected), max(y_expected)], [min(y_expected), max(y_expected)], 'k--', lw=4)
plt.title('r2')
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