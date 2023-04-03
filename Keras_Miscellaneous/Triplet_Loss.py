# Triplet Loss Network
# Aneja Lab | Yale College
# Justin Du
# Created (06/16/20), more documentation can be found - 

import numpy as np
import random
import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Dense, Dropout, Activation, Flatten, Masking
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, concatenate
from keras import regularizers

# I have marked most of the areas to pay attention to with a To-Do

##############################
# CONSTANTS - READ CAREFULLY #
##############################

# TODO Update these constants
# Path to the folder with all the patients
PATIENT_PATH =


# 0. Time the code

import time
start_time = time.time()

# 0b. Constants
# TODO Update these

# Ex. dt_triplet_callbacks.h5, dt_triplet_callbacks.json
TRIPLET_MODEL_SAVE =
TRIPLET_MODEL_ARCHITECTURE_SAVE =

# 0c. DL Constants
# TODO Update these

PATIENCE = 10
LEARNING_RATE = .001
MIN_LR = 1e-8
# Taken from Jeff SBRT - https://github.com/Aneja-Lab-Yale/SBRT/blob/master/SBRT_Final/Main/SBRT_Default_Constants_Jeff.py
DROPOUT_PROB = 0.5
L2_PENALTY = 1e-4
BATCH_SIZE = 3
EPOCHS = 20
# Why is this calculated the way it is? -> Convention
STEPS_PER_EPOCH = int(30 / BATCH_SIZE)
EMB_DIM = 1024

# 3. Create Triplets

# Load in the files to train
# Input image dimensions
INPUT_SHAPE = (224, 224, 3)
ZEROS_SHAPE = (BATCH_SIZE, 224, 224, 3)
EMB_DIM = 1024
def create_batch(batch_size):
    # Return a list of anchors, positives, and negatives
    anchors = np.zeros(ZEROS_SHAPE)
    positives = np.zeros(ZEROS_SHAPE)
    negatives = np.zeros(ZEROS_SHAPE)
    
    for i in range(0, batch_size):
        # Get index of a random anchor
        index = random.randint(0, len(PATIENTS_NP) - 1)
        # No target_slice_number_for_now
        # padded_image = pad_image(PATIENT_PATH, patients_np[index], TARGET_SLICE_NUMBER)
        image = get_image(PATIENT_PATH, PATIENTS_NP[index])
        
        # TO-DO -> Choose how you pick your anchor and positive images
        # anc =
        # pos = 
        # neg = 
        # Zero Normal
        anc = (anc - anc.mean()) / anc.std()
        pos = (pos - pos.mean()) / pos.std()
        neg = (neg - neg.mean()) / neg.std()

        # Update the arrays
        anchors[i] = anc
        positives[i] = pos
        negatives[i] = neg
    return [anchors, positives, negatives]

# 4. Embedding Model

# TO-DO -> Think about what architecture you want to use
# Decide on the embedding size, dropout rate, l2 penalty

resnet_base=ResNet50(weights='imagenet', include_top=False)
x= resnet_base.output
x= Dropout(DROPOUT_PROB)(x)
x= GlobalAveragePooling2D()(x)
x= Dropout(DROPOUT_PROB)(x)
embeddings= Dense(EMB_DIM, activation='relu', kernel_regularizer = regularizers.l2(L2_PENALTY))(x)

embedding_model = Model(inputs=resnet_base.input, outputs=embeddings)

# Freezing layers...
# TO-DO -> Decide what layers you want to train
for layer in embedding_model.layers[:152]:
    layer.trainable= False

# 4a. Test predict

# 5. Siamese Network

in_anc = Input(shape=(INPUT_SHAPE))
in_pos = Input(shape=(INPUT_SHAPE))
in_neg = Input(shape=(INPUT_SHAPE))

em_anc = embedding_model(in_anc)
em_pos = embedding_model(in_pos)
em_neg = embedding_model(in_neg)

# Concatenate them
# Use axis = 1 because axis = 0 is for the batch size or number of samples
out = concatenate([em_anc, em_pos, em_neg], axis=1)

net = Model(
    [in_anc, in_pos, in_neg],
    out
)

net.summary()

# 6. Triplet Loss

# What is used in the FaceNet paper
def triplet_loss(alpha, emb_dim):
    def loss(y_true, y_pred):
        # Only get the dimensions we want aka the embedding dimensions
        # Basically since the prediction is a concatenated value of all three, we need to just
        # index it based off of the embedding dimension
        anc, pos, neg = y_pred[:, :emb_dim], y_pred[:, emb_dim:2*emb_dim], y_pred[:, 2*emb_dim:]
        # distance between positive and anchor
        # What does reduce mean do? finds the average distance
        # How do you do this in Keras?
        dp = tf.reduce_mean(tf.square(anc - pos), axis=1)
        dn = tf.reduce_mean(tf.square(anc - neg), axis=1)
        
        # Max because we don't want it to ever be 0
        return tf.maximum(dp - dn + alpha, 0.)
    return loss

# 7. Data Generator

def data_generator(batch_size, emb_dim):
    while True:
        x = create_batch(batch_size)
        # Because three embeddings multiply by 3
        y = np.zeros((batch_size, 3*emb_dim))
        yield x, y

#8. Compiling the model

# Compile the network
# Set the alkpha and the embedding dimension
# TODO -> Decide on optimizer, alpha value, etc.
net.compile(loss=triplet_loss(alpha = 0.2, emb_dim=EMB_DIM), optimizer = "adam")

# Go ahead and save the model architecture
model_json = embedding_model.to_json()
with open(TRIPLET_MODEL_ARCHITECTURE_SAVE, "w") as json_file:
    json_file.write(model_json)
print("Saved model architecture for the embedding model")

# 9. Training the Model - COMMENTED OUT FOR NOW

# No callbacks right now
# Is this the same as what we had before?
triplet_fit = net.fit_generator(
    data_generator(BATCH_SIZE, EMB_DIM),
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    verbose=False,
    # TODO Add your own call backs, callbacks=model_cb
)

# Save the weights
embedding_model.save_weights(TRIPLET_MODEL_SAVE)
print("Saved model to disk for the embedding model")

# See how long it took
print("--- %s seconds ---" % (time.time() - start_time))