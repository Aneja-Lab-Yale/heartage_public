# Aneja Lab Common Code
# For Loop Models
# Creates multiple models through nested for loops using user inputs for various inputs (models, weights, dropout, initial learning rate, data augmentation)
# Outputs a CSV file for each model with its parameters and metrics (variable mets)
# Example for running the function on last line
# Aneja Lab | Yale School of Medicine
# Victor Lee
# Created (8/22/2020)
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, VGG16, VGG19, MobileNet
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the parameters for each model in the following lines
models = [ResNet50, MobileNet, VGG16, VGG19, ResNet101, ResNet152]
weights = ['imagenet', None]
dropout = [0, 0.2, 0.5, 0.7]
initial_lr = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
data_augmentation = [True, False]

#These are the metrics that will be outputted to the CSV file
mets=[
    tf.keras.metrics.TruePositives(name="True_Pos"),
    tf.keras.metrics.FalsePositives(name="False_Pos"),
    tf.keras.metrics.TrueNegatives(name="True_Neg"),
    tf.keras.metrics.FalseNegatives(name="False_Neg"),
    tf.keras.metrics.BinaryAccuracy(name="Binary_Accuracy"),
    tf.keras.metrics.Precision(name="Precision_Specificity"),
    tf.keras.metrics.Recall(name="Recall_Sensitivity"),
    tf.keras.metrics.AUC(name="AUC"),
    tf.keras.metrics.Accuracy(name="accuracy")
    ]

# Adds on a few layers at the end of each model
def base_addon(
        model_variable,
        c,
        d,
):
    x = model_variable.output
    x = Dropout(c)(x)  # user defined
    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(c)(x)  # user defined
    predictions = Dense(10, activation='softmax')(x)  # user defined--based on problem
    final_model = Model(inputs=model_variable.input, outputs=predictions)  # architechture for resnet is made
    for layer in final_model.layers[:-4]:
        layer.trainable = False
    final_model.compile(loss='categorical_crossentropy',
                        optimizer=Adam(learning_rate=d), #todo LR scheduler
                        metrics=mets)
    return final_model

def train_test(final_model, e, filepath):
    # create data generator and iterator, added 8/6/2020, then added as a parameter into fit function
    datagen = ImageDataGenerator(rotation_range=15, horizontal_flip=True, zoom_range=0.2)
    it_train = datagen.flow(x_train, y_train, batch_size=batch_size)
    if e == True:
        history = final_model.fit(it_train,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=(x_test, y_test),
                                  shuffle=True,
                                  callbacks=tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=0, save_weights_only=True, mode='auto', save_freq='epoch'),
                                  )
    elif e == False:
        history = final_model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(x_test, y_test),
                                shuffle=True,
                                callbacks= tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=0, save_weights_only=True, mode='auto', save_freq='epoch'),
                                )

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv("loss_history.csv", index=False, header=history.history.keys)

    scores = final_model.evaluate(x_test, y_test, verbose=1)
    return scores

def run_all_models(
        models,
        weights,
        dropout,
        initial_lr,
        data_augmentation
):
    counter = 0
    model_id = np.array(["Model Number Tested", "Model", "Weights", "Dropout", "Initial LR", "Data Augmentation"])
    for a in models:
        for b in weights:
            for c in dropout:
                for d in initial_lr:
                    for e in data_augmentation:
                        counter = counter + 1
                        print("Model Number Tested: ", counter, "Model: ", a, "Weights: ", b, "Dropout: ", c, "Initial LR:", d, "Data Augmentation: ", e)
                        model_id = np.vstack((model_id, [counter,a,b,c,d,e]))
                        final_model = base_addon(a(weights=b, include_top=False), c, d)
                        filepath = "saved_models/model_number_{}.h5".format(counter)
                        train_test(final_model, e, filepath)
                        data = pd.read_csv("loss_history.csv", names=['loss','True_Pos','False_Pos','True_Neg','False_Neg','Binary_Accuracy','Precision_Specificity','Recall_Sensitivity','AUC','accuracy','val_loss','val_True_Pos','val_False_Pos','val_True_Neg','val_False_Neg','val_Binary_Accuracy','val_Precision_Specificity','val_Recall_Sensitivity','val_AUC','val_accuracy'])
                        df = pd.DataFrame(data=model_id)
                        horizontal_stack = pd.concat([df, data], axis=1)
                        pd.DataFrame(horizontal_stack).to_csv("model_id_loss_history_combined.csv", index=False, header=False)
                        tf.keras.backend.clear_session()

run_all_models(models=models, weights=weights, dropout=dropout, initial_lr=initial_lr, data_augmentation=data_augmentation)
