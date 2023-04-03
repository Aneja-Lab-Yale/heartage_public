# Aneja Lab Common Code
# Keras Callbacks
# Aneja Lab | Yale School of Medicine
# Sanjay Aneja, MD
# Created (3/6/20)
# Updated (10/7/20)
import tensorflow as tf


# Callbacks Keras
# Created (5/4/17)
# Debugged (10/7/20)
# Created By (Sanjay Aneja, MD)
def callbacks_model(model_save_path,
                    csv_log_file,
                    patience,
                    minimum_lrate,
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
            monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
            baseline=None, restore_best_weights=False),
        ## Stop training when quality hasn't improved, patients = number of epochs without improvement
        #tf.keras.callbacks.LearningRateScheduler(lrate_schedule, verbose=0),
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


