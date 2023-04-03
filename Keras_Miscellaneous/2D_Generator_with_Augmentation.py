# Generalization Phenotype
# Generic generator for model training
# Aneja Lab | Yale School of Medicine
# Aidan Gilson
# Created (06/17/20)
# Updated (07/07/2020)

# Import
# Employed (6/23/20)
# Updated (7/01/2020)
# <editor-fold desc="Import">
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
from random import shuffle
from imgaug import augmenters as iaa
# </editor-fold>


# Yields patients from file with augmentations
# Employed (06/17/20)
# Updated (07/07/20)
# Created By Aidan Gilson
def image_generator(inputPath, patients, labels, bs, convert_to_one_hot=False, classes=None, aug=False,
                    fit_number=None, added_func=False, rotation_range=0., zoom_range=0., width_shift_range=0.,
                    height_shift_range=0., shear_range=0., horizontal_flip=False, fill_mode="nearest",
                    preprocessing_function=None):
    """
    Description:
        This generator is used to return a set of patients and labels from a pool of patients for model training.
    Param:
        inputPath = string, a path the the folder containing each patients image(s)
        patients = list of strings, a list of patients in that folder that can be used to generate the desired set. Names
            should be the file name excluding the .npy extention. I.E for Patient001.npy, pass in "Patient001"
        labels = list of labels, the labels for the patients provided in the same order as the patients
        bs = int, batch size
        convert_to_one_hot = bool. True if data labels need to be converted to one_hot encoding before returned
        classes = int, number of classes in the label set. classes will default to the unique elements in the labels if
            not specified. Definitely specify it though.
        aug = bool. True if augmentation of the images is wanted
        fit_number = int, number of patients that the image augmentor should be fit on. Note, all images that it is fit
            to need to be loaded into memory so using the full dataset is unreasonable. Use the smallest statistically
            significant value possible. If not specified, will default to the batch size
        added_func = bool, true if a custom function should be passed into the augmentor. Function should be specified
            with preprocessing_function. If preprocessing_function is not specified, gaussian blur function will be added
        remaining = varied, parameters for image augmentation if desired. Specifications in the Keras ImageDataGenerator
            documentation
    Return:
        Returns a list images or image sets and their corresponding labels.
        """

    # Gets an initial random ordering of the patients
    indecies = [x for x in range(len(patients))]
    shuffle(indecies)
    index = 0

    # Creates the image augmentation function as specified
    if aug:
        if added_func:
            if preprocessing_function is None:
                preprocessing_function = gaussian_noise
        augmentation = ImageDataGenerator(rotation_range=rotation_range, zoom_range=zoom_range,
                                          width_shift_range=width_shift_range, height_shift_range=height_shift_range,
                                          shear_range=shear_range, horizontal_flip=horizontal_flip, fill_mode=fill_mode,
                                          preprocessing_function=preprocessing_function)
        # Determines the number of patients to fit the augmentation to
        if fit_number is None:
            fit_number = bs
            if fit_number > len(patients):
                print("Augmentation was attempting to fit on more samples then are available. Will default to fitting "
                      "on full dataset.")
                fit_number = len(patients)

        fit_images = []
        for i in range(fit_number):
            patient = patients[indecies[i]]
            patient_path = inputPath + patient + ".npy"
            image = np.load(patient_path, allow_pickle=True)
            fit_images.append(image)
        fit_images = np.array(fit_images)
        augmentation.fit(fit_images)

    # The generator will continuously create and augment images as needed. I recommend that when training a model,
    # set the number of training images to whatever is desired, independent of dataset size.
    while True:
        images = []
        outcomes = []

        # for index in batch_patients
        while len(images) < bs:
            patient_index = indecies[index]
            # Loads patient from folder
            patient = patients[patient_index]
            patient_path = inputPath + patient + ".npy"
            image = np.load(patient_path, allow_pickle=True)

            # Finds the label for the patient
            label = labels[patient_index]

            # Adds them to the running list of patients and labels.
            images.append(image)
            outcomes.append(label)
            index += 1

            # After all patients are used, reshuffle to begin again
            if index == len(patients):
                index = 0
                shuffle(indecies)

        outcomes = np.array(outcomes)

        # Converts the labels to one_hot encoding if needed
        if convert_to_one_hot:
            if classes is None:
                classes = len(set(outcomes))
            outcomes = np_utils.to_categorical(outcomes, int(classes))

        # Performs data augmentation if desired
        if aug:
            (images, outcomes) = next(augmentation.flow(np.array(images),
                                                        outcomes, batch_size=bs))
        images = np.array(images)

        yield images, outcomes


# 1: Gaussian Noise Implementation
# Employed (03/18/20)
# Debugged (03/20/20)
# Created By (Sachin Umrao, Ph.D.)
def gaussian_noise(images,
                   gn_scale=0.1 * 255):
    """
    Description:
        This function adds gaussian noise on images..
    Param:
        Images = Images in array form
		gn_scale = distribution for gaussian noice
    Return:
        returns images in array form
    """

    batch_images = []
    g_noise = iaa.AdditiveGaussianNoise(scale=gn_scale)
    for img in images:
        batch_images.append(g_noise(image=img))
    return np.array(batch_images)
