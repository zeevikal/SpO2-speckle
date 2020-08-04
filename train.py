from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Dropout, Dense, Input, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential

from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

from tqdm import tqdm
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from imutils import paths
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib
import argparse
import pickle
import glob
import cv2
import os

np.random.seed(42)
matplotlib.use("Agg")

# Frames images size
FRAME_SIZE = 32
data_path = './data'

x_data = list()
y_data = list()

x_train = list()
y_train = list()
x_val = list()
y_val = list()

train_list = glob.glob('{0}/train/*.png'.format(data_path))
val_list = glob.glob('{0}/val/*.png'.format(data_path))


def prep_train_val(train_list, val_list):
    for i in tqdm(range(0, len(train_list) - 1)):
        img1 = train_list[i]
        img2 = train_list[i + 1]
        img1_path_list = img1.split(sep='_')
        img2_path_list = img2.split(sep='_')
        vid1 = '{0}_{1}_{2}'.format(img1_path_list[1], img1_path_list[2], img1_path_list[3])
        vid2 = '{0}_{1}_{2}'.format(img2_path_list[1], img2_path_list[2], img2_path_list[3])
        cls = '{0}_{1}'.format(img1_path_list[0].split(sep=os.sep)[1].split(sep='-')[0],
                               img2_path_list[0].split(sep=os.sep)[1].split(sep='-')[1])
        if vid1 != vid2: continue

        im1 = cv2.imread(img1)
        im2 = cv2.imread(img2)
        im1 = np.mean(im1, axis=-1)
        im2 = np.mean(im2, axis=-1)
        corr = correlate2d(im1, im2, mode='same')
        x_train.append(corr)

        if '96' in cls:
            y_train.append(0)
        else:
            y_train.append(1)

    for i in tqdm(range(0, len(val_list) - 1)):
        img1 = val_list[i]
        img2 = val_list[i + 1]
        img1_path_list = img1.split(sep='_')
        img2_path_list = img2.split(sep='_')
        vid1 = '{0}_{1}_{2}'.format(img1_path_list[1], img1_path_list[2], img1_path_list[3])
        vid2 = '{0}_{1}_{2}'.format(img2_path_list[1], img2_path_list[2], img2_path_list[3])
        cls = '{0}_{1}'.format(img1_path_list[0].split(sep=os.sep)[1].split(sep='-')[0],
                               img2_path_list[0].split(sep=os.sep)[1].split(sep='-')[1])

        if vid1 != vid2: continue

        im1 = cv2.imread(img1)
        im2 = cv2.imread(img2)
        im1 = np.mean(im1, axis=-1)
        im2 = np.mean(im2, axis=-1)
        corr = correlate2d(im1, im2, mode='same')
        x_val.append(corr)

        if '96' in cls:
            y_val.append(0)
        else:
            y_val.append(1)

    train_data = np.array(x_train)
    train_labels_orig = np.array(y_train)
    val_data = np.array(x_val)
    val_labels_orig = np.array(y_val)

    train_data = np.expand_dims(train_data, axis=-1)
    val_data = np.expand_dims(val_data, axis=-1)

    return train_data, train_labels_orig, val_data, val_labels_orig


def prep_train_val_on_movement(train_list, val_list):
    for i in tqdm(range(0, len(train_list) - 1)):
        img1 = train_list[i]
        img2 = train_list[i + 1]
        img1_path_list = img1.split(sep='_')
        img2_path_list = img2.split(sep='_')
        vid1 = '{0}_{1}_{2}'.format(img1_path_list[1], img1_path_list[2], img1_path_list[3])
        vid2 = '{0}_{1}_{2}'.format(img2_path_list[1], img2_path_list[2], img2_path_list[3])
        cls = '{0}_{1}'.format(img1_path_list[0].split(sep=os.sep)[1].split(sep='-')[0],
                               img2_path_list[0].split(sep=os.sep)[1].split(sep='-')[1])
        if vid1 != vid2: continue

        im1 = cv2.imread(img1)
        im2 = cv2.imread(img2)
        image = im2 - im1
        x_train.append(image)

        if '96' in cls:
            y_train.append(0)
        else:
            y_train.append(1)

    for i in tqdm(range(0, len(val_list) - 1)):
        img1 = val_list[i]
        img2 = val_list[i + 1]
        img1_path_list = img1.split(sep='_')
        img2_path_list = img2.split(sep='_')
        vid1 = '{0}_{1}_{2}'.format(img1_path_list[1], img1_path_list[2], img1_path_list[3])
        vid2 = '{0}_{1}_{2}'.format(img2_path_list[1], img2_path_list[2], img2_path_list[3])
        cls = '{0}_{1}'.format(img1_path_list[0].split(sep=os.sep)[1].split(sep='-')[0],
                               img2_path_list[0].split(sep=os.sep)[1].split(sep='-')[1])

        if vid1 != vid2: continue

        im1 = cv2.imread(img1)
        im2 = cv2.imread(img2)
        image = im2 - im1
        x_val.append(image)

        if '96' in cls:
            y_val.append(0)
        else:
            y_val.append(1)

    train_data = np.array(x_train)
    train_labels_orig = np.array(y_train)
    val_data = np.array(x_val)
    val_labels_orig = np.array(y_val)

    train_data = np.expand_dims(train_data, axis=-1)
    val_data = np.expand_dims(val_data, axis=-1)

    return train_data, train_labels_orig, val_data, val_labels_orig


def prep_train_val_correlate2d():
    for i in tqdm(range(0, len(train_list) - 1)):
        img1 = train_list[i]
        img2 = train_list[i + 1]
        img1_path_list = img1.split(sep='_')
        img2_path_list = img2.split(sep='_')
        vid1 = '{0}_{1}_{2}'.format(img1_path_list[1], img1_path_list[2], img1_path_list[3])
        vid2 = '{0}_{1}_{2}'.format(img2_path_list[1], img2_path_list[2], img2_path_list[3])
        cls = '{0}_{1}'.format(img1_path_list[0].split(sep=os.sep)[1].split(sep='-')[0],
                               img2_path_list[0].split(sep=os.sep)[1].split(sep='-')[1])
        if vid1 != vid2: continue

        im1 = cv2.imread(img1)
        im2 = cv2.imread(img2)
        im1 = np.mean(im1, axis=-1)
        im2 = np.mean(im2, axis=-1)
        corr = correlate2d(im1, im2, mode='same')
        x_train.append(corr)

        if '96' in cls:
            y_train.append(0)
        else:
            y_train.append(1)

    for i in tqdm(range(0, len(val_list) - 1)):
        img1 = val_list[i]
        img2 = val_list[i + 1]
        img1_path_list = img1.split(sep='_')
        img2_path_list = img2.split(sep='_')
        vid1 = '{0}_{1}_{2}'.format(img1_path_list[1], img1_path_list[2], img1_path_list[3])
        vid2 = '{0}_{1}_{2}'.format(img2_path_list[1], img2_path_list[2], img2_path_list[3])
        cls = '{0}_{1}'.format(img1_path_list[0].split(sep=os.sep)[1].split(sep='-')[0],
                               img2_path_list[0].split(sep=os.sep)[1].split(sep='-')[1])

        if vid1 != vid2: continue

        im1 = cv2.imread(img1)
        im2 = cv2.imread(img2)
        im1 = np.mean(im1, axis=-1)
        im2 = np.mean(im2, axis=-1)
        corr = correlate2d(im1, im2, mode='same')
        x_val.append(corr)

        if '96' in cls:
            y_val.append(0)
        else:
            y_val.append(1)

    train_data = np.array(x_train)
    train_labels_orig = np.array(y_train)
    val_data = np.array(x_val)
    val_labels_orig = np.array(y_val)

    train_data = np.expand_dims(train_data, axis=-1)
    val_data = np.expand_dims(val_data, axis=-1)

    return train_data, train_labels_orig, val_data, val_labels_orig


def load_dataset(train_path='./data_zip/train_3rd_day_corr.npz', val_path='./data_zip/val_3rd_day_corr.npz',
                 train_data=None, train_labels_orig=None, val_data=None, val_labels_orig=None):
    if os.path.exists(train_path):
        train_file = np.load(train_path)
        train_data = train_file['x']
        train_labels_orig = train_file['y']
    else:
        np.savez_compressed(train_path, x=train_data, y=train_labels_orig)
    if os.path.exists(val_path):
        val_file = np.load(val_path)
        val_data = val_file['x']
        val_labels_orig = val_file['y']
    else:
        np.savez_compressed(val_path, x=val_data, y=val_labels_orig)
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_data = np.expand_dims(train_data, axis=-1)
        val_data = np.expand_dims(val_data, axis=-1)
        return train_data, train_labels_orig, val_data, val_labels_orig


def prep_labels(train_labels_orig, val_labels_orig):
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels_orig)
    val_labels = lb.fit_transform(val_labels_orig)
    # print(lb.classes_)
    return train_labels, val_labels


def cnn_model(lb):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our 2D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(1))
    model.add(Dense(len(lb.classes_)))
    model.add(Activation('softmax'))

    # model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model


def resnet50_model(lb):
    # load the ResNet-50 network, ensuring the head FC layer sets are left off
    baseModel = ResNet50(weights=None, include_top=False, input_tensor=Input(shape=(FRAME_SIZE, FRAME_SIZE, 1)))

    # construct the head of the model that will be placed on top of the the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # compile our model (this needs to be done after our setting our layers to being non-trainable)

    # opt = Adam(lr=1e-4, momentum=0.9)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=0.0001), metrics=["accuracy"])
    return model


def train(train_labels_orig, model, train_data, train_labels, val_data, val_labels):
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels_orig), train_labels_orig)
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(val_labels_orig), val_labels_orig)
    EPOCHS = 10
    checkpoint_filepath = './models/full_data_best_classifier_20200513/model.hdf5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    H = model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                  batch_size=256, epochs=EPOCHS, callbacks=[model_checkpoint_callback], class_weight=class_weight)
    with open('./models/full_data_best_classifier_20200513_HISTORY', 'wb') as f:
        pickle.dump(H.history, f)
    model.save('./models/full_data_best_classifier_20200513.h5')
