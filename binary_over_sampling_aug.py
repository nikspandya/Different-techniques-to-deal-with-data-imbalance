from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.callbacks import TensorBoard
import h5py
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from helpers import f1
from imblearn.over_sampling import SMOTE
from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
num_classes = 2
num_epochs = 10
img_rows, img_cols = 128, 128
# load data
x_train = np.load("D:\\hiwi_work\\binary_split\\X_train.npy")
y_train = np.load("D:\\hiwi_work\\binary_split\\y_train.npy")
x_val = np.load("D:\\hiwi_work\\binary_split\\x_val.npy")
y_val = np.load("D:\\hiwi_work\\binary_split\\y_val.npy")
x_test = np.load("D:\\hiwi_work\\binary_split\\X_test.npy")
y_test = np.load("D:\\hiwi_work\\binary_split\\y_test.npy")

#make (samples, features) format
x_train_2d = np.reshape(x_train, (x_train.shape[0], -1))
x_val_2d = np.reshape(x_val, (x_val.shape[0], -1))
x_test_2d = np.reshape(x_test, (x_test.shape[0], -1))

smote = SMOTE('minority')
x_sm_t, y_sm_t = smote.fit_sample(x_train_2d, y_train)
x_sm_v, y_sm_val = smote.fit_sample(x_val_2d, y_val)
x_sm_test, y_sm_test = smote.fit_sample(x_test_2d, y_test)
# reshape again for cnn
x_sm_train = x_sm_t.reshape(3782,128,128,3)
x_sm_val = x_sm_v.reshape(458,128,128,3)
x_sm_test = x_sm_test.reshape(428,128,128,3)

#some pre-processing to feed data to keras model
#some pre-processing to feed data to keras model
if K.image_data_format() == 'channels_first':
    x_train_bal = x_train.reshape(x_sm_train.shape[0], 3, img_rows, img_cols)
    x_val_bal = x_val.reshape(x_sm_val.shape[0], 3, img_rows, img_cols)
    x_test_bal = x_test.reshape(x_sm_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train_bal = x_sm_train.reshape(x_sm_train.shape[0], img_rows, img_cols, 3)
    x_val_bal = x_sm_val.reshape(x_sm_val.shape[0], img_rows, img_cols, 3)
    x_test_bal = x_sm_test.reshape(x_sm_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_sm_train = x_sm_train.astype('float32')
x_sm_val = x_sm_val.astype('float32')
x_sm_test = x_sm_test.astype('float32')
# normalize data
x_sm_train /= 255
x_sm_val /= 255
x_sm_test /= 255

# convert class vectors to binary class matrices
y_train_encoded = keras.utils.to_categorical(y_sm_t, num_classes)
y_val_encosed = keras.utils.to_categorical(y_sm_val, num_classes)
y_test_encoded = keras.utils.to_categorical(y_sm_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='tanh'))

tensorboard = TensorBoard(log_dir="D:\\hiwi_work\\tb_logs\\model_original_binary_smote_aug")

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(x_sm_train)

model.compile(optimizer=Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', f1])

model.fit_generator(datagen.flow(x_sm_train, y_train_encoded,
          batch_size=batch_size),
          epochs=num_epochs,
          steps_per_epoch=len(x_train) / batch_size,
          verbose=1,
          validation_data=(x_sm_val, y_val_encosed), callbacks=[tensorboard])
score = model.evaluate(x_sm_test, y_test_encoded, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("D:\\hiwi_work\\Models\\model_original_smote_aug.h5")
