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
from sklearn.utils import class_weight


batch_size = 128
num_classes = 2
num_epochs = 10
img_rows, img_cols = 128, 128
# load data
x_train = np.load("Path_to_x_train.npy")
y_train = np.load("Path_to_y_train.npy")
x_val = np.load("Path_to_x_val.npy")
y_val = np.load("Path_to_y_val.npy")
x_test = np.load("Path_to_x_test.npy")
y_test = np.load("Path_to_y_test.npy")

#some pre-processing to feed data to keras model
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
# normalize data
x_train /= 255
x_val /= 255
x_test /= 255

#compute_class_weights and provide to model.fit() latter
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

#build simple model with some cnn layers 
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
model.add(Dense(num_classes, activation='sigmoid'))

tensorboard = TensorBoard(log_dir="path_to_store_tensorboard_logs")

model.compile(optimizer=Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', f1])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_data=(x_val, y_val), class_weight= class_weights, callbacks=[tensorboard])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("path_to_save_model.h5")
