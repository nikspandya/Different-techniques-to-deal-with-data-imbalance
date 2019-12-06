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
from helpers import f1, balanced_sample_maker

batch_size = 128
num_classes = 2
num_epochs = 10
img_rows, img_cols = 128, 128
# load data
x_train = np.load("path_to_X_train.npy")
y_train = np.load("path_to_y_train.npy")
x_val = np.load("path_to_x_val.npy")
y_val = np.load("path_to_y_val.npy")
x_test = np.load("path_to_X_test.npy")
y_test = np.load("path_to_y_test.npy")

train_data = balanced_sample_maker(x_train, y_train, sample_size=2000)
x_train_bal = train_data[0]
y_train_bal = train_data[1]



#some pre-processing to feed data to keras model
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)

x_train = x_train_bal.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
# normalize data
x_train /= 255
x_val /= 255
x_test /= 255

# Onehot encoding
y_train_encoded = keras.utils.to_categorical(y_train_bal, num_classes)
y_val_encosed = keras.utils.to_categorical(y_val, num_classes)
y_test_encoded = keras.utils.to_categorical(y_test, num_classes)

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

tensorboard = TensorBoard(log_dir="path_to_save_tensorboard_logs")

model.compile(optimizer=Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', f1])

model.fit(x_train, y_train_encoded,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_data=(x_val, y_val_encosed), callbacks=[tensorboard])
score = model.evaluate(x_test, y_test_encoded, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("path_to_save_model.h5")
