from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import h5py
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenetv2 import MobileNetV2
from helpers import f1

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

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def build_model():
    input_tensor = Input(shape=(img_rows, img_cols, 3))
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(img_rows, img_cols, 3),
        pooling='avg')

    for layer in base_model.layers[:141]:
        layer.trainable = False
    for layer in base_model.layers[141:]:
        layer.trainable = True
base_model.load_weights("C:\\Users\\niksp\\.keras\\models\\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5")
    op = Dense(64, activation='relu')(base_model.output)
    op = Dropout(.20)(op)
    output_tensor = Dense(num_classes, activation='tanh')(op)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

tensorboard = TensorBoard(log_dir="D:\\hiwi_work\\tb_logs\\mobilenet_trL_B")

model= build_model()
model.compile(optimizer=Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', f1])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_data=(x_val, y_val), callbacks=[tensorboard])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("D:\\hiwi_work\\Models\\mobilenet_trL_B.h5")
