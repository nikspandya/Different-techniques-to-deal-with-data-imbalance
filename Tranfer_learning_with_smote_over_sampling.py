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
from keras.applications.inception_v3 import InceptionV3

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

def create_balanced_sample_smote(x, y):
    """x: input image
       y: labels
       return: class ballanced images numpy array with labels
    """
    x_2d = np.reshape(x, (x.shape[0], -1))
    smote = SMOTE('balanced')
    x_sm, y_sm = smote.fit_sample(x_2d, y)
    x_sm_resized = x_sm.reshape(x_sm.shape[0], 128, 128, 3)
    return x_sm_resized, y_sm
x_sm_train, y_sm_t = create_balanced_sample_smote(x_train, y_train)

#some pre-processing 
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)
    
x_sm_train = x_sm_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
# normalize data
x_sm_train /= 255
x_val /= 255
x_test /= 255

# Onehot encoding
y_train_encoded = keras.utils.to_categorical(y_sm_t, num_classes)
y_val_encoded = keras.utils.to_categorical(y_val, num_classes)
y_test_encoded = keras.utils.to_categorical(y_test, num_classes)

def build_model():
    input_tensor = Input(shape=(img_rows, img_cols, 3))
    base_model = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(img_rows, img_cols, 3),
        pooling='avg')
    base_model.load_weights("path_to_model.h5")
    for layer in base_model.layers[:249]:
        layer.trainable = False
    for layer in base_model.layers[249:]:
        layer.trainable = True #train only some last cnn

    op = Dense(64, activation='relu')(base_model.output)
    op = Dropout(.20)(op)
    output_tensor = Dense(num_classes, activation='sigmoid')(op)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

tensorboard = TensorBoard(log_dir="path_to_save_tensorboard_logs")

model = build_model()
model.compile(optimizer=Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', f1])

model.fit(x_sm_train, y_train_encoded,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_data=(x_val, y_val_encoded), callbacks=[tensorboard])
score = model.evaluate(x_test, y_test_encoded, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("path_to_save_model.h5")
