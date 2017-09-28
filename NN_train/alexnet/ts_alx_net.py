import os

import h5py 
import keras
from keras.models import Sequential, Model 
from keras.layers import Dense, Flatten, Dropout, Input, merge, Activation 
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import keras.backend as K

from keras.callbacks import ModelCheckpoint, LearningRateScheduler 


import numpy as np

import keras_utils 

from keras import backend as K
from keras.engine import Layer
from keras.layers.core import Lambda
#from keras.layers.core import Merge
from keras.layers import Merge

def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        div = X.shape[axis] // ratio_split

        if axis == 0:
            output = X[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = X[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = X[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = X[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError('This axis is not possible')

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)

def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """

    def f(X):
        b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1))
                                              , (0, half))
        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:, i:i + ch, :, :]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)

def build_alexnet_orig(weights_path=None, heatmap=False, orig=False, mod=0):
    if heatmap:
        inputs = Input(shape=(3, None, None))
    else:
        inputs = Input(shape=(3, 227, 227))

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu', name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    #conv_2 = Convolution2D(256, 5, 5, activation='relu', name='conv_2')(conv_2)

    conv_2 = merge([
                       Convolution2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    #conv_4 = Convolution2D(384, 3, 3, activation='relu', name='conv_4')(conv_4)

    conv_4 = merge([
                       Convolution2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')


    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    #conv_5 = Convolution2D(256, 3, 3, activation='relu', name='conv_5')(conv_5)

    conv_5 = merge([
                       Convolution2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

    if heatmap:
        dense_1 = Convolution2D(4096, 6, 6, activation='relu', name='dense_1')(dense_1)
        dense_2 = Convolution2D(4096, 1, 1, activation='relu', name='dense_2')(dense_1)
        dense_3 = Convolution2D(1000, 1, 1, name='dense_3')(dense_2)
        prediction = Softmax4D(axis=1, name='softmax')(dense_3)
    else:
        dense_1 = Flatten(name='flatten')(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        if orig:
            prediction = Dense(1000, name='out')(dense_3)
        else:
            if mod == 1:
                dense_3 = Dense(1024, activation='relu', name='dense_3')(dense_3)
            prediction = Dense(3, name='out')(dense_3)

    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path)

    return model


def build_alexnet(weights_path=None, heatmap=False, orig=False, num_out=3):
    if heatmap:
        inputs = Input(shape=(3, None, None))
    else:
        inputs = Input(shape=(3, 227, 227))

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu', name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    #conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = Convolution2D(256, 5, 5, activation='relu', name='conv_2')(conv_2)
    '''
    conv_2 = merge([
                       Convolution2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')
    '''
    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    #conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = Convolution2D(384, 3, 3, activation='relu', name='conv_4')(conv_4)
    '''
    conv_4 = merge([
                       Convolution2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')
    '''

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = Convolution2D(256, 3, 3, activation='relu', name='conv_5')(conv_5)
    '''
    conv_5 = merge([
                       Convolution2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')
    '''
    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

    if heatmap:
        dense_1 = Convolution2D(4096, 6, 6, activation='relu', name='dense_1')(dense_1)
        dense_2 = Convolution2D(4096, 1, 1, activation='relu', name='dense_2')(dense_1)
        dense_3 = Convolution2D(1000, 1, 1, name='dense_3')(dense_2)
        prediction = Softmax4D(axis=1, name='softmax')(dense_3)
    else:
        dense_1 = Flatten(name='flatten')(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        if orig:
            prediction = Dense(1000, name='out')(dense_3)
        else:
            if num_out == 1024:
                dense_3 = Dense(1024, activation='relu', name='dense_3')(dense_3)
            if num_out == 5:
                prediction = Dense(5, activation='softmax', name='out_5')(dense_3)
            elif num_out == 3:
                prediction = Dense(3, activation='softmax', name='out')(dense_3)

    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path)

    return model
if __name__=='__main__':

    net = build_alexnet()
    print (net.summary())

    #net.load_weights('models/hp_model.h5')

