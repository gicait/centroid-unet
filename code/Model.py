import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv1D, Concatenate, Conv2DTranspose, GaussianNoise, Lambda, Dropout, Flatten, Dense, Reshape, TimeDistributed, Permute, Softmax, Multiply, BatchNormalization, UpSampling2D
from keras import backend as K
import tensorflow as tf
from keras.models import model_from_json

def getModel(input_shape):
    # Reference - https://github.com/zhixuhao/unet

    e_in = Input(shape = input_shape)

    e_cnn = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_in)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_skip1 = Dropout(0.2)(e_cnn)
    e_cnn = MaxPooling2D(pool_size=(2, 2))(e_skip1)

    e_cnn = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_skip2 = Dropout(0.2)(e_cnn)
    e_cnn = MaxPooling2D(pool_size=(2, 2))(e_skip2)

    e_cnn = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_skip3 = Dropout(0.2)(e_cnn)
    e_cnn = MaxPooling2D(pool_size=(2, 2))(e_skip3)

    e_cnn = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_skip4 = Dropout(0.2)(e_cnn)
    e_cnn = MaxPooling2D(pool_size=(2, 2))(e_skip4)

    e_cnn = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Dropout(0.2)(e_cnn)
    e_cnn = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)

    '''Decoder'''

    e_cnn = UpSampling2D(size = (2,2))(e_cnn)
    e_cnn = Concatenate()([e_cnn,e_skip4])
    e_cnn = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Dropout(0.2)(e_cnn)

    e_cnn = UpSampling2D(size = (2,2))(e_cnn)
    e_cnn = Concatenate()([e_cnn,e_skip3])
    e_cnn = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Dropout(0.2)(e_cnn)

    e_cnn = UpSampling2D(size = (2,2))(e_cnn)
    e_cnn = Concatenate()([e_cnn,e_skip2])
    e_cnn = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Dropout(0.2)(e_cnn)

    e_cnn = UpSampling2D(size = (2,2))(e_cnn)
    e_cnn = Concatenate()([e_cnn,e_skip1])
    e_cnn = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Dropout(0.2)(e_cnn)

    e_cnn = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)
    e_cnn = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(e_cnn)
    e_cnn = BatchNormalization()(e_cnn)

    e_out = Conv2D(1, 1, activation = 'sigmoid')(e_cnn)

    model = Model(input = e_in, output = e_out)

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model
