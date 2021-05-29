from keras.initializers import glorot_uniform
from keras.layers import Input, Conv2D, UpSampling2D, Dense, concatenate, MaxPooling2D, Conv2DTranspose,\
    BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from global_params import *

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + dice_coeff_smooth_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + dice_coeff_smooth_factor)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def dimension_reduction_inception_small_filters(inputs, num_filters):
    num_filters_part = int(num_filters / 4)
    tower_one = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    tower_one = Conv2D(num_filters_part, (1, 1), activation='relu', padding='same')(tower_one)
    tower_one = BatchNormalization(axis=3)(tower_one)

    tower_two = Conv2D(num_filters_part, (1, 1), activation='relu', padding='same')(inputs)
    tower_two = Conv2D(num_filters_part, (1, 3), activation='relu', padding='same')(tower_two)
    tower_two = Conv2D(num_filters_part, (3, 1), activation='relu', padding='same')(tower_two)
    tower_two = BatchNormalization(axis=3)(tower_two)

    tower_three = Conv2D(num_filters_part, (1, 1), activation='relu', padding='same')(inputs)
    tower_three = Conv2D(num_filters_part, (1, 5), activation='relu', padding='same')(tower_three)
    tower_three = Conv2D(num_filters_part, (5, 1), activation='relu', padding='same')(tower_three)
    tower_three = BatchNormalization(axis=3)(tower_three)

    tower_four = Conv2D(num_filters_part, (1, 1), activation='relu', padding='same')(inputs)
    tower_four = Conv2D(num_filters_part, (1, 10), activation='relu', padding='same')(tower_four)
    tower_four = Conv2D(num_filters_part, (10, 1), activation='relu', padding='same')(tower_four)
    tower_four = BatchNormalization(axis=3)(tower_four)

    x = concatenate([tower_one, tower_two, tower_three, tower_four], axis=3)
    return x


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = dimension_reduction_inception_small_filters(inputs, 32)
    conv1 = dimension_reduction_inception_small_filters(conv1, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = dimension_reduction_inception_small_filters(pool1, 64)
    conv2 = dimension_reduction_inception_small_filters(conv2, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = dimension_reduction_inception_small_filters(pool2, 128)
    conv3 = dimension_reduction_inception_small_filters(conv3, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = dimension_reduction_inception_small_filters(pool3, 256)
    conv4 = dimension_reduction_inception_small_filters(conv4, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = dimension_reduction_inception_small_filters(pool4, 512)
    conv5 = dimension_reduction_inception_small_filters(conv5, 512)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = dimension_reduction_inception_small_filters(up6, 256)
    conv6 = dimension_reduction_inception_small_filters(conv6, 256)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = dimension_reduction_inception_small_filters(up7, 128)
    conv7 = dimension_reduction_inception_small_filters(conv7, 128)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = dimension_reduction_inception_small_filters(up8, 64)
    conv8 = dimension_reduction_inception_small_filters(conv8, 64)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = dimension_reduction_inception_small_filters(up9, 32)
    conv9 = dimension_reduction_inception_small_filters(conv9, 32)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='main_output')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=seg_model_lr), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def get_classification_model():
    input_tensor = Input((128, 128, 1))
    x = UpSampling2D((2, 2), interpolation='nearest')(input_tensor)
    x = Conv2D(3, (33, 33), padding='valid', kernel_initializer=glorot_uniform())(x)
    res_model = ResNet50(weights='imagenet')
    x = res_model(x)
    x = Dense(1, activation='sigmoid')(x)
    model_classify = Model(inputs=input_tensor, outputs=x)
    return model_classify

