from model import*
from project_utils import*
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from global_params import *


f_loss = focal_loss(focal_loss_gamma, focal_loss_alpha)


def load_and_preprocess_data():
    imgs_train, imgs_mask_train = load_train_data()
    imgs_val, imgs_mask_val = load_val_data()
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)
    imgs_val = preprocess(imgs_val)
    imgs_mask_val = preprocess(imgs_mask_val)

    imgs_train = imgs_train.astype('float32')
    imgs_val = imgs_val.astype('float32')

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_val -= mean
    imgs_val /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    imgs_mask_val = imgs_mask_val.astype('float32')
    imgs_mask_val /= 255.

    return imgs_train, imgs_mask_train, imgs_val, imgs_mask_val


def train_and_save_classification_model(x_train, y_train, x_val, y_val):
    model_classify = get_classification_model()
    model_classify.compile(optimizer=Adam(lr=classification_model_lr), loss=f_loss, metrics=['accuracy'])
    model_checkpoint = ModelCheckpoint(classification_model_weights_path + 'base_model_classify.h5',
                                       monitor='val_loss', save_best_only=True)

    if train_classification:
        history = model_classify.fit(x_train, y_train, batch_size=classification_model_batch_size,
                                     epochs=classification_model_epochs, verbose=1, shuffle=True,
                                     validation_data=(x_val, y_val), callbacks=[model_checkpoint])
    else:
        model_classify.load_weights(classification_model_weights_path + 'base_model_classify.h5')
        history = []

    return model_classify, history


def train_and_save_seg_model():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train, imgs_val, imgs_mask_val = load_and_preprocess_data()
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint(seg_model_weights_path + 'base_model.h5', monitor='val_loss',
                                       save_best_only=True)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=seg_model_batch_size, epochs=seg_model_epochs,
                        verbose=1, shuffle=True, validation_data=(imgs_val, imgs_mask_val),
                        callbacks=[model_checkpoint, EarlyStopping(monitor='val_loss', mode='min', patience=20)])

    print('-' * 30)
    return model, history


def main():

    # loading data for classification model
    print('loading classification data...')
    iso_train_masks = np.load(np_files_path + 'iso_masks_train.npy')
    train_classes = np.load(np_files_path + 'iso_masks_train_classes.npy')
    iso_val_masks = np.load(np_files_path + 'iso_masks_val.npy')
    val_classes = np.load(np_files_path + 'iso_masks_val_classes.npy')

    # training and saving classification model
    print('training and saving classification model...')
    model_classify, model_classify_history = train_and_save_classification_model(iso_train_masks, train_classes,
                                                                                 iso_val_masks, val_classes)

    # training segmentation model
    print('training and saving segmentation model...')
    history_seg, model = train_and_save_seg_model()

    print('done')

if __name__ == '__main__':
    main()