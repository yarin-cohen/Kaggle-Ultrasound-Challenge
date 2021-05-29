import numpy as np
from project_utils import *
from model import *
from global_params import *
from skimage.io import imsave


def save_segmentation_test_images(imgs, imgs_ids, model):

    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test = model.predict(imgs, verbose=1)
    np.save(np_files_path + 'imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)
    for image, image_id in zip(imgs_mask_test, imgs_ids):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(predictions_dir, str(image_id) + '_pred.png'), image)


def main():
    print('-' * 30)
    print('loading segmentation model and weights...')
    seg_model = get_unet()
    seg_model.load_weights(seg_model_weights_path + 'base_model.h5')

    print('-' * 30)
    print('preprocessing data...')
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)
    imgs_test = imgs_test.astype('float32')
    imgs_test -= np.mean(imgs_test)  # CHECK IF OK. WAS JUST mean
    imgs_test /= np.std(imgs_test)  # CHECK IF OK. WAS JUST std

    print('-' * 30)
    print('predicting segmentation model results and saving images...')
    save_segmentation_test_images(imgs_test, imgs_id_test, seg_model)


if __name__ == '__main__':
    main()

