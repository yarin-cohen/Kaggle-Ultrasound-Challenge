from __future__ import print_function
import cv2
from skimage.io import imread
import copy
from tqdm import tqdm_notebook as tqdm
import os
from skimage.transform import resize
import numpy as np
from global_params import *


def create_train_data(data_path):
    if not os.path.exists(np_files_path):
        os.mkdir(np_files_path)

    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)
    imgs = np.ndarray((total, image_rows_resized, image_cols_resized), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows_resized, image_cols_resized), dtype=np.uint8)

    i = 0
    count = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        if image_name[2] == '_' and int(image_name[0:2]) > 37:
            continue
        
        count +=1
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    
    print('Loading done.')
    imgs = imgs[:count, :, :]
    imgs_mask = imgs_mask[:count, :, :]
    np.save(np_files_path + 'imgs_train.npy', imgs)
    np.save(np_files_path + 'imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def create_val_data(data_path):
    if not os.path.exists(np_files_path):
        os.mkdir(np_files_path)

    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)
    imgs = np.ndarray((total, image_rows_resized, image_cols_resized), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows_resized, image_cols_resized), dtype=np.uint8)

    i = 0
    count = 0
    print('-'*30)
    print('Creating validation images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        if not (image_name[2] == '_' and int(image_name[0:2]) > 37):
            continue
            
        count += 1
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    
    print('Loading done.')
    #print('count: '+ str(count))
    imgs = imgs[:count, :, :]
    imgs_mask = imgs_mask[:count, :, :]
    np.save(np_files_path + 'imgs_val.npy', imgs)
    np.save(np_files_path + 'imgs_mask_val.npy', imgs_mask)
    print('Saving to .npy files done.')
    

def load_train_data():
    imgs_train = np.load(np_files_path + 'imgs_train.npy')
    imgs_mask_train = np.load(np_files_path + 'imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def load_val_data():
    imgs_val = np.load(np_files_path+ 'imgs_val.npy')
    imgs_mask_val = np.load(np_files_path + 'imgs_mask_val.npy')
    return imgs_val, imgs_mask_val


def create_false_mask(img, org_locs, shifted_locs, height, width):
    
    while True:
        corner1 = np.random.randint(len(img) - height)
        corner2 = np.random.randint(len(img[0]) - width)
        if abs(corner1 - np.min(org_locs[0])) > np.floor(height*corners_limit_coef) and \
                abs(corner2 - np.min(org_locs[1])) > np.floor(width*corners_limit_coef):
            continue
        m1 = np.min(org_locs[0])
        m2 = np.min(org_locs[1])
        for j in range(len(org_locs[0])):
            org_locs[0][j] = org_locs[0][j] - m1 + corner1
            org_locs[1][j] = org_locs[1][j] - m2 + corner2
        false_mask = np.multiply(128, np.ones((height, width)))
        false_mask[shifted_locs] = img[org_locs]
        break
    return false_mask
    

def create_iso_masks_from_data(imgs, masks, num_neg, to_ignore_neg=1):
    
    iso_masks = np.zeros((6*len(imgs), iso_masks_size, iso_masks_size))
    classes = np.zeros((6*len(imgs), 1))
    count = 0
    for k in tqdm(range(len(imgs))):

        mask = masks[k] 
        img = imgs[k]

        if len(mask[mask > 0]) > 0:
            locs = np.where(mask > 0)
            org_locs = copy.deepcopy(locs)
            height = np.max(locs[0]) - np.min(locs[0]) + 1

            width = np.max(locs[1]) - np.min(locs[1]) + 1
            new_img = np.multiply(128, np.ones((height, width)))
            m1 = np.min(locs[0])
            m2 = np.min(locs[1])
            for j in range(len(locs[0])):
                locs[0][j] = locs[0][j] - m1
                locs[1][j] = locs[1][j] - m2
            
            new_img[locs] = img[locs]
            
            iso_masks[count] = cv2.resize(new_img, (iso_masks_size, iso_masks_size))
            classes[count] = 1
            count += 1
            for j in range(num_neg):
                false_mask = create_false_mask(img, org_locs, locs, height, width)
                iso_masks[count] = cv2.resize(false_mask, (iso_masks_size, iso_masks_size))
                classes[count] = 0
                count += 1
        elif not to_ignore_neg:
            iso_masks[count] = img[100:228, 100:228]
            classes[count] = 0
            count += 1

    iso_masks = iso_masks[:count, :, :]
    classes = classes[:count]
    return iso_masks, classes

    
def create_iso_masks():
    if not os.path.exists(np_files_path):
        os.mkdir(np_files_path)
    imgs_train, imgs_mask_train = load_train_data()
    imgs_val, imgs_mask_val = load_val_data()
    
    iso_masks_train, train_classes = create_iso_masks_from_data(imgs_train, imgs_mask_train, 10)
    iso_masks_val, val_classes = create_iso_masks_from_data(imgs_val, imgs_mask_val, 1)
    iso_masks_train = np.expand_dims(iso_masks_train, axis=3)
    iso_masks_val = np.expand_dims(iso_masks_val, axis=3)
    np.save(np_files_path + 'iso_masks_train.npy', iso_masks_train)
    np.save(np_files_path + 'iso_masks_val.npy', iso_masks_val)
    np.save(np_files_path + 'iso_masks_train_classes.npy', train_classes)
    np.save(np_files_path + 'iso_masks_val_classes.npy', val_classes)


def create_test_data(test_data_path):
    if not os.path.exists(np_files_path):
        os.mkdir(np_files_path)

    test_data_path = os.path.join(test_data_path, 'test')
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows_resized, image_cols_resized), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(test_data_path, image_name), as_gray=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(np_files_path + 'imgs_test.npy', imgs)
    np.save(np_files_path + 'imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load(np_files_path + 'imgs_test.npy')
    imgs_id = np.load(np_files_path + 'imgs_id_test.npy')
    return imgs_test, imgs_id


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def get_object_existence(mask_array):
    return np.array([int(np.sum(mask_array[i, :, :]) > 0) for i in range(len(mask_array))])


def calc_pixel_sum(mask_train_pred):
    pixel_sum = np.zeros((len(mask_train_pred), 1))
    for k in range(len(mask_train_pred)):
        pixel_sum[k] = np.sum(mask_train_pred[k, :, :, :])
    return pixel_sum


def get_data_normalized(imgs, masks, mean, std):
    imgs = preprocess(imgs)
    masks = preprocess(masks)
    imgs = imgs.astype('float32')
    masks = masks.astype('float32')
    imgs -= mean
    imgs /= std
    masks /= 255. # scale masks to [0, 1]
    return imgs, masks


if __name__ == '__main__':
    TEST_DATA_PATH = 'E:/Machine Learning/git uploading/kaggle ultrasound challenge/inputs/test'
    create_test_data(TEST_DATA_PATH)

