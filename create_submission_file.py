from project_utils import *
from model import *
from global_params import *

model_classify = get_classification_model()
model_classify.load_weights(classification_model_weights_path + 'base_model_classify.h5')


def prep(img):
    img = img.astype('float32')
    img = (img > mask_presence_threshold).astype(np.uint8)  # threshold
    img = resize(img, (image_rows_resized, image_cols_resized), preserve_range=True)
    return img


def calc_pixel_sum(mask_train_pred):
    pixel_sum = np.zeros((len(mask_train_pred), 1))
    for k in range(len(mask_train_pred)):
        pixel_sum[k] = np.sum(mask_train_pred[k, :, :, :])
    return pixel_sum


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''

    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z + 1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s + 1, l + 1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


if __name__ == '__main__':
    imgs_test, imgs_id_test = load_test_data()
    imgs_mask_test = np.load(np_files_path + 'imgs_mask_test.npy')

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]
    imgs_mask_test = imgs_mask_test[argsort]
    masks_test_pred_post = np.zeros((len(imgs_test), image_rows_resized, image_cols_resized))
    org_imgs = np.zeros((len(imgs_test), image_rows_resized, image_cols_resized))
    for k in range(len(imgs_mask_test)):
        masks_test_pred_post[k, :, :] = cv2.resize(imgs_mask_test[k, :, :], (image_cols_resized, image_rows_resized))
        org_imgs[k, :, :] = cv2.resize(imgs_test[k, :, :], (image_cols_resized, image_rows_resized))

    masks_test_pred_post[masks_test_pred_post < mask_presence_threshold] = 0
    mask_test_iso, class1 = create_iso_masks_from_data(org_imgs, masks_test_pred_post, 0, to_ignore_neg=0)
    mask_test_iso = np.expand_dims(mask_test_iso, axis=3)
    mask_existance = model_classify.predict(mask_test_iso)
    total = imgs_test.shape[0]
    ids = []
    rles = []

    mask_for_pixel = copy.deepcopy(imgs_mask_test)
    mask_for_pixel[mask_for_pixel >= pixel_sum_calc_threshold] = 1
    mask_for_pixel[mask_for_pixel < pixel_sum_calc_threshold] = 0
    pp = calc_pixel_sum(mask_for_pixel)

    for i in range(total):
        img = imgs_mask_test[i, :, :, 0]
        if mask_existance[i] >= mask_presence_probability_threshold and pp[i] > mask_pixel_sum_min:
            img = prep(img)
        else:
            img = np.zeros(img.shape)
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')