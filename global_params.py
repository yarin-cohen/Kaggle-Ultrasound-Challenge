# data params
img_rows = 80
img_cols = 112
image_rows_resized = 420
image_cols_resized = 580
iso_masks_size = 128

# operation params
train_classification = 1

# directories params
seg_model_weights_path = 'enter path here'
classification_model_weights_path = 'enter path here'
predictions_dir = 'enter path here'
np_files_path = 'enter path here'

# model params
classification_model_batch_size = 32
classification_model_epochs = 5
classification_model_lr = 1e-5

seg_model_batch_size = 64
seg_model_epochs = 50
seg_model_lr = 1e-3

# loss functions params
dice_coeff_smooth_factor = 1
focal_loss_gamma = 7
focal_loss_alpha = 0.9

# mask existence thresholds
mask_presence_threshold = 0.3
mask_presence_probability_threshold = 0.5
pixel_sum_calc_threshold = 0.1
mask_pixel_sum_min = 100
corners_limit_coef = 0.9


