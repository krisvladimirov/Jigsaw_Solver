from Network import JigNetwork as jn
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import cv2
from Network import PreprocessImage as pi


TRAIN_INFO_PATH = "src\\valid\\valid.csv"

# link to download H5 files :
# https://drive.google.com/drive/folders/1HP2oNFS9XHI8JDRvk8IU-fr7qYgBxZ8a?fbclid=IwAR02V6eqdlG4xUhzIQJ1WJ14v8dORlF4ZGYjAZX3kmP4j7BuH8iqfESlk7M
# ? not sure ? model = create_vgg_model() sample 100 ?[acc = 0.95]
H5_PATH_V1 = "jig_net_v1.h5"
# model = create_vgg_model() [acc = 0.74]
H5_PATH_V2 = "jig_net_v2.h5"
# model = create_resnet_self() [acc = 0.95]
H5_PATH_V3 = "jig_net_v3.h5"
# model = keras_resnet() [acc = 0.92]
H5_PATH_V4 = "jignet_v4_keras_resnet.h5"
# model = keras_resnet(), self_generator [acc = 0.50]
H5_PATH_V5 = "jignet_v5_keras_resnet_generator.h5"
# model = keras_resnet(), image rotation, 20 epoch, using same directory for valid and training [acc = 0.88]
H5_PATH_V6 = "jignet_v6_keras_resnet_generator_shuffle_index.h5"
# model = keras_resnet(), self_generator [5 cross validation] [acc = 0.99] but the result is not that good
H5_PATH_V7 = "jignet_v7_keras_resnet_self_generator.h5"
# model = keras_resnet(), keras generator, image preprocess, with 4 image sample, 10 epoch, train 240k straight
H5_PATH_V8 = "jignet_v8_keras_resnet_image_augmented_4image.h5"
# model = keras_resnet() random weights, keras generator, iumage preprocess, 4 imgage sample only,  10 epoch, train 120k 
H5_PATH_V9 = "jignet_v9_keras_resnet_image_augmented_4_only.h5"
# model = keras_resnet() with pretrain v4, keras generator, no flip, 4 image sample only,  10 epoch, train 120k
H5_PATH_V10 = "jignet_v10_keras_resnet_image_no_augmented_4_only.h5"
# model = keras_resnet() with pretrain v4, keras generator, iumage preprocess, 4 imgage sample only,  10 epoch, train 120k
H5_PATH_V11 = "jignet_v11_keras_resnet_image_augmented_4_only.h5"
# model = self keras_resnet, exactly same as v3, 2 sample
H5_PATH_V12 = "jignet_v12_keras_resnet_image_augmented_4_only.h5"
# model = self keras_resnet, only sample for 4
H5_PATH_V13 = "jignet_v12_keras_resnet_image_augmented_4_only.h5"
# model = keras_resnet, only sample for 2
H5_PATH_V14 = "jignet_v14_keras_resnet_2_only.h5"
# model = keras_resnet, only sample for 4, 120k samples [acc = 0.98]
H5_PATH_V15 = "jignet_v15_keras_resnet_4_only.h5"
# model = keras_resnet, all, 240k samples
H5_PATH_V16 = "jignet_v16_keras_resnet_all.h5"
# model = keras resnet, all 240k samples, crosss validation, self generator
H5_PATH_V17 = "jignet_v17_keras_resnet_all.h5"
# model = keras resnet, 240k samples, 4 only , crosss validation, self generator
H5_PATH_V18 = "jignet_v18_keras_resnet_4_only.h5"

delete_path = "delete.h5"

""" TEST PREPROCESS IMAGE """
# pi.create_image_info_csv('test4_black')
# pi.create_image_info_csv('train4_black')
# pi.create_image_info_csv('valid4_black')
# samples, labels = pi.generate_samples_4_with_black_part("src\\original\\test\\Places365_test_00010001.jpg", image_size=64)
# print(np.shape(samples))
# print(np.shape(labels))
# pi.save_generated_sample_4("test", "test4_black", with_black=True)
# pi.save_generated_sample_4("train", "train4_black", with_black=True)
# pi.save_generated_sample_4("valid", "valid4_black", with_black=True)
# pi.save_generated_sample_4("train2", "train")
# pi.save_generated_sample_4("valid", "valid")
# pi.save_generated_sample_4("train2", "train4")

""" TEST SINGLE IMAGE """
# model = jn.keras_resnet()
# model.load_weights(H5_PATH_V7)
# negative_image1 = cv2.imread("C:\\Users\\Mark\\Desktop\\JigsawNetwork\\PuzzleNetwork\\src\\test\\negative\\1041.jpg")
# negative_image2 = cv2.imread("C:\\Users\\Mark\\Desktop\\JigsawNetwork\\PuzzleNetwork\\src\\test\\negative\\1125.jpg")
# positive_image1 = cv2.imread("C:\\Users\\Mark\\Desktop\\JigsawNetwork\\PuzzleNetwork\\src\\test\\positive\\1125.jpg")
#
# prediction1 = jn.predict_image(negative_image1, model)
# prediction2 = jn.predict_image(negative_image2, model)
# prediction3 = jn.predict_image(positive_image1, model)
#
# print("Negative Image 1 = {0}".format(np.round(prediction1*100)))
# print("Negative Image 2 = {0}".format(np.round(prediction2*100)))
# print("Positive Image 1 = {0}".format(np.round(prediction3*100)))
#
# jn.test_model(model, H5_PATH_V7)


""" TRAINING BY USING DIFFERENT MODEL """
# model = jn.create_resnet_self()
# model = jn.create_vgg_model()
# model = jn.keras_resnet()
# jn.train_model_keras_generator(model, H5_PATH_V16)
# model.summary()
# jn.test_model(model, H5_PATH_V12, 2)
jn.cross_validation(H5_PATH_V17, "src\\train4_black\\train4_black.csv")
# train_generator = jn.image_generator(data, 2, train_index)
# valid_generator = jn.image_generator(data, 2, test_index)
# jn.train_model(model, H5_PATH_V6, train_generator, valid_generator)
# jn.cross_validation(model, H5_PATH_V7)
# test_model(model, h5_filepath)

# data = pd.read_csv("src\\train2\\train.csv")
# train_indices, test_indices = k_fold(data, 5)
# train_gen = image_generator(data, 2, train_indices)


""" RESULT """
# V1 (model = create_vgg_model())
# Epoch 1 - 299s 529ms/step - loss: 0.2755 - acc: 0.9228 - val_loss: 0.5681 - val_acc: 0.8000
# Epoch 2 - 101s 179ms/step - loss: 0.1957 - acc: 0.9389 - val_loss: 0.8319 - val_acc: 0.7000
# Epoch 3 - 101s 179ms/step - loss: 0.1668 - acc: 0.9457 - val_loss: 0.5805 - val_acc: 0.7500
# Epoch 4 - 102s 179ms/step - loss: 0.1561 - acc: 0.9517 - val_loss: 0.9471 - val_acc: 0.7500
# Epoch 5 - 101s 179ms/step - loss: 0.1467 - acc: 0.9509 - val_loss: 0.6128 - val_acc: 0.8800

# V2 (model = create_vgg_model())
# Epoch 1 - 4717s 629ms/step - loss: 0.7446 - acc: 0.7239 - val_loss: 0.6605 - val_acc: 0.7562
# Epoch 2 - 2567s 342ms/step - loss: 0.7283 - acc: 0.7424 - val_loss: 0.7229 - val_acc: 0.7750
# Epoch 3 - 2568s 342ms/step - loss: 0.7379 - acc: 0.7449 - val_loss: 0.7190 - val_acc: 0.7375
# Epoch 4 - 2560s 341ms/step - loss: 0.7361 - acc: 0.7469 - val_loss: 0.6464 - val_acc: 0.7875

# V3 (model = create_resnet_self())
# Epoch 5 - 2573s 343ms/step - loss: 0.7373 - acc: 0.7470 - val_loss: 0.9264 - val_acc: 0.7312
# Epoch 1 -
# Epoch 2 -
# Epoch 3 -
# Epoch 4 - 2570s 343ms/step - loss: 0.0957 - acc: 0.9662 - val_loss: 0.0780 - val_acc: 0.9875
# Epoch 5 - 2562s 342ms/step - loss: 0.0806 - acc: 0.9713 - val_loss: 0.1409 - val_acc: 0.943

# V4 - (model = keras_resnet()) Keras ResNet (with drop out 0.7)
# Epoch 1 - 2598s 346ms/step - loss: 1.1705 - acc: 0.5950 - val_loss: 3.1197 - val_acc: 0.4875
# Epoch 2 -
# Epoch 3 - 2589s 345ms/step - loss: 0.2050 - acc: 0.9254 - val_loss: 0.1812 - val_acc: 0.9500
# Epoch 4 - 2595s 346ms/step - loss: 0.1411 - acc: 0.9507 - val_loss: 0.0971 - val_acc: 0.9500
# Epoch 5 - 2589s 345ms/step - loss: 0.2050 - acc: 0.9254 - val_loss: 0.1812 - val_acc: 0.9500

# V5 - (model = keras_resnet()) Keras ResNet (with drop out 0.7) Same with V4 but using own image generator
# Epoch 1 - 1237s - loss: 1.2935 - acc: 0.5020 - val_loss: 0.6914 - val_acc: 0.5000
# Epoch 2 - 1223s - loss: 1.2153 - acc: 0.5024 - val_loss: 0.6957 - val_acc: 0.4938
# Epoch 3 - 1222s - loss: 1.1579 - acc: 0.5063 - val_loss: 0.6896 - val_acc: 0.5563
# Epoch 4 - 1223s - loss: 1.1478 - acc: 0.5057 - val_loss: 0.9780 - val_acc: 0.5125
# Epoch 5 - 1220s - loss: 1.2121 - acc: 0.5055 - val_loss: 0.6967 - val_acc: 0.5125

# V6.1 - (model = keras_resnet()) Keras ResNet (with drop out 0.7) Same with V4 but using own image generator 2
# Epoch 1 - 1237s - loss: 3.3508 - acc: 0.7920 - val_loss: 8.5627 - val_acc: 0.4688
# Epoch 2 - 1226s - loss: 9.6709 - acc: 0.4000 - val_loss: 8.1598 - val_acc: 0.4938
# Epoch 3 - 1226s - loss: 6.4472 - acc: 0.6000 - val_loss: 7.2531 - val_acc: 0.5500
# Epoch 4 -
# Epoch 5 -

# V6.2 - (model = keras_resnet()) Keras ResNet Same with V4 but using own image generator 2 permute the indices
# Epoch 1 - 1238s - loss: 1.2433 - acc: 0.5070 - val_loss: 1.2459 - val_acc: 0.4938
# Epoch 2 - 1224s - loss: 1.2391 - acc: 0.5004 - val_loss: 1.0605 - val_acc: 0.5000
# Epoch 3 - 1224s - loss: 1.1727 - acc: 0.5083 - val_loss: 0.6860 - val_acc: 0.5188
# Epoch 4 - 1223s - loss: 1.1703 - acc: 0.5096 - val_loss: 1.1749 - val_acc: 0.4875
# Epoch 5 - 1224s - loss: 1.1675 - acc: 0.5106 - val_loss: 1.7006 - val_acc: 0.5750

# V6.2 (stored) - (model = keras_resnet()) Keras ResNet Same with V4 but using imageDataGenerator split in one directory
# Epoch 1 - 2382s - loss: 0.9235 - acc: 0.6761 - val_loss: 1.0719 - val_acc: 0.5062
# Epoch 2 - 2171s - loss: 1.2201 - acc: 0.5026 - val_loss: 1.2918 - val_acc: 0.4875
# Epoch 3 - 2106s - loss: 1.1639 - acc: 0.5031 - val_loss: 1.1576 - val_acc: 0.5437
# Epoch 4 - 2102s - loss: 1.1553 - acc: 0.5005 - val_loss: 0.8780 - val_acc: 0.4750
# Epoch 5 - 2100s - loss: 1.1596 - acc: 0.4997 - val_loss: 1.0719 - val_acc: 0.4750
# Epoch 6 - 2101s - loss: 1.1595 - acc: 0.4997 - val_loss: 0.9686 - val_acc: 0.4813
# Epoch 7 - 2103s - loss: 1.1624 - acc: 0.5050 - val_loss: 1.2744 - val_acc: 0.4250
# Epoch 8 - 2103s - loss: 1.1795 - acc: 0.5048 - val_loss: 0.7817 - val_acc: 0.4625
# Epoch 9 - 2103s - loss: 1.1497 - acc: 0.5075 - val_loss: 0.9621 - val_acc: 0.5188
# Epoch 10- 2110s - loss: 1.1313 - acc: 0.5066 - val_loss: 0.8562 - val_acc: 0.5188
# Epoch 11- 2106s - loss: 1.1864 - acc: 0.5018 - val_loss: 0.8841 - val_acc: 0.5062
# Epoch 12- 2104s - loss: 1.1553 - acc: 0.5058 - val_loss: 0.7707 - val_acc: 0.4938
# Epoch 13- 2103s - loss: 1.1708 - acc: 0.5052 - val_loss: 1.1652 - val_acc: 0.4750
# Epoch 14- 2101s - loss: 1.1429 - acc: 0.5006 - val_loss: 1.1434 - val_acc: 0.5250
# Epoch 15- 2111s - loss: 1.1526 - acc: 0.5270 - val_loss: 0.8613 - val_acc: 0.5625
# Epoch 16- 2104s - loss: 0.8562 - acc: 0.8204 - val_loss: 0.2723 - val_acc: 0.9187
# Epoch 17- 2113s - loss: 0.6867 - acc: 0.8980 - val_loss: 0.3135 - val_acc: 0.9187
# Epoch 18- 2112s - loss: 0.6823 - acc: 0.8970 - val_loss: 0.3641 - val_acc: 0.9437
# Epoch 19- 2115s - loss: 0.4949 - acc: 0.8883 - val_loss: 0.1832 - val_acc: 0.9563
# Epoch 20

# V7 - (model = keras_resnet()) Keras ResNet Same with V4 but using using cross validation, self generator !!!!
# Epoch 1/20 - 1236s - loss: 1.1096 - acc: 0.6340 - val_loss: 0.2440 - val_acc: 0.9500
# Epoch 2/20 - 1221s - loss: 0.5783 - acc: 0.8336 - val_loss: 0.2180 - val_acc: 0.9437
# Epoch 3/20 - 1222s - loss: 0.1510 - acc: 0.9473 - val_loss: 0.0364 - val_acc: 0.9938
# Epoch 4/20 - 1222s - loss: 0.1248 - acc: 0.9563 - val_loss: 0.6257 - val_acc: 0.9500
# Epoch 5/20 - 1222s - loss: 0.0919 - acc: 0.9682 - val_loss: 0.0722 - val_acc: 0.9750
# Epoch 6/20 - 1222s - loss: 0.0851 - acc: 0.9706 - val_loss: 0.0552 - val_acc: 0.9812
# Epoch 7/20 - 1223s - loss: 0.0787 - acc: 0.9728 - val_loss: 0.0349 - val_acc: 0.9875
# Epoch 8/20 - 1223s - loss: 0.0700 - acc: 0.9766 - val_loss: 0.0259 - val_acc: 0.9875
# Epoch 9/20 - 1223s - loss: 0.0635 - acc: 0.9786 - val_loss: 0.2821 - val_acc: 0.9062
# Epoch 10/20 - 1222s - loss: 0.0617 - acc: 0.9792 - val_loss: 0.0293 - val_acc: 0.9875
# Epoch 11/20 - 1223s - loss: 0.0547 - acc: 0.9816 - val_loss: 0.1493 - val_acc: 0.9625
# Epoch 12/20 - 1224s - loss: 0.0509 - acc: 0.9821 - val_loss: 0.0496 - val_acc: 0.9688
# Epoch 13/20 - 1224s - loss: 0.0502 - acc: 0.9826 - val_loss: 0.0307 - val_acc: 0.9875
# Epoch 14/20 - 1224s - loss: 0.0538 - acc: 0.9818 - val_loss: 0.0434 - val_acc: 0.9875
# Epoch 15/20 - 1224s - loss: 0.0448 - acc: 0.9843 - val_loss: 0.0119 - val_acc: 0.9938
# Epoch 16/20 - 1224s - loss: 0.0475 - acc: 0.9833 - val_loss: 0.0160 - val_acc: 0.9938
# Epoch 17/20 - 1223s - loss: 0.0421 - acc: 0.9856 - val_loss: 0.1517 - val_acc: 0.9688
# Epoch 18/20 - 1222s - loss: 0.0396 - acc: 0.9860 - val_loss: 0.0412 - val_acc: 0.9875
# Epoch 19/20 - 1221s - loss: 0.0364 - acc: 0.9874 - val_loss: 0.0344 - val_acc: 0.9875
# Epoch 20/20 - 1221s - loss: 0.0376 - acc: 0.9880 - val_loss: 0.0279 - val_acc: 0.9812
# -------------------- Cross Validation 2 --------------------
# Epoch 1/20 - 1238s - loss: 0.0531 - acc: 0.9831 - val_loss: 0.0849 - val_acc: 0.9750
# Epoch 2/20 - 1226s - loss: 0.0428 - acc: 0.9857 - val_loss: 0.0238 - val_acc: 1.0000
# Epoch 3/20 - 1227s - loss: 0.0362 - acc: 0.9872 - val_loss: 0.0277 - val_acc: 0.9812
# Epoch 4/20 - 1227s - loss: 0.0308 - acc: 0.9895 - val_loss: 0.0039 - val_acc: 1.0000
# Epoch 5/20 - 1231s - loss: 0.0346 - acc: 0.9885 - val_loss: 0.0419 - val_acc: 0.9875
# Epoch 6/20 - 1230s - loss: 0.0467 - acc: 0.9843 - val_loss: 0.0412 - val_acc: 0.9875
# Epoch 7/20 - 1228s - loss: 0.0235 - acc: 0.9920 - val_loss: 0.0853 - val_acc: 0.9938
# Epoch 8/20 - 1228s - loss: 0.0206 - acc: 0.9926 - val_loss: 0.1034 - val_acc: 0.9812
# Epoch 9/20 - 1228s - loss: 0.0223 - acc: 0.9924 - val_loss: 0.1344 - val_acc: 0.9625
# Epoch 10/20 - 1228s - loss: 0.0173 - acc: 0.9942 - val_loss: 0.1149 - val_acc: 0.9688
# Epoch 11/20 - 1228s - loss: 0.0181 - acc: 0.9936 - val_loss: 0.0286 - val_acc: 0.9938
# Epoch 12/20 - 1228s - loss: 0.0164 - acc: 0.9943 - val_loss: 0.0017 - val_acc: 1.0000
# Epoch 13/20 - 1228s - loss: 0.0170 - acc: 0.9943 - val_loss: 0.0140 - val_acc: 0.9938
# Epoch 14/20 - 1355s - loss: 0.0152 - acc: 0.9950 - val_loss: 0.0456 - val_acc: 0.9875
# Epoch 15/20 - 1243s - loss: 0.0148 - acc: 0.9951 - val_loss: 0.1932 - val_acc: 0.9750
# Epoch 16/20 - 1243s - loss: 0.0148 - acc: 0.9951 - val_loss: 0.1932 - val_acc: 0.9750
# Epoch 16/20 - 1365s - loss: 0.0179 - acc: 0.9942 - val_loss: 0.0032 - val_acc: 1.0000
# Epoch 17/20 - 1438s - loss: 0.0123 - acc: 0.9959 - val_loss: 0.0243 - val_acc: 0.9938
# Epoch 18/20 - 1322s - loss: 0.0120 - acc: 0.9960 - val_loss: 0.0094 - val_acc: 0.9938
# Epoch 19/20 - 1225s - loss: 0.0124 - acc: 0.9960 - val_loss: 0.0649 - val_acc: 0.9875
# Epoch 20/20 - 1228s - loss: 0.0123 - acc: 0.9961 - val_loss: 0.0104 - val_acc: 0.9938
# ------------------- Cross Validation 3 --------------------
# Epoch 1/20 - 1375s - loss: 0.0312 - acc: 0.9905 - val_loss: 0.0084 - val_acc: 1.0000
# Epoch 2/20 - 1376s - loss: 0.0221 - acc: 0.9928 - val_loss: 0.0214 - val_acc: 0.9938
# Epoch 3/20 - 1388s - loss: 0.0187 - acc: 0.9940 - val_loss: 0.0131 - val_acc: 1.0000
# Epoch 4/20 - 1382s - loss: 0.0158 - acc: 0.9949 - val_loss: 0.0129 - val_acc: 0.9938
# Epoch 5/20 - 1381s - loss: 0.0150 - acc: 0.9949 - val_loss: 0.0144 - val_acc: 0.9938
# Epoch 6/20 - 1391s - loss: 0.0148 - acc: 0.9953 - val_loss: 0.0245 - val_acc: 0.9875
# Epoch 7/20 - 1387s - loss: 0.0137 - acc: 0.9958 - val_loss: 0.0204 - val_acc: 0.9938
# Epoch 8/20 - 1389s - loss: 0.0136 - acc: 0.9959 - val_loss: 0.0078 - val_acc: 0.9938
# Epoch 9/20 - 1379s - loss: 0.0124 - acc: 0.9961 - val_loss: 0.3821 - val_acc: 0.9125
# Epoch 10/20 - 1386s - loss: 0.0130 - acc: 0.9960 - val_loss: 0.1448 - val_acc: 0.9688
# Epoch 11/20 - 1390s - loss: 0.0131 - acc: 0.9959 - val_loss: 0.0095 - val_acc: 0.9938
# Epoch 12/20 - 1378s - loss: 0.0120 - acc: 0.9965 - val_loss: 0.0555 - val_acc: 0.9938
# Epoch 13/20 - 1379s - loss: 0.0126 - acc: 0.9965 - val_loss: 0.0052 - val_acc: 1.0000
# Epoch 14/20 - 1589s - loss: 0.0113 - acc: 0.9967 - val_loss: 0.0214 - val_acc: 0.9938
# Epoch 15/20 - 1398s - loss: 0.0112 - acc: 0.9968 - val_loss: 0.0775 - val_acc: 0.9812
# Epoch 16/20 - 1412s - loss: 0.0111 - acc: 0.9969 - val_loss: 0.1053 - val_acc: 0.9938
# Epoch 17/20 - 1397s - loss: 0.0107 - acc: 0.9969 - val_loss: 0.0564 - val_acc: 0.9938
# Epoch 18/20 - 1396s - loss: 0.0121 - acc: 0.9969 - val_loss: 0.0240 - val_acc: 0.9875
# Epoch 19/20 - 1416s - loss: 0.0100 - acc: 0.9972 - val_loss: 0.0696 - val_acc: 0.9938
# Epoch 20/20 - 1381s - loss: 0.0130 - acc: 0.9968 - val_loss: 0.0068 - val_acc: 0.9938
# -------------------- Cross Validation 4 --------------------
# Epoch 1/20 - 1366s - loss: 0.0242 - acc: 0.9931 - val_loss: 0.0106 - val_acc: 0.9938
# Epoch 2/20 - 1387s - loss: 0.0135 - acc: 0.9957 - val_loss: 2.1836e-05 - val_acc: 1.0000
# Epoch 3/20 - 1384s - loss: 0.0132 - acc: 0.9961 - val_loss: 0.0069 - val_acc: 1.0000
# Epoch 4/20 - 1389s - loss: 0.0117 - acc: 0.9965 - val_loss: 0.0070 - val_acc: 0.9938
# Epoch 5/20

# V8 - Keras Resnet, rotate image, 4 image pieces
# Epoch 1/10 - 12204s - loss: 1.2250 - acc: 0.5046 - val_loss: 0.8045 - val_acc: 0.5000
# Epoch 2/10 - 11124s - loss: 1.1984 - acc: 0.5074 - val_loss: 0.9728 - val_acc: 0.5586
# Epoch 3/10 - 10232s - loss: 1.1633 - acc: 0.5139 - val_loss: 1.9575 - val_acc: 0.5000
# Epoch 4/10 - 8746s - loss: 1.1748 - acc: 0.5014 - val_loss: 0.8639 - val_acc: 0.5547
# Epoch 5/10 - 7882s - loss: 1.1592 - acc: 0.5050 - val_loss: 1.4494 - val_acc: 0.4922
# Epoch 6/10 - 7893s - loss: 1.1762 - acc: 0.5050 - val_loss: 0.8389 - val_acc: 0.4492
# Epoch 7/10 - 7903s - loss: 1.1657 - acc: 0.5049 - val_loss: 0.7378 - val_acc: 0.5234
# Epoch 8/10 - 7902s - loss: 1.1221 - acc: 0.5088 - val_loss: 0.7926 - val_acc: 0.4766
# Epoch 9/10 - 9226s - loss: 1.1592 - acc: 0.5014 - val_loss: 1.0954 - val_acc: 0.5078
# Epoch 9/10 - 8803s - loss: 1.1658 - acc 0.5008 - val_loss: 0.7966 - val_acc: 0.5312

# V9 - Keras resnet, only 4 image
# Epoch 1/10 - 7707s - loss: 1.2400 - acc: 0.4994 - val_loss: 0.8643 - val_acc: 0.4904
# Epoch 2/10 - 5942s - loss: 1.1934 - acc: 0.5006 - val_loss: 0.9923 - val_acc: 0.5021
# Epoch 3/10 - 5991s - loss: 1.1640 - acc: 0.4996 - val_loss: 0.9758 - val_acc: 0.4983
# Epoch 4/10 - 5902s - loss: 1.1661 - acc: 0.5005 - val_loss: 0.8499 - val_acc: 0.4980
# Epoch 5/10 - 5776s - loss: 1.1589 - acc: 0.4998 - val_loss: 1.4372 - val_acc: 0.5004
# Epoch 6/10 - 5795s - loss: 1.1674 - acc: 0.4990 - val_loss: 0.9102 - val_acc: 0.4962
# Epoch 7/10 - 5772s - loss: 1.1511 - acc: 0.5007 - val_loss: 0.9394 - val_acc: 0.4985
# Epoch 8/10 - 5781s - loss: 1.1625 - acc: 0.4981 - val_loss: 0.8919 - val_acc: 0.5093
# Epoch 9/10 - 5847s - loss: 1.1586 - acc: 0.4992 - val_loss: 0.9459 - val_acc: 0.4965
# Epoch 10/10 - 5808s - loss: 1.1596 - acc: 0.5017 - val_loss: 0.9417 - val_acc: 0.4992

# V10.0 - Keras resnet only 4 imag, random negative image
# Epoch 1/5 - 6288s - loss: 1.5366 - acc: 0.5017 - val_loss: 0.7868 - val_acc: 0.4979
# Epoch 2/5 - 4148s - loss: 1.5198 - acc: 0.5025 - val_loss: 1.6815 - val_acc: 0.4953
# Epoch 3/5 - 4150s - loss: 1.4551 - acc: 0.5002 - val_loss: 0.8895 - val_acc: 0.5033


# V10.1 - Keras resnet only 4 image, no suffle, random negative image
# Epoch 1/5 - 4250s - loss: 1.5283 - acc: 0.5003 - val_loss: 1.2204 - val_acc: 0.4993
# Epoch 2/5 - 4060s - loss: 1.4570 - acc: 0.5027 - val_loss: 1.1466 - val_acc: 0.5083
# Epoch 3/5 - 4057s - loss: 1.4748 - acc: 0.5026 - val_loss: 0.9390 - val_acc: 0.4937
# Epoch 4/5 - 4061s - loss: 1.4723 - acc: 0.5012 - val_loss: 1.0614 - val_acc: 0.4950
# Epoch 5/5 - 4059s - loss: 1.4315 - acc: 0.5001 - val_loss: 1.7627 - val_acc: 0.4960

# V11.0 - Keras resnet
# Epoch 1/5 - 4195s - loss: 1.5145 - acc: 0.5008 - val_loss: 0.8428 - val_acc: 0.4959
# Epoch 2/5 - 4102s - loss: 1.4448 - acc: 0.4999 - val_loss: 0.6931 - val_acc: 0.5066
# Epoch 3/5 - 4119s - loss: 1.4367 - acc: 0.5012 - val_loss: 1.0098 - val_acc: 0.4988
# Epoch 4/5

# V12.0 - self keras resnet - sample 4 jignet_v12_self_resnet_4_only.h5
# Epoch 1/5 - 4187s - loss: 0.8504 - acc: 0.5218 - val_loss: 0.7316 - val_acc: 0.6390
# Epoch 2/5 - 4095s - loss: 0.3788 - acc: 0.8412 - val_loss: 0.7943 - val_acc: 0.6037
# Epoch 3/5 - 4129s - loss: 0.3535 - acc: 0.8550 - val_loss: 0.7743 - val_acc: 0.5967
# Epoch 4/5 - 4085s - loss: 0.3474 - acc: 0.8580 - val_loss: 0.7637 - val_acc: 0.5562
# Epoch 5/5 - 4124s - loss: 0.3435 - acc: 0.8601 - val_loss: 0.9903 - val_acc: 0.5382
# Evaluation  0.5042464681943688

# V12.0 - self keras resnet - sample 2
# Epoch 1/10- 4144s - loss: 0.6098 - acc: 0.7054 - val_loss: 0.8656 - val_acc: 0.4733
# Epoch 2/10 - 4049s - loss: 0.3617 - acc: 0.8510 - val_loss: 0.8040 - val_acc: 0.6305
# Epoch 3/10 - 4044s - loss: 0.3475 - acc: 0.8577 - val_loss: 0.7651 - val_acc: 0.6063
# Epoch 4/10 - 4057s - loss: 0.3412 - acc: 0.8612 - val_loss: 0.9040 - val_acc: 0.5495
# Epoch 5/10 - 4058s - loss: 0.3359 - acc: 0.8634 - val_loss: 0.9833 - val_acc: 0.5163
# Epoch 6/10 - 4061s - loss: 0.3342 - acc: 0.8645 - val_loss: 1.2147 - val_acc: 0.5355
# Epoch 7/10 - 4054s - loss: 0.3311 - acc: 0.8655 - val_loss: 0.8610 - val_acc: 0.6089
# Epoch 8/10 - 6702s - loss: 0.3300 - acc: 0.8664 - val_loss: 0.8246 - val_acc: 0.5938
# Epoch 9/10 - 4078s - loss: 0.3281 - acc: 0.8673 - val_loss: 1.0768 - val_acc: 0.5951
# Epoch 10/10
# Evaludation 0.78

# V14
# Epoch 1/10 - 4760s - loss: 1.1271 - acc: 0.5076 - val_loss: 0.7260 - val_acc: 0.5048
# Epoch 2/10 - 2120s - loss: 1.1296 - acc: 0.5009 - val_loss: 1.1258 - val_acc: 0.4962
# Epoch 3/10 - 2113s - loss: 1.1055 - acc: 0.5087 - val_loss: 0.9609 - val_acc: 0.5459

# V14 keras_resnet, only sample for 2
# Epoch 1/10 - 2096s - loss: 0.4546 - acc: 0.8266 - val_loss: 0.1997 - val_acc: 0.9217
# Epoch 2/10 - 2065s - loss: 0.1472 - acc: 0.9459 - val_loss: 0.5275 - val_acc: 0.9090
# Epoch 3/10 - 2063s - loss: 0.1100 - acc: 0.9603 - val_loss: 0.1292 - val_acc: 0.9507
# Epoch 4/10 - 2059s - loss: 0.1161 - acc: 0.9591 - val_loss: 0.1058 - val_acc: 0.9598
# Epoch 5/10 - 2069s - loss: 0.0935 - acc: 0.9662 - val_loss: 0.1669 - val_acc: 0.9470
# Epoch 6/10 - 2074s - loss: 0.0799 - acc: 0.9714 - val_loss: 0.0778 - val_acc: 0.9729
# Epoch 7/10 - 2066s - loss: 0.0760 - acc: 0.9735 - val_loss: 0.0799 - val_acc: 0.9702
# Epoch 8/10 - 2066s - loss: 0.0660 - acc: 0.9765 - val_loss: 0.1629 - val_acc: 0.9439
# Epoch 9/10 - 2066s - loss: 0.0618 - acc: 0.9778 - val_loss: 0.0736 - val_acc: 0.9724
# Epoch 10/10 - 2069s - loss: 0.0603 - acc: 0.9787 - val_loss: 0.0831 - val_acc: 0.9705
# Accuracy =  0.9854409289042684

# V15.0 keras resnet, only sample for 4 - wrong samples
# Epoch 1/10 - 2132s - loss: 0.9836 - acc: 0.4992 - val_loss: 0.7058 - val_acc: 0.5076
# Epoch 2/10 - 2067s - loss: 0.6930 - acc: 0.5130 - val_loss: 0.6933 - val_acc: 0.5168
# Epoch 3/10 - 2066s - loss: 0.6812 - acc: 0.5533 - val_loss: 0.8357 - val_acc: 0.4700
# Epoch 4/10 - 2087s - loss: 0.6259 - acc: 0.6233 - val_loss: 0.8550 - val_acc: 0.4277
# Epoch 5/10 - 2080s - loss: 0.5830 - acc: 0.6618 - val_loss: 0.9418 - val_acc: 0.4495
# Epoch 6/10 - 2072s - loss: 0.5599 - acc: 0.6888 - val_loss: 1.0110 - val_acc: 0.4543
# Epoch 7/10 - 2064s - loss: 0.5320 - acc: 0.7175 - val_loss: 1.0310 - val_acc: 0.4834
# Epoch 8/10 - 2071s - loss: 0.4963 - acc: 0.7493 - val_loss: 1.0448 - val_acc: 0.4998
# Epoch 9/10 - 2063s - loss: 0.4737 - acc: 0.7683 - val_loss: 1.2697 - val_acc: 0.4767
# Epoch 10/10 - 2070s - loss: 0.4601 - acc: 0.7796 - val_loss: 1.2462 - val_acc: 0.4964
# Accuracy =  0.6869384360789459

# V15 keras resnet 4 piece image only
# Epoch 1/10 - 4715s - loss: 0.5428 - acc: 0.7867 - val_loss: 0.2149 - val_acc: 0.9235
# Epoch 2/10 - 2099s - loss: 0.1261 - acc: 0.9540 - val_loss: 0.1176 - val_acc: 0.9565
# Epoch 3/10 - 2096s - loss: 0.0984 - acc: 0.9651 - val_loss: 0.1029 - val_acc: 0.9639
# Epoch 4/10 - 2093s - loss: 0.0771 - acc: 0.9731 - val_loss: 0.1265 - val_acc: 0.9568
# Epoch 5/10 - 2100s - loss: 0.0633 - acc: 0.9777 - val_loss: 0.0682 - val_acc: 0.9746
# Epoch 6/10 - 2093s - loss: 0.0550 - acc: 0.9808 - val_loss: 0.0684 - val_acc: 0.9764
# Epoch 7/10 - 2098s - loss: 0.0493 - acc: 0.9826 - val_loss: 0.0752 - val_acc: 0.9747
# Epoch 8/10 - 2105s - loss: 0.0454 - acc: 0.9842 - val_loss: 0.0499 - val_acc: 0.9829
# Epoch 9/10 - 2099s - loss: 0.0412 - acc: 0.9858 - val_loss: 0.0488 - val_acc: 0.9827
# Epoch 10/10 - 2095s - loss: 0.0383 - acc: 0.9865 - val_loss: 0.0575 - val_acc: 0.9793
# Accuracy =  0.9856905126928688

# V16 keras resnet 2 and 4 image
# Epoch 1/10 - 12453s - loss: 0.4704 - acc: 0.8275 - val_loss: 0.2240 - val_acc: 0.9220
# Epoch 2/10 - 8207s - loss: 0.1512 - acc: 0.9438 - val_loss: 0.1162 - val_acc: 0.9573
# Epoch 3/10 - 8205s - loss: 0.1072 - acc: 0.9621 - val_loss: 0.1143 - val_acc: 0.9580
# Epoch 4/10 - 8229s - loss: 0.0846 - acc: 0.9698 - val_loss: 0.1542 - val_acc: 0.9418
# Epoch 5/10 - 8234s - loss: 0.0747 - acc: 0.9736 - val_loss: 0.0776 - val_acc: 0.9731
# Epoch 6/10 - 8225s - loss: 0.0661 - acc: 0.9770 - val_loss: 0.0814 - val_acc: 0.9707
# Epoch 7/10 - 8276s - loss: 0.0624 - acc: 0.9780 - val_loss: 0.0735 - val_acc: 0.9749
# Epoch 8/10 - 8241s - loss: 0.0579 - acc: 0.9797 - val_loss: 0.0753 - val_acc: 0.9721
# Epoch 9/10 - 8237s - loss: 0.0546 - acc: 0.9808 - val_loss: 0.0661 - val_acc: 0.9763
# Epoch 10/10 - 8237s - loss: 0.0522 - acc: 0.9817 - val_loss: 0.1238 - val_acc: 0.9574
# TODO evaluate this

# V17 keras resnet self generator 2 and 4 image
# -------------------- Cross Validation 1 --------------------
# Epoch 1/10 - 3614s - loss: 0.7035 - acc: 0.6358 - val_loss: 0.2373 - val_acc: 0.9110
# Epoch 2/10 - 6419s - loss: 0.1491 - acc: 0.9450 - val_loss: 0.1230 - val_acc: 0.9550
# Epoch 3/10 - 1738s - loss: 0.0875 - acc: 0.9692 - val_loss: 0.1019 - val_acc: 0.9623
# Epoch 4/10 - 1739s - loss: 0.0683 - acc: 0.9766 - val_loss: 0.0658 - val_acc: 0.9775
# Epoch 5/10 - 1739s - loss: 0.0614 - acc: 0.9790 - val_loss: 0.0888 - val_acc: 0.9711
# Epoch 6/10 - 1738s - loss: 0.0520 - acc: 0.9821 - val_loss: 0.0713 - val_acc: 0.9760
# Epoch 7/10 - 1739s - loss: 0.0477 - acc: 0.9835 - val_loss: 0.0732 - val_acc: 0.9752
# Epoch 8/10 - 1739s - loss: 0.0402 - acc: 0.9860 - val_loss: 0.0753 - val_acc: 0.9755
# Epoch 9/10 - 1740s - loss: 0.0361 - acc: 0.9875 - val_loss: 0.0636 - val_acc: 0.9798
# Epoch 10/10 - 1741s - loss: 0.0324 - acc: 0.9889 - val_loss: 0.0720 - val_acc: 0.9790
# Accuracy =  0.9790295670301603
# -------------------- Cross Validation 2 --------------------
# Epoch 1/10 - 1747s - loss: 0.5597 - acc: 0.8072 - val_loss: 0.1950 - val_acc: 0.9317
# Epoch 2/10 - 1736s - loss: 0.1318 - acc: 0.9526 - val_loss: 0.2464 - val_acc: 0.9148
# Epoch 3/10 - 1737s - loss: 0.0855 - acc: 0.9701 - val_loss: 0.1949 - val_acc: 0.9364
# Epoch 4/10 - 1738s - loss: 0.0663 - acc: 0.9772 - val_loss: 0.1074 - val_acc: 0.9668
# Epoch 5/10 - 1738s - loss: 0.0578 - acc: 0.9804 - val_loss: 0.0841 - val_acc: 0.9721
# Epoch 6/10 - 1738s - loss: 0.0497 - acc: 0.9829 - val_loss: 0.1134 - val_acc: 0.9613
# Epoch 7/10 - 1739s - loss: 0.0432 - acc: 0.9854 - val_loss: 0.0808 - val_acc: 0.9735
# Epoch 8/10 - 1740s - loss: 0.0381 - acc: 0.9868 - val_loss: 0.0903 - val_acc: 0.9709
# Epoch 9/10 - 1741s - loss: 0.0336 - acc: 0.9887 - val_loss: 0.0737 - val_acc: 0.9777
# Epoch 10/10 - 1741s - loss: 0.0301 - acc: 0.9896 - val_loss: 0.0928 - val_acc: 0.9716
# Accuracy =  0.9715743397534365
# -------------------- Cross Validation 3 --------------------
# Epoch 1/10 - 1753s - loss: 0.7709 - acc: 0.6448 - val_loss: 0.3229 - val_acc: 0.8851
# Epoch 2/10 - 1741s - loss: 0.1628 - acc: 0.9400 - val_loss: 0.2073 - val_acc: 0.9253
# Epoch 3/10 - 1741s - loss: 0.0991 - acc: 0.9645 - val_loss: 0.1324 - val_acc: 0.9519
# Epoch 4/10 - 1740s - loss: 0.0738 - acc: 0.9741 - val_loss: 0.0878 - val_acc: 0.9704
# Epoch 5/10 - 1741s - loss: 0.0638 - acc: 0.9778 - val_loss: 0.0710 - val_acc: 0.9760
# Epoch 6/10 - 1741s - loss: 0.0552 - acc: 0.9807 - val_loss: 0.1033 - val_acc: 0.9679
# Epoch 7/10 - 1741s - loss: 0.0482 - acc: 0.9834 - val_loss: 0.1023 - val_acc: 0.9689
# Epoch 8/10 - 1742s - loss: 0.0463 - acc: 0.9842 - val_loss: 0.0643 - val_acc: 0.9782
# Epoch 9/10 - 1742s - loss: 0.0393 - acc: 0.9867 - val_loss: 0.0642 - val_acc: 0.9785
# Epoch 10/10 - 1743s - loss: 0.0351 - acc: 0.9881 - val_loss: 0.0686 - val_acc: 0.9783
# Accuracy =  0.9783215284943333
# -------------------- Cross Validation 4 --------------------
# Epoch 1/10 - 1755s - loss: 0.5183 - acc: 0.7481 - val_loss: 0.1923 - val_acc: 0.9245
# Epoch 2/10 - 1742s - loss: 0.1267 - acc: 0.9544 - val_loss: 0.2099 - val_acc: 0.9267
# Epoch 3/10 - 1744s - loss: 0.0776 - acc: 0.9727 - val_loss: 0.1249 - val_acc: 0.9538
# Epoch 4/10 - 1744s - loss: 0.0632 - acc: 0.9784 - val_loss: 0.0664 - val_acc: 0.9774
# Epoch 5/10 - 1745s - loss: 0.0555 - acc: 0.9806 - val_loss: 0.0547 - val_acc: 0.9817
# Epoch 6/10 - 1745s - loss: 0.0481 - acc: 0.9834 - val_loss: 0.0511 - val_acc: 0.9832
# Epoch 7/10 - 1745s - loss: 0.0431 - acc: 0.9852 - val_loss: 0.0645 - val_acc: 0.9779
# Epoch 8/10 - 1745s - loss: 0.0387 - acc: 0.9866 - val_loss: 0.0802 - val_acc: 0.9732
# Epoch 9/10 - 1744s - loss: 0.0356 - acc: 0.9878 - val_loss: 0.2101 - val_acc: 0.9257
# Epoch 10/10 - 1744s - loss: 0.0312 - acc: 0.9892 - val_loss: 0.2265 - val_acc: 0.9206
# Accuracy =  0.920512285544345
# -------------------- Cross Validation 5 --------------------
# Epoch 1/10 - 1754s - loss: 0.5555 - acc: 0.8335 - val_loss: 0.1929 - val_acc: 0.9276
# Epoch 2/10 - 1742s - loss: 0.1322 - acc: 0.9519 - val_loss: 0.1241 - val_acc: 0.9535
# Epoch 3/10 - 1743s - loss: 0.0833 - acc: 0.9707 - val_loss: 0.0790 - val_acc: 0.9728
# Epoch 4/10 - 1744s - loss: 0.0672 - acc: 0.9765 - val_loss: 0.0778 - val_acc: 0.9727
# Epoch 5/10 - 1744s - loss: 0.0576 - acc: 0.9806 - val_loss: 0.0940 - val_acc: 0.9681
# Epoch 6/10 - 1778s - loss: 0.0507 - acc: 0.9825 - val_loss: 0.0739 - val_acc: 0.9745
# Epoch 7/10 - 1751s - loss: 0.0488 - acc: 0.9843 - val_loss: 0.0933 - val_acc: 0.9714
# Epoch 8/10 - 1744s - loss: 0.0483 - acc: 0.9836 - val_loss: 0.0654 - val_acc: 0.9798
# Epoch 9/10 - 1742s - loss: 0.0393 - acc: 0.9868 - val_loss: 0.0749 - val_acc: 0.9754
# Epoch 10/10 - 1746s - loss: 0.0337 - acc: 0.9885 - val_loss: 0.0609 - val_acc: 0.9799
# Accuracy =  0.9799250271706619
# Average

# V18 Keras resnet, self generator, 4 only with black part
# -------------------- Cross Validation 1 --------------------
# Epoch 1/10 - 4548s - loss: 0.4889 - acc: 0.7517 - val_loss: 0.2050 - val_acc: 0.9230
# Epoch 2/10 - 1732s - loss: 0.1482 - acc: 0.9441 - val_loss: 0.1336 - val_acc: 0.9485
# Epoch 3/10 - 1738s - loss: 0.1119 - acc: 0.9588 - val_loss: 0.1116 - val_acc: 0.9570
# Epoch 4/10 - 1744s - loss: 0.0972 - acc: 0.9647 - val_loss: 0.1310 - val_acc: 0.9523
# Epoch 5/10 - 1734s - loss: 0.0827 - acc: 0.9699 - val_loss: 2.1788 - val_acc: 0.7424
# Epoch 6/10 - 1732s - loss: 0.0743 - acc: 0.9730 - val_loss: 0.0971 - val_acc: 0.9639
# Epoch 7/10 - 1731s - loss: 0.0660 - acc: 0.9760 - val_loss: 0.0945 - val_acc: 0.9647
# Epoch 8/10 - 1731s - loss: 0.0589 - acc: 0.9782 - val_loss: 3.8652 - val_acc: 0.7394
# Epoch 9/10 - 1729s - loss: 0.0564 - acc: 0.9790 - val_loss: 3.8969 - val_acc: 0.7431
# Epoch 10/10 - 1729s - loss: 0.0506 - acc: 0.9811 - val_loss: 0.1387 - val_acc: 0.9513
# Accuracy =  0.9512911249279529
# -------------------- Cross Validation 2 --------------------
# Epoch 1/10 - 1742s - loss: 0.4865 - acc: 0.7580 - val_loss: 0.2557 - val_acc: 0.9060
# Epoch 2/10 - 1731s - loss: 0.1486 - acc: 0.9440 - val_loss: 0.1334 - val_acc: 0.9490
# Epoch 3/10 - 1731s - loss: 0.1136 - acc: 0.9581 - val_loss: 0.1387 - val_acc: 0.9472
# Epoch 4/10 - 1731s -loss: 0.0957 - acc: 0.9652 - val_loss: 0.1453 - val_acc: 0.9504
# Epoch 5/10 - 1731s - loss: 0.0813 - acc: 0.9702 - val_loss: 0.2048 - val_acc: 0.9275
# Epoch 6/10 - 1731s - loss: 0.0740 - acc: 0.9732 - val_loss: 0.1223 - val_acc: 0.9581
# Epoch 7/10 - 1732s - loss: 0.0672 - acc: 0.9753 - val_loss: 0.0999 - val_acc: 0.9671
# Epoch 8/10 - 1732s - loss: 0.0591 - acc: 0.9785 - val_loss: 0.0923 - val_acc: 0.9686
# Epoch 9/10 - 1732s - loss: 0.0528 - acc: 0.9807 - val_loss: 0.1436 - val_acc: 0.9518
# Epoch 10/10 - 1732s - loss: 0.0486 - acc: 0.9825 - val_loss: 3.4321 - val_acc: 0.7370
# Accuracy =  0.7368596399863131
# -------------------- Cross Validation 3 --------------------
# Epoch 1/10 - 1741s - loss: 0.7446 - acc: 0.6206 - val_loss: 0.4767 - val_acc: 0.7889
# Epoch 2/10 - 1730s - loss: 0.1735 - acc: 0.9332 - val_loss: 0.1516 - val_acc: 0.9434
# Epoch 3/10 - 1730s - loss: 0.1176 - acc: 0.9565 - val_loss: 0.1385 - val_acc: 0.9457
# Epoch 4/10 - 1731s - loss: 0.0984 - acc: 0.9642 - val_loss: 0.1891 - val_acc: 0.9318
# Epoch 5/10 - 1730s - loss: 0.0857 - acc: 0.9687 - val_loss: 0.0986 - val_acc: 0.9657
# Epoch 6/10 - 1730s - loss: 0.0747 - acc: 0.9727 - val_loss: 0.2689 - val_acc: 0.9090
# Epoch 7/10 - 1731s - loss: 0.0666 - acc: 0.9757 - val_loss: 0.0998 - val_acc: 0.9666
# Epoch 8/10 - 1731s - loss: 0.0595 - acc: 0.9782 - val_loss: 0.2008 - val_acc: 0.9410
# Epoch 9/10 - 1731s - loss: 0.0536 - acc: 0.9808 - val_loss: 0.0874 - val_acc: 0.9688
# Epoch 10/10 - 1731s - loss: 0.0507 - acc: 0.9816 - val_loss: 0.5300 - val_acc: 0.8499
# Accuracy =  0.8497917520061526
# -------------------- Cross Validation 4 --------------------
# Epoch 1/10
