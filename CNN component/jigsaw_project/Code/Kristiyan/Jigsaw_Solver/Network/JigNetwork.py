import keras
import cv2
import numpy as np
import pandas as pd
from Network import MyImageGenerator
from Network.PreprocessImage import count_number_file
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Add, Activation, ZeroPadding2D, BatchNormalization,\
    Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from keras.initializers import glorot_uniform
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from skimage.transform import resize


# Global variable

# TEMP_PATH = "src\\temp"
#
# TRAIN_PATH = "src\\train"
# VALID_PATH = "src\\valid"
# TEST_PATH = "src\\test"
#
# TRAIN2_PATH = "src\\train2"
# VALID2_PATH = "src\\valid2"
# TEST2_PATH = "src\\test2"
#
# TRAIN4_PATH = "src\\train4"
# VALID4_PATH = "src\\valid4"
# TEST4_PATH = "src\\test4"
#
# TRAIN_INFO_PATH = "src\\train\\train.csv"
# TEMP_INFO_PATH = "src\\temp\\temp.csv"
#
#
# CLASSES = ['negative', 'positive']
# BS = 20
# EPOCH = 10
# PST_IMG, NGT_IMG = count_number_file(TRAIN_PATH)
# PST_IMG_VALID, NGT_IMG_VALID = count_number_file(VALID_PATH)
# VALIDATION_SPLIT = 0.2
# STEP_PER_EPOCH = ((PST_IMG + NGT_IMG) // BS)
# STEP_PER_EPOCH_VALID = ((PST_IMG_VALID + NGT_IMG_VALID) // BS)
#
# use this to create resnet model
def keras_resnet(input_shape=(64, 64, 3), number_classes=2):
    """
    Using Keras ResNet50 model architecture and append last layer.
    :param input_shape: size of input images
    :param number_classes: number output
    :return: neural network model architecture
    """
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape, pooling='avg')
    x = base_model.output
    # x = Dropout(0.7)(x)
    predictions = Dense(number_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

#
# def convolutional_block(X, f, filters, stage, block, s=2):
#     """
#     Implementation of the convolutional block as defined in Figure 4
#
#     Arguments:
#     X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     f -- integer, specifying the shape of the middle CONV's window for the main path
#     filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#     stage -- integer, used to name the layers, depending on their position in the network
#     block -- string/character, used to name the layers, depending on their position in the network
#     s -- Integer, specifying the stride to be used
#
#     Returns:
#     X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
#     """
#
#     # defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     # Retrieve Filters
#     F1, F2, F3 = filters
#
#     # Save the input value
#     X_shortcut = X
#
#     ##### MAIN PATH #####
#     # First component of main path
#     X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid", name=conv_name_base + '2a',
#                kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
#     X = Activation('relu')(X)
#
#     # Second component of main path
#     X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), name=conv_name_base + '2b', padding="same",
#                kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
#     X = Activation('relu')(X)
#
#     # Third component of main path
#     X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c', padding="valid",
#                kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
#
#     ##### SHORTCUT PATH ####
#     X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '1', padding="valid",
#                         kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
#     X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
#
#     # Final step: Add shortcut value to main path, and pass it through a RELU activation
#     X = Add()([X_shortcut, X])
#     X = Activation("relu")(X)
#
#     return X
#
#
# def identity_block(X, f, filters, stage, block):
#     """
#     Implementation of the identity block as defined in Figure 3
#
#     Arguments:
#     X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     f -- integer, specifying the shape of the middle CONV's window for the main path
#     filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#     stage -- integer, used to name the layers, depending on their position in the network
#     block -- string/character, used to name the layers, depending on their position in the network
#
#     Returns:
#     X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
#     """
#
#     # defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     # Retrieve Filters
#     F1, F2, F3 = filters
#
#     # Save the input value. You'll need this later to add back to the main path.
#     X_shortcut = X
#
#     # First component of main path
#     X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
#                kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
#     X = Activation('relu')(X)
#
#     # Second component of main path
#     X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
#                kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
#     X = Activation('relu')(X)
#
#     # Third component of main path
#     X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=conv_name_base + '2c',
#                kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
#
#     # Final step: Add shortcut value to main path, and pass it through a RELU activation
#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)
#
#     return X
#
#
# def create_resnet_self(input_shape=(64, 64, 3), classes=2):
#     X_input = Input(input_shape)
#
#     # Add 3 padding around the input image
#     X = ZeroPadding2D((3, 3))(X_input)
#
#     # Stage 1 -
#     X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name='bn_conv1')(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D((3, 3), strides=(2, 2))(X)
#
#     # Stage 2
#     X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
#     X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
#     X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
#
#     # Stage 3
#     X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
#     X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='b')
#     X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='c')
#     X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='d')
#
#     # Stage 4
#     X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
#     X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='b')
#     X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='c')
#     X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='d')
#     X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='e')
#     X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='f')
#
#     # Stage 5
#     X = convolutional_block(X, f=3, filters=[256, 256, 2048], stage=5, block='a', s=3)
#     X = identity_block(X, 3, filters=[256, 256, 2048], stage=5, block='b')
#     X = identity_block(X, 3, filters=[256, 256, 2048], stage=5, block='c')
#
#     # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
#     X = AveragePooling2D((2, 2), name='avg_pool')(X)
#
#     # TODO: try drop out
#     # output layer
#     X = Flatten()(X)
#     X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
#
#     # Create model
#     model = Model(inputs=X_input, outputs=X, name='ResNet50')
#     return model
#
#
def create_vgg_model(input_shape=(64, 64, 3),  classes=2):
    # Generate a model with all layers (with top)
    vgg16 = keras.applications.vgg16.VGG16(weights=None, include_top=True, input_shape=input_shape)
    x = vgg16.layers[-2].output
    x = Dropout(0.7)(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Then create the corresponding model
    my_model = Model(inputs=vgg16.input, outputs=x)
    return my_model
#
#
# # TODO remove valid generatr (combine valid and train to produce larger dataset for training)
# def train_model(model, h5_file_path, train_generator, valid_generator, test_generator, train_n, valid_n):
#     """
#     Train the model by given a model structure and path of h5 file going to save.
#     :param model: model architecture (ResNet, VGG or others)
#     :param h5_file_path: file location to store your weight of the model
#     :param train_generator
#     :param valid_generator
#     :param test_generator
#     :return:
#     """
#     model.compile(Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])
#
#     # steps_per_epoch  value as the total number of training data points (490) divided by the batch size (30).
#     # Once Keras hits this step count it knows that it’s a new epoch.
#     model.fit_generator(train_generator,
#                         steps_per_epoch=train_n//BS,
#                         validation_data=valid_generator,
#                         validation_steps=valid_n//BS,
#                         epochs=EPOCH,
#                         verbose=2)
#
#     # save trained result (weight) in to a given file path
#     model.save(h5_file_path)
#
#     score = model.evaluate_generator(test_generator, steps=(valid_n//BS)+1, verbose=2)
#     print("Accuracy = ", score[1])
#     return score[1]
#
#
# def train_model_keras_generator_validation_split(model, h5_file_path):
#     idg = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rescale=0.2, validation_split=VALIDATION_SPLIT)
#
#     train_batches = ImageDataGenerator().flow_from_directory(TRAIN_PATH, target_size=(64, 64), classes=CLASSES,
#                                                              batch_size=BS, shuffle=True, subset="training")
#
#     valid_batches = ImageDataGenerator().flow_from_directory(TRAIN_PATH, target_size=(64, 64), classes=CLASSES,
#                                                              batch_size=BS, shuffle=True, subset="validation")
#
#     test_batches = ImageDataGenerator().flow_from_directory(TEST_PATH, target_size=(64, 64), classes=CLASSES,
#                                                             batch_size=10, shuffle=True)
#
#     train_model(model, h5_file_path, train_batches, valid_batches, test_batches, train_batches.n, valid_batches.n)
#
#
# def train_model_keras_generator(model, h5_file_path, sample_type=0):
#     """
#     Train the network regarding the sample type and store the weight in given path.
#     :param model: model architecture
#     :param h5_file_path: name of the h5 file
#     :param sample_type: type of sample, 2 indicates 2 connected image, 4 indicates 4 connected image,
#     0 indicates both.
#     :return:
#     """
#     if sample_type == 2:
#         train_path = TRAIN2_PATH
#         valid_path = VALID2_PATH
#         test_path = TEST2_PATH
#     elif sample_type == 4:
#         train_path = TRAIN4_PATH
#         valid_path = VALID4_PATH
#         test_path = TEST4_PATH
#     else:
#         train_path = TRAIN_PATH
#         valid_path = VALID_PATH
#         test_path = TEST_PATH
#     # idg = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rescale=0.2)
#
#     train_batches = ImageDataGenerator(horizontal_flip=True).flow_from_directory(train_path,
#                                                                                  target_size=(64, 64),
#                                                                                  classes=CLASSES,
#                                                                                  batch_size=BS)
#
#     valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
#                                                              target_size=(64, 64),
#                                                              classes=CLASSES,
#                                                              batch_size=BS,
#                                                              shuffle=False)
#
#     test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(64, 64), classes=CLASSES,
#                                            batch_size=BS, shuffle=False)
#
#     train_model(model, h5_file_path, train_batches, valid_batches, test_batches, train_batches.n, valid_batches.n)
#
#
# def cross_validation(h5_file_path, csv_file_path= TRAIN_INFO_PATH, k=5):
#     # retrieve data from training info csv file.
#     # idg = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rescale=0.2, validation_split=0.2)
#     data = pd.read_csv(csv_file_path)
#     X = data["image_path"]
#     y = data["label"]
#
#     skf = StratifiedKFold(n_splits=k, shuffle=True)
#     count = 1
#     total_score = 0
#
#     # train the model k times with different image generator
#     for train_index, test_index in skf.split(X, y):
#         print("-------------------- Cross Validation {0} --------------------".format(count))
#         model = keras_resnet()
#         # permute the indices arrangement for a better training
#         train_index = np.random.permutation(train_index)
#         train_generator = MyImageGenerator(data, train_index, BS, 2)
#         valid_generator = MyImageGenerator(data, test_index, BS, 2)
#         h5_file_path_cv = "{0}_cv{1}.h5".format(h5_file_path.split('.')[0], count)
#         total_score += train_model(model, h5_file_path_cv, train_generator.image_generator(), valid_generator.image_generator(),
#                     valid_generator.image_generator(), len(train_generator.n), len(valid_generator.n))
#         count += 1
#         # clear keras session to prevent leak of memory
#         K.clear_session()
#     print("Average score = "+str(total_score/k))
#
#
# def test_model(model, h5_file_path, sample_type=0):
#     if sample_type == 2:
#         test_path = TEST2_PATH
#     elif sample_type == 4:
#         test_path = TEST4_PATH
#     else:
#         test_path = TEST_PATH
#     test_batches = ImageDataGenerator().flow_from_directory(test_path,
#                                                             batch_size=BS,
#                                                             target_size=(64, 64),
#                                                             classes=CLASSES,
#                                                             shuffle=False)
#     model.load_weights(h5_file_path)
#     test_batches.reset()
#     pred = model.predict_generator(test_batches, verbose=1, steps=test_batches.n//BS)
#     predicted_class_indices = np.argmax(pred, axis=1)
#     print(pred)
#
#     # test_images, test_labels = next(test_batches)
#     # test_labels = test_labels[:, 0]
#     # predictions = model.predict(test_images, steps=1, verbose=2)
#     # print("predictions = {0}".format(np.round(predictions[:, 0]*100)))
#     # print("predictions = {0}".format(np.round(predictions[:, 0])))
#     # print("test lables = {0}".format(test_labels))
#     #
#     # cm = confusion_matrix(test_labels, np.round(predictions[:, 0]))
#     # print(cm)
#     # predict = model.predict_generator(test_batches, (24000//BS)+1)
#     # print(predict)

# use this to create resnet model and load the weight
def get_model(h5_file_path="jignet_v4_keras_resnet.h5"):
    """
    Get the model and load the weight.
    :param h5_file_path: file location stored the model's weight
    :return: keras model with weight
    """
    model = keras_resnet()
    model.load_weights(h5_file_path)
    return model


# pass image and get the predict result
def predict_image(image, model):
    """
    Given an image, predict whether it is a adjacent pieces or it isn't. Return an array with probability of each
    class. [negative_probability, positive_probability]
    :param image: combined image
    :param h5_file_path: model weight in h5 format
    :return: [negative_probability, positive_probability]
    """
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return model.predict(np.array([image]).reshape((1, 64, 64, 3)))[0]
