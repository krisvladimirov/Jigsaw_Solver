import pandas as pd
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical


class MyImageGenerator:
    def __init__(self, data, n, batch_size, classes=2):
        self.data = data
        self.n = n
        self.batch_size = batch_size
        self.classes = classes

    def get_image(self, index, data):
        """
        Get sample details - image, label from given data frame and index .
        :param index: index number of the data wanted to get
        :param data: dataframe
        :return: an image store in matrix, label of the image
        """
        image = cv2.imread(data['image_path'].values[index])
        label = data['label'].values[index]
        return cv2.resize(image, (64, 64)), label

    def image_generator(self):
        """
        A image generator to retrieve a batch of images and label from folder.
        :param data: dataframe
        :param classes: int, number of classes
        :param n: int, number of classes
        :return: batch of image and label
        """

        while True:
            # loop from 0 to total number of indices, increment by batch size
            for b in range(0, len(self.n), self.batch_size):
                # slice out current batch according to the batch size
                current_batch = self.n[b:(b + self.batch_size)]
                x_train = np.empty(shape=(0, 64, 64, 3))  # samples
                y_train = []  # labels
                for i in current_batch:
                    image, label = self.get_image(i, self.data)
                    x_train = np.append(x_train, [image], axis=0)
                    y_train = np.append(y_train, [label])
                y_train = to_categorical(y_train, num_classes=self.classes)
                yield (x_train, y_train)

    def preprocess_input(self, image):
        # --- Rescale
        # Image
        # --- Rotate
        # Image
        # --- Resize
        # Image
        # --- Flip
        # Image
        # --- PCA
        # etc.
        return (image)