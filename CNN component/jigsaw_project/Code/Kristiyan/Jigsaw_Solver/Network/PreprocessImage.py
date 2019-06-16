import cv2
import numpy as np
import os
import csv
from random import randint, choice
from os.path import isfile, join

"""
All image should store in src folder, src/original store all images before cut into pieces. src/test store testing 
samples which 
Directory Tree:
-SRC
 |_original 
 |  |_test
 |  |_train
 |  |_valid
 |_test
 |  |_positive
 |  |_negative
 |_train
 |  |_positive
 |  |_negative
 |_valid
    |_positive
    |_negative
 
"""


def flatten_image(image, piece_size, is_half=True):
    """
    Generate a list of image pieces by given a input image. It will split image into
    ((piece_size/2) x piece_size) pieces. For example piece_size = 64, output image pieces size
    will be 32 x 64. This allow user to concat two image pieces and generate a square piece.
    :param image:
    :param piece_size:
    :param is_half: whether want to cut the pieces of image in half of a square
    :return:

    Usage::
        >>> from PreprocessImage import flatten_image
        >>> flat_image = flatten_image(image, 32, True)

    """
    piece_size_col = piece_size
    if is_half:
        piece_size_col = piece_size // 2

    rows, columns = image.shape[0] // piece_size, image.shape[1] // piece_size_col

    pieces = []

    # Crop pieces from original image
    for y in range(rows):
        for x in range(columns):
            left, top, w, h = x * piece_size_col, y * piece_size, (x + 1) * piece_size_col, (y + 1) * piece_size
            piece = np.empty((piece_size, piece_size_col, image.shape[2]))
            piece[:piece_size, :piece_size_col, :] = image[top:h, left:w, :]
            pieces.append(np.array(piece, dtype=np.uint8))

    return pieces, rows, columns


def generate_samples_4(file_path, with_black=False, image_size=64):
    """
    Generate 4 combined pieces samples.
    :param file_path:
    :param with_black:
    :param image_size:
    :return: an array of pieces, an array of labels (0=negative, 1 = positive)
    """
    image = cv2.imread(file_path)
    flat_image, row, col = flatten_image(image, image_size, False)
    train_samples = []
    train_label = []
    black_image = np.zeros(shape=(image_size, image_size, 3))
    for i in range(row-1):
        for j in range(1, col-1):

            img_index = i*col + j
            img_index_bottom = (i+1)*col + j
            # concatenate horizontally
            combine_img_top = np.concatenate((flat_image[img_index - 1], flat_image[img_index]), axis=1)
            combine_img_bottom = np.concatenate((flat_image[img_index_bottom - 1], flat_image[img_index_bottom]), axis=1)
            # concatenate vertically (positive sample)
            positive_combine_img = np.concatenate((combine_img_top, combine_img_bottom), axis=0)
            train_samples.append(positive_combine_img)
            train_label.append(1)   # positive label

            if j-1 != col:
                neg_index = (i * col) + j + 1
            else:
                neg_index = (i * col) + j - 2

            two_image_size = image_size*2
            # choose random number between 0-3
            permute_index1 = randint(0, 3)
            permute_index2 = choice([i for i in range(0, 3) if i != permute_index1])


            negative_combine_img1 = np.copy(positive_combine_img)
            # replace the place with out image
            if permute_index1 == 0:
                negative_combine_img1[0:image_size, 0:image_size, :] = flat_image[neg_index]
            elif permute_index1 == 1:
                negative_combine_img1[image_size:two_image_size, 0:image_size, :] = flat_image[neg_index]
            elif permute_index1 == 2:
                negative_combine_img1[0:image_size, image_size:two_image_size, :] = flat_image[neg_index]
            else:
                negative_combine_img1[image_size:two_image_size, image_size:two_image_size, :] = flat_image[neg_index]
            # store into sample
            train_samples.append(negative_combine_img1)
            train_label.append(0)  # negative label

            if with_black:
                # positive image with one black part
                positive_combine_img2 = np.copy(positive_combine_img)
                if permute_index1 == 0:
                    positive_combine_img2[0:image_size, 0:image_size, :] = black_image
                elif permute_index1 == 1:
                    positive_combine_img2[image_size:two_image_size, 0:image_size, :] = black_image
                elif permute_index1 == 2:
                    positive_combine_img2[0:image_size, image_size:two_image_size, :] = black_image
                else:
                    positive_combine_img2[image_size:two_image_size, image_size:two_image_size, :] = black_image

                # store into sample
                train_samples.append(positive_combine_img2)
                train_label.append(1)  # positive label

                negative_combine_img2 = np.copy(negative_combine_img1)
                # negative image with one black part
                if permute_index2 == 0:
                    negative_combine_img2[0:image_size, 0:image_size, :] = black_image
                elif permute_index2 == 1:
                    negative_combine_img2[image_size:two_image_size, 0:image_size, :] = black_image
                elif permute_index2 == 2:
                    negative_combine_img2[0:image_size, image_size:two_image_size, :] = black_image
                else:
                    negative_combine_img2[image_size:two_image_size, image_size:two_image_size, :] = black_image
                # store into sample
                train_samples.append(negative_combine_img2)
                train_label.append(0)  # negative label

    return train_samples, train_label


def generate_samples(file_path, image_size):
    """
    Generate dataset to train the model.
    :param file_path:
    :param image_size:
    :return: list of positive image and list of negative image
    """
    image = cv2.imread(file_path)
    flat_image, row, col = flatten_image(image, image_size)
    train_samples = []
    train_label = []
    for i in range(row):
        for j in range(1, col):
            img_index = i*col + j
            # concatenate horizontally
            combine_img = np.concatenate((flat_image[img_index - 1], flat_image[img_index]), axis=1)
            train_samples.append(combine_img)
            train_label.append(1)   # positive label
            for k in range(j+1, col):
                neg_img_index = i*col+j-1
                neg_combine_img = np.concatenate((flat_image[neg_img_index], flat_image[i*col+k]), axis=1)
                train_samples.append(neg_combine_img)
                train_label.append(0)   # negative label

    return train_samples, train_label


def save_generated_sample_2(sample_type, store_path, image_size=128):
    """
    Generate samples with 2 random piece piece together.
    :param sample_type: folder name in original file
    :param store_path: folder name to store samples
    :param image_size: size of the image require to generate
    :return: -
    """
    # original image directory path
    image_file_directory = join(os.getcwd(), 'src\\original\\{0}'.format(sample_type))

    # get all images path from src/original and store in a list
    only_files = [f for f in os.listdir(image_file_directory) if isfile(join(image_file_directory, f))]
    positive_count, negative_count = count_number_file("src\\"+store_path)
    for f in only_files:
        samples, labels = generate_samples(join(image_file_directory, f), image_size)
        for i in range(len(samples)):
            if labels[i] == 1:
                cv2.imwrite("src\\{0}\\positive\\{1}.jpg".format(store_path, positive_count), samples[i])
                positive_count += 1
            else:
                cv2.imwrite("src\\{0}\\negative\\{1}.jpg".format(store_path, negative_count), samples[i])
                negative_count += 1


def save_generated_sample_4(src_path, dst_path, with_black=False, image_size=64):
    """
    Generate samples with 4 random pieces piece together. sample will store in positive and negative folder.
    :param src_path: folder name in original folder
    :param dst_path: folder name to store the images
    :param with_black:
    :param image_size:
    :return: _
    """
    # original image directory path
    image_file_directory = join(os.getcwd(), 'src\\original\\{0}'.format(src_path))

    # get all images path from src/original and store in a list
    only_files = [f for f in os.listdir(image_file_directory) if isfile(join(image_file_directory, f))]
    positive_count, negative_count = count_number_file("src\\" + dst_path)
    for f in only_files:
        samples, labels = generate_samples_4(join(image_file_directory, f), with_black, image_size)
        for i in range(len(samples)):
            if labels[i] == 1:
                cv2.imwrite("src\\{0}\\positive\\{1}.jpg".format(dst_path, positive_count), samples[i])
                positive_count += 1
            else:
                cv2.imwrite("src\\{0}\\negative\\{1}.jpg".format(dst_path, negative_count), samples[i])
                negative_count += 1


def count_number_file(path_name):
    pst_img = os.listdir(join(path_name, 'positive'))
    ngt_img = os.listdir(join(path_name, 'negative'))
    return len(pst_img), len(ngt_img)


def get_generated_samples(sample_type):
    """

    :param sample_type: train/valid/test
    :return:
    """
    pst_img_c, ngt_img_c = count_number_file(sample_type)
    pst_path_name = "{0}\\positive".format(sample_type)
    ngt_path_name = "{0}\\negative".format(sample_type)

    # get all positive labels from the directory
    samples_path = ["{0}\\{1}".format(pst_path_name, f) for f in os.listdir(pst_path_name) if isfile(join(pst_path_name, f))]
    labels = [1 for _ in range(pst_img_c)]

    # get all negative labels from directory
    samples_path = samples_path + ["{0}\\{1}".format(ngt_path_name, f) for f in os.listdir(ngt_path_name) if isfile(join(ngt_path_name, f))]
    labels = labels + [0 for _ in range(ngt_img_c)]

    return samples_path, labels


def create_image_info_csv(sample_type):
    """
    Create a csv file to store all info into a csv file with image path and its label.
    We will use it during generate image batch for training or testing a model.
    :param sample_type: train or test or valid
    :return: None
    """
    path = "src\\{0}".format(sample_type)
    with open('{0}\\{1}.csv'.format(path, sample_type), mode='w') as file:
        file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(["image_path", "label"])
        samples_path, labels = get_generated_samples(path)
        for i in range(len(samples_path)):
            file_writer.writerow([samples_path[i], labels[i]])


# create_image_info_csv("temp")
# image_temp = cv2.imread("C:\\Users\\Mark\\Desktop\\JigsawNetwork\\PuzzleNetwork\\src\\test\\negative\\1041.jpg")

# store_path = 'src\\test'
# num_pst_img, num_ngt_img = count_number_file(store_path)
# save_generated_sample(0, 0, "train")
# save_generated_sample(0, 0, "test")
# save_generated_sample(0, 0, "valid")
# count_number_file('src\\train')
# create_image_info_csv('train')
# x, y = generate_samples("src\\original\\done\\building6.jpg", 128
# generate_samples_4("src\\original\\test\\Places365_test_00010001.jpg", 64)
# print(np.shape(x))

