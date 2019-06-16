import numpy
import math
from scipy.spatial.distance import mahalanobis
import Constants as Constants
from  Network import JigNetwork as jn
import cv2


def ssd_compatibility(side_a, side_b):
    """

    :param side_a: One of the many sides we are compering
    :param side_b: One of the many sides we are compering
    :return: A dissimilarity value between two sides, the lower it is the bigger the similarity is
    """
    return numpy.sum(((side_a[:, 0:3] - side_b[:, 0:3]) ** 2))


def ssd_rotation_compatibility(side_a, side_b, num_rotation_a, num_rotation_b):
    """

    :param side_a: One of the many sides we are compering
    :param side_b: One of the many sides we are compering
    :param num_rotation_a: Number of rotation carrying out by the side a
    :param num_rotation_b: Number of rotation carrying out by the side b
    :return: A dissimilarity value between two sides, the lower it is the bigger the similarity is
    """
    assert side_a.shape == side_b.shape, 'images must be of same dimensions'
    assert num_rotation_a in range(0, 4), 'invalid number of rotations'
    assert num_rotation_b in range(0, 4), 'invalid number of rotations'

    # Rotate images based on orientation - this is easier than extracting
    # the sequences based on an orientation case switch

    image1_signed = numpy.rot90(side_a, num_rotation_b)
    image2_signed = numpy.rot90(side_b, num_rotation_b)

    return numpy.sum(((image1_signed[:, 0] - image2_signed[:, 0]) ** 2))


def mgc_rotation_compatibility(image1, image2, image1_num_rotations, image2_num_rotations):
    """
    Rotations number of image 1
    - 0: measure MGC between the right of image 1 ...
    - 1: measure MGC between the bottom of image 1 ...
    - 2: measure MGC between the left of image 1 ...
    - 3: measure MGC between the top of image 1 ...
    ... and ...
    Rotations number of image 2
    - 0: measure MGC between the left of image 2 ...
    - 1: measure MGC between the top of image 2 ...
    - 2: measure MGC between the right of image 2 ...
    - 3: measure MGC between the bottom of image 2 ...
    Both images are first rotated into position according to the specified
    number of rotations, such that the right side of image 1 and the left side of
    image 2 are the boundaries of interest. This preprocessing step simplifies
    the subsequent calculation of the MGC, but increases computation time.
    Therefore, a straightforward optimisation would be to extract boundary
    sequences directly.
    NOTE: code is take references from https://github.com/mhmoed/jigsaw/blob/development/jigsaw/linprog.py.
    :param image1_num_rotations: number rotations of image 1.
    :param image2_num_rotations: number rotations of image 2.
    :param image1: first image.
    :param image2: second image.
    :return MGC.
    """
    assert image1.shape == image2.shape, 'images must be of same dimensions'

    image1_signed = numpy.rot90(image1, image1_num_rotations)
    image2_signed = numpy.rot90(image2, image2_num_rotations)

    # Get mean gradient of image1

    g_i_l = image1_signed[:, -1] - image1_signed[:, -2]
    g_i_r = image2_signed[:, 0] - image2_signed[:, 1]

    mu_l = numpy.sum(g_i_l) / len(g_i_l)
    mu_r = numpy.sum(g_i_r) / len(g_i_r)

    s_l = numpy.cov(g_i_l.T) + numpy.eye(3) * 10e-6
    s_r = numpy.cov(g_i_r.T) + numpy.eye(3) * 10e-6

    g_ij_lr = image2_signed[:, 0] - image1_signed[:, -1]
    g_ij_rl = image1_signed[:, -1] - image2_signed[:, 0]

    d_lr = sum(mahalanobis(row, mu_l, numpy.linalg.inv(s_l)) for row in g_ij_lr)
    d_rl = sum(mahalanobis(row, mu_r, numpy.linalg.inv(s_r)) for row in g_ij_rl)

    return d_lr + d_rl
    # Get covariance matrix S
    # Small values are added to the diagonal of S to resolve non-invertibility
    # of S. This will not influence the final result.
    #
    # s = numpy.cov(g_i_l.T) + numpy.eye(3) * 10e-6
    #
    # # Get G_ij_LR
    #
    # g_ij_lr = image2_signed[:, 0] - image1_signed[:, -1]
    #
    # return sum(mahalanobis(row, mu, numpy.linalg.inv(s)) for row in g_ij_lr)

def _avg_difference(npiece, side):
    """

    :param npiece:
    :type npiece:
    :param side:
    :type side:
    :return:
    :rtype:
    """
    if side == Constants.LEFT:
        difference = npiece[:, 0] - npiece[:, 1]
    elif side == Constants.RIGHT:
        difference = npiece[:, -1] - npiece[:, -2]
    elif side == Constants.TOP:
        difference = npiece[0, :] - npiece[1, :]
    else:
        difference = npiece[-1, :] - npiece[-2, :]

    return sum(difference)/float(len(difference))


def _gradient(pieces_difference, average_side_difference):
    """

    :param pieces_difference:
    :type pieces_difference:
    :param average_side_difference:
    :type average_side_difference:
    :return:
    :rtype:
    """
    grad = pieces_difference - average_side_difference
    grad_t = numpy.transpose(grad)
    cov = numpy.cov(grad_t)
    try:
        cov_inv = numpy.linalg.inv(cov)
    except numpy.linalg.LinAlgError as e:
        cov_inv = numpy.ones((3, 3))

    return grad.dot(cov_inv).dot(grad_t)


def mgc(np1, np2, relation):
    """

    :param np1:
    :type np1:
    :param np2:
    :type np2:
    :param relation:
    :type relation:
    :return:
    :rtype:
    """
    if relation == Constants.LEFT:
        grad_12 = _gradient(np2[:, 0] - np1[:, -1], _avg_difference(np1, Constants.RIGHT))
        grad_21 = _gradient(np1[:, -1] - np2[:, 0], _avg_difference(np2, Constants.LEFT))
    else:
        grad_12 = _gradient(np2[0, :] - np1[-1, :], _avg_difference(np1, Constants.BOTTOM))
        grad_21 = _gradient(np1[-1, :] - np2[0, :], _avg_difference(np2, Constants.TOP))

    return numpy.sum(grad_12 + grad_21)


def ssd(np1, np2, relation):
    """

    :param np1:
    :type np1:
    :param np2:
    :type np2:
    :param relation:
    :type relation:
    :return:
    :rtype:
    """
    if relation == Constants.LEFT:
        difference = np1[:, -1] - np2[:, 0]
    else:
        difference = np1[-1, :] - np2[0, :]
    exponent = numpy.vectorize(lambda x: math.pow(x, 2))
    dissimilarity = numpy.sum(exponent(difference))
    return math.sqrt(dissimilarity)


def cnn_compatibility(image1, image2, relation):
    """

    :param image1:
    :type image1:
    :param image2:
    :type image2:
    :param relation:
    :type relation:
    :return:
    :rtype:
    """
    # TODO - Figure this out the problem is here,need to include a condition for LR TB
    if relation == Constants.LEFT_RIGHT:
        image = numpy.concatenate((image2, image1), axis=1)
        return round(jn.predict_image(image, Constants.model)[0] * 1000)
    if relation == Constants.RIGHT_LEFT :
        image = numpy.concatenate((image1, image2), axis=1)
        return round(jn.predict_image(image, Constants.model)[0] * 1000)
    elif relation == Constants.BOTTOM_TOP :
        rotated_a = numpy.rot90(image1, k=1)
        rotated_b = numpy.rot90(image2, k=1)
        image = numpy.concatenate((rotated_a, rotated_b), axis=1)
        return round(jn.predict_image(image, Constants.model)[0] * 1000)
    elif relation == Constants.TOP_BOTTOM:
        rotated_a = numpy.rot90(image1, k=1)
        rotated_b = numpy.rot90(image2, k=1)
        image = numpy.concatenate((rotated_b, rotated_a), axis=1)
        return round(jn.predict_image(image, Constants.model)[0] * 1000)


def mgc_ssd_compatibility(image1, image2, relation):
    """

    :param image1:
    :type image1:
    :param image2:
    :type image2:
    :param relation:
    :type relation:
    :return:
    :rtype:
    """
    # TODO - Figure this out the problem is here,need to include a condition for LR TB
    if relation == Constants.RIGHT_LEFT or relation == Constants.LEFT_RIGHT:
        return ssd(image1, image2, Constants.LEFT) * mgc(image1, image2, Constants.LEFT)
    elif relation == Constants.BOTTOM_TOP or relation == Constants.TOP_BOTTOM:
        # rotated_a = numpy.rot90(image1, k=1)
        # rotated_b = numpy.rot90(image2, k=1)
        return ssd(image1, image2, Constants.BOTTOM) * mgc(image1, image2, Constants.BOTTOM)
