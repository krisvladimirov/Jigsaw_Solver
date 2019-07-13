import numpy
import math
import Constants as Constants

"""
    Refer to Jigsaw Puzzles with Pieces of Unknown Orientation
    Authors: Andrew C. Gallagher
    DOI: 10.1109/CVPR.2012.6247699
    
    Refer to Robust Solvers for Square Jigsaw Puzzles
    Authors: Debajyoti Mondal, Yang Wang, Stephane Durocher
    DOI: 10.1109/CRV.2013.54
"""


def avg_difference(puz_piece, side):
    """
        Computes the mean distribution of the gradients along a particular side of a puzzle piece
    :param puz_piece: The puzzle piece passed in for work
    :type puz_piece: numpy.ndarray
    :param side: One of the four sides for which the mean distribution of gradients will be computed
    :type side: int
    :return: The mean distribution of gradients for each colour channel summed up
    :rtype: numpy.ndarray
    """
    if side == Constants.LEFT:
        difference = puz_piece[:, 0] - puz_piece[:, 1]
    elif side == Constants.RIGHT:
        difference = puz_piece[:, -1] - puz_piece[:, -2]
    elif side == Constants.TOP:
        difference = puz_piece[0, :] - puz_piece[1, :]
    else:
        difference = puz_piece[-1, :] - puz_piece[-2, :]

    return sum(difference)/float(len(difference))


def gradient(difference_between_pieces, average_side_difference):
    """
        Computes the mahalanobis distance between the gradients from one side of the piece to the opposing side of a
        adjacent puzzle piece
    :param difference_between_pieces: The the difference in colour intensities between two pieces u and v
    :type difference_between_pieces: numpy.ndarray
    :param average_side_difference: The mean distribution of gradients for each colour channel summed up for one puzzle piece
    :type average_side_difference: numpy.ndarray
    :return: The dissimilarity between between two pieces. It must be noted it is not symmetric. See papers
    :rtype: numpy.ndarray
    """
    grad = difference_between_pieces - average_side_difference
    grad_t = numpy.transpose(grad)
    cov = numpy.cov(grad_t)
    try:
        cov_inv = numpy.linalg.inv(cov)
    except numpy.linalg.LinAlgError as e:
        cov_inv = numpy.ones((3, 3))

    return grad.dot(cov_inv).dot(grad_t)


def mgc(piece_u, piece_v, relation):
    """
        Computes the final mahalanobis gradient compatibility
    :param piece_u: One of the puzzle pieces
    :type piece_u: numpy.ndarray
    :param piece_v: The other one of the puzzle pieces
    :type piece_v: numpy.ndarray
    :param relation: The connecting relation between both puzzle pieces
    :type relation: int
    :return: The dissimilarity between two pieces, the lower the better. Negative number are permitted.
    :rtype: float
    """
    if relation == Constants.LEFT:
        grad_12 = gradient(piece_v[:, 0] - piece_u[:, -1], avg_difference(piece_u, Constants.RIGHT))
        grad_21 = gradient(piece_u[:, -1] - piece_v[:, 0], avg_difference(piece_v, Constants.LEFT))
    else:
        grad_12 = gradient(piece_v[0, :] - piece_u[-1, :], avg_difference(piece_u, Constants.BOTTOM))
        grad_21 = gradient(piece_u[-1, :] - piece_v[0, :], avg_difference(piece_v, Constants.TOP))

    return numpy.sum(grad_12 + grad_21)


def ssd(piece_u, piece_v, relation):
    """
        Computes sum of squared differences between two puzzle pieces u and v
    :param piece_u: One of the puzzle pieces
    :type piece_u: numpy.ndarray
    :param piece_v: The other one of the puzzle pieces
    :type piece_v: numpy.ndarray
    :param relation: The connecting relation between both puzzle pieces
    :type relation: int
    :return: Sum of squared difference between the colours of two puzzle sides
    :rtype: float
    """
    if relation == Constants.LEFT:
        difference = piece_u[:, -1] - piece_v[:, 0]
    else:
        difference = piece_u[-1, :] - piece_v[0, :]
    exponent = numpy.vectorize(lambda x: math.pow(x, 2))
    dissimilarity = numpy.sum(exponent(difference))
    return math.sqrt(dissimilarity)


def mgc_ssd_compatibility(image1, image2, relation):
    """
        Combines SSD with MGC compatibility, it is a bit more accurate than just MGC
    :param image1: One of the puzzle pieces
    :type image1: numpy.ndarray
    :param image2: The other one of the puzzle pieces
    :type image2: numpy.ndarray
    :param relation: The connecting relation between both puzzle pieces
    :type relation: int
    :return: The MGC + SSD dissimilarity between two puzzle sides
    :rtype: float
    """
    if relation == Constants.RIGHT_LEFT or relation == Constants.LEFT_RIGHT:
        return ssd(image1, image2, Constants.LEFT) * mgc(image1, image2, Constants.LEFT)
    elif relation == Constants.BOTTOM_TOP or relation == Constants.TOP_BOTTOM:
        return ssd(image1, image2, Constants.BOTTOM) * mgc(image1, image2, Constants.BOTTOM)
