import numpy
import math
import Code.Kristiyan.Jigsaw_Solver.Constants as Constants


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


def _mgc(np1, np2, relation):
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


def _ssd(np1, np2, relation):
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
    if relation == Constants.RIGHT_LEFT or relation == Constants.LEFT_RIGHT:
        return _ssd(image1, image2, Constants.LEFT) * _mgc(image1, image2, Constants.LEFT)
        # Only for SSD results
        # return _ssd(image1, image2, Constants.LEFT)
    elif relation == Constants.BOTTOM_TOP or relation == Constants.TOP_BOTTOM:
        return _ssd(image1, image2, Constants.BOTTOM) * _mgc(image1, image2, Constants.BOTTOM)
        # Only for SSD results
        # return _ssd(image1, image2, Constants.BOTTOM)
