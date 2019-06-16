import Chunk as Chunk

READ = "read"
WRITE = "write"
KNOWN_ORIENTATION = "known"
UNKNOWN_ORIENTATION = "unknown"
YES = "yes"
NO = "no"
EMPTY_PATH = ""

settings = {}

# Shu's CNN model 5th iteration
model = None

"""
    combo relates to the 16 relations between two pieces
"""
combo = [
    [0, 1, 2, 3],       # Left
    [4, 5, 6, 7],       # Top
    [8, 9, 10, 11],     # Right
    [12, 13, 14, 15]    # Bottom
]

#
reverse_combo = [
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 0), (3, 1), (3, 2), (3, 3)
]

"""
    index 0 -> i.e. Left side is looking for Right side, thus orientation 2
    index 1 -> i.e. Top side is looking for Bottom side, thus orientation 3
    index 2 -> i.e. Right side is looking for Left side, thus orientation 0
    index 3 -> i.e. Bottom side is looking for Top side, thus orientation 1
"""
combo_without_rotation = [
    2, 3, 0, 1
]



# Keeps track of the biggest Chunk
BIGGEST_CHUNK = None
#
VALUE_INITIALIZER = -1
# 3 colour channels BGR
COLOUR_CHANNELS = 3
# 10 pixels
PUZZLE_PIECE_SPACE = 10
# Epsilon
EPSILON = 0.1**6
# Really big number
INFINITY = 10**9
# Maximum allowable pieces on the y axis
HEIGHT_RANGE = VALUE_INITIALIZER
# Maximum allowable pieces on the x axis
WIDTH_RANGE = VALUE_INITIALIZER
# The actual dimensions of a patch i.e. 200x200 pixels
PATCH_DIMENSIONS = -1

LEFT            = 0
TOP             = 1
RIGHT           = 2
BOTTOM          = 3

"""
    Combinations between pieces
"""
# Left -
LEFT_LEFT       = 0
LEFT_TOP        = 1
LEFT_RIGHT      = 2
LEFT_BOTTOM     = 3

# Top -
TOP_LEFT        = 4
TOP_TOP        = 5
TOP_RIGHT       = 6
TOP_BOTTOM      = 7

# Right -
RIGHT_LEFT      = 8
RIGHT_TOP       = 9
RIGHT_RIGHT     = 10
RIGHT_BOTTOM    = 11

# Bottom -
BOTTOM_LEFT     = 12
BOTTOM_TOP      = 13
BOTTOM_RIGHT    = 14
BOTTOM_BOTTOM   = 15


# For no rotations
RIGHT_LEFT_OFF_SET = (0, 1)
BOTTOM_TOP_OFF_SET = (1, 0)


def convert_relation(side_a, side_b):
    """
        Namespace
    :param side_a:
    :param side_b:
    :return:
    """
    if side_a == TOP and side_b == BOTTOM:
        return BOTTOM, TOP, True
    elif side_a == LEFT and side_b == RIGHT:
        return RIGHT, LEFT, True
    else:
        return side_a, side_b, False


def get_combo_without_rotation(side_a):
    """
        Return side A's 'companion' or in other words its opposite
    :param side_a:
    :return:
    """
    return combo_without_rotation[side_a]


def get_relation(side_a, side_b):
    """
        Namespace
    :param side_a:
    :param side_b:
    :return:
    """
    return combo[side_a][side_b]


def get_reverse_combo(relation):
    """
        Namespace
    :param relation:
    :return:
    """
    return reverse_combo[relation]


def get_off_set(side_a, side_b):
    """

    :param side_a:
    :type side_a: int
    :param side_b:
    :type side_b: int
    :return: The off set of the relation
    :rtype: tuple
    """
    if side_a == 3 and side_b == 1:
        return BOTTOM_TOP_OFF_SET
    elif side_a == 2 and side_b == 0:
        return RIGHT_LEFT_OFF_SET
