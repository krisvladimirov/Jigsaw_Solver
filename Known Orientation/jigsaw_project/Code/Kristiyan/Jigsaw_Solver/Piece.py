import numpy
import Side as Side
import Constants as Constants


# Simple object holder for a puzzle piece
class Piece:
    def __init__(self, piece, index):
        self.index = index              # An index to identify a unique piece
        self.piece = piece              # The actual jigsaw piece
        self.piece_shape = piece.shape  # The shape of the piece

    # def initialize_sides(self):
    #     """
    #         Would retrieve the side of each piece
    #         Whenever using SSD we would only require the one "pixel border"
    #         Whenever using MGC we would require two "pixel borders" so the gradients could be calculated
    #     :return: A list of the 4 sides of the piece
    #     """
    #     height, width, _ = self.piece.shape
    #     """
    #         Orientation guidelines:
    #         0 -> left side of the patch     (Constants.Left)
    #         1 -> top side of the patch      (Constants.Top)
    #         2 -> right side of the patch    (Constants.Right)
    #         3 -> bottom side of the patch   (Constants.Bottom)
    #     """
    #     """***SSD***"""
    #     self.top_side = Side.Side((self.piece[0:1, 0:width]), Constants.TOP, self.index)
    #     self.bottom_side = Side.Side((self.piece[(height - 1):height, 0:width]), Constants.BOTTOM, self.index)
    #     self.left_side = Side.Side(
    #         numpy.reshape(self.piece[0:height, 0:1], (1, width, 3)),
    #         Constants.LEFT,
    #         self.index
    #     )
    #     self.right_side = Side.Side(
    #         numpy.reshape(self.piece[0:height, (width - 1):width], (1, width, 3)),
    #         Constants.RIGHT,
    #         self.index
    #     )
    #     return [self.left_side, self.top_side, self.right_side, self.bottom_side]