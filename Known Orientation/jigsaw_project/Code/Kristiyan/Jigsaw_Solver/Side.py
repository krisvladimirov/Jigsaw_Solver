import cv2 as openCV


class Side():
    # TODO Rename the side variable to something more meaningful
    def __init__(self, content, orientation, index):
        """
        Initializes a Side with all its parameters
        :param side: One from the four sides of each piece represented as a numpy array
        :param orientation: The orientation of the side in the process of extraction. Note, this is unlikely the correct
        rotation of a side.
        """
        self.content = content
        self.parent_piece_index = index
        self.orientation = orientation
        self.is_it_matched = False
        self.is_it_border = False
        self.connection_with = None
