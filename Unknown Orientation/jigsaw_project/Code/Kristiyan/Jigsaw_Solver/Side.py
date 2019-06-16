# Not used
class Side:
    def __init__(self, content, orientation, index):
        """
            Initializes a Side with all its parameters
        :param content:
        :type content: numpy.ndarray
        :param orientation:
        :type orientation: int
        :param index:
        :type index: int
        """
        self.content = content
        self.parent_piece_index = index
        self.orientation = orientation
