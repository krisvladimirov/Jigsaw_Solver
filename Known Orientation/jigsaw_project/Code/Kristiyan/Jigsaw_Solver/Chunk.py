import numpy
import Constants as Constants
from operator import add
import cv2 as openCV


class Chunk:

    def __init__(self, piece_index):
        """

        :param piece_index:
        :type piece_index: int
        """
        self.parent_index = piece_index
        self.chunk = [piece_index]
        self.piece_coordinates = {piece_index: (0, 0)}
        self.uninitialized = True
        self.current_height = 1
        self.current_width = 1

    def add_together(self, piece_u, piece_v, side_a, side_b, chunk_v):
        """

        :param piece_u:
        :type piece_u: int
        :param piece_v:
        :type piece_v: int
        :param side_a:
        :type side_a: int
        :param side_b:
        :type side_b: int
        :param chunk_v:
        :type chunk_v: Chunk
        :return: The evaluation boolean of whether or not the merge is successful
        :rtype: bool
        """
        off_set = Constants.get_off_set(side_a, side_b)
        return self.update_four(off_set, piece_u, piece_v, chunk_v)

    def update_four(self, off_set, piece_u, piece_v, chunk_v):
        """
            Follows the procedures of checking if two chunks are supposed to be matched, if so it matches them.
        :param off_set: The relation between two pieces
        :type off_set: tuple
        :param piece_u: A piece from chunk_u (i.e. the self.Chunk)
        :type piece_u: int
        :param piece_v: A piece from chunk_v (i.e. the chunk_v
        :type piece_v: int
        :param chunk_v: The chunk from the other side of the relation
        :type chunk_v: Chunk
        :return: Whether or not two pieces can be matched without causing a collision or going out of boundary
        :rtype: bool
        """

        # Make copies so we can work on the merging and if the merge is unsuccessful
        # We can simply discard the copies
        piece_coordinates_v = chunk_v.piece_coordinates.copy()
        piece_coordinates_u = self.piece_coordinates.copy()
        position_u = self.piece_coordinates[piece_u]
        position_v = piece_coordinates_v[piece_v]
        y_u = position_u[0]
        x_u = position_u[1]
        y_v = position_v[0]
        x_v = position_v[1]

        # Dimension parameters
        current_height_u = self.current_height
        current_width_u = self.current_width
        current_height_v = chunk_v.current_height
        current_width_v = chunk_v.current_width

        if off_set == Constants.RIGHT_LEFT_OFF_SET:
            # Outer condition
            if y_u == y_v:
                # They are the same level on the y axis
                # No movement required
                if x_u == x_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                elif x_u < x_v:
                    correction_off_set = (0, x_v - x_u - 1)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                elif x_u > x_v:
                    correction_off_set = (0, x_u - x_v + 1)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

            # Outer condition
            elif y_u < y_v:
                # They are not the same level on the y axis
                correction_off_set = (y_v - y_u, 0)
                piece_coordinates_u, current_height_u, current_width_u = \
                    self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                if x_u == x_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                elif x_u < x_v:
                    correction_off_set = (0, x_v - x_u - 1)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                elif x_u > x_v:
                    correction_off_set = (0, x_u - x_v + 1)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

            # Outer condition
            elif y_u > y_v:
                # They are not the same level on the y axis
                correction_off_set = (y_u - y_v, 0)
                piece_coordinates_v, current_height_v, current_width_v = \
                    self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                if x_u == x_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                elif x_u < x_v:
                    correction_off_set = (0, x_v - x_u - 1)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                elif x_u > x_v:
                    correction_off_set = (0, x_u - x_v + 1)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

        # Relation condition
        elif off_set == Constants.BOTTOM_TOP_OFF_SET:
            # Outer condition
            if x_u == x_v:
                # They are the same level on the x axis
                # No movement required
                if y_u == y_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                elif y_u < y_v:
                    correction_off_set = (y_v - y_u - 1, 0)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                elif y_u > y_v:
                    correction_off_set = (y_u - y_v + 1, 0)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

            # Outer condition
            elif x_u < x_v:
                correction_off_set = (0, x_v - x_u)
                piece_coordinates_u, current_height_u, current_width_u = \
                    self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                if y_u == y_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                elif y_u < y_v:
                    correction_off_set = (y_v - y_u - 1, 0)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                elif y_u > y_v:
                    correction_off_set = (y_u - y_v + 1, 0)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

            # Outer condition
            elif x_u > x_v:
                correction_off_set = (0, x_u - x_v)
                piece_coordinates_v, current_height_v, current_width_v = \
                    self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                if y_u == y_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                elif y_u < y_v:
                    correction_off_set = (y_v - y_u - 1, 0)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                elif y_u > y_v:
                    correction_off_set = (y_u - y_v + 1, 0)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

        # A correct placement is one that does not cause any piece overlapping/collisions or it causes pieces to go over
        # the boundary of the puzzle (if known)
        correct_placement = self.is_collision(piece_coordinates_u, piece_coordinates_v)

        if Constants.BIGGEST_CHUNK is not None:
            out_of_boundary = self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                      chunk_v, current_height_v, current_width_v,
                                                      current_height_u, current_width_u)
            if not correct_placement and not out_of_boundary:
                self.current_height, self.current_width = self.find_dimensions(current_height_v, current_width_v,
                                                                               current_height_u, current_width_u)
                self.merge_chunks(piece_coordinates_u, piece_coordinates_v)
                return False
            else:
                return True
        else:
            if not correct_placement:
                self.current_height, self.current_width = self.find_dimensions(current_height_v, current_width_v,
                                                                               current_height_u, current_width_u)
                self.merge_chunks(piece_coordinates_u, piece_coordinates_v)
                return False
            else:
                return True

    def merge_chunks(self, piece_coordinates_u, piece_coordinates_v):
        """
            Performs the actual merging of two chunks. This is done by first combining the two dictionaries containing
            piece:coordinate entries, then the best dimensions are recalculated for the new chunk and updated. Finally,
            the two chunks are merged together by traversing the dictionary and adding the pieces into the matrix
        :param piece_coordinates_u:
        :type piece_coordinates_u: dict
        :param piece_coordinates_v:
        :type piece_coordinates_v: dict
        :return:
        :rtype:
        """
        self.piece_coordinates = {**piece_coordinates_u, **piece_coordinates_v}
        self.update_chunk()
        self.uninitialized = False

    def find_dimensions(self, chunk_v_height, chunk_v_width, chunk_u_height, chunk_u_width):
        """

        :param chunk_v_height:
        :type chunk_v_height: int
        :param chunk_v_width:
        :type chunk_v_width: int
        :param chunk_u_height:
        :type chunk_u_height: int
        :param chunk_u_width:
        :type chunk_u_width: int
        :return:
        :rtype:
        """
        temporary_height = self.current_height
        temporary_width = self.current_width

        # Assign new dimension values if all test above have passed
        if chunk_v_height > temporary_height:
            temporary_height = chunk_v_height
        if chunk_v_width > temporary_width:
            temporary_width = chunk_v_width
        if chunk_u_height > temporary_height:
            temporary_height = chunk_u_height
        if chunk_u_width > temporary_width:
            temporary_width = chunk_u_width

        # self.current_height = temporary_height
        # self.current_width = temporary_width
        return temporary_height, temporary_width

    def update_chunk(self):
        """
            Place all the new pieces in the (U) chunk
        :return:
        """
        # Updates the dimensions of the chunk
        self.chunk = numpy.full((self.current_height, self.current_width), fill_value=Constants.VALUE_INITIALIZER,
                                dtype="int16")

        # Repopulates it with the new pieces
        for key, value in self.piece_coordinates.items():
            self.chunk[value] = key

    def is_out_of_boundary(self, coordinates, chunk_v, chunk_v_height, chunk_v_width, chunk_u_height, chunk_u_width):
        """
            First it checks to see if any of the new height or width of either of the chunks are out of boundary.
            If they are a True boolean is returned which will then be passed up the method calls to the evaluation.
            In the case that they are within the boundaries the merging process will continue as 'planned' xD
            chunk_u
        :param coordinates:
        :type coordinates: dict
        :param chunk_v:
        :type chunk_v: Chunk
        :param chunk_v_height:
        :type chunk_v_height: int
        :param chunk_v_width:
        :type chunk_v_width: int
        :param chunk_u_height:
        :type chunk_u_height: int
        :param chunk_u_width:
        :type chunk_u_width: int
        :return:
        :rtype:
        """

        # This one should only check if there are pieces out of boundary and not set the self.current_height and
        # self.current_width

        temporary_height, temporary_width = self.find_dimensions(chunk_v_height, chunk_v_width, chunk_u_height,
                                                                 chunk_u_width)
        will_fit = False

        # Special case where either u or v is the biggest chunk so it wouldn't make much sense to perform the mask
        # against the biggest chunk
        if self == Constants.BIGGEST_CHUNK or chunk_v == Constants.BIGGEST_CHUNK:
            return False

        # Check if there is any overlap between the pieces
        if Constants.BIGGEST_CHUNK is not None:
            # Making a copy matrix of the this.chunk
            copy_chunk = numpy.full((temporary_height, temporary_width),
                                    fill_value=Constants.VALUE_INITIALIZER,
                                    dtype="int")
            for key, value in coordinates.items():
                copy_chunk[value] = key

            # TODO - Document this part because I don't remember how it works sadly, to tired
            copy_chunk[copy_chunk > -1] = 1
            copy_chunk[copy_chunk < 1] = 0
            # TODO - Document this part because I don't remember how it works sadly, to tired
            biggest_chunk_copy = numpy.copy(Constants.BIGGEST_CHUNK.chunk)
            biggest_chunk_copy[biggest_chunk_copy > -1] = 0
            # TODO - Make into while loops
            for y in range(Constants.BIGGEST_CHUNK.current_height - temporary_height + 1):
                for x in range(Constants.BIGGEST_CHUNK.current_width - temporary_width + 1):
                    temporary = biggest_chunk_copy[y:y + temporary_height, x:x + temporary_width] + copy_chunk
                    if 1 not in temporary:
                        will_fit = True
                        break

            # TODO - If the secondary check passes we need to assign the copied values
            # TODO - Actually we don't need to assign anything as the merge function will deal with all of it
            # TODO - We only need to not update the dimensions if the secondary check fails

            if will_fit:
                return False
            else:
                return True
        else:
            return False

    @staticmethod
    def adjust_piece_coordinates(dictionary, off_set):
        """
            Adjust the piece_coordinates to accommodate the union of two chunks
        :param dictionary:
        :type dictionary:
        :param off_set:
        :type off_set:
        :return:
        :rtype: tuple
        """
        height = 1
        width = 1
        for key, value in dictionary.items():
            old_value = dictionary[key]
            new_value = tuple(map(add, old_value, off_set))
            # While calculating the new pieces we look at how height changes and if it exceeds the value of the chunk
            # that it is going to be merged into
            if new_value[0] + 1 > height:
                height = new_value[0] + 1
            if new_value[1] + 1 > width:
                width = new_value[1] + 1
            dictionary[key] = new_value

        return dictionary, height, width

    @staticmethod
    def is_collision(dict_a, dict_b):
        """
            Check for the intersection between two sets
            Each set contains the coordinate values of each piece that the current act of merging two chunks together
            If the intersection set is empty that means that there are not collisions, thus we can proceed and merge
            chunks.
            In the case the the intersection set is not empty we skip, thus canceling the merging
        :param dict_a:
        :type dict_a: dict
        :param dict_b:
        :type dict_b: dict
        :return:
        :rtype: bool
        """

        intersection = set(dict_a.values()) & set(dict_b.values())
        if not intersection:
            # Empty - No collisions
            return False
        else:
            # Not Empty - Collisions
            return True

    def update_piece_coordinates(self):
        """
            Updates the coordinates of each piece in the biggest chunk after the trimming process
        :return:
        :rtype:
        """
        for i in range(self.current_height):
            for j in range(self.current_width):
                key = self.chunk[i][j]
                if key != -1:
                    self.piece_coordinates[key] = (i, j)
