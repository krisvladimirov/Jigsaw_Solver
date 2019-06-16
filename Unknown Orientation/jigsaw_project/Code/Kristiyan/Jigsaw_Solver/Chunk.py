import numpy
import Constants as Constants
from operator import add


class Chunk:
    # Globally stores the current rotations of each piece
    global_piece_rotations = None  # Dictionary
    # Globally stores where each side is current at depending on how many times it has been rotated
    global_side_location = None

    # TODO - How much to rotate i % 4
    # TODO - Updating the relationship would be (i + calculated_amount) % 4
    """
        The idea is that when rotating a piece we don't have to exceed more than 3 rotations
        per piece. So if we need to rotate
    """

    def __init__(self, piece_index):
        """

        :param piece_index:
        :type piece_index: int
        """
        self.parent_index = piece_index
        self.chunk = numpy.array([[piece_index]])
        self.piece_coordinates = {piece_index: (0, 0)}
        # TODO - Explain what this does because it is important
        # Exploiting the boolean properties
        # 0 == False
        # 1 == True
        self.rotational_state = False  # Important parameter for rotated pieces
        self.uninitialized = True
        self.current_height = 1
        self.current_width = 1

    def add(self, piece_u, piece_v, k_rotations_a, k_rotations_b, mgc_specific_relation, chunk_v):
        """

        :param piece_u:
        :type piece_u: int
        :param piece_v:
        :type piece_v: int
        :param k_rotations_a:
        :type k_rotations_a: int
        :param k_rotations_b:
        :type k_rotations_b: int
        :param mgc_specific_relation:
        :type mgc_specific_relation: int
        :param chunk_v:
        :type chunk_v: Chunk
        :return: The evaluation boolean of whether or not the merge is successful
        :rtype: bool
        """
        # Getting the side_a and side_b based on the mgc_specific_relation
        side_a, side_b = Constants.get_reverse_combo(mgc_specific_relation)
        off_set = Constants.get_off_set(side_a, side_b)
        # off_set = Constants.get_off_set(side_a, side_b)
        return self.update_three(off_set, piece_u, piece_v, k_rotations_a, k_rotations_b, chunk_v)

    def update_three(self, off_set, piece_u, piece_v, k_rotations_a, k_rotations_b, chunk_v):
        """
            Whenever both chunks are initialized
        :param off_set:
        :type off_set: tuple
        :param piece_u:
        :type piece_u: int
        :param piece_v:
        :type piece_v: int
        :param k_rotations_a:
        :type k_rotations_a: int
        :param k_rotations_b:
        :type k_rotations_b: int
        :param chunk_v:
        :type chunk_v: Chunk
        :return:
        :rtype: bool
        """
        rotated_copy_chunk_a = None
        rotated_copy_chunk_b = None
        piece_coordinates_u = dict()
        piece_coordinates_v = dict()

        # Dimension parameters
        current_height_u = 0
        current_width_u = 0
        current_height_v = 0
        current_width_v = 0

        # Tell us if there is any collisions
        evaluate = False

        global_piece_rotations_copy = Chunk.global_piece_rotations.copy()

        rotated_copy_chunk_a = numpy.rot90(self.chunk, k=k_rotations_a)
        current_height_u, current_width_u = Chunk.update_piece_coordinates(rotated_copy_chunk_a, piece_coordinates_u)  # Pass in an empty dict

        rotated_copy_chunk_b = numpy.rot90(chunk_v.chunk, k=k_rotations_b)
        current_height_v, current_width_v = Chunk.update_piece_coordinates(rotated_copy_chunk_b, piece_coordinates_v)  # Pass in an empty dict

        position_u = piece_coordinates_u[piece_u]
        position_v = piece_coordinates_v[piece_v]
        y_u = position_u[0]
        x_u = position_u[1]
        y_v = position_v[0]
        x_v = position_v[1]

        # # Make copies so we can work on the merging and if the merge is unsuccessful
        # # We can simply discard the copies
        # piece_coordinates_v = chunk_v.piece_coordinates.copy()
        # piece_coordinates_u = self.piece_coordinates.copy()

        # Relation condition

        if off_set == Constants.RIGHT_LEFT_OFF_SET:
            # Outer condition
            if y_u == y_v:
                # They are the same level on the y axis, no movement required
                if x_u == x_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

                elif x_u < x_v:
                    correction_off_set = (0, x_v - x_u - 1)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

                elif x_u > x_v:
                    correction_off_set = tuple(map(add, (0, position_u[1]), off_set))
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

            # Outer condition
            elif y_u < y_v:
                # They are not the same level on the y axis
                correction_off_set = (y_v - y_u, 0)
                piece_coordinates_u, current_height_u, current_width_u = \
                    self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                if x_u == x_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

                elif x_u < x_v:
                    correction_off_set = (0, x_v - x_u - 1)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No collision
                    else:
                        evaluate = True  # Collision

                elif x_u > x_v:
                    correction_off_set = tuple(map(add, (0, position_u[1]), off_set))
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

            # Outer condition
            elif y_u > y_v:
                # They are not the same level on the y axis
                correction_off_set = (y_u - y_v, 0)
                piece_coordinates_v, current_height_v, current_width_v = \
                    self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                if x_u == x_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

                elif x_u < x_v:
                    correction_off_set = (0, x_v - x_u - 1)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

                elif x_u > x_v:
                    correction_off_set = tuple(map(add, (0, position_u[1]), off_set))
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v)\
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

        # Relation condition
        elif off_set == Constants.BOTTOM_TOP_OFF_SET:
            # Outer condition
            if x_u == x_v:
                # They are the same level on the x axis
                # No movement required
                if y_u == y_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

                elif y_u < y_v:
                    correction_off_set = (y_v - y_u - 1, 0)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

                elif y_u > y_v:
                    correction_off_set = (y_u - y_v + 1, 0)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

            # Outer condition
            elif x_u < x_v:
                correction_off_set = (0, x_v - x_u)
                piece_coordinates_u, current_height_u, current_width_u = \
                    self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                if y_u == y_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

                elif y_u < y_v:
                    correction_off_set = (y_v - y_u - 1, 0)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

                elif y_u > y_v:
                    correction_off_set = (y_u - y_v + 1, 0)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

            # Outer condition
            elif x_u > x_v:
                correction_off_set = (0, x_u - x_v)
                piece_coordinates_v, current_height_v, current_width_v = \
                    self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                if y_u == y_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

                elif y_u < y_v:
                    correction_off_set = (y_v - y_u - 1, 0)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

                elif y_u > y_v:
                    correction_off_set = (y_u - y_v + 1, 0)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        evaluate = False  # No Collision
                    else:
                        evaluate = True  # Collision

        if not evaluate:
            # Also update the rotations
            self.merge_chunks(piece_coordinates_u, piece_coordinates_v)
            # Update the global rotations
            # Essentially, take the pieces from both chunks and update their rotations
            # Additionally, record the new positions at which each side is located
            self.update_rotations(list(piece_coordinates_u.keys()), k_rotations_a)
            self.update_rotations(list(piece_coordinates_v.keys()), k_rotations_b)
        else:
            # Collision
            pass

        return evaluate
    # Well if the secondary check fails we would have already changed the global dimensions of this class
    # we need to prevent this

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
        self.update_dimensions()
        self.update_chunk()
        self.uninitialized = False

    def update_dimensions(self):
        """
            Updates the dimensions of the chunk
        :return:
        """
        self.chunk = numpy.full((self.current_height, self.current_width), fill_value=Constants.VALUE_INITIALIZER,
                                dtype="int16")

    def update_chunk(self):
        """
            Place all the new pieces in the (U) chunk
        :return:
        """
        for key, value in self.piece_coordinates.items():
            # Why is the key a numpy.int type ???
            self.chunk[value] = key

    def is_out_of_boundary(self, coordinates, chunk_v, chunk_v_height, chunk_v_width, chunk_u_height, chunk_u_width):
        """
            First it checks to see if any of the new height or width of either of the chunks are out of boundary.
            If they are a True boolean is returned which will then be passed up the method calls to the evaluation.
            In the case that they are within the boundaries the merging process will continue as 'planned' xD
            TODO - Check it again might need refactoring of the logic
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
        # TODO - Update for rotation

        temporary_height = self.current_height
        temporary_width = self.current_width
        will_fit = False
        # TODO - The main problem now is the boundary we need to know how is the rectangle frame oriented
        # i.e. Puzzle with 5 x 3 can also be 3 x 5
        # Check if we have gone out of boundary
        if chunk_v_height > Constants.HEIGHT_RANGE \
                or chunk_v_width > Constants.WIDTH_RANGE \
                or chunk_u_height > Constants.HEIGHT_RANGE \
                or chunk_u_width > Constants.WIDTH_RANGE:
            return True  # Out of the puzzles boundary

        # Assign new dimension values if all tttest above have passed
        if chunk_v_height > temporary_height:
            temporary_height = chunk_v_height
        if chunk_v_width > temporary_width:
            temporary_width = chunk_v_width
        if chunk_u_height > temporary_height:
            temporary_height = chunk_u_height
        if chunk_u_width > temporary_width:
            temporary_width = chunk_u_width

        # Special case where either u or v is the biggest case so it wouldn't make much sense to perform the mask
        # against the biggest chunk
        if self == Constants.BIGGEST_CHUNK or chunk_v == Constants.BIGGEST_CHUNK:
            self.current_height = temporary_height
            self.current_width = temporary_width
            return False

        # Check if there is any overlap between the pieces
        if Constants.BIGGEST_CHUNK is not None:
            # Making a copy matrix of the this.chunk
            copy_chunk = numpy.full((temporary_height, temporary_width),
                                    fill_value=Constants.VALUE_INITIALIZER,
                                    dtype="int")
            for key, value in coordinates.items():
                copy_chunk[value] = key

            copy_chunk[copy_chunk > -1] = 1
            copy_chunk[copy_chunk < 1] = 0

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
                self.current_height = temporary_height
                self.current_width = temporary_width
                return False
            else:
                return True
        else:
            self.current_height = temporary_height
            self.current_width = temporary_width
            # Check if we have reached the biggest_chunk limits, if we have set this object to be the biggest_chunk
            # if self.current_height == Constants.HEIGHT_RANGE and self.current_width == Constants.WIDTH_RANGE:
            #     print("Triggered")
            #     Constants.BIGGEST_CHUNK = self

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
            # Empty
            return False
        else:
            # Not Empty
            return True

    @staticmethod
    def update_piece_coordinates(chunk, copy_piece_coordinates):
        """
            Updates the coordinates of each piece inside a chunk whenever it has been rotated. Additionally,
            copy_piece coordinates is an empty dict, we just fill it with correct values (presumable)
        :param chunk:
        :type chunk: numpy.ndarray
        :param copy_piece_coordinates:
        :type copy_piece_coordinates: dict
        :return: Copy of the piece_coordinates which are also rotated and return the new height and the new width
        :rtype: tuple
        """

        dimensions = chunk.shape  # In the case of only one piece it will return a tuple with 1 values
        height = 0
        width = 0
        if len(dimensions) == 1:
            height = 1
            width = 1
        else:
            height, width = chunk.shape
        for i in range(height):
            for j in range(width):
                # Check if -1, we don't want -1, No one wants them, They do not have friends, they are weird
                key = chunk[i][j]
                if key != -1:
                    copy_piece_coordinates[key] = (i, j)

        return height, width

    @staticmethod
    def update_rotations(pieces_to_update, k_rotations):
        """
            Updates the rotation of each piece in in the matrix space
        :param pieces_to_update:
        :type pieces_to_update: list
        :param k_rotations:
        :type k_rotations: int
        :return:
        :rtype:
        """
        # A lot of spaghetti on the next line
        for key in pieces_to_update:
            old_value = Chunk.global_piece_rotations[key]
            new_value = (k_rotations + old_value) % 4
            Chunk.global_piece_rotations[key] = new_value

        for piece in pieces_to_update:
            new_side_locations = list(
                map(lambda side_num: (side_num - k_rotations) % 4, Chunk.global_side_location[piece])
            )
            Chunk.global_side_location[piece] = new_side_locations

    @staticmethod
    def correct_rotation(k_rotations):
        """
            Correct the rotation of the assembled puzzle based on the user's judgement
            Correct the rotations of each piece ONCE the puzzle has been assembled based on the user's judgement
        :return:
        :rtype:
        """

        for key, value in Chunk.global_piece_rotations.items():
            Chunk.global_piece_rotations[key] = (k_rotations + value) % 4
        # Should I correct it for the side rotations also?

    def correct_coordinates(self, k_rotations):
        """

        :param k_rotations:
        :type k_rotations:
        :return:
        :rtype:
        """
        # Rotate the chunk
        self.chunk = numpy.rot90(self.chunk, k=k_rotations)
        height, width = self.chunk.shape
        for i in range(height):
            for j in range(width):
                # Check if -1, we don't want -1, No one wants them, They do not have friends, they are weird
                key = self.chunk[i][j]
                if key != -1:
                    self.piece_coordinates[key] = (i, j)
