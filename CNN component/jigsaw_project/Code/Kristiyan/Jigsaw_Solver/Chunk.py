import numpy
from operator import add
import cv2 as openCV
import Constants as Constants
from Network import JigNetwork as Network


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

    def add(self, piece_u, piece_v, side_a, side_b, chunk_v):
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
        return self.update_three(off_set, piece_u, piece_v, chunk_v)

    def update_three(self, off_set, piece_u, piece_v, chunk_v):
        """
            Whenever both chunks are initialized
        :param off_set:
        :type off_set: tuple
        :param piece_u:
        :type piece_u: int
        :param piece_v:
        :type piece_v: int
        :param chunk_v:
        :type chunk_v: Chunk
        :return:
        :rtype: bool
        """

        old_height_u = self.current_height
        old_width_u = self.current_width
        old_height_v = chunk_v.current_height
        old_width_v = chunk_v.current_width

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
        # Relation condition

        collision = False

        if off_set == Constants.RIGHT_LEFT_OFF_SET:
            # Outer condition
            if y_u == y_v:
                # They are the same level on the y axis
                # No movement required
                if x_u == x_v:
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

                elif x_u < x_v:
                    correction_off_set = (0, x_v - x_u - 1)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, chunk_v.current_height, chunk_v.current_width,
                                                            current_height_u, current_width_u):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

                elif x_u > x_v:
                    correction_off_set = tuple(map(add, (0, position_u[1]), off_set))
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

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
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

                elif x_u < x_v:
                    correction_off_set = (0, x_v - x_u - 1)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, chunk_v.current_height, chunk_v.current_width,
                                                            current_height_u, current_width_u):
                        collision = False  # No collision
                    else:
                        collision = True  # Collision

                elif x_u > x_v:
                    correction_off_set = tuple(map(add, (0, position_u[1]), off_set))
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

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
                                                            chunk_v, current_height_v, current_width_v):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

                elif x_u < x_v:
                    correction_off_set = (0, x_v - x_u - 1)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

                elif x_u > x_v:
                    correction_off_set = tuple(map(add, (0, position_u[1]), off_set))
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v)\
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

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
                                                            chunk_v, current_height_v, current_width_v):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

                elif y_u < y_v:
                    correction_off_set = (y_v - y_u - 1, 0)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, chunk_v.current_height, chunk_v.current_width,
                                                            current_height_u, current_width_u):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

                elif y_u > y_v:
                    correction_off_set = (y_u - y_v + 1, 0)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

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
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

                elif y_u < y_v:
                    correction_off_set = (y_v - y_u - 1, 0)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, chunk_v.current_height, chunk_v.current_width,
                                                            current_height_u, current_width_u):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

                elif y_u > y_v:
                    correction_off_set = (y_u - y_v + 1, 0)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

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
                                                            chunk_v, current_height_v, current_width_v):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

                elif y_u < y_v:
                    correction_off_set = (y_v - y_u - 1, 0)
                    piece_coordinates_u, current_height_u, current_width_u = \
                        self.adjust_piece_coordinates(piece_coordinates_u, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v,
                                                            current_height_u, current_width_u):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

                elif y_u > y_v:
                    correction_off_set = (y_u - y_v + 1, 0)
                    piece_coordinates_v, current_height_v, current_width_v = \
                        self.adjust_piece_coordinates(piece_coordinates_v, correction_off_set)

                    if not self.is_collision(piece_coordinates_u, piece_coordinates_v) \
                            and not self.is_out_of_boundary({**piece_coordinates_u, **piece_coordinates_v},
                                                            chunk_v, current_height_v, current_width_v):
                        collision = False  # No Collision
                    else:
                        collision = True  # Collision

        if not collision:
            if not self.do_NN({**piece_coordinates_u, **piece_coordinates_v}):
                self.merge_chunks(piece_coordinates_u, piece_coordinates_v)
                return False  # No Collision
            else:
                # TODO - revert the old width and height of both chunks
                self.current_height = old_height_u
                self.current_width = old_width_u
                chunk_v.current_height = old_height_v
                chunk_v.current_width = old_width_v
                return True  # Collision
        else:
            return True  # Collision


    # Well if the secondary check fails we would have already changed the global dimensions of this class
    # we need to prevent this

    def do_NN(self, coordinates):
        """

        :param coordinates:
        :type coordinates: dict
        :return:
        :rtype:
        """
        image = numpy.zeros((Constants.PATCH_DIMENSIONS * self.current_height,
                             Constants.PATCH_DIMENSIONS * self.current_width,
                             Constants.COLOUR_CHANNELS), dtype="uint8")

        # image = numpy.empty(shape=(chunk.current_height, chunk.current_width, 3), dtype="uint8")
        if self.current_width * self.current_height == len(coordinates.keys()):
            for key in coordinates.keys():
                h, w = coordinates[key]
                y0 = h * Constants.PATCH_DIMENSIONS
                x0 = w * Constants.PATCH_DIMENSIONS
                h1 = y0 + Constants.PATCH_DIMENSIONS
                w1 = x0 + Constants.PATCH_DIMENSIONS

                # image[y0:y1, x0:x1] = self.pieces[key].piece
                image[y0:h1, x0:w1] = Constants.pieces[key].piece

        # result = 0.00
        # print("current height " + str(self.current_height))
        # print("current weight " + str(self.current_width))
        # print(image.shape)
        # images4 = numpy.empty(shape=(0, Constants.PATCH_DIMENSIONS*2, Constants.PATCH_DIMENSIONS*2, 3))  # samples
        # images2h = numpy.empty(shape=(0, Constants.PATCH_DIMENSIONS, Constants.PATCH_DIMENSIONS*2, 3))  # samples
        # images2v = numpy.empty(shape=(0, Constants.PATCH_DIMENSIONS*2, Constants.PATCH_DIMENSIONS, 3))  # samples
        # for w in range(0, self.current_width, 2):
        #     for h in range(0, self.current_height, 2):
        #         h0 = h * Constants.PATCH_DIMENSIONS
        #         w0 = w * Constants.PATCH_DIMENSIONS
        #         h1 = h0 + (Constants.PATCH_DIMENSIONS * 2)
        #         w1 = w0 + (Constants.PATCH_DIMENSIONS * 2)
        #         if self.current_width >= w+2 and self.current_height >= h+2:
        #             images4 = numpy.append(images4, [image[h0:h1, w0:w1]], axis=0)
        #             # print("image 1 " + str(numpy.shape(image[h0:h1, w0:w1])))
        #             # openCV.imshow("Image passed to Network 1", image[h0:h1, w0:w1])
        #             # openCV.waitKey(0)
        #             # openCV.destroyAllWindows()
        #         elif self.current_width >= w+2:
        #             images2h = numpy.append(images2h, [image[h0:h0+Constants.PATCH_DIMENSIONS, w0:w1]], axis=0)
        #             # print("image 2 " + str(numpy.shape(image[h0:h0+Constants.PATCH_DIMENSIONS, w0:w1])))
        #             # openCV.imshow("Image passed to Network 2", image[h0:h0+Constants.PATCH_DIMENSIONS, w0:w1])
        #             # openCV.waitKey(0)
        #             # openCV.destroyAllWindows()
        #         elif self.current_height >= h+2:
        #             images2v = numpy.append(images2v, [image[h0:h1, w0:w0+Constants.PATCH_DIMENSIONS]], axis=0)
        #             # print("image 3 " + str(numpy.shape(image[h0:h1, w0:w0+Constants.PATCH_DIMENSIONS])))
        #             # openCV.imshow("Image passed to Network 3", image[h0:h1, w0:w0+Constants.PATCH_DIMENSIONS])
        #             # openCV.waitKey(0)
        #             # openCV.destroyAllWindows()
        # result4 = Network.predict_image(images4, Constants.model)
        # result2h = Network.predict_image(images2h, Constants.model)
        # result2v = Network.predict_image(images2v, Constants.model)
        # # result = Network.predict_image(image, Constants.model)[0]
        # # print(result)
        # result = numpy.min(result2h, axis=0)[1]
        # result = min(result, numpy.min(result2v, axis=0)[1])
        # result = min(result, numpy.min(result4, axis=0)[1])
        #
        result = 0.0
        if result > 0.60:
            # High Accuracy
            return False
        else:
            # Low Accuracy
            # TODO - To make it run without the CNN compatibility set to False
            return False
            # openCV.imshow("Image passed to Network", image)
            # openCV.waitKey()
            # openCV.destroyAllWindows()
            # result = Network.predict_image(image)
            #
            # print("Negative outcome:", result[0])
            # print("Positive outcome", result[1])

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
            self.chunk[value] = key

    def is_out_of_boundary(self, coordinates, chunk_v, chunk_v_height, chunk_v_width, chunk_u_height=0,
                           chunk_u_width=0):
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
        temporary_height = self.current_height
        temporary_width = self.current_width
        will_fit = False

        # Check if we have gone out of boundary
        if chunk_v_height > Constants.HEIGHT_RANGE \
                or chunk_v_width > Constants.WIDTH_RANGE \
                or chunk_u_height > Constants.HEIGHT_RANGE \
                or chunk_u_width > Constants.WIDTH_RANGE:
            return True  # Out of the puzzles boundary

        # Assign new dimension values if all test above have passed
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
        # TODO - While calculating the new pieces we look at how height changes and if it exceeds the value of the chunk
        # that it is going to be merged into
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
