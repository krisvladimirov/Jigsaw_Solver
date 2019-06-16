import numpy
import cv2 as openCV
import Piece as Piece
import Compatibility as Compatibility
import Constants as Constants
import Chunk as Chunk
import math
import sys
from operator import add
import time

"""
    What has been done so far
    1. Calculating weights 
        1.1 Calculating weights with SSD is working
        1.2 Calculating weights with MGC is working
    2. Constructing the graph
    3. Obtaining the minimum spanning tree
"""

"""
    Data type explanation:
        -   all_pieces[]
        -   edges[]
        -   all_sides[] shape of list
                -> [(Side, piece_index),
                    (Side, piece_index), 
                    (Side, piece_index), 
                    (Side, piece_index),
                    ..]
                
        -   weights = []
        
"""

"""
    Types of assembly constraints implemented:
    1. Piece based collision
        -   Checks whether or not moving a patch from one place to another will cause overlapping
        -   This is prevented by simply ignoring the particular relationship and moving to the next in line
    2. Out of boundary
        -   Check whether or not moving a patch will cause it to go out of the maximum allowed dimensions
        -   i.e. if the puzzle is 6x4 (w,h) there shouldn't be any chunks that go over those limits
        -   This is prevented by simply ignoring the particular relationship and moving to the next in line
    
    Currently both types are being tested for non rotated pieces.
    Might need a slight changes to be adapted for rotated pieces.
"""


class Solver:

    def __init__(self):
        """
            Namespace
        """
        self.original_height = -1
        self.original_width = -1
        # Stores all 4 relations between 2 pieces
        self.weights_0_4 = []
        # Stores only half of the relations between 2 pieces
        self.weights_2_4 = []
        self.edges_2_4 = []
        self.edges_0_4 = []
        self.pieces = []
        self.sides = []
        self.minimum_spanning_tree = []

        """
            In other words these are the all the minimum spanning trees
        """
        self.chunks = []

        """
            Saving the positions of the unsolved jigsaw pieces so we can later compare with the solved one
        """
        self.initial_positions = []
        """
            self.rotations will be a list holding data related to the rotation of pieces
            its length will be equal to the number of pieces in the puzzle
            this list will record any rotations that have been applied to a puzzle piece (in order for it to be 
            concatenated to another one)
            
        """
        self.rotations = []
        self.solution = None
        self.comparison = []
        self.new_minimal_spanning_tree = []
        # Sets to keep track of the how the MST's are formed
        self.parent_set = None
        self.trees = None
        self.steps = 0

    def start_solving(self, extracted_pieces, dimensions, og_dimensions, weight_path, option=None):
        """
            Option 0 - normal solving without pre-loading the weights
            Option 1 - solving with pre-loaded weights
            Option 2 - computing matchlift data
            Option 3 - normal solving without pre-loading the weights and matchlift data
            Option 4 - solving with pre-loaded weights and matchlift data
            Option 6 - computing and saving weights for later use
        :param extracted_pieces: The extracted pieces to be assembled
        :type extracted_pieces:
        :param dimensions: The dimensions of each piece
        :type dimensions:
        :param og_dimensions:
        :type og_dimensions:
        :return:
        :rtype:
        """
        Constants.PATCH_DIMENSIONS = dimensions[0]
        self.original_height = og_dimensions[0]
        self.original_width = og_dimensions[1]
        for i in range(len(extracted_pieces)):
            # processed = self.sharpen(extracted_pieces[i])
            # single_piece = Piece.Piece(processed, i)
            # self.pieces.append(single_piece)
            single_piece = Piece.Piece(extracted_pieces[i], i)
            self.pieces.append(single_piece)
            # piece_sides = single_piece.initialize_sides()
            # self.add_sides(piece_sides, i)

        self.initialize_positions_rotations()
        self.initialize_parameters()

        if option == 0 or option == 3:
            self.get_mgc_rotated()
            self.sort_edges()
            self.create_chunks()
            self.find_mst()
        elif option == 1 or option == 4:
            self.load_weights(weight_path)
            self.sort_edges()
            self.create_chunks()
            self.find_mst()
        elif option == 6:
            self.get_mgc_rotated()
        elif option == 2:
            pass
        elif option is None:
            print("Option is None, go to Run.py and set it!")

    def load_weights(self, path):
        self.weights_0_4 = numpy.load(path)
        for piece_a in self.pieces:
            image_a = piece_a.piece
            for piece_b in self.pieces:
                if piece_a != piece_b:
                    image_b = piece_b.piece
                    for side_a in range(0, 4):
                        for side_b in range(0, 4):
                            # side_a which are Left and Top will be rotated to their opposites
                            # side_a which are Right and Bottom will not be rotated
                            # side_b which are Right and Top will be rotated
                            # side_b which are Left and Top will not be rotated
                            relation = Constants.get_relation(side_a, side_b)
                            single_edge = (piece_a.index, piece_b.index, relation)
                            rotations_a, rotations_b, mgc_specific_relation, piece_swap = \
                                Constants.get_mgc_rotation(side_a, side_b)
                            transformed_a = numpy.rot90(image_a, k=rotations_a)
                            transformed_b = numpy.rot90(image_b, k=rotations_b)
                            # if piece_swap:
                            #     dissimilarity = Compatibility.mgc_ssd_compatibility(transformed_b, transformed_a,
                            #                                                         mgc_specific_relation)
                            # else:
                            #     dissimilarity = Compatibility.mgc_ssd_compatibility(transformed_a, transformed_b,
                            #                                                         mgc_specific_relation)
                            self.edges_0_4.append(single_edge)
                            # self.weights_0_4[piece_a.index][piece_b.index][relation] = dissimilarity

    def sharpen(self, piece_image):
        """

        :param piece_image:
        :type piece_image:
        :return:
        :rtype:
        """
        kernel_size = (9, 9)
        sigma = 1.5
        amount = 1.0
        threshold = 0
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = openCV.GaussianBlur(piece_image, kernel_size, sigma)
        sharpened = float(amount + 1) * piece_image - float(amount) * blurred
        sharpened = numpy.maximum(sharpened, numpy.zeros(sharpened.shape))
        sharpened = numpy.minimum(sharpened, 255 * numpy.ones(sharpened.shape))
        sharpened = sharpened.round().astype(numpy.uint8)
        if threshold > 0:
            low_contrast_mask = numpy.absolute(piece_image - blurred) < threshold
            numpy.copyto(sharpened, piece_image, where=low_contrast_mask)
        return sharpened

    def add_sides(self, piece_sides, index):
        """
            Namespace
        :param piece_sides:
        :param index:
        :return:
        """
        for side_object in piece_sides:
            single_side = (side_object, index)
            self.sides.append(single_side)

    # Fixed for non rotated pieces
    def initialize_parameters(self):
        """
            Initializes the array in which we are going to store all the weight as well as the matrix in which we will
            store the assembled pieces of the puzzle
        :return:
        :rtype:
        """
        # TODO - Update this so it is initialized at the end when we have the biggest chunk no at the beginning
        # self.solution = numpy.zeros((Constants.PATCH_DIMENSIONS * Constants.HEIGHT_RANGE,
        #                              Constants.PATCH_DIMENSIONS * Constants.WIDTH_RANGE,
        #                              Constants.COLOUR_CHANNELS), dtype="uint16")
        piece_count = len(self.pieces)
        self.weights_0_4 = numpy.full((piece_count, piece_count, 16), fill_value=Constants.INFINITY, dtype="float")

        self.parent_set = [i for i in range(len(self.pieces))]
        self.trees = [{i} for i in range(len(self.pieces))]

    def get_mgc_rotated(self):
        """
            Calculates and normalizes the weights for rotated pieces
        :return:
        :rtype:
        """
        # for piece_a in self.pieces:
        #     image_a = piece_a.piece
        #     for piece_b in self.pieces:
        #         if piece_a != piece_b:
        #             image_b = piece_b.piece
        #             possibilities = numpy.full((16, 16), fill_value=Constants.INFINITY, dtype="float")
        #             possibilities_index = 0
        #             for rot_a in range(0, 4):
        #                 for rot_b in range(0, 4):
        #                     for side_a in range(0, 4):
        #                         for side_b in range(0, 4):
        #                             # side_a which are Left and Top will be rotated to their opposites
        #                             # side_a which are Right and Bottom will not be rotated
        #                             # side_b which are Right and Top will be rotated
        #                             # side_b which are Left and Top will not be rotated
        #                             relation = Constants.get_relation(side_a, side_b)
        #                             rotations_a, rotations_b, mgc_specific_relation, piece_swap = \
        #                                 Constants.get_mgc_rotation(side_a, side_b)
        #                             transformed_a = numpy.rot90(image_a, k=rotations_a)
        #                             transformed_b = numpy.rot90(image_b, k=rotations_b)
        #                             if piece_swap:
        #                                 dissimilarity = Compatibility.mgc_ssd_compatibility(transformed_b,
        #                                                                                     transformed_a,
        #                                                                                     mgc_specific_relation)
        #                             else:
        #                                 dissimilarity = Compatibility.mgc_ssd_compatibility(transformed_a,
        #                                                                                     transformed_b,
        #                                                                                     mgc_specific_relation)
        #                             possibilities[possibilities_index][relation] = dissimilarity
        #                     possibilities_index += 1
        #                     the_min = possibilities.min()
        #                     found = False
        #                     i = 0
        #                     while not found:
        #                         if possibilities[i].min() == the_min:
        #                             found = True
        #                         else:
        #                             i += 1
        #                     for j in range(16):
        #                         self.edges_0_4.append((piece_a.index, piece_b.index, j))
        #                     self.weights_0_4[piece_a.index][piece_b.index] = possibilities[i]
        #                     k_rotations_a, k_rotations_b = Constants.get_reverse_combo(i)
        #                     Chunk.Chunk.global_piece_rotations
        t = time.process_time()
        test_set = set()
        for piece_a in self.pieces:
            image_a = piece_a.piece
            for piece_b in self.pieces:
                if piece_a != piece_b:
                    image_b = piece_b.piece
                    for side_a in range(0, 4):
                        for side_b in range(0, 4):
                            # side_a which are Left and Top will be rotated to their opposites
                            # side_a which are Right and Bottom will not be rotated
                            # side_b which are Right and Top will be rotated
                            # side_b which are Left and Top will not be rotated
                            relation = Constants.get_relation(side_a, side_b)
                            single_edge = (piece_a.index, piece_b.index, relation)
                            rotations_a, rotations_b, mgc_specific_relation, piece_swap = \
                                Constants.get_mgc_rotation(side_a, side_b)
                            transformed_a = numpy.rot90(image_a, k=rotations_a)
                            transformed_b = numpy.rot90(image_b, k=rotations_b)
                            if piece_swap:
                                dissimilarity = Compatibility.mgc_ssd_compatibility(transformed_b, transformed_a,
                                                                                    mgc_specific_relation)
                            else:
                                dissimilarity = Compatibility.mgc_ssd_compatibility(transformed_a, transformed_b,
                                                                                    mgc_specific_relation)
                            self.edges_0_4.append(single_edge)
                            self.weights_0_4[piece_a.index][piece_b.index][relation] = dissimilarity
                            test_set.add(dissimilarity)

        # print(len(test_set))
        # Normalization step
        normalized_weights_0_4 = numpy.array(self.weights_0_4)
        for i, j, rel in self.edges_0_4:
            min_weight = min(self.weights_0_4[i, :, rel].min(), self.weights_0_4[:, j, rel].min())
            normalized_weights_0_4[i, j, rel] = self.weights_0_4[i, j, rel] / (min_weight + Constants.EPSILON)
        self.weights_0_4 = normalized_weights_0_4
        elapsed_time = time.process_time() - t
        print("Elapsed time for ", str(Constants.HEIGHT_RANGE*Constants.WIDTH_RANGE), " pieces of 100 pixel size:", elapsed_time, "s")

    def save_weights_to_npy(self, root_folder, child_folder, name):
        """
            Saves the calculated weights into a numpy file for later use
        :return:
        :rtype:
        """

        numpy.save(root_folder + child_folder + name + "_"
                   + str(Constants.WIDTH_RANGE * Constants.HEIGHT_RANGE) + "_90.npy", self.weights_0_4)

    def recalculate_weights(self, pieces_of_interest):
        """

        :param pieces_of_interest:
        :type pieces_of_interest:
        :return:
        :rtype:
        """
        # Clearing the lists
        self.edges_0_4.clear()
        for u in pieces_of_interest:
            for v in pieces_of_interest:
                if u != v:
                    for side_a in range(0, 4):
                        for side_b in range(0, 4):
                            relation = Constants.get_relation(side_a, side_b)
                            single_edge = (u, v, relation)
                            self.edges_0_4.append(single_edge)

    # TODO - Checked!
    def sort_edges(self):
        """
            Sorts the edges from the smallest error to the biggest error
            This is done so Kruskal's algorithm can be applied
        :return:
        """
        self.edges_0_4.sort(key=lambda x: self.weights_0_4[x])

    def find_mst(self):
        """

        :return:
        :rtype:
        """
        not_in_one = True
        infinity_counter = 0
        t = time.process_time()
        while not_in_one:
            self.sort_edges()
            self.kruskal_alg()
            self.get_biggest_chunk()
            # TODO - get the pieces from the smaller chunks
            # TODO - What are refused pieces ?
            refused_pieces = self.get_pieces_without_a_place()
            if not refused_pieces \
                    and len(Constants.BIGGEST_CHUNK.piece_coordinates) == (Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE):
                # Whenever all pieces are located in one chunk
                not_in_one = False
            else:
                # TODO - find the border pieces of the biggest chunk
                border_pieces = self.find_border_pieces(Constants.BIGGEST_CHUNK.chunk)
                # TODO - calculate the weights again for the pieces we are interested, also the border pieces
                self.reinitialize_parameters(refused_pieces)
                self.recalculate_weights(border_pieces.union(refused_pieces))
            infinity_counter = infinity_counter + 1
            print("Infinity counter at ->", infinity_counter)
            elapsed_time = time.process_time() - t
        print("Elapsed time for ", str(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE),
              " pieces of 100 pixel size:", elapsed_time, "s")
        self.assembly_image(Constants.BIGGEST_CHUNK.piece_coordinates)

    def reinitialize_parameters(self, refused_pieces):
        """
            Resets the chunk
            It also resets the parent of a refused piece to be the refused pieces itself.
            For example if 2 has not obtained a position in the main chunk assembly yet but has a connection with 3,
            thus 3 is 2's parent, we reset this in the parent set.
            Resets the trees
        :param refused_pieces:
        :type refused_pieces:
        :return:
        :rtype:
        """
        for i in refused_pieces:
            chunk = Chunk.Chunk(i)
            self.chunks[i] = chunk
            self.parent_set[i] = i
            self.trees[i] = {i}

    def find_border_pieces(self, assembly_matrix):
        """

        :param assembly_matrix:
        :type assembly_matrix: ndarray
        :return:
        :rtype: set
        """
        # The height or width of the assembly matrix might not be the constant height and width, thus
        # we take the current values of the assembly matrix
        h, w = assembly_matrix.shape
        # Make this an array because it can't be iterated
        empty_spots = numpy.argwhere(assembly_matrix < 0)  # From 2d to 1d vector holding coordinates
        border_piece = set()
        for coordinate in empty_spots:
            # The four possible off-sets in such order R(), B(), L(), T()
            for off_set in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                y, x = map(add, tuple(coordinate), off_set)
                # Prevents the coordinate from going out of bound
                if (y >= 0 and x >= 0) and (y < h and x < w):
                    lookup_value = assembly_matrix[y][x]
                    if lookup_value != -1:
                        border_piece.add(lookup_value)

        return border_piece

    def create_chunks(self):
        """
            Basically creates the chunks or sub-images
        :return:
        """
        for i in range(len(self.pieces)):
            chunk = Chunk.Chunk(i)
            self.chunks.append(chunk)

    # TODO -  Checked!
    def kruskal_alg(self):
        """
            Implementation of Kruskal's algorithm for finding the minimum spanning tree
        :return: None
        :type: None
        """
        special_steps = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
        unsuccessful_merges = set()
        end_index = 0
        up_to = len(self.pieces) - 1  # We need V -1 edges for the minimal spanning tree
        for u_vertex, v_vertex, relation in self.edges_0_4:
            self.steps = self.steps + 1
            if end_index < up_to:
                weight = self.weights_0_4[u_vertex, v_vertex, relation]
                u = self.find_parent(parent=self.parent_set, i=u_vertex)
                v = self.find_parent(parent=self.parent_set, i=v_vertex)
                # Removes cycles from the 'graph'
                if u != v:
                    u_tree = self.trees[u]
                    v_tree = self.trees[v]

                    side_a, side_b = Constants.get_reverse_combo(relation)
                    # _, _, piece_swap = Constants.convert_relation(side_a, side_b)
                    k_rotations_a, k_rotations_b, mgc_specific_relation, piece_swap = \
                        Constants.get_assembly_rotation(u_vertex, v_vertex, side_a, side_b)
                    # TODO - Check if the relation of the pieces has to be converted, if it does then we should just
                    # look at the correct indices
                    evaluation = False  # Bool to store the output of the merger function either Failed or it Hold
                    chunk_u = self.chunks[u]
                    chunk_v = self.chunks[v]
                    # TODO - look if this works for rotation
                    # TODO - we always want either a R->L or B->T never anything else
                    # if piece_swap:
                    #     evaluation = chunk_v.add(v_vertex, u_vertex, k_rotations_a, k_rotations_b, mgc_specific_relation,
                    #                              chunk_u)
                    # else:
                    #     evaluation = chunk_u.add(u_vertex, v_vertex, k_rotations_a, k_rotations_b, mgc_specific_relation,
                    #                              chunk_v)
                    try:
                        if piece_swap:
                            evaluation = chunk_v.add(v_vertex, u_vertex, k_rotations_b, k_rotations_a,
                                                     mgc_specific_relation,
                                                     chunk_u)
                        else:
                            evaluation = chunk_u.add(u_vertex, v_vertex, k_rotations_a, k_rotations_b,
                                                     mgc_specific_relation,
                                                     chunk_v)
                    except IndexError:
                        print(self.steps)
                        sys.exit(1)
                    except KeyError:
                        print(self.steps)
                        sys.exit(1)

                    if not evaluation:
                        self.new_minimal_spanning_tree.append((u_vertex, v_vertex, relation, weight))
                        end_index = end_index + 1
                        if piece_swap:
                            united = v_tree.union(u_tree)
                            self.trees[v] = united
                            # Explicitly set the object that has been merged into None so we do not have to
                            # keep track of it
                            self.trees[u] = None
                            self.chunks[u] = None
                            self.parent_set[u] = v
                            # self.assembly_image(chunk_v.piece_coordinates, len(self.new_minimal_spanning_tree) - 1)
                        else:
                            united = u_tree.union(v_tree)
                            self.trees[u] = united
                            # Explicitly set the object that has been merged into None so we do not have to
                            # keep track of it
                            self.trees[v] = None
                            self.chunks[v] = None
                            self.parent_set[v] = u
                            # self.assembly_image(chunk_u.piece_coordinates, len(self.new_minimal_spanning_tree) - 1)

                        # TODO - Account for swapping or maybe not?

                    else:
                        # Collision between chunks
                        # We did not add anything nor we updated the end_index
                        unsuccessful_merges.add((u, v))
                else:
                    # A cycle is found
                    pass
            else:
                # MST is fully constructed we can exit the loop
                # Could probably go on and find all cycles inside the graph?
                break
        # print(self.new_minimal_spanning_tree)

    def find_parent(self, parent, i):
        if parent[i] == i:
            return i
        return self.find_parent(parent, parent[i])

    def initialize_positions_rotations(self):
        """
            Initializes the height and width of the puzzle as well as the rotation look up tables for each piece
            as it is important to know
            i.e. 16 piece is 4x4: height is 4 and is width
        :return:
        """
        # Try to approximate the pieces on each axis
        # If it is not a perfect square root, then we can conclude that the puzzle is not symmetric i.e. N by N
        total_pieces = len(self.pieces)
        sqrt_piece_count = math.sqrt(total_pieces)
        counter = 0

        if total_pieces % sqrt_piece_count == 0.0:
            Constants.WIDTH_RANGE = int(sqrt_piece_count)
            Constants.HEIGHT_RANGE = int(sqrt_piece_count)
            self.initial_positions = numpy.full((Constants.WIDTH_RANGE, Constants.HEIGHT_RANGE),
                                                fill_value=Constants.VALUE_INITIALIZER, dtype="int8")

        else:
            # If the puzzle is not symmetric we get the factors
            self.get_factors(len(self.pieces))
            self.initial_positions = numpy.full((Constants.WIDTH_RANGE, Constants.HEIGHT_RANGE),
                                                fill_value=Constants.VALUE_INITIALIZER, dtype="int8")

        # Doing it the lazy way
        for i in range(0, Constants.WIDTH_RANGE):
            for j in range(0, Constants.HEIGHT_RANGE):
                self.initial_positions[i, j] = counter
                counter = counter + 1

        rotations = dict()
        self.initial_positions = numpy.rot90(self.initial_positions)
        for key in range(len(self.pieces)):
            # When the rotation of a piece is set to 0 that means that the piece is at the current rotation in which
            # it was read in by the detector
            rotations[key] = 0
        locations = []

        for i in range(len(self.pieces)):
            locations.append([0, 1, 2, 3])
        # The indexes of locations correspond to the piece numbers
        # The indexes of the list appended to locations correspond to the four sides of a piece 0 (L) 1(T) 2(R) 3(B)
        # The values at each index in this list correspond to the place where this side is current at i.e. O (L) can be
        # at 2 (R) if it has been rotated 2 times by 90 degrees

        Chunk.Chunk.global_piece_rotations = rotations
        Chunk.Chunk.global_side_location = locations

    def get_factors(self, count):
        """

        :param count: Amount of puzzle pieces we are working with
        :type count: int
        :return:
        """
        factors = []
        for height in range(1, count + 1):
            if count % height == 0:
                pair = (int(count / height), height)
                factors.append(pair)

        not_found = True
        index = 0
        x_axis_pieces = Constants.VALUE_INITIALIZER
        y_axis_pieces = Constants.VALUE_INITIALIZER
        while not_found:
            # item is one of the many tuples found
            # Each tuple contains factors of len(self.pieces)
            item = factors[index]
            x = item[0]
            y = item[1]

            # Approximating the space that is is between each puzzle piece on the y and x axis
            # There is a 10 pixel distance between each piece and the borders i.e
            """
                                    10px             10px
                            10px    piece1    10px   piece2   10px
                                    10px             10px
                            10px    piece3    10px   piece4   10px
                                    10px             10px
            """
            y_space = (y + 1) * 10
            x_space = (x + 1) * 10

            x_axis_pieces = int((self.original_width - x_space) / Constants.PATCH_DIMENSIONS)
            y_axis_pieces = int((self.original_height - y_space) / Constants.PATCH_DIMENSIONS)

            if x_axis_pieces == x and y_axis_pieces == y:
                print("Dimensions found")
                print("Y:{0}  X:{1}".format(y_axis_pieces, x_axis_pieces))
                not_found = False
            else:
                index = index + 1

        # Some spaghetti to fix the current issues we have where we do not take into account
        # the 2 possible border frames

        Constants.ROTATED_DIMENSIONS = [(y_axis_pieces, x_axis_pieces), (x_axis_pieces, y_axis_pieces)]
        Constants.HEIGHT_RANGE = y_axis_pieces  # The correct dimensions from the image
        Constants.WIDTH_RANGE = x_axis_pieces  # The correct dimensions from the image

    def assembly_image(self, dictionary, index=0):
        """
            Puts together the solved_2_4 puzzle by looking at the piece indices
            Iterate over the self.positions which is the assembly matrix, where for each (x, y) get the piece index
        :param dictionary
        :type dictionary: dict
        :return: None
        """
        # TODO - This has to be updated to take the dimensions of the biggest chunk
        solution = numpy.zeros((Constants.PATCH_DIMENSIONS * Constants.HEIGHT_RANGE,
                               Constants.PATCH_DIMENSIONS * Constants.WIDTH_RANGE,
                               Constants.COLOUR_CHANNELS), dtype="uint8")

        for key in dictionary.keys():
            y, x = dictionary[key]
            y0 = y * Constants.PATCH_DIMENSIONS
            x0 = x * Constants.PATCH_DIMENSIONS
            y1 = y0 + Constants.PATCH_DIMENSIONS
            x1 = x0 + Constants.PATCH_DIMENSIONS

            # TODO - Rotate the piece before putting into the image
            transformed_piece = numpy.rot90(self.pieces[key].piece, k=Chunk.Chunk.global_piece_rotations[key])
            solution[y0:y1, x0:x1] = transformed_piece
            # Iterate over the dictionary
        if index != 0:
            openCV.imwrite(
                "mst_steps/tree_" + str(index) + ".png", solution)
        else:
            pass
        openCV.imshow("Test", solution)
        openCV.imwrite("../solved/big_cat_" + str(Constants.WIDTH_RANGE * Constants.HEIGHT_RANGE) +
                       "_rot90_" + str(self.steps) + "_steps" + ".png", solution)
        # openCV.imwrite("../solved_rotation/tttest/" + "edge" + str(index) + ".png", solution)
        self.solution = solution

        # file = open("test.txt", mode="w")
        # # Save the weights to txt file
        # for i in range(Constants.HEIGHT_RANGE):
        #     for j in range(Constants.WIDTH_RANGE):
        #         for rel in range(16):
        #             if rel not in {2, 8}
        #             weight = self.weights_0_4[i][j][rel]
        #             if
        #             file.write(self.weights_0_4)

    def get_biggest_chunk(self):
        """

        :return:
        :rtype: Chunk
        """
        max_size = -1
        for chunk in self.chunks:
            if chunk is not None:
                chunk_len = len(chunk.piece_coordinates)
                if max_size < chunk_len:
                    max_size = chunk_len
                    Constants.BIGGEST_CHUNK = chunk

    def get_pieces_without_a_place(self):
        """
            Gets all pieces without a location
        :return:
        :rtype: set
        """
        index = 0
        no_location_pieces = set()
        for chnk in self.chunks:
            if chnk is not None and chnk != Constants.BIGGEST_CHUNK:
                pieces = set(numpy.ravel(chnk.chunk))
                if -1 in pieces:
                    pieces.remove(-1)
                no_location_pieces = no_location_pieces.union(pieces)
                self.chunks[index] = None
            index = index + 1

        return no_location_pieces

    def rotate_after_completion(self, k_rotations):
        """

        :param k_rotations:
        :type k_rotations: int
        :return:
        :rtype:
        """
        self.solution = numpy.rot90(self.solution, k=k_rotations)
        Constants.BIGGEST_CHUNK.correct_rotation(k_rotations)
        Constants.BIGGEST_CHUNK.correct_coordinates(k_rotations)
        openCV.imwrite("../solved/corrected_big_cat_processed_2nd_" + str(Constants.WIDTH_RANGE * Constants.HEIGHT_RANGE) +
                       "_rot90_" + str(self.steps) + "_steps" + ".png", self.solution)
