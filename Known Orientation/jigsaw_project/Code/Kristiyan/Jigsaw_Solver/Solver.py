import numpy
import cv2 as openCV
import Piece as Piece
import Compatibility as Compatibility
import Constants as Constants
import Chunk as Chunk
import math
import h5py
import sys
from operator import add
from collections import deque
import time
import pathlib

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

pieces = []


class Solver:
    def __init__(self):
        # TODO - New variables
        self.weights = None
        self.original_height = Constants.VALUE_INITIALIZER
        self.original_width = Constants.VALUE_INITIALIZER
        # Hold all edges
        self.all_edges = []
        self.important_edges = []
        self.pieces = []
        # self.sides = []
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
            this list will record any rotations that have been applied to a puzzle piece (in order for it to be concatenated
            to another one)
            
        """
        self.rotations = []

        self.comparison = []
        self.new_minimal_spanning_tree = []
        # Sets to keep track of the how the MST's are formed
        self.parent_set = None
        self.trees = None
        self.steps = 0

        """
            Matchlift data
        """
        self.matchlift_weights_0_4 = None
        self.matchlift_edges_0_4 = []

    def prepare_solver(self, extracted_pieces, dimensions, og_dimensions):
        """

        :param extracted_pieces:
        :param dimensions:
        :param og_dimensions:
        :return:
        """

        # Assign the dimension of the square puzzle piece
        Constants.PATCH_DIMENSIONS = dimensions[0]
        # The original height of the puzzle frame
        self.original_height = og_dimensions[0]
        # The original width of the puzzle frame
        self.original_width = og_dimensions[1]
        # Create a puzzle piece object for easy access and manipulation, record its index also
        for i in range(len(extracted_pieces)):
            single_piece = Piece.Piece(extracted_pieces[i], i)
            # TODO - Why am I doing it in two places???
            self.pieces.append(single_piece)

        # TODO - To be removed as it violates the out-of-boundary condition
        self.initialize_positions_rotations()
        self.initialize_parameters()

    def start_solving(self, extracted_pieces, dimensions, og_dimensions):
        """
            Option 0 - normal solving without pre-loading the weights
            Option 1 - solving with pre-loaded weights
            Option 2 - computing matchlift data
            Option 3 - normal solving without pre-loading the weights and matchlift data
            Option 4 - solving with pre-loaded weights and matchlift data
            Option 6 - computing and saving weights for later use
        :param extracted_pieces: The extracted pieces to be assembled
        :param dimensions: The dimensions of each piece
        :return:
        """
        Constants.PATCH_DIMENSIONS = dimensions[0]
        self.original_height = og_dimensions[0]
        self.original_width = og_dimensions[1]
        for i in range(len(extracted_pieces)):
            single_piece = Piece.Piece(extracted_pieces[i], i)
            self.pieces.append(single_piece)
            # piece_sides = single_piece.initialize_sides()

        # TODO - Remove this
        self.initialize_positions_rotations()

        self.initialize_parameters()

    def load_matchlift_data(self, path_to_file, correspondence, num_of_pieces):
        """

        :param path_to_file:
        :type path_to_file:
        :param correspondence:
        :type correspondence:
        :return:
        :rtype:
        """
        file = h5py.File(path_to_file)
        destination = file["/matching/value"]
        total_number = len(destination) * correspondence

        storage = deque()
        for index in range(total_number):
            for val in file[destination[index][0]][()]:
                storage.append(val[0])
        self.save_matchlift_weights(storage, num_of_pieces, correspondence)

    def load_weights(self):
        """

        :return:
        :rtype:
        """
        # Check if the path exists first
        if not pathlib.Path.exists(Constants.settings["weight"]["path_to_weights"]):
            raise Exception("Please specify correctly the \"output_path\" attribute of \"weights\"!")

        self.weights = numpy.load(Constants.settings["weight"]["path_to_weights"])

        for piece_a in self.pieces:
            for piece_b in self.pieces:
                # Check if two pieces are the same, if they are skip the comparison
                if piece_a != piece_b:
                    # This loops gets the necessary 4 rotations of piece_a
                    for side_a in range(0, 4):
                        # Get the opposing side of piece b based on the side of piece a
                        side_b = Constants.get_combo_without_rotation(side_a)
                        # Checks if the pieces need to be swapped in order for the weights to be calculated correctly
                        _, _, piece_swap = Constants.convert_relation(side_a, side_b)
                        # rot_a, rot_b = self.get_rotation_mgc(side_a, side_b)
                        relation = Constants.get_relation(side_a, side_b)
                        single_edge = (piece_a.index, piece_b.index, relation)
                        if piece_swap:
                            # Add the symmetric opposing edges also
                            self.all_edges.append(single_edge)
                        else:
                            # Add all edges, just in case they are needed
                            self.all_edges.append(single_edge)
                            # Add only the 2 edges between piece a and piece b, unnecessary to add symmetric edges also
                            self.important_edges.append(single_edge)

    def save_matchlift_weights(self, storage, num_of_pieces, num_correspondences):
        """

        :param storage:
        :type storage:
        :param num_of_pieces:
        :type num_of_pieces:
        :param num_correspondences:
        :type num_correspondences:
        :return:
        :rtype:
        """
        print(storage)
        self.matchlift_weights_0_4 = numpy.zeros((num_of_pieces, num_of_pieces, 16), dtype="float64")
        for piece_a_index in range(num_of_pieces):
            for piece_b_index in range(num_of_pieces):
                for side_a in range(0, 4):
                    side_b = Constants.get_combo_without_rotation(side_a)
                    # _, _, piece_swap = Constants.convert_relation(side_a, side_b)  # TODO - Do I really need it?
                    relation = Constants.get_relation(side_a, side_b)
                    # Usually the correspondence should be set to 1 for easier computation
                    if piece_a_index != piece_b_index:
                        weight = storage.popleft()
                        # TODO - Important parameter
                        if weight > 0.0:
                            single_edge = (piece_a_index, piece_b_index, relation)
                            self.matchlift_edges_0_4.append(single_edge)
                        self.matchlift_weights_0_4[piece_a_index, piece_b_index, relation] = weight


        # Sorting the weights and edges
        # self.matchlift_edges_0_4.sort(key=lambda x: self.matchlift_weights_0_4[x], reverse=True)

    # TODO - Fixed for non rotated pieces
    def initialize_parameters(self):
        """
            Initializes the array in which we are going to store all the weight as well as the matrix in which we will
            store the assembled pieces of the puzzle
        :return:
        :rtype:
        """

        piece_count = len(self.pieces)
        self.weights = numpy.full((piece_count, piece_count, 16), fill_value=Constants.INFINITY, dtype="float")
        # Keeps track of the parent of each piece in the MST tree
        self.parent_set = [i for i in range(len(self.pieces))]
        # Keeps track of all trees that are formed
        self.trees = [{i} for i in range(len(self.pieces))]

    def get_mgc_matchlift(self):
        """
            Calculate the MGC between all pieces including 4 or more correspondencies for matchlift
        :return:
        :rtype:
        """

        edges_with_correspondences = []
        num_correspondence = Constants.settings["matchlift"]["num_correspondences"]

        if num_correspondence < 1:
            raise Exception(
                "The number of correspondences can't be smaller than 1, "
                "check \"num_correspondences\" in the settings.json")

        # TODO - HEIGHT Range and WIDTH Range will be changed
        weights_matchlift = numpy.full((Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE,
                                        Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE, 16,
                                        num_correspondence), fill_value=0,
                                       dtype="float")
        for piece_a in self.pieces:
            image_a = piece_a.piece
            # This loops get the necessary 4 rotations of piece_a
            for piece_b in self.pieces:
                if piece_a != piece_b:
                    for side_a in range(0, 4):
                        image_b = piece_b.piece  # Get the image from the piece object
                        side_b = Constants.get_combo_without_rotation(side_a)
                        _, _, piece_swap = Constants.convert_relation(side_a, side_b)
                        relation = Constants.get_relation(side_a, side_b)
                        for correspondence in range(0, num_correspondence):
                            cropped_a = self.crop_it(image_a, correspondence, relation, num_correspondence)
                            cropped_b = self.crop_it(image_b, correspondence, relation, num_correspondence)
                            single_edge = (piece_a.index, piece_b.index, relation, correspondence)
                            if piece_swap:
                                dissimilarity = Compatibility.mgc_ssd_compatibility(cropped_b, cropped_a, relation)
                            else:
                                dissimilarity = Compatibility.mgc_ssd_compatibility(cropped_a, cropped_b, relation)

                            edges_with_correspondences.append(single_edge)
                            weights_matchlift[piece_a.index, piece_b.index, relation, correspondence] = dissimilarity

        normalized_weights = numpy.full((Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE,
                                         Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE, 16, num_correspondence),
                                         fill_value=0, dtype="float")

        for i, j, rel, cor in edges_with_correspondences:
            min_weight = min(weights_matchlift[i, :, rel, cor].min(), weights_matchlift[:, j, rel, cor].min())
            normalized_weights[i, j, rel, cor] = normalized_weights[i, j, rel, cor] / (min_weight + Constants.EPSILON)

        file = open(Constants.settings["matchlift"]["output_path"] + Constants.settings["name_of_image"] + "_" + str(
            Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE) + "_" + str(num_correspondence) + "_cor.txt",
                    mode="w")
        for i in range(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE):
            for j in range(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE):
                for rel in range(16):
                    for correspondence in range(num_correspondence):
                        weight = normalized_weights[i][j][rel][correspondence]
                        if rel == Constants.RIGHT_LEFT or rel == Constants.LEFT_RIGHT or rel == Constants.BOTTOM_TOP or rel == Constants.TOP_BOTTOM:
                            file.write(str(weight) + "\n")

    @staticmethod
    def crop_it(image, index_crop, rel, correspondence):
        """
                    Get a normal puzzle piece, and effectively cuts it into equal areas from where the correspondence will be
                    calculated
                :param image:
                :type image:
                :param index_crop:
                :type index_crop:
                :param rel:
                :type rel:
                :param correspondence:
                :type correspondence:
                :return:
                :rtype:
                """

        height, width, _ = image.shape
        cropped = None

        if rel == Constants.LEFT_RIGHT or rel == Constants.RIGHT_LEFT:
            height_interval = height / correspondence
            y_start = int(index_crop * height_interval)
            y_end = int((index_crop + 1) * height_interval)
            cropped = image[y_start:y_end, :]
        elif rel == Constants.TOP_BOTTOM or rel == Constants.BOTTOM_TOP:
            width_interval = width / correspondence
            x_start = int(index_crop * width_interval)
            x_end = int((index_crop + 1) * width_interval)
            cropped = image[:, x_start:x_end]

        return cropped

    # Used for Unknown orientation
    # Used for Matchlift Unknown orientation
    def get_mgc_rotated(self):
        """
            Calculates the MGC between all pieces while taking the rotation into consideration
        :return:
        """
        t = time.process_time()
        """
            The process of calculating the weight is performed for all 16 relations between the pieces as the 
            orientation is unknown. This function is very similar to get_mgc, but I have decided to split them as the 
            logic will become very confusing if one function is used for both
        """
        for piece_a in self.pieces:
            image_a = piece_a.piece
            for piece_b in self.pieces:
                # Check if two pieces are the same, if they are skip the comparison
                if piece_a != piece_b:
                    image_b = piece_b.piece
                    # Go through the 4 sides of piece A
                    for side_a in range(0, 4):
                        # Go through the 4 sides of piece B
                        for side_b in range(0, 4):
                            # Whenever side_a is Left or Top, it will be rotated to their opposites
                            # Whenever side_a is Right and Bottom, it will not be rotated
                            # Whenever side_b is Right and Top, it will be rotated
                            # Whenever side_b is Left and Top, it will not be rotated
                            relation = Constants.get_relation(side_a, side_b)
                            single_edge = (piece_a.index, piece_b.index, relation)
                            # Finds how many times pieces should be rotated for mgc_ssd
                            rotations_a, rotations_b, mgc_specific_relation, piece_swap = \
                                Constants.get_mgc_rotation(side_a, side_b)
                            # Rotate piece so mgc_ssd can be correctly used
                            transformed_a = numpy.rot90(image_a, k=rotations_a)
                            transformed_b = numpy.rot90(image_b, k=rotations_b)
                            if piece_swap:
                                dissimilarity = Compatibility.mgc_ssd_compatibility(transformed_b, transformed_a,
                                                                                    mgc_specific_relation)
                            else:
                                dissimilarity = Compatibility.mgc_ssd_compatibility(transformed_a, transformed_b,
                                                                                    mgc_specific_relation)
                            # Identical to self.all_edges int his case
                            self.important_edges.append(single_edge)
                            self.weights[piece_a.index][piece_b.index][relation] = dissimilarity

        # Normalization step
        # TODO - Change this a bit if calculating matchlift weights
        normalized_weights = numpy.array(self.weights)
        for i, j, rel in self.important_edges:
            min_weight = min(self.weights[i, :, rel].min(), self.weights[:, j, rel].min())
            normalized_weights[i, j, rel] = self.weights[i, j, rel] / (min_weight + Constants.EPSILON)
        self.weights = normalized_weights
        elapsed_time = time.process_time() - t
        print("Elapsed time for ", str(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE), " pieces of 100 pixel size:",
              elapsed_time, "s")

    # Used for Known orientation
    # Used for Matchlift Known orientation
    def get_mgc(self):
        """
            Calculates the MGC between all pieces
        :return:
        """
        t = time.process_time()

        """
            The process of calculating the weights is performed for all 4 relations even two they are symmetric,
            thus wasting a bit of space but that is not an issue see later
        """
        for piece_a in self.pieces:
            image_a = piece_a.piece
            for piece_b in self.pieces:
                # Check if two pieces are the same, if they are skip the comparison
                if piece_a != piece_b:
                    # This loop gets the necessary 4 rotations of piece_a
                    for side_a in range(0, 4):
                        # Get the image from the Piece object
                        image_b = piece_b.piece
                        # Get the opposing side of piece b based on the side of piece a
                        side_b = Constants.get_combo_without_rotation(side_a)
                        # Checks if the pieces need to be swapped in order for the weights to be calculated correctly
                        # Look at Compatibility.mgc_ssd_compatibility to understand why
                        _, _, piece_swap = Constants.convert_relation(side_a, side_b)
                        # TODO - Figure out what that was doing?
                        # rot_a, rot_b = self.get_rotation_mgc(side_a, side_b)
                        relation = Constants.get_relation(side_a, side_b)
                        single_edge = (piece_a.index, piece_b.index, relation)
                        if piece_swap:
                            dissimilarity = Compatibility.mgc_ssd_compatibility(image_b, image_a, relation)
                            # Add the symmetric opposing edges also
                            self.all_edges.append(single_edge)
                            self.weights[piece_a.index, piece_b.index, relation] = dissimilarity
                        else:
                            dissimilarity = Compatibility.mgc_ssd_compatibility(image_a, image_b, relation)
                            # Add all edges, just in case they are needed
                            self.all_edges.append(single_edge)
                            # Add only the 2 edges between piece a and piece b, unnecessary to add symmetric edges also
                            self.important_edges.append(single_edge)
                            self.weights[piece_a.index, piece_b.index, relation] = dissimilarity

        # TODO - Change this a bit if calculating matchlift weights
        # Normalization step
        normalized_weights = numpy.array(self.weights)
        for i, j, rel in self.important_edges:
            min_weight = min(self.weights[i, :, rel].min(), self.weights[:, j, rel].min())
            normalized_weights[i, j, rel] = self.weights[i, j, rel] / (min_weight + Constants.EPSILON)

        self.weights = normalized_weights

        elapsed_time = time.process_time() - t
        print("Elapsed time for computing mgc", str(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE), " pieces of", Constants.PATCH_DIMENSIONS, "pixel size:",
              elapsed_time, "s")

        # TODO - Decide what to do with this
        # self.optimise_weights_from_matchlift()

    def optimise_weights_from_matchlift(self):
        for i, j, rel in self.matchlift_edges_0_4:
            self.weights_0_4[i, j, rel] = float(0)

    # Used for Known orientation
    # Used for Unknown orientation
    def save_weights_to_npy(self):
        """
            Saves the calculated weights into a numpy file for later use
        :return:
        :rtype:
        """

        # Check if the directory exists
        # if not pathlib.Path.is_dir(Constants.settings["output_path"]):
        #     raise Exception("Please specify correctly the \"output_path\" attribute of \"weights\"!")

        if Constants.settings["puzzle_type"] == Constants.KNOWN_ORIENTATION:
            self.get_mgc()
            string = Constants.settings["weight"]["output_path"] \
                + Constants.settings["name_of_image"] + str(len(self.pieces)) + "_no.npy"
        elif Constants.settings["puzzle_type"] == Constants.UNKNOWN_ORIENTATION:
            self.get_mgc_rotated()
            string = Constants.settings["weight"]["output_path"] + Constants.settings["name_of_image"] \
                + str(len(self.pieces)) + "_90.npy"
        else:
            raise Exception("Please specify the type of the puzzle correctly! Either \"known\" for puzzles with known "
                            "orientation or \"unknown\" for puzzles with unknown orientation.")
        numpy.save(string, self.weights)

    def recalculate_weights(self, pieces_of_interest):
        """
            TODO - Correct this as it is partially wrong
        :param pieces_of_interest:
        :type pieces_of_interest:
        :return:
        :rtype:
        """
        # Clearing the lists
        self.important_edges.clear()
        self.all_edges.clear()
        for u in pieces_of_interest:
            for v in pieces_of_interest:
                if u != v:
                    for side_a in range(0, 4):
                        # image_a = self.pieces[u].piece
                        # image_b = self.pieces[v].piece
                        side_b = Constants.get_combo_without_rotation(side_a)
                        _, _, piece_swap = Constants.convert_relation(side_a, side_b)
                        # rot_a, rot_b = self.get_rotation_mgc(side_a, side_b)
                        relation = Constants.get_relation(side_a, side_b)
                        single_edge = (u, v, relation)
                        if piece_swap:
                            self.all_edges.append(single_edge)
                        else:
                            self.important_edges.append(single_edge)
                            self.all_edges.append(single_edge)

    def sort_edges(self):
        """
            Sorts the edges from the smallest error to the biggest error
            This is done so Kruskal's algorithm can be applied
        :return:
        """
        self.important_edges.sort(key=lambda x: self.weights[x])
        self.all_edges.sort(key=lambda x: self.weights[x])

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
            # Get the pieces from the smaller chunks
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
        self.assembly_image(Constants.BIGGEST_CHUNK.piece_coordinates)
        elapsed_time = time.process_time() - t
        print("Elapsed time for solving big_cat_", str(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE), "_no", " pieces of", Constants.PATCH_DIMENSIONS, "pixel size:",
              elapsed_time, "s")

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

    # TODO - To be further updated for Unknown orientation
    def kruskal_alg(self):
        """
            Implementation of Kruskal's algorithm for finding the minimum spanning tree
            This is not a pure implementation of the algorithm, it is modified for the needs of the program
        :return: None
        :type: None
        """
        unsuccessful_merges = set()
        end_index = 0
        up_to = len(self.pieces) - 1  # We need V -1 edges for the minimal spanning tree
        for u_vertex, v_vertex, relation in self.important_edges:
            self.steps = self.steps + 1
            if end_index < up_to:
                weight = self.weights[u_vertex, v_vertex, relation]
                u = self.find_parent(parent=self.parent_set, i=u_vertex)
                v = self.find_parent(parent=self.parent_set, i=v_vertex)
                # Omits the formation of cycles
                if u != v:
                    #
                    u_tree = self.trees[u]
                    v_tree = self.trees[v]

                    # The sides of the pieces making the connection
                    side_a, side_b = Constants.get_reverse_combo(relation)
                    _, _, piece_swap = Constants.convert_relation(side_a, side_b)

                    evaluation = False  # Get the chunks we are working with
                    chunk_u = self.chunks[u]
                    chunk_v = self.chunks[v]

                    if piece_swap:
                        evaluation = chunk_v.add(v_vertex, u_vertex, side_b, side_a, chunk_u)
                    else:
                        evaluation = chunk_u.add(u_vertex, v_vertex, side_a, side_b, chunk_v)

                    if not evaluation:
                        if piece_swap:
                            united = v_tree.union(u_tree)
                            self.trees[v] = united
                            # Explicitly set the object that has been merged into None so we do not have to
                            # keep track of it
                            self.trees[u] = None
                            self.chunks[u] = None
                            self.parent_set[u] = v
                        else:
                            united = u_tree.union(v_tree)
                            self.trees[u] = united
                            # Explicitly set the object that has been merged into None so we do not have to
                            # keep track of it
                            self.trees[v] = None
                            self.chunks[v] = None
                            self.parent_set[v] = u

                        # TODO - Account for swapping or maybe not?
                        self.new_minimal_spanning_tree.append((u_vertex, v_vertex, relation, weight))

                        end_index = end_index + 1

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

    def find_parent(self, parent, i):
        if parent[i] == i:
            return i
        return self.find_parent(parent, parent[i])

    def initialize_positions_rotations(self):
        """
            Initializes the height and width of the puzzle as it is important to know
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
            self.initial_positions = numpy.full((int(sqrt_piece_count), int(sqrt_piece_count)),
                                                fill_value=Constants.VALUE_INITIALIZER, dtype="uint16")

        else:
            # If the puzzle is not symmetric we get the factors
            self.get_factors(len(self.pieces))
            self.initial_positions = numpy.full((Constants.WIDTH_RANGE, Constants.HEIGHT_RANGE),
                                                fill_value=Constants.VALUE_INITIALIZER, dtype="uint16")

        # Doing it the lazy way
        for i in range(0, Constants.WIDTH_RANGE):
            for j in range(0, Constants.HEIGHT_RANGE):
                self.initial_positions[i, j] = counter
                counter = counter + 1

        self.initial_positions = numpy.rot90(self.initial_positions)

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
            y_space = (y + 1) * 10
            x_space = (x + 1) * 10

            x_axis_pieces = int((self.original_width - x_space) / Constants.PATCH_DIMENSIONS)
            y_axis_pieces = int((self.original_height - y_space) / Constants.PATCH_DIMENSIONS)

            if x_axis_pieces == x and y_axis_pieces == y:
                print("Dimensions found")
                print("Y - Height:{0}  X - Width:{1}".format(y_axis_pieces, x_axis_pieces))
                not_found = False
            else:
                index = index + 1

        Constants.HEIGHT_RANGE = y_axis_pieces
        Constants.WIDTH_RANGE = x_axis_pieces

    def assembly_image(self, dictionary):
        """
            Puts together the solved puzzle by looking at the piece indices
            Iterate over the self.positions which is the assembly matrix, where for each (x, y) get the piece index
        :param dictionary
        :type dict
        :return: None
        """

        solution = numpy.zeros((Constants.PATCH_DIMENSIONS * Constants.HEIGHT_RANGE,
                               Constants.PATCH_DIMENSIONS * Constants.WIDTH_RANGE,
                               Constants.COLOUR_CHANNELS), dtype="uint8")

        for key in dictionary.keys():
            y, x = dictionary[key]
            y0 = y * Constants.PATCH_DIMENSIONS
            x0 = x * Constants.PATCH_DIMENSIONS
            y1 = y0 + Constants.PATCH_DIMENSIONS
            x1 = x0 + Constants.PATCH_DIMENSIONS

            solution[y0:y1, x0:x1] = self.pieces[key].piece

        openCV.imwrite(Constants.settings["output_path"] + Constants.settings["name_of_image"] + "_"
                       + str(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE) + "_"
                       + str(self.steps) + "_no.png", solution)

        # file = open("test.txt", mode="w")
        # for i in range(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE):
        #     for j in range(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE):
        #         for rel in range(16):
        #             weight = self.weights_0_4[i][j][rel]
        #             if rel == Constants.RIGHT_LEFT or rel == Constants.LEFT_RIGHT or rel == Constants.BOTTOM_TOP or rel == Constants.TOP_BOTTOM:
        #                 file.write(str(weight) + "\n")

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
