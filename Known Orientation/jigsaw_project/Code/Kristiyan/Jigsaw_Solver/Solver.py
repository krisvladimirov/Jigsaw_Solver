import numpy
import cv2 as openCV
import Piece as Piece
import Compatibility as Compatibility
import Constants as Constants
import Chunk as Chunk
import math
import h5py
import sys
from operator import add, sub
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

# pieces = []


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
            self.pieces.append(single_piece)

        # TODO - To be removed as it violates the out-of-boundary condition (kind of)
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
        if not pathlib.Path.exists(pathlib.Path(Constants.settings["solving"]["path_to_weights"])):
            raise Exception("Please specify correctly the \"output_path\" attribute of \"weights\"!")

        self.weights = numpy.load(Constants.settings["solving"]["path_to_weights"])

        # TODO - Update for weights which include the unknown orientation, thus 16 combinations
        for piece_a in self.pieces:
            for piece_b in self.pieces:
                # Check if two pieces are the same, if they are skip the comparison
                if piece_a != piece_b:
                    # This loops gets the necessary 4 rotations of piece_a
                    # TODO - More specifically this part has to be changed
                    for side_a in range(0, 4):
                        # Get the opposing side of piece b based on the side of piece a
                        # TODO - This part would also have to be updated for unknown orientations
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
    # TODO - Fixed for non rotated pieces

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

    def get_mgc_matchlift(self, writing_settings):
        """
            Calculate the MGC between all pieces including 4 or more correspondencies for matchlift
        :return:
        :rtype:
        """
        # TODO
        edges_with_correspondences = []
        num_correspondence = writing_settings["matchlift"]["num_correspondences"]

        if num_correspondence < 1:
            raise Exception(
                "The number of correspondences can't be smaller than 1, "
                "check \"num_correspondences\" in the settings.json")

        # # TODO - HEIGHT Range and WIDTH Range will be changed
        # weights_matchlift = numpy.full((Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE,
        #                                 Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE, 16,
        #                                 num_correspondence), fill_value=0,
        #                                dtype="float")
        # for piece_a in self.pieces:
        #     image_a = piece_a.piece
        #     # This loops get the necessary 4 rotations of piece_a
        #     for piece_b in self.pieces:
        #         if piece_a != piece_b:
        #             for side_a in range(0, 4):
        #                 image_b = piece_b.piece  # Get the image from the piece object
        #                 side_b = Constants.get_combo_without_rotation(side_a)
        #                 _, _, piece_swap = Constants.convert_relation(side_a, side_b)
        #                 relation = Constants.get_relation(side_a, side_b)
        #                 for correspondence in range(0, num_correspondence):
        #                     cropped_a = self.crop_it(image_a, correspondence, relation, num_correspondence)
        #                     cropped_b = self.crop_it(image_b, correspondence, relation, num_correspondence)
        #                     single_edge = (piece_a.index, piece_b.index, relation, correspondence)
        #                     if piece_swap:
        #                         dissimilarity = Compatibility.mgc_ssd_compatibility(cropped_b, cropped_a, relation)
        #                     else:
        #                         dissimilarity = Compatibility.mgc_ssd_compatibility(cropped_a, cropped_b, relation)
        #
        #                     edges_with_correspondences.append(single_edge)
        #                     weights_matchlift[piece_a.index, piece_b.index, relation, correspondence] = dissimilarity
        #
        # normalized_weights = numpy.full((Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE,
        #                                  Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE, 16, num_correspondence),
        #                                  fill_value=0, dtype="float")
        #
        # for i, j, rel, cor in edges_with_correspondences:
        #     min_weight = min(weights_matchlift[i, :, rel, cor].min(), weights_matchlift[:, j, rel, cor].min())
        #     normalized_weights[i, j, rel, cor] = normalized_weights[i, j, rel, cor] / (min_weight + Constants.EPSILON)
        #
        # file = open(Constants.settings["matchlift"]["output_path"] + Constants.settings["name_of_image"] + "_" + str(
        #     Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE) + "_" + str(num_correspondence) + "_cor.txt",
        #             mode="w")
        # for i in range(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE):
        #     for j in range(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE):
        #         for rel in range(16):
        #             for correspondence in range(num_correspondence):
        #                 weight = normalized_weights[i][j][rel][correspondence]
        #                 if rel == Constants.RIGHT_LEFT or rel == Constants.LEFT_RIGHT or rel == Constants.BOTTOM_TOP or rel == Constants.TOP_BOTTOM:
        #                     file.write(str(weight) + "\n")

    # Used for Unknown orientation
    # Used for Matchlift Unknown orientation
    def get_mgc_rotated(self):
        """
            Calculates the MGC between all pieces while taking the rotation into consideration
        :return:
        """

        print("Computing the weights (dissimilarities) between puzzle pieces with", Constants.settings["puzzle_type"],
              "orientation...")
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
        normalized_weights = numpy.array(self.weights)
        for i, j, rel in self.important_edges:
            min_weight = min(self.weights[i, :, rel].min(), self.weights[:, j, rel].min())
            normalized_weights[i, j, rel] = self.weights[i, j, rel] / (min_weight + Constants.EPSILON)
        self.weights = normalized_weights
        elapsed_time = time.process_time() - t
        print("Elapsed time for computing the weights (dissimilarities) between",
              str(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE), "pieces with", Constants.settings["puzzle_type"],
              "orientation and pixel size:",
              Constants.PATCH_DIMENSIONS, "in", elapsed_time, "S")

    # Used for Known orientation
    # Used for Matchlift Known orientation
    def get_mgc(self):
        """
            Calculates the MGC between all pieces
        :return:
        """
        print("Computing the weights (dissimilarities) between puzzle pieces with", Constants.settings["puzzle_type"],
              "orientation...")
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

        # Normalization step
        normalized_weights = numpy.array(self.weights)
        for i, j, rel in self.important_edges:
            min_weight = min(self.weights[i, :, rel].min(), self.weights[:, j, rel].min())
            normalized_weights[i, j, rel] = self.weights[i, j, rel] / (min_weight + Constants.EPSILON)

        self.weights = normalized_weights

        elapsed_time = time.process_time() - t
        print("Elapsed time for computing the weights (dissimilarities) between",
              str(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE), "pieces with", Constants.settings["puzzle_type"],
              "orientation and pixel size:",
              Constants.PATCH_DIMENSIONS, "in", elapsed_time, "S")

        # TODO - Decide what to do with this
        # self.optimise_weights_from_matchlift()

    def optimise_weights_from_matchlift(self):
        for i, j, rel in self.matchlift_edges_0_4:
            self.weights_0_4[i, j, rel] = float(0)

    def save_weights_to_npy(self, write_settings):
        """
            Saves the calculated weights into a numpy file for later use
            Used for Known orientation
            Used for Unknown orientation
        :param write_settings:
        :type write_settings: dict
        :return:
        :rtype:
        """

        # Check if the directory exists
        if not pathlib.Path.is_dir(pathlib.Path(write_settings["weights"]["output_path"])):
            raise Exception("Please specify correctly the \"output_path\" attribute of \"weights\"!")

        if Constants.settings["puzzle_type"] == Constants.KNOWN_ORIENTATION:
            self.get_mgc()
            string = write_settings["weights"]["output_path"] + Constants.settings["name_of_image"] + "_" + str(
                len(self.pieces)) + "_no.npy"
        elif Constants.settings["puzzle_type"] == Constants.UNKNOWN_ORIENTATION:
            self.get_mgc_rotated()
            string = write_settings["weights"]["output_path"] + Constants.settings["name_of_image"] \
                + "_" + str(len(self.pieces)) + "_90.npy"
        else:
            raise Exception("Please specify the type of the puzzle correctly! Either \"known\" for puzzles with known "
                            "orientation or \"unknown\" for puzzles with unknown orientation.")
        numpy.save(string, self.weights)

    def recalculate_weights(self, border_pieces, border_empty_spots, trimmed_pieces):
        """

        :param border_pieces: We don't need them
        :type border_pieces: We don't need them
        :param border_empty_spots: Kye -> coordinate, Value -> list of neighbours
        :type border_empty_spots: dict
        :param trimmed_pieces:
        :type trimmed_pieces: list
        :return:
        :rtype:
        """
        # Clearing the lists
        self.important_edges.clear()
        self.all_edges.clear()

        biggest_chunk_matrix = Constants.BIGGEST_CHUNK.chunk
        piece_coordinates_biggest_chunk = Constants.BIGGEST_CHUNK.piece_coordinates
        # Key -> coordinate, value -> puzzle piece
        best_fit_empty_positions = {}

        # The error we observe when placing a piece in an empty position. The total error are the combined dissimilarities
        # between the placed piece and any neighbours
        total_error = Constants.INFINITY
        # Temporary variable to hold the best suitable puzzle piece for an empty position
        best_match = Constants.VALUE_INITIALIZER

        new_border_pieces = []

        for key, neighbours in border_empty_spots.items():
            # For every key (the key is a coordinate) value is a list holding all neighbours of a coordinate
            y, x = key
            for trm_piece in trimmed_pieces:
                error = 0
                for n_piece in neighbours:
                    # The neighbour coordinates
                    n_y, n_x = piece_coordinates_biggest_chunk[n_piece]
                    # Offset coordinate from empty position to neighbour
                    off_set = (n_y - y, n_x - x)
                    # There is no care about the boundaries as the trimming process has dealt with
                    # fixing/moving/updating the boundaries
                    side_empty_spot = Constants.off_set_to_puzzle_side[off_set]
                    side_neighbour = Constants.get_combo_without_rotation(side_empty_spot)
                    relation = Constants.get_relation(side_empty_spot, side_neighbour)
                    weight = self.weights[trm_piece, n_piece, relation]
                    error = error + weight

                if error < total_error:
                    total_error = error
                    best_match = trm_piece

            total_error = Constants.INFINITY
            best_fit_empty_positions[key] = best_match
            trimmed_pieces.remove(best_match)
            # Update biggest chunk
            biggest_chunk_matrix[key] = best_match
            # Set the best_match for each empty spot

        # Now the pieces that we just placed will become the border pieces
        # TODO - Use the pieces that are left in trimmed pieces
        print()


        # # TODO - Once the top part is done
        # # TODO - 1. Take the rest of the pieces that do not have a place
        #
        #
        # # TODO - 1-  pieces_of_interest with other pieces of interest should not be considered
        # # TODO - 1.1 - first we consider pieces_of_interest only with refused_pieces, gives list_one
        # # TODO - 2 - secondly we consider connections between only between refused_pieces, gives list_two
        # # TODO - 3 - we sort the produced lists of each step individually
        # # TODO - 4 - append the second list and the end of the first
        #
        # # TODO - Choose more sensible names
        # list_one_all_edges = []
        # list_one_important_edges = []
        # list_two_all_edges = []
        # list_two_important_edges = []
        #
        # # Step 1 and 1.1
        # for u in pieces_of_interest:
        #     for v in refused_pieces:
        #         # Just a safety precaution, even thought it will never be false, to be tested
        #         if u != v:
        #             for side_a in range(0, 4):
        #                 side_b = Constants.get_combo_without_rotation(side_a)
        #                 _, _, piece_swap = Constants.convert_relation(side_a, side_b)
        #                 # rot_a, rot_b = self.get_rotation_mgc(side_a, side_b)
        #                 relation = Constants.get_relation(side_a, side_b)
        #                 single_edge = (u, v, relation)
        #                 if piece_swap:
        #                     list_one_all_edges.append(single_edge)
        #                 else:
        #                     list_one_all_edges.append(single_edge)
        #                     list_one_important_edges.append(single_edge)
        #
        # # Step 2
        # for u in refused_pieces:
        #     for v in refused_pieces:
        #         if u != v:
        #             for side_a in range(0, 4):
        #                 side_b = Constants.get_combo_without_rotation(side_a)
        #                 _, _, piece_swap = Constants.convert_relation(side_a, side_b)
        #                 # rot_a, rot_b = self.get_rotation_mgc(side_a, side_b)
        #                 relation = Constants.get_relation(side_a, side_b)
        #                 single_edge = (u, v, relation)
        #                 if piece_swap:
        #                     list_two_all_edges.append(single_edge)
        #                 else:
        #                     list_two_all_edges.append(single_edge)
        #                     list_two_important_edges.append(single_edge)
        #
        # # Step 3
        # # print("Before sorting")
        # # print("Important edges")
        # # print(list_two_important_edges)
        # # print("All edges")
        # # print(list_two_all_edges)
        # self.sort_edges([list_one_important_edges, list_one_all_edges])
        # self.sort_edges([list_two_important_edges, list_two_all_edges])
        #
        # # print("After sorting")
        # # print("Important edges")
        # # print(list_two_important_edges)
        # # print("All edges")
        # # print(list_two_all_edges)
        #
        # # Step 4
        # self.all_edges = list_one_all_edges + list_two_all_edges
        # self.important_edges = list_one_important_edges + list_two_important_edges
        #
        # # for u in pieces_of_interest:
        # #     for v in pieces_of_interest:
        # #         if u != v:
        # #             for side_a in range(0, 4):
        # #                 # image_a = self.pieces[u].piece
        # #                 # image_b = self.pieces[v].piece
        # #                 side_b = Constants.get_combo_without_rotation(side_a)
        # #                 _, _, piece_swap = Constants.convert_relation(side_a, side_b)
        # #                 # rot_a, rot_b = self.get_rotation_mgc(side_a, side_b)
        # #                 relation = Constants.get_relation(side_a, side_b)
        # #                 single_edge = (u, v, relation)
        # #                 if piece_swap:
        # #                     self.all_edges.append(single_edge)
        # #                 else:
        # #                     self.important_edges.append(single_edge)
        # #                     self.all_edges.append(single_edge)


    def sort_edges(self, lists_to_sort):
        """
            Gets a list containing lists of edges as show:
            lists_to_sort = [self.important_edges, self.all_edges, and so on],
            and it sorts the edges according to the weights.
            The edges are sorted from the smallest edge weight (error) to the biggest error
            This is done so Kruskal's algorithm can be applied
        :param lists_to_sort:
        :type: list
        :return:
        """
        for lst in lists_to_sort:
            lst.sort(key=lambda x: self.weights[x])

    def find_mst(self):
        """

        :return:
        :rtype:
        """
        # Keeps track of whether or not all pieces are located in one chunk
        not_in_one = True
        # Records time
        # TODO - Add infinity counter to JSON parameters
        infinity_counter = 0
        t = time.process_time()
        self.sort_edges([self.important_edges, self.all_edges])
        while not_in_one:
            self.kruskal_alg()
            self.find_biggest_chunk()
            # trimmed_pieces = self.trim_biggest_chunk()
            # # refused_pieces = []
            # if not trimmed_pieces \
            #         and len(Constants.BIGGEST_CHUNK.piece_coordinates) == (
            #         Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE):
            #     # Change the condition whenever all pieces are located in one chunk
            #     not_in_one = False
            # else:
            #     border_pieces, border_empty_spots = self.find_border_pieces(Constants.BIGGEST_CHUNK.chunk)
            #     # self.reinitialize_parameters(refused_pieces)
            #     self.recalculate_weights(border_pieces, border_empty_spots, trimmed_pieces)
            #
            # infinity_counter = infinity_counter + 1

            not_in_one = False
            print("Infinity counter at ->", infinity_counter)
        # self.assembly_image(Constants.BIGGEST_CHUNK.piece_coordinates)
        self.just_assemble_it(Constants.BIGGEST_CHUNK.piece_coordinates)
        elapsed_time = time.process_time() - t
        print("Elapsed time solving puzzle with", str(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE), "piece of",
              Constants.PATCH_DIMENSIONS, "pixel size in", elapsed_time, "S")

    def trim_biggest_chunk(self):
        """

        :return:
        :rtype:
        """

        # the numpy matrix holding piece location
        biggest_chunk = Constants.BIGGEST_CHUNK.chunk
        parent = Constants.BIGGEST_CHUNK.parent_index

        # kurva = self.find_parent(self.parent_set, 116)

        # TODO - Comment on this part
        # TODO - Something is wrong here
        fewest_empty_space_counter = Constants.INFINITY
        start_y = Constants.VALUE_INITIALIZER
        start_x = Constants.VALUE_INITIALIZER
        end_y = Constants.VALUE_INITIALIZER
        end_x = Constants.VALUE_INITIALIZER

        height, width = biggest_chunk.shape
        diff_in_height = height - Constants.HEIGHT_RANGE
        diff_in_width = width - Constants.WIDTH_RANGE

        # TODO - decide what to do with these
        if diff_in_height < 0:
            pass
        if diff_in_width < 0:
            pass

        for x in range(diff_in_width + 1):
            for y in range(diff_in_height + 1):
                temp_trimmed_chunk = biggest_chunk[y:y + Constants.HEIGHT_RANGE, x:x + Constants.WIDTH_RANGE]
                num_of_empty_places = numpy.count_nonzero(temp_trimmed_chunk == -1)
                if num_of_empty_places < fewest_empty_space_counter:
                    fewest_empty_space_counter = num_of_empty_places
                    start_x = x
                    start_y = y
                    end_x = x + Constants.WIDTH_RANGE
                    end_y = y + Constants.HEIGHT_RANGE

        all_pieces = [i for i in range(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE)]
        trimmed_pieces = list()
        non_trimmed_pieces = set(biggest_chunk[start_y:end_y, start_x:end_x].flatten())

        if -1 in non_trimmed_pieces:
            non_trimmed_pieces.remove(-1)

        for i in all_pieces:
            if i not in non_trimmed_pieces:
                trimmed_pieces.append(i)

        # If the parent piece is trimmed from the biggest chunk
        if parent in trimmed_pieces:
            # TODO - If the parent piece is inside the biggest chunk after trimming -> no problem
            # TODO - If the parent piece is outside the biggest chunk after trimming and does not have any parents
            #  -> It is its own parent as
            print("Yep it is in")
            pass
        # If the parent is inside the biggest chunk, perform normal procedure
        else:
            # Reset the chunks of pieces that have been trimmed
            # Reset the parent of the trimmed pieces to be itself
            # Reset the trees of the trimmed pieces, thus create a new tree for each trimmed piece
            # TODO - Missed important steep
            # TODO - Need to update any piece that has a parent part of the trimmed pieces that ain't good

            # TODO - 1.1 Find any children of of every trimmed piece
            # TODO - 1.2 Find the parents of the trimmed pieces

            # Set the parent of all pieces in the biggest chunk to be the same, i.e. the parent of the biggest chunk
            for index in non_trimmed_pieces:
                self.parent_set[index] = parent

            for index in trimmed_pieces:
                chunk = Chunk.Chunk(index)
                self.chunks[index] = chunk
                self.parent_set[index] = index
                self.trees[index] = {index}

            # Update the chunk after trimming some pieces
            Constants.BIGGEST_CHUNK.chunk = biggest_chunk[start_y:end_y, start_x:end_x]
            # Update the biggest chunk dimensions
            Constants.BIGGEST_CHUNK.current_height = Constants.HEIGHT_RANGE
            Constants.BIGGEST_CHUNK.current_width = Constants.WIDTH_RANGE
            # Update the coordinates of pieces inside the biggest chunk after the trimming process
            Constants.BIGGEST_CHUNK.piece_coordinates.clear()
            Constants.BIGGEST_CHUNK.update_piece_coordinates()

        return trimmed_pieces

    # def reinitialize_parameters(self, refused_pieces):
    #     """
    #         Resets the chunks of refused pieces\n
    #         Resets the parent of a refused piece to be itself
    #         Resets the trees, thus creating a new tree for each refused piece
    #     :param refused_pieces:
    #     :type refused_pieces:
    #     :return:
    #     :rtype:
    #     """
    #     for index in refused_pieces:
    #         chunk = Chunk.Chunk(index)
    #         self.chunks[index] = chunk
    #         self.parent_set[index] = index
    #         self.trees[index] = {index}

    @staticmethod
    def find_border_pieces(chunk_matrix):
        """
            Finds all border pieces of a chunk
        :param chunk_matrix:
        :type chunk_matrix: ndarray
        :return:
        :rtype:
        """
        tmp = Constants.BIGGEST_CHUNK.chunk
        # Take the height and width of the chunk we are working with
        h, w = chunk_matrix.shape
        # An array of tuples where there is and empty position (y, x)
        empty_spots = numpy.argwhere(chunk_matrix < 0)
        border_empty_spots = {}
        border_pieces = set()
        for coordinate in empty_spots:
            border_empty_spots[tuple(coordinate)] = list()
        for coordinate in empty_spots:
            # The four possible off-sets in such order [Right, Bottom, Left, Top]
            for off_set in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                # y, x coordinate where we will check for a border piece
                y, x = map(add, tuple(coordinate), off_set)
                # Prevents the coordinate from going out of bound
                if (y >= 0 and x >= 0) and (y < h and x < w):
                    lookup_value = chunk_matrix[y][x]
                    # If the y, x position is not empty we have found a border piece
                    if lookup_value != -1:
                        border_pieces.add(lookup_value)
                        border_empty_spots[tuple(coordinate)].append(lookup_value)

        border_empty_spots = dict(filter(lambda elem: len(elem[1]) > 0, border_empty_spots.items()))
        return border_pieces, border_empty_spots

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
                        evaluation = chunk_v.add_together(v_vertex, u_vertex, side_b, side_a, chunk_u)
                    else:
                        evaluation = chunk_u.add_together(u_vertex, v_vertex, side_a, side_b, chunk_v)

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

                        off_set = Constants.get_off_set(side_a, side_b)
                        coordinate_u = chunk_u.piece_coordinates[u_vertex]
                        coordinate_v = chunk_u.piece_coordinates[v_vertex]

                        difference = tuple(map(sub, coordinate_v, coordinate_u))

                        if off_set != difference:
                            raise Exception("Something got fucked during chunk merging!")

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
            self.initial_positions = numpy.full((Constants.HEIGHT_RANGE, Constants.WIDTH_RANGE),
                                                fill_value=Constants.VALUE_INITIALIZER, dtype="uint16")

        # Doing it the lazy way
        for i in range(0, Constants.HEIGHT_RANGE):
            for j in range(0, Constants.WIDTH_RANGE):
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

        openCV.imwrite(Constants.settings["solving"]["output_path"] + Constants.settings["name_of_image"] + "_"
                       + str(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE) + "_"
                       + str(self.steps) + "_no.png", solution)

        # file = open("test.txt", mode="w")
        # for i in range(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE):
        #     for j in range(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE):
        #         for rel in range(16):
        #             weight = self.weights_0_4[i][j][rel]
        #             if rel == Constants.RIGHT_LEFT or rel == Constants.LEFT_RIGHT or rel == Constants.BOTTOM_TOP or rel == Constants.TOP_BOTTOM:
        #                 file.write(str(weight) + "\n")

    def just_assemble_it(self, dictionary):
        """
            Used for testing purposes to assemble a incomplete image
        :return:
        :rtype:
        """
        height, width = Constants.BIGGEST_CHUNK.chunk.shape
        solution = numpy.zeros((Constants.PATCH_DIMENSIONS * height,
                                Constants.PATCH_DIMENSIONS * width,
                                Constants.COLOUR_CHANNELS), dtype="uint8")

        for key in dictionary.keys():
            y, x = dictionary[key]
            y0 = y * Constants.PATCH_DIMENSIONS
            x0 = x * Constants.PATCH_DIMENSIONS
            y1 = y0 + Constants.PATCH_DIMENSIONS
            x1 = x0 + Constants.PATCH_DIMENSIONS

            solution[y0:y1, x0:x1] = self.pieces[key].piece

        openCV.imwrite("no_trimming_testing_" + Constants.settings["solving"]["output_path"] + Constants.settings["name_of_image"] + "_"
                       + str(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE) + "_"
                       + str(self.steps) + "_no.png", solution)

    def assemble_after_each_step(self):
        pass

    # def trim_biggest_chunk(self):
    #     """
    #
    #     :return:
    #     :rtype:
    #     """
    #
    #     # TODO - Make a new biggest chunk that has only pieces that are within the max allowable HEIGHT and WIDTH
    #     # TODO - Try and keep the parent the same as in the chunk we are copying from
    #             # TODO - Select a new parent if that is not possible and perform the appropriate parent changes in self.parents
    #     new_chunk = numpy.full((Constants.HEIGHT_RANGE, Constants.WIDTH_RANGE), fill_value=Constants.VALUE_INITIALIZER,
    #                            dtype="int16")
    #
    #     for y in range(Constants.HEIGHT_RANGE):
    #         for x in range(Constants.WIDTH_RANGE):
    #             pass

    def find_biggest_chunk(self):
        """
            Finds the chunk containing most puzzle pieces and saves it in the constants for later use
        :return:
        :rtype:
        """
        max_size = -1
        biggest_chunk_at_index = 0
        for index in range(len(self.chunks)):
            # Skip uninitialized chunks in the array
            if self.chunks[index] is not None:
                chunk_len = len(self.chunks[index].piece_coordinates)
                # Compare sizes and find biggest chunk
                if max_size < chunk_len:
                    max_size = chunk_len
                    biggest_chunk_at_index = index

        Constants.BIGGEST_CHUNK = self.chunks[biggest_chunk_at_index]

    def get_pieces_without_a_place(self):
        """
            Gets all pieces without a position in the biggest chunk
        :return:
        :rtype: set
        """
        piece_without_location = set()
        for index in range(len(self.chunks)):
            if self.chunks[index] is not None and self.chunks[index] != Constants.BIGGEST_CHUNK:
                pieces = set(numpy.ravel(self.chunks[index].chunk))
                # Remove empty positions, as we are not interested in them
                if Constants.VALUE_INITIALIZER in pieces:
                    pieces.remove(Constants.VALUE_INITIALIZER)
                piece_without_location = piece_without_location.union(pieces)
                # Deallocate the chunk for later use
                self.chunks[index] = None

        return piece_without_location
