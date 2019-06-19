import Chunk as Chunk
import Constants as Constants


class Evaluation:
    def __init__(self):
        self.original_piece_locations = []
        self.original_piece_orientation = []
        # Index each piece and put a set which will contain the neighbours
        # TODO - Update how WIDTH and HEIGHT are calculated (based on paper) and what I have written in my dissertation
        self.original_piece_neighbours = [set() for f in range(Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE)]

    def load_data(self):
        """

        :return:
        """
        try:
            # Read data associated with positions
            with open(Constants.settings["evaluation"]["path_to_positions"], mode="r") as handler:
                raw_string = handler.readlines()
                for single_line in raw_string:
                    line = single_line.rstrip("\n")
                    y, x = line.split(",")
                    self.original_piece_locations.append((int(y), int(x)))
        except(IOError, OSError) as e:
            print("Could not open the file associated with the evaluation of piece position/location! "
                  "Check path in settings.json")

        try:
            with open(Constants.settings["evaluation"]["path_to_neighbours"], mode="r") as handler:
                raw_string = handler.readlines()
                piece_index = 0
                counter = 0
                for single_line in raw_string:
                    line = single_line.rstrip("\n")
                    list_of_neighbours = line.split(",")
                    for neighbour in list_of_neighbours:
                        self.original_piece_neighbours[piece_index].add(int(neighbour))
                        counter = counter + 1
                    piece_index += 1

                print(counter)
        except (IOError, OSError) as e:
            print("Could not open the file associated with the evaluation of piece neighbours! "
                  "Check path in settings.json")

        if Constants.settings["puzzle_type"] == Constants.UNKNOWN_ORIENTATION:
            try:
                # Read data associated with rotations if unknown orientaito
                with open(Constants.settings["evaluation"]["path_to_rotations"]) as handler:
                    raw = handler.readlines()
                    for single_line in raw:
                        k_rotations = single_line.rsplit("\n")
                        self.original_piece_orientation.append(int(k_rotations[0]))
            except (IOError, OSError) as e:
                print("Could not open the file associated with the evaluation of piece rotations! "
                      "Check path in settings.json")
        else:
            # Known orientation skip this step
            pass

    def evaluate(self):
        """

        :return:
        :rtype:
        """

        total_number_of_pieces = Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE

        matched_pieces, unmatched_pieces = self.piece_evaluation()
        correct_neighbours, incorrect_neighbours = self.neighbour_evaluation()
        rotation_info = None
        if Constants.settings["puzzle_type"] == Constants.UNKNOWN_ORIENTATION:
            matched_rotations, unmatched_rotations = self.rotation_evaluation()
            # Forth line will contain the how many pieces are correctly rotated, how many are incorrectly rotated,
            # and the percentage of correctly rotated pieces
            rotation_info = "Piece rotation (correct/incorrect):" + str(matched_rotations) + " / " + str(
                unmatched_rotations) + "\nRatio of correctly rotated pieces to total number of pieces: " + str(
                self.round_up((matched_rotations / total_number_of_pieces) * 100, number_of_digits=1)) + "\n"
        total_number_of_neighbours = correct_neighbours + incorrect_neighbours

        # Puzzle info will contain the puzzle height, puzzle width, total amount of pieces,
        # patch size (i.e. how big a puzzle piece is)
        # puzzle height and width are how many pieces there would be on each axis
        puzzle_info = "Puzzle height (in pieces): " + str(Constants.HEIGHT_RANGE) + "\nPuzzle width (in pieces): " \
            + str(Constants.WIDTH_RANGE) + "\nPuzzle piece size (in pixels): " + str(Constants.PATCH_DIMENSIONS) \
            + "\nTotal number of pieces: " + str(total_number_of_pieces) + "\n"

        # Position info will contain how many pieces are correctly placed, how many are incorrectly placed,
        # and the percentage of correctly placed pieces
        position_info = "Piece placement (correct/incorrect): " + str(matched_pieces) + " / " + str(unmatched_pieces) \
            + "\nRatio of correctly placed to total number: " \
            + str(self.round_up((matched_pieces / total_number_of_pieces) * 100, number_of_digits=1)) + "\n"

        # Neighbour info will contain how many pieces have their correct neighbour, how many have an incorrect neighbour
        # and the percentage of correct neighbours placements
        neighbour_info = "Neighbour placement (correct/incorrect): " + str(correct_neighbours) + " / " + str(
            incorrect_neighbours) + "\nRatio of correctly neighbours to total number of neighbours: " + str(
            self.round_up((correct_neighbours / total_number_of_neighbours) * 100, number_of_digits=1)) + "\n"

        # 2 line will always be the piece evaluation
        # 3 line will always be the neighbour evaluation
        # 4 line will always be the rotation evaluation
        file = open(Constants.settings["evaluation"]["save_evaluation_to"], mode="w")
        file.write(puzzle_info)
        file.write(position_info)
        file.write(neighbour_info)
        if Constants.settings["puzzle_type"] == Constants.UNKNOWN_ORIENTATION:
            file.write(rotation_info)

        file.close()

    def piece_evaluation(self):
        """

        :return:
        :rtype:
        """
        positions = Constants.BIGGEST_CHUNK.piece_coordinates
        match = 0
        unmatched = 0
        for key, value in positions.items():
            if value == self.original_piece_locations[key]:
                match += 1
            else:
                unmatched += 1

        return match, unmatched

    def neighbour_evaluation(self):
        """

        :return:
        :rtype:
        """
        matrix_chunk = Constants.BIGGEST_CHUNK.chunk
        match = 0
        unmatched = 0

        # For every piece starting from 0
        for key in range(len(self.original_piece_neighbours)):
            # Find the location/coordinates/position of piece i in the biggest assembled chunk
            y, x = Constants.BIGGEST_CHUNK.piece_coordinates[key]
            # Iterate over the 3x3 region, where piece i is in the center of this region
            for yy in range(y - 1, y + 2, 1):
                for xx in range(x - 1, x + 2, 1):
                    if not ((xx < 0 or xx > Constants.WIDTH_RANGE - 1) or (yy < 0 or yy > Constants.HEIGHT_RANGE - 1)):
                        # Grab the neighbouring piece index located at (yy, xx) from the solved chunk
                        possible_neighbour = matrix_chunk[yy][xx]
                        # Compute to see if it 1 unit away from a piece we are looking at
                        # If it is more than one we do not consider it a neighbour, i.e. cases where is diagonally
                        # placed to another one
                        distance = abs((y + x) - (yy + xx))
                        if distance == 1:
                            # Compare with the ground truth to see if it is an actual neighbour
                            if possible_neighbour in self.original_piece_neighbours[key]:
                                match += 1
                            else:
                                unmatched += 1

        return match, unmatched

    def rotation_evaluation(self):
        """

        :return:
        :rtype:
        """
        rotations = Chunk.Chunk.global_piece_rotations
        match = 0
        unmatched = 0
        for key, value in rotations.items():
            if (value + self.original_piece_orientation[key]) % 4 == 0:
                match += 1
            else:
                unmatched += 1

        return match, unmatched

    def round_up(self, number, number_of_digits=1):
        """

        :param number:
        :type number:
        :param number_of_digits:
        :type number_of_digits:
        :return:
        :rtype:
        """
        # start by just rounding the number, as sometimes this rounds it up
        result = round(number, number_of_digits if number_of_digits else 0)
        if result < number:
            # whoops, the number was rounded down instead, so correct for that
            if number_of_digits:
                # use the type of number provided, e.g. float, decimal, fraction
                Numerical = type(number)
                # add the digit 1 in the correct decimal place
                result += Numerical(10) ** -number_of_digits
                # may need to be tweaked slightly if the addition was inexact
                result = round(result, number_of_digits)
            else:
                result += 1  # same as 10 ** -0 for precision of zero digits
        return result
