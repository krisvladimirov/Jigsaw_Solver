import Chunk as Chunk
import Constants as Constants


class Evaluation:
    def __init__(self):
        self.original_piece_locations = []
        self.original_piece_orientation = []
        # Index each piece and put a set which will contain the neighbours
        self.original_piece_neighbours = []

    def load_data(self):
        """

        :return:
        """
        try:
            with open(Constants.settings["evaluation"]["path_to_locations"], mode="r") as handler:
                raw_string = handler.readlines()
                for single_line in raw_string:
                    line = single_line.rstrip("\n")
                    y, x = line.split(",")
                    self.original_piece_locations.append((int(y), int(x)))
        except(IOError, OSError) as e:
            print("Could not open the file associated with the evaluation of piece placement/location! "
                  "Check path in settings.json")

        try:
            # TODO - Decide on how this will be saved
            with open(Constants.settings["evaluation"]["path_to_neighbours"], mode="r") as handler:
                raw_string = handler.readlines()
                # TODO - Decide the format
        except (IOError, OSError) as e:
            print("Could not open the file associated with the evaluation of piece neighbours! "
                  "Check path in settings.json")

        if Constants.settings["puzzle_type"] == Constants.KNOWN_ORIENTATION:
            try:
                with open(Constants.settings["evaluation"]["path_to_rotations"]) as handler:
                    raw = handler.readlines()
                    for single_line in raw:
                        k_rotations = single_line.rsplit("\n")
                        self.original_piece_orientation.append(int(k_rotations[0]))
            except (IOError, OSError) as e:
                print("Could not open the file associated with the evaluation of piece rotations! "
                      "Check path in settings.json")
        else:
            pass

    def evaluate(self, where_to):
        """
        :param where_to:
        :type where_to:
        :return:
        :rtype:
        """

        total_number_of_pieces = Constants.HEIGHT_RANGE * Constants.WIDTH_RANGE

        matched_pieces, unmatched_pieces = self.piece_evaluation()
        matched_rotations, unmatched_rotations = self.rotation_evaluation()

        # First line will contain the puzzle height, puzzle width, total amount of pieces,
        # patch size (i.e. how big a puzzle piece is)
        # puzzle height and width are how many pieces there would be on each axis
        first_line = str(Constants.HEIGHT_RANGE) + "," + str(Constants.WIDTH_RANGE) + "," \
            + str(total_number_of_pieces) + "," + str(Constants.PATCH_DIMENSIONS) + "\n"

        # Second line will contain how many pieces are correctly placed, how many are incorrectly placed,
        # and the percentage of correctly placed pieces
        second_line = str(matched_pieces) + "," + str(unmatched_pieces) + "," \
            + str(self.round_up((matched_pieces / total_number_of_pieces) * 100, number_of_digits=1)) + "\n"

        # TODO - DO IT
        third_line = "THE_NEIGHBOUR Ratio"

        # Forth line will contain the how many pieces are correctly rotated, how many are incorrectly rotated,
        # and the percentage of correctly rotated pieces
        forth_line = str(matched_rotations) + "," + str(unmatched_rotations) + "," \
            + str(self.round_up((matched_rotations / total_number_of_pieces) * 100, number_of_digits=1)) + "\n"

        # 2 line will always be the piece evaluation
        # 3 line will always be the neighbour evaluation
        # 4 line will always be the rotation evaluation
        file = open(where_to, mode="w")
        file.write(first_line)
        file.write(second_line)
        file.write(third_line)
        if Constants.settings["puzzle_type"] == Constants.UNKNOWN_ORIENTATION:
            file.write(forth_line)

        file.close()

    def piece_evaluation(self):
        positions = Constants.BIGGEST_CHUNK.piece_coordinates
        match = 0
        unmatched = 0
        for key, value in positions.items():
            if value == self.original_piece_locations[key]:
                match += 1
            else:
                unmatched += 1

        # print("Location")
        # print("How many matched:", match)
        # print("How many unmatched", unmatched)
        return match, unmatched

    def neighbour_evaluation(self):
        # TODO
        match = 0
        unmatched = 0

        return match, unmatched

    def rotation_evaluation(self):
        rotations = Chunk.Chunk.global_piece_rotations
        match = 0
        unmatched = 0
        for key, value in rotations.items():
            if (value + self.original_piece_orientation[key]) % 4 == 0:
                match += 1
            else:
                unmatched += 1

        # print("")
        # print("Orientation")
        # print("How many matched:", match)
        # print("How many unmatched", unmatched)
        return match, unmatched

    def round_up(self, number, number_of_digits=1):
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
