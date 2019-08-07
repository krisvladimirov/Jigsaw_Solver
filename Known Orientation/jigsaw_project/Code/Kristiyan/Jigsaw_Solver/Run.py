import cv2 as openCV
import Detector as Detector
import Solver as Solver
import json
import Evaluation as Evaluation
import Constants as Constants


def load_puzzle():
    """
        Loads the image containing the puzzle
    :return:
    """
    # Loading the image containing the puzzle
    print("Loading: ", Constants.settings["path_to_image"])
    print("Starting extracting of pieces...")
    # Extracting the puzzle pieces
    extracted_pieces, dimensions, og_dimensions = Detector.main(Constants.settings["path_to_image"])
    print(str(len(extracted_pieces)), "pieces have been extracted successfully, with dimensions -> Height:",
          dimensions[0], " Width:", dimensions[1])
    return extracted_pieces, dimensions, og_dimensions


def solve(solver):
    """
        TODO
    :param solver:
    :type solver: S
    :return:
    :rtype:
    """
    # Check if weights are provided
    if str.lower(Constants.settings["solving"]["path_to_weights"]) != Constants.EMPTY_PATH:
        solver.load_weights()  # Load weights for the solver, if provided
    else:
        solver.get_mgc()  # Computes weights, if not provided

    # Check if matchlift data has been provided
    if str.lower(Constants.settings["solving"]["path_to_matchlift_data"]) != Constants.EMPTY_PATH:
        # Load matchlift data for the optimizer, if provided
        # Otherwise don't do anything
        # TODO - Complete the below function
        solver.load_matchlift_data()

    solver.create_chunks()  # Build the chunks for the MST
    solver.find_mst()  # Performs the constrained MST algorithm


def write_data(solver):
    """
        Saves piece weights or matchlift weights
    :param solver:
    :return:
    """

    # Puzzle weights
    if Constants.settings["writing"]["weights"]["perform"] == Constants.YES:
        solver.save_weights_to_npy(Constants.settings["writing"])
    elif Constants.settings["writing"]["weights"]["perform"] != Constants.YES \
            or Constants.settings["writing"]["weights"]["perform"] != Constants.NO:
        print("Did not pre-calculate weights, \"mode\" == write "
              "but \"weight\" perform is \"no\".\nCheck the settings.json if a mistake was made\n")

    # Matchlift data
    if Constants.settings["writing"]["matchlift"]["perform"] == Constants.YES:
        # TODO
        solver.get_mgc_matchlift(Constants.settings["writing"])
    elif Constants.settings["writing"]["matchlift"]["perform"] != Constants.NO \
            or Constants.settings["writing"]["matchlift"]["perform"] != Constants.YES:
        print("Did not calculate correspondences for matchlift, \"mode\" == write "
              "but \"matchlift\" perform is \"yes\".\nCheck the settings.json if a mistake was made\n")


def ask_to_rotate(solver):
    """
        TODO
    :param solver:
    :type solver:
    :return:
    :rtype:
    """
    actions = {"No", "Yes", "no", "yes"}
    actions_1 = {"Yes", "yes"}
    performed_action = "No"
    print("")
    height, width, _ = solver.solution.shape
    resized = openCV.resize(solver.solution, (int(width * 0.25), int(height * 0.25)), interpolation=openCV.INTER_AREA)
    openCV.imshow("Solved", resized)
    openCV.waitKey(0)
    openCV.destroyAllWindows()
    while performed_action in actions:
        performed_action = input("Would you like to rotate the image: Yes/No/Exit\nYour choice: ")
        while performed_action in actions_1:
            print("[ 1 -> rotate by 90 degrees, 2 -> rotate by 180 degrees, 3 -> rotate by 270 degrees ]")
            rotate_by_how_much = input("By how much: ")
            if rotate_by_how_much.isdigit():
                if int(rotate_by_how_much) > 3 or int(rotate_by_how_much) < 1:
                    print("Not in the specified range, try again or type No to stop")
                else:
                    print("Rotating by:", rotate_by_how_much)
                    # solver.solution = numpy.rot90(solver.solution, k=int(rotate_by_how_much))
                    solver.rotate_after_completion(int(rotate_by_how_much))
                    height, width, _ = solver.solution.shape
                    resized = openCV.resize(solver.solution, (int(width * 0.4), int(height * 0.4)),
                                            interpolation=openCV.INTER_AREA)
                    openCV.imshow("Solved and rotated by: " + rotate_by_how_much, resized)
                    openCV.waitKey(0)
                    openCV.destroyAllWindows()
                    performed_action = input("If you want to rotate it again: Yes; No to stop\nYour choice: ")
            else:
                performed_action = "No"
        performed_action = "exit"

    print("Finished")


def start():
    """
        Reads the json data provided by the user
        The json contains:
        1. The mode of the operation -> You can either 'solve' a puzzle or 'write' data i.e. MGC data or Matchlift data
        2. The type of puzzle the program will solve -> 'known' orientation or 'unknown' orientation
        3. Name of the image -> under what name the processed image will be saved as
        4. Solving -> defines the parameters related to solving a puzzle
            4.1 The path to the image -> where you are loading the image from
            4.2 The path to the weights -> where you are loading the weights from
            4.3 The path to matchlift -> where you are loading the matchlift data from
            4.4 The output path of the image -> where the assembled image will be saved
            4.5 Dimensions -> Not implemented yet
        5. Writing -> all the data associated with pre-calculating MGC or computing Matchlift
            -> each perform specifies whether or not a certain operation should be performed
            -> output path specifies where the data will be saved
            -> number of correspondences is a specific matchlift parameter
        6. Evaluation of assembled puzzle -> all data needed to perform an evaluation of a assembled puzzle
            -> perform specifies whether or not the evaluation should be done
            -> Path to the correct positions of the puzzle
            -> Path to the correct neighbours of the puzzle
            -> Path to the correct rotations of the puzzle
    :return:
    """

    # Opens the json file containing the settings and saves it directly to a dictionary
    with open("settings.json", 'r') as read_file:
        Constants.settings = json.load(read_file)

    # Solver object, also prepares the solver for what would follow next
    solver = Solver.Solver()
    extracted_pieces, dimensions, og_dimensions = load_puzzle()
    solver.prepare_solver(extracted_pieces, dimensions, og_dimensions)

    ev = None

    # Will throw and error if the file of wrong size
    if str.lower(Constants.settings["evaluation"]["perform"]) == Constants.YES \
            and str.lower(Constants.settings["mode"]) == Constants.SOLVE:
        ev = Evaluation.Evaluation()
        ev.load_data()
        ev.check_file_size(solver)

    # Solving a puzzle
    if str.lower(Constants.settings["mode"]) == Constants.SOLVE:
        solve(solver)

    # Writing some data, i.e. weights or matchlift
    elif str.lower(Constants.settings["mode"]) == Constants.WRITE:
        write_data(solver)
    else:
        raise Exception("Please specify the mode correctly! Either read if you want to \"solve\" a puzzle or \"write\" "
                        "if you want to save some data, i.e. matchlift data or just puzzle weights.")

    # Check if there is an evaluation provided
    # Perform evaluation if it is provided
    if str.lower(Constants.settings["mode"]) != Constants.WRITE:
        if str.lower(Constants.settings["evaluation"]["perform"]) == Constants.YES:
            ev.evaluate()

        elif str.lower(Constants.settings["evaluation"]["perform"]) != Constants.NO \
                and str.lower(Constants.settings["evaluation"]["perform"]) != Constants.YES:
            print(Constants.settings["evaluation"]["perform"])
            raise Exception("Please specify correctly the \"perform\" attribute of \"evaluation\"!. If you want to perform "
                            "evaluation on the solved puzzle type \"yes\", if not type \"no\".")


if __name__ == "__main__":
    start()
