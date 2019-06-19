import argparse
import os
import cv2 as openCV
import Detector as Detector
import Solver as Solver
import json
import Evaluation as Evaluation
import glob
import Constants as Constants


def main():

    # do_matchlift_weights()
    do_normal_solving()
    # pre_calculate_weights()


def pre_calculate_weights():
    path_kris = "../output/no_rotation/kris/"
    path_greece = "../output/no_rotation/greece/"
    path_big_cat = "../output/no_rotation/big_cat/"
    path_star_wars = "../output/no_rotation/star_wars/"

    # kris_files = [f for f in glob.glob(path_kris + "*.png")]
    # greece_files = [f for f in glob.glob(path_greece + "*.png")]
    # star_wars_files = [f for f in glob.glob(path_star_wars + "*.png")]
    # big_cat_files = [f for f in glob.glob(path_big_cat + "*.png")]

    # for path in kris_files:
    #     extracted_pieces, dimensions, og_dimensions = Detector.main(path)
    #     solver = Solver.Solver()
    #     solver.start_solving(extracted_pieces, dimensions, og_dimensions)
    #     solver.save_weights_to_npy(root_folder="../weights", child_folder="/kris/", name="kris")
    #     solver = None
    #
    # print("Completed kris")
    #
    # for path in greece_files:
    #     extracted_pieces, dimensions, og_dimensions = Detector.main(path)
    #     solver = Solver.Solver()
    #     solver.start_solving(extracted_pieces, dimensions, og_dimensions)
    #     solver.save_weights_to_npy(root_folder="../weights", child_folder="/greece/", name="greece")
    #     solver = None
    #
    # print("Completed greece")

    # for path in star_wars_files:
    #     extracted_pieces, dimensions, og_dimensions = Detector.main(path)
    #     solver = Solver.Solver()
    #     solver.start_solving(extracted_pieces, dimensions, og_dimensions)
    #     solver.save_weights_to_npy(root_folder="../weights", child_folder="/star_wars/", name="star_wars")
    #     solver = None
    #
    # print("Completed star_wars")
    #
    # for path in big_cat_files:
    #     extracted_pieces, dimensions, og_dimensions = Detector.main(path)
    #     solver = Solver.Solver()
    #     solver.start_solving(extracted_pieces, dimensions, og_dimensions)
    #     solver.save_weights_to_npy(root_folder="../weights", child_folder="/big_cat/", name="big_cat")
    #     solver = None
    #
    # print("Completed big_cat")


def do_normal_solving():

    """

    :return:
    :rtype:
    """
    """
        LOOK AT Solver.start_solving() function for more info on option!
    """
    OPTION = 0
    path_to_image = ""
    path_to_weight = ""
    path_to_locations = ""
    path_to_rotations = ""
    save_evaluation_to = ""
    # path_to_matchlift_data = "big_cat_9.mat"
    # num_correspondences = 1
    # output_path = ""

    print("Loading: ", path_to_image)
    print("Starting extracting of pieces")
    extracted_pieces, dimensions, og_dimensions = Detector.main(path_to_image)
    print(str(len(extracted_pieces)), "pieces have been extracted successfully")

    print("Preparing Solver program...")
    solver = Solver.Solver()
    print("Loading the matchlift weights...")
    # solver.read_cycle_data(path_to_matchlift_data, num_correspondences, len(extracted_pieces))
    # print("Finished reading machlift data")
    # print(solver.matchlift_weights_0_4)
    print("Starting Solver...")
    solver.start_solving(extracted_pieces, dimensions, og_dimensions, path_to_weight, OPTION)

    # print("Evaluation process commence...")
    # ev = Evaluation.Evaluation()
    # ev.load_data(path_to_locations, path_to_rotations)
    # ev.evaluate(save_evaluation_to)
    # ev.piece_evaluation()


def do_matchlift_weights():
    path = "../output/no_rotation/big_cat_16_no.png"
    path_to_weight = "../weights/kris/kris_16_no.npy"
    output_path = "./"
    name_of_image = "big_cat"
    how_many_correspondences = 1

    print("Loading: ", path)
    print("Starting extracting of pieces")
    extracted_pieces, dimensions, og_dimensions = Detector.main(path)
    print(str(len(extracted_pieces)), "pieces have been extracted successfully")
    print("Preparing Solver for Matchlift...")
    solver = Solver.Solver()
    print("Solver computing correspondences for MatchLift...")
    solver.start_solving(extracted_pieces, dimensions, og_dimensions, path_to_weight)
    solver.get_mgc_matchlift(output_path + name_of_image, how_many_correspondences)


def load_puzzle():
    """

    :return:
    """
    # Loading the image containing the puzzle
    print("Loading: ", Constants.settings["path_to_image"])
    print("Starting extracting of pieces...")
    # Extracting the puzzle pieces
    extracted_pieces, dimensions, og_dimensions = Detector.main(Constants.settings["path_to_image"])
    print(str(len(extracted_pieces)), "pieces have been extracted successfully")

    # Check if their are existing weights present
    if Constants.settings["weight"]["path_to_weights"] != Constants.EMPTY_PATH:
        # TODO - Check if the paths are correct
        # Load the weights
        # Calculate the edges between the weights
        pass
    else:

        # Calculate weights and edges
        pass

    return extracted_pieces, dimensions, og_dimensions


def write_data(solver):
    """
        Saves piece weights or matchlift weights
    :param solver:
    :return:
    """

    # Normal weights
    if str.lower(Constants.settings["weight"]["perform"]) == Constants.YES:
        # Check if the path exists if it doesn't create one
        print("Preparing the solver for MGC weight calculation...")
        solver.save_weights_to_npy()
        print("Weights were computed.")
    elif str.lower(Constants.settings["weight"]["perform"]) != Constants.NO \
            or str.lower(Constants.settings["weight"]["perform"]) != Constants.YES:
        print("Did not pre-calculate weights, \"mode\" == write "
              "but \"weight\" perform is \"no\".\nCheck the settings.json if a mistake was made\n")

    # Matchlift
    if str.lower(Constants.settings["matchlift"]["perform"]) == Constants.YES:
        # Do the calculation of the matchlift
        print("Preparing solver for MatchLift...")
        solver.get_mgc_matchlift()
        print("Matchlift correspondences were computed.")

    elif str.lower(Constants.settings["matchlift"]["perform"]) != Constants.NO \
            or str.lower(Constants.settings["matchlift"]["perform"]) != Constants.YES:
        print("Did not calculate correspondences for matchlift, \"mode\" == write "
              "but \"matchlift\" perform is \"yes\".\nCheck the settings.json if a mistake was made\n")


def perform_evaluation():
    """
        Function enforcing puzzle evaluation after it has been created
    :return:
    """
    print("Evaluation process commence...")
    ev = Evaluation.Evaluation()
    ev.load_data()
    ev.evaluate()
    ev.piece_evaluation()


def ask_to_rotate(solver):
    """

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
    resized = openCV.resize(solver.solution, (int(width * 0.4), int(height * 0.4)), interpolation=openCV.INTER_AREA)
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
        1. The mode of the operation -> either read data and solve the puzzle or write data
        2. The type of puzzle the program will solve -> Known orientation or Unknown orientation
        3. The path to the image -> where it will be loaded from
        4. The name of the image -> how the processed image will be saved as
        5. The output path to the processed image -> where it will be saved
        6. Path to any pre-calculated weights (Optional)
        7. Path to the correct positions of the puzzle -> for evaluation purposes
        8. Path to the correct neighbours of the puzzle -> for evaluation purposes
        9. Path to the correct rotations of the puzzle -> for evaluation purposes
        10. Path to where the evaluation will be saved
        (Optional)
        11. Path to matchlift data
        12. Number of correspondences of the matchlift data
    :return:
    """
    # Save the json as a dictionary in the Constants for easy access
    with open("settings.json", 'r') as read_file:
        Constants.settings = json.load(read_file)

    solver = Solver.Solver()
    extracted_pieces, dimensions, og_dimensions = load_puzzle()
    solver.prepare_solver(extracted_pieces, dimensions, og_dimensions)

    # Solving a puzzle
    if str.lower(Constants.settings["mode"]) == Constants.SOLVE:
        # Check if weights are provided
        if str.lower(Constants.settings["weight"]["path_to_weights"]) != Constants.EMPTY_PATH:
            # Load weights for the solver
            solver.load_weights()
        else:
            # Computes weights
            solver.get_mgc()

        # Check if matchlift data has been provided
        if str.lower(Constants.settings["weight"]["path_to_matchlift_data"]) != Constants.EMPTY_PATH:
            # Load matchlift data for the optimizer
            solver.load_matchlift_data()
        else:
            pass

        # TODO - Matchlift optimizer
        solver.sort_edges()
        solver.find_mst()
    # Writing some data, i.e. weights matchlift
    elif str.lower(Constants.settings["mode"]) == Constants.WRITE:
        write_data(solver)
    else:
        raise Exception("Please specify the mode correctly! Either read if you want to \"solve\" a puzzle or \"write\" "
                        "if you want to save some data, i.e. matchlift data or just puzzle weights.")

    # Check if there is an evaluation provided
    # Perform evaluation if it is provided
    if str.lower(Constants.settings["evaluation"]["perform"]) == Constants.YES:
        perform_evaluation()
    # Crashesh for some reason
    elif str.lower(Constants.settings["evaluation"]["perform"]) != Constants.NO \
            and str.lower(Constants.settings["evaluation"]["perform"]) != Constants.YES:
        print(Constants.settings["evaluation"]["perform"])
        raise Exception("Please specify correctly the \"perform\" attribute of \"evaluation\"!. If you want to perform "
                        "evaluation on the solved puzzle type \"yes\", if not type \"no\".")


if __name__ == "__main__":
    # main()
    start()
