import argparse
import os
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
    # Loading the image containing the puzzle
    print("Loading: ", Constants.settings["path_to_image"])
    print("Starting extracting of pieces")
    # Extracting the puzzle pieces
    extracted_pieces, dimensions, og_dimensions = Constants.settings["path_to_image"]
    print(str(len(extracted_pieces)), "pieces have been extracted successfully")

    # Check if their are existing weights present
    if Constants.settings["path_to_weight"] != Constants.EMPTY_PATH:
        # TODO - Check if the paths are correct
        # Load the weights
        # Calculate the edges between the weights
        pass
    else:

        # Calculate weights and edges
        pass


def write_data(extracted_pieces, dimensions, og_dimensions):
    """
        Saves piece weights or matchlift weights
    :return:
    """

    # Normal weights
    if Constants.settings["weight"]["perform"] == Constants.YES:
        # Do the pre-calculation of the weights
        solver = Solver.Solver()
        # Check if the path exists if it doesn't create one
        solver.save_weights_to_npy()
    else:
        print("Did not pre-calculate weights, \"mode\" == write "
              "but \"weight\" perform is \"yes\".\nCheck the settings.json if a mistake was made\n")

    # Matchlift
    if Constants.settings["matchlift"]["perform"] == Constants.YES:
        # Do the calculation of the matchlift
        print("Preparing solver for MatchLift...")
        solver = Solver.Solver()
        print("Solver computing correspondences for Matchlift...")


        solver = None
    else:
        print("Did not calculate correspondences for matchlift, \"mode\" == write "
              "but \"weight\" perform is \"yes\".\nCheck the settings.json if a mistake was made\n")


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
        7. Path to the correct locations of the puzzle -> for evaluation purposes
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

    if Constants.settings["mode"] == Constants.READ:
        # read
        pass
    else:
        # write
        pass

    # Check if there is an evaluation provided
    if Constants.settings["evaluation"]["perform"] == Constants.YES:
        # Perform evaluation
        pass
    else:
        # Don't perform evaluation
        pass


if __name__ == "__main__":
    # main()
    start()
