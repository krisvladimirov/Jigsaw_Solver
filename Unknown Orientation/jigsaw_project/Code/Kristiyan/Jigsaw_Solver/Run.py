import argparse
import os
import Detector as Detector
import Solver as Solver
import Evaluation as Evaluation
import cv2 as openCV
import glob
import numpy


def main():
    """

    :return:
    :rtype:
    """
    # pre_calculate_weights()

    """
            LOOK AT Solver.start_solving() function for more info on option!
    """
    OPTION = 0
    path_to_image = ""
    weight_path = ""
    path_to_locations = ""
    path_to_rotations = ""
    save_evaluation_to = ""

    print("Loading: ", path_to_image)
    print("Starting extracting of pieces")
    extracted_pieces, dimensions, og_dimensions = Detector.main(path_to_image)  # To be changed to arguments

    print(str(len(extracted_pieces)), "pieces have been extracted successfully")
    print("Preparing Solver program...")
    solver = Solver.Solver()
    print("Starting Solver...")
    solver.start_solving(extracted_pieces, dimensions, og_dimensions, weight_path, OPTION)

    ask_to_rotate(solver)
    # return solver.solution
    ev = Evaluation.Evaluation()
    ev.load_data(path_to_locations, path_to_rotations)
    ev.evaluate(save_evaluation_to)
    ev.piece_evaluation()
    ev.rotation_evaluation()


def pre_calculate_weights():
    path_kris = "../output/rotated/kris/"
    path_greece = "../output/rotated/greece/"
    path_big_cat = "../output/rotated/big_cat/"
    path_star_wars = "../output/rotated/star_wars/"

    kris_files = [f for f in glob.glob(path_kris + "*.png")]
    greece_files = [f for f in glob.glob(path_greece + "*.png")]
    star_wars_files = [f for f in glob.glob(path_star_wars + "*.png")]
    big_cat_files = [f for f in glob.glob(path_big_cat + "*.png")]

    for path in kris_files:
        extracted_pieces, dimensions, og_dimensions = Detector.main(path)
        solver = Solver.Solver()
        solver.start_solving(extracted_pieces, dimensions, og_dimensions)
        solver.save_weights_to_npy(root_folder="../weights", child_folder="/kris/", name="kris")
        solver = None

    print("Completed kris")

    for path in greece_files:
        extracted_pieces, dimensions, og_dimensions = Detector.main(path)
        solver = Solver.Solver()
        solver.start_solving(extracted_pieces, dimensions, og_dimensions)
        solver.save_weights_to_npy(root_folder="../weights", child_folder="/greece/", name="greece")
        solver = None

    print("Completed greece")

    for path in star_wars_files:
        extracted_pieces, dimensions, og_dimensions = Detector.main(path)
        solver = Solver.Solver()
        solver.start_solving(extracted_pieces, dimensions, og_dimensions)
        solver.save_weights_to_npy(root_folder="../weights", child_folder="/star_wars/", name="star_wars")
        solver = None

    print("Completed star_wars")

    for path in big_cat_files:
        extracted_pieces, dimensions, og_dimensions = Detector.main(path)
        solver = Solver.Solver()
        solver.start_solving(extracted_pieces, dimensions, og_dimensions)
        solver.save_weights_to_npy(root_folder="../weights", child_folder="/big_cat/", name="big_cat")
        solver = None

    print("Completed big_cat")


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


if __name__ == "__main__":
    main()
