import argparse
import os
import Detector as Detector
import Solver as Solver
import Constants as Constants
from Network import JigNetwork as Network


# TODO - Metric to evaluate if a an assembled piece is in its correct spot

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True, help="path to original image or puzzle directory")
    parser.add_argument("-o", "--output", required=True, help="where to save generated puzzle or solved puzzle image")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error("-i/--input should be an jigsaw puzzle, in the form of an image, to be solved")
    if not os.path.isdir(args.output):
        parser.error("-o/--output should be a directory where the reconstructed puzzle will be saved "
                     "(created if not exists)")

    return args


def main():
    # arguments = parse_arguments()
    # if not os.path.exists(arguments.input):
    #     os.mkdir(arguments.output)
    # extracted_pieces = None
    # dimensions = None
    model_path = "../models/jignet_v23_resnet_all_black_cv1.h5"

    # path = "../output/no_rotation/big_cat_100_no.png"
    path = "../output/no_rotation/star_wars_84_no.png"
    output_path = "../matchlift/"
    name_of_image = "cat_image_"
    how_many_correspondences = 4
    # path_to_locations = "../evaluation/location/cats/big_cat_9_90.txt"
    # path_to_rotations = "../evaluation/rotation/cats/big_cat_9_90.txt"
    # save_evaluation_to = "../evaluation/report/cats/big_cat_not_processed_9_90.txt"

    print("Loading: ", path)
    print("Starting extracting of pieces")
    extracted_pieces, dimensions, og_dimensions = Detector.main(path)

    print(str(len(extracted_pieces)), "pieces have been extracted successfully")
    # TODO - Uncomment to perform CNN
    # print("Preparing CNN model constraint...")
    Constants.model = Network.get_model(model_path)
    print("CNN model loaded")
    print("Preparing Solver program...")
    solver = Solver.Solver()
    print("Starting Solver...")
    solver.start_solving(extracted_pieces, dimensions, og_dimensions)
    # solver.get_mgc_matchlift(output_path + name_of_image, how_many_correspondences)

    """
        Uncomment if you want to do the evaluation but remember to give it the correct paths to where you have saved the
        evaluation txt's and where you want to save them
    """
    # return solver.solution
    # ev = Evaluation.Evaluation()
    # ev.load_data(path_to_locations, path_to_rotations)
    # ev.evaluate(save_evaluation_to)
    # ev.piece_evaluation()
    # ev.rotation_evaluation()


if __name__ == "__main__":
    main()