import cv2 as openCV
import numpy
import math
import matplotlib.pyplot as plt
import random
import os
from collections import deque

"""
    Progress so far:
     STATUS : WORKING   1. Cutting an image into multiple slices X, where X is a perfect squares working
     STATUS : WORKING   2. Shuffling images and inserting them into a new image
     STATUS : WORKING   3. Rotating a full image
                        -> (some black borders due to interpolation I guess)
                        -> It is due to not giving the center point properly check function for clarification
     STATUS : WORKING   3. Rotational slices of an image
     STATUS : WORKING   4. Patch making
     STATUS : ONGOING   5. Final test
"""

"""
    OpenCV is BGR
"""

"""
    I have commented almost every part of the code.
    There might be some confusion when reading, but since I am not a native English speaker
    
"""

# How many slices to be made by default
default_slice_value = 4

# Slice spacing in pixels
slice_spacing = 25

downscaling_y = 0.5
downscaling_x = 0.5

"""
"""


class Slicer:
    def __init__(self, file_write_name_positions, file_write_name_rotations, puzzle_location, output_file_name,
                 file_read_name, angle, patch_size):
        self.location = file_write_name_positions
        self.rotations = file_write_name_rotations
        self.puzzle_location = puzzle_location
        self.output_file_name = output_file_name
        self.input_image = openCV.imread(file_read_name)
        assert self.input_image is not None, "Please supply correct path to image!"
        self.angle = angle
        self.patch_size = patch_size
        self.sliced_images = deque([])

    def patch_slice(self):
        """

        :return:
        :rtype:
        """
        # number_of_slices = actual amount of slices on the y and x axis
        coordinate = []
        resized_image, final_image, number_of_slices = self.check_dimensions()
        y_axis, x_axis = Slicer.get_gap_points(resized_image.shape, number_of_slices)
        piece_index = 0
        x = 0
        y = number_of_slices[0] - 1
        # A bit of spaghetti code when saving the original values but it works
        for i in range(len(x_axis)):
            x_point = x_axis[i]
            for j in range(len(y_axis)):
                y_point = y_axis[j]
                # Corrected it to be on the resized image, I think I had it wrong on the self.input_image
                sliced = resized_image[y_point[0]:y_point[1], x_point[0]:x_point[1]]
                # openCV.imshow("sliced", sliced)
                # openCV.waitKey(0)
                # openCV.destroyAllWindows()
                self.sliced_images.append(sliced)
                # Adding to dict
                coordinate.append((y, x))
                piece_index = piece_index + 1
                y = y - 1

            x = x + 1
            y = number_of_slices[0] - 1

        if self.angle is None:
            # number_of_slices = actual amount of slices on the y and x axis
            self.save(final_image, list(self.sliced_images), number_of_slices, coordinate)
        elif self.angle == 90 or self.angle is "random":
            # We already have the final image dimensions with the rotated patches
            sliced_and_rotated, rotations_by_piece = self.slice_rotational(None, self.angle)
            self.save(final_image, list(sliced_and_rotated), number_of_slices, coordinate, rotations_by_piece)

        # location = "../output/rotated/test" + str(number_of_slices[0] * number_of_slices[1]) + "_90" + ".png"
        st = self.puzzle_location + "/" + self.output_file_name + "_" + str(number_of_slices[0] * number_of_slices[1]) \
            + "_" + str(self.angle) + ".png"
        openCV.imwrite(st, final_image)

        print("Saved at:", self.location)

    def check_dimensions(self):
        """
            Checks if we have the correct dimensions to cut the image into patches
        :return:
        :rtype:
        """
        height, width, _ = self.input_image.shape
        # Keeps track of the original image and updates it if there are any new changes
        transformed_image = self.input_image
        # Calculate the number of slices that would be able to fit it and adjust image *IF NECESSARY*
        slices_axis_y = math.floor(height / self.patch_size)
        slices_axis_x = math.floor(width / self.patch_size)

        # Checking if the image will be divided correctly
        # If it is not divided into equal patches, resizing is required
        new_height = int(self.patch_size * slices_axis_y)
        new_width = int(self.patch_size * slices_axis_x)

        # The resized image is the current image we will be working with
        # The final image is the one where we would be saving all the patches to

        if new_height != height or new_width != width:
            transformed_image = openCV.resize(self.input_image, (new_width, new_height), openCV.INTER_AREA)

        if self.angle is "random" or self.angle == 90:
            # 1. Calculate the diagonal of a patch, that way we would know its max Height and max Width when rotated
            # 2. Calculate the max Height and max Width
            # 3. Create a blank image big enough to hold all rotated patches
            if self.angle != 90:
                diagonal = math.ceil(self.patch_size * math.sqrt(2))
                h = int(new_height / self.patch_size) * diagonal
                w = int(new_width / self.patch_size) * diagonal
                number_of_slices = (slices_axis_y, slices_axis_x)
                final_image = self.initialize_new_image(h, w, number_of_slices)
            else:
                number_of_slices = (int(new_height / self.patch_size), int(new_width / self.patch_size))
                final_image = self.initialize_new_image(new_height, new_width, number_of_slices)
            return transformed_image, final_image, number_of_slices
        else:
            number_of_slices = (int(new_height / self.patch_size), int(new_width / self.patch_size))
            final_image = self.initialize_new_image(new_height, new_width, number_of_slices)
            return transformed_image, final_image, number_of_slices

    def slice_rotational(self, slices=None, angle=None):
        """

        :param slices:
        :type slices:
        :param angle:
        :type angle: str, int
        :return:
        :rtype:
        """
        sliced_images_rotated = deque([])
        # Keeps track of the maximum height and width from all slices
        # Used to calculated the height and width of the new_image
        max_height = 0
        max_width = 0
        ang = 0
        rotations_by_piece = []

        while len(self.sliced_images) > 0:
            sliced = self.sliced_images.popleft()
            if angle is "random":
                ang = random.randrange(30, 331, 15)
            elif angle == 90:
                ang = random.randrange(0, 361, 90)
            rotated_image = self.rotate_image_boundary(sliced, ang)
            shape = rotated_image.shape
            # Checks for the maximum height and width of a slice
            if max_height < shape[0]:
                max_height = shape[0]
            if max_width < shape[1]:
                max_width = shape[1]
            sliced_images_rotated.append(rotated_image)
            rotations_by_piece.append(int(ang / 90) % 4)
            # openCV.imshow(str(ang), rotated_image)
            # openCV.waitKey(0)
            # openCV.destroyAllWindows()
        # print("Max_y", max_height, "Max_x", max_width, end="\n\n")
        if slices is None:
            return sliced_images_rotated, rotations_by_piece
        else:
            return sliced_images_rotated, max_height * math.sqrt(slices), max_width * math.sqrt(slices)

    @staticmethod
    def initialize_new_image(height, width, slices):
        """
            Calculates the new dimensions of the image. This is done by taking into account how many slices have to be
            cut. Creates a new 'empty' image ready to be populated with actual data
        :param height:
        :type height:
        :param width:
        :type width:
        :param slices:
        :type slices:
        :return:
        :rtype:
        """
        if type(slices) is tuple:
            # Tuple representing how many slices on the y and x axis
            new_height = ((slices[0] + 1) * slice_spacing) + height
            new_width = ((slices[1] + 1) * slice_spacing) + width
            new_image = numpy.ones([new_height, new_width, 3], numpy.uint8) * 255
            return new_image
        else:
            new_height = math.ceil((math.sqrt(slices) + 1) * slice_spacing) + math.ceil(
                math.ceil(height / math.sqrt(slices)) * math.sqrt(slices))
            new_width = math.ceil((math.sqrt(slices) + 1) * slice_spacing) + math.ceil(
                math.ceil(width / math.sqrt(slices)) * math.sqrt(slices))
            new_image = numpy.ones([new_height, new_width, 3], numpy.uint8) * 255
            return new_image

    @staticmethod
    def rotate_image_boundary(image, angle):
        """
            Rotate and image at an angle
        :param image:
        :param angle:
        :return:
        """
        height, width, _ = image.shape
        center = (((width - 1) / 2), ((height - 1) / 2))
        matrix = openCV.getRotationMatrix2D(center, angle, 1)
        cos = numpy.abs(matrix[0, 0])
        sin = numpy.abs(matrix[0, 1])

        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        matrix[0, 2] += (new_width / 2) - (width / 2)
        matrix[1, 2] += (new_height / 2) - (height / 2)

        rotated = openCV.warpAffine(image, matrix, (new_width, new_height), borderValue=(255, 255, 255))
        return rotated

    def save(self, image_to_write, transformed_images, slices, coordinate, rotations=None):
        """

        :param image_to_write:
        :type image_to_write:
        :param transformed_images:
        :type transformed_images:
        :param slices:
        :type slices:
        :param coordinate:
        :type coordinate:
        :param rotations:
        :type rotations:
        :return:
        :rtype:
        """
        # "*********************************"
        for i in range(len(transformed_images)):
            openCV.imwrite("../output/puzzle/" + str(i) + ".png", transformed_images[i])
        "*********************************"

        img_height, _, _ = image_to_write.shape

        # The starting points used to determine the region in which a sliced image would be added to the shuffled image
        starting_point_y = 0
        starting_point_x = 0

        # The ending points used to determine the region which a sliced image would not pass
        ending_point_y = img_height
        ending_point_x = 0

        # Keeps track of the height and width of the sliced image which is next in line(deque)
        height = 0
        width = 0

        right_ending_point_x = 0

        if type(slices) is tuple:
            range_for_i = slices[0]
            range_for_j = slices[1]
        else:
            range_for_i = int(math.sqrt(slices))
            range_for_j = int(math.sqrt(slices))

        file_locations = open(self.location + self.output_file_name + "_" + str(range_for_i * range_for_j) + "_" + str(self.angle) + ".txt", mode="w")

        # for i in range(len(sliced_images)):
        #     openCV.imshow("im: " + str(i), sliced_images[i])
        #     openCV.waitKey(0)
        #     openCV.destroyAllWindows()

        # Shuffle the indexes of the sliced images
        list_of_indexes = deque(range(range_for_i * range_for_j))
        random.shuffle(list_of_indexes)

        # Shuffles the image slices inside the deque (Randomly)
        # random.shuffle(sliced_images)

        # Save piece locations to txt
        for i in list_of_indexes:
            y, x = coordinate[i]
            to_write = str(y) + "," + str(x) + "\n"
            file_locations.write(to_write)

        file_locations.close()

        # Save piece rotations to txt
        if rotations is not None:
            file_rotations = open(self.rotations + self.output_file_name + "_" + str(range_for_i * range_for_j) + "_" + str(self.angle) + ".txt", mode="w")
            for i in list_of_indexes:
                rot = rotations[i]
                to_write = str(rot) + "\n"
                file_rotations.write(to_write)
            file_rotations.close()

        for j in range(range_for_j):
            for i in range(range_for_i):

                # Acquires the height and width of the sliced image which is next in line
                height = Slicer.peek(transformed_images, 0)
                width = Slicer.peek(transformed_images, 1)

                starting_point_y = ending_point_y - height

                ending_point_x = starting_point_x + width

                if right_ending_point_x < ending_point_x:
                    right_ending_point_x = ending_point_x

                y1 = starting_point_y - ((i + 1) * slice_spacing)
                y2 = ending_point_y - ((i + 1) * slice_spacing)
                x1 = starting_point_x + ((j + 1) * slice_spacing)
                x2 = ending_point_x + ((j + 1) * slice_spacing)
                index = list_of_indexes.popleft()
                image_to_write[y1:y2, x1:x2] = transformed_images[index]

                ending_point_y = starting_point_y

            if len(transformed_images) != 0:
                starting_point_x = right_ending_point_x
            starting_point_y = 0
            ending_point_y = img_height

    @staticmethod
    def peek(sliced_images, axis):
        """
            Peeks at the next in line element without removing it
            This is done to acquire the height and width of an image which are used to calculate the correct place of a slice
            0 -> y axis
            1 or any other -> x axis
        :param sliced_images:
        :param axis:
        :return:
        """
        height, width, _ = sliced_images[-1].shape
        if axis == 0:
            return height
        else:
            return width

    @staticmethod
    def get_gap_points(size_of_image, slices):
        """
            Gets tha gap points
            EXTENSIVELY TESTED TO ENSURE THE CORRECT POINTS ARE FOUND
        :param size_of_image:
        :type size_of_image: tuple
        :param slices:
        :type slices: tuple, int
        :return:
        :rtype: tuple
        """
        # Gets the height and width of the image
        height, width, _ = size_of_image

        # Calculates at what interval the cut will be made
        # No need for rounding, ceiling or flooring, check in loops
        if type(slices) is tuple:
            range_for_y = slices[0]
            range_for_x = slices[1]
            height_interval = height / slices[0]  # In this case the slices variable represents the value of the patch, i.e. 14x14
            width_interval = width / slices[1]   # In this case the slices variable represents the value of the patch, i.e. 14x14
        else:
            range_for_y = int(math.sqrt(slices))
            range_for_x = int(math.sqrt(slices))
            height_interval = height/int(math.sqrt(slices))
            width_interval = width/int(math.sqrt(slices))

        # Used to save 2D points that would be used to crop an image
        y_axis = []
        x_axis = []

        # Used a variable outside of the loop to keep track of the 2D points
        # Each iteration a new 'y' object local to the for loop is created and basically everything ends up in flames
        # And we don't want that
        # Same applies for x_gap
        y_gap = 0
        for i in range(range_for_y):
            # After each iteration y is updated with the new 1D point
            y = y_gap
            # Appending the the points at the y axis of the image
            # Important to floor the current value of y since we don't want a 1 of error
            # The function int(x), x IS A FLOATING POINT number, acts as math.floor
            y_axis.append((math.floor(y), int(y + height_interval)))
            # Assign the height interval to y_gap
            y_gap += height_interval

        x_gap = 0
        for i in range(range_for_x):
            # After each iteration x is updated with the new 1D point
            x = x_gap
            # Appending the the points at the x axis of the image
            # Important to floor the current value of y since we don't want a 1 of error
            # The function int(x), x IS A FLOATING POINT number, acts as math.floor
            x_axis.append((math.floor(x), int(x + width_interval)))
            # Assign the width interval to x_gap
            x_gap += width_interval

        # Reverse the y_axis points so we could go from bottom to top when traversing and cutting the pieces
        y_axis.reverse()

        # For testing purposes
        print("Y points", y_axis)
        print("X points", x_axis, end="\n\n")
        return y_axis, x_axis


def testing():
    """
    For now it would only be working with +- 90 degree change in angle
    :return:
    """
    degrees = 90
    center = (23.5, 23.5)

    img = numpy.ones((48, 48, 3), numpy.uint8) * 255
    mat = openCV.getRotationMatrix2D(center, degrees, 1.0)
    img = openCV.warpAffine(img, mat, (48, 48))

    plt.imshow(img)
    plt.show()


def main():


    to_where_positions = "../evaluation/location/kris/"
    to_where_rotations = "../evaluation/rotation/kris/"
    to_where_puzzle = "../output/rotated/kris/"
    output_file_name = "kris_presentation"
    from_where = "../input/kris.jpg"
    dim = 1546
    rotation = 90
    obj = Slicer(to_where_positions, to_where_rotations, to_where_puzzle, output_file_name, from_where, rotation, dim)
    obj.patch_slice()
    obj = None
    # kris_dimensions = [1548, 1032, 774, 516, 344, 308, 258, 220, 150]
    # big_cat_dimensions = [960, 640, 480, 384, 320, 240, 192, 160, 136, 128]
    # greece_dimensions = [1008, 800, 756, 600, 378, 252, 216]
    # star_wars_dimensions = [720, 400, 360, 200, 180, 140, 120]
    #
    # to_where_positions = "../evaluation/location/kris/"
    # to_where_rotations = "../evaluation/rotation/kris/"
    # to_where_puzzle = "../output/rotated/kris/"
    # output_file_name = "kris"
    # from_where = "../input/kris.jpg"
    # rotation = 90
    #
    # """ kris image """
    # for dim in kris_dimensions:
    #     obj = Slicer(to_where_positions, to_where_rotations, to_where_puzzle, output_file_name, from_where, rotation, dim)
    #     obj.patch_slice()
    #     obj = None
    #
    # to_where_positions = "../evaluation/location/greece/"
    # to_where_rotations = "../evaluation/rotation/greece/"
    # to_where_puzzle = "../output/rotated/greece/"
    # output_file_name = "greece"
    # from_where = "../input/greece.jpg"
    # rotation = 90
    #
    # """ greece image """
    # for dim in greece_dimensions:
    #     obj = Slicer(to_where_positions, to_where_rotations, to_where_puzzle, output_file_name, from_where, rotation, dim)
    #     obj.patch_slice()
    #     obj = None
    #
    # to_where_positions = "../evaluation/location/big_cat/"
    # to_where_rotations = "../evaluation/rotation/big_cat/"
    # to_where_puzzle = "../output/rotated/big_cat/"
    # output_file_name = "big_cat"
    # from_where = "../input/cat_big.jpg"
    # rotation = 90
    #
    # """ big cat image"""
    # for dim in big_cat_dimensions:
    #     obj = Slicer(to_where_positions, to_where_rotations, to_where_puzzle, output_file_name, from_where, rotation, dim)
    #     obj.patch_slice()
    #     obj = None
    #
    # to_where_positions = "../evaluation/location/star_wars/"
    # to_where_rotations = "../evaluation/rotation/star_wars/"
    # to_where_puzzle = "../output/rotated/star_wars/"
    # output_file_name = "star_wars"
    # from_where = "../input/star_wars.png"
    # rotation = 90
    #
    # """ star wars image """
    # for dim in star_wars_dimensions:
    #     obj = Slicer(to_where_positions, to_where_rotations, to_where_puzzle, output_file_name, from_where, rotation, dim)
    #     obj.patch_slice()
    #     obj = None


if __name__ == "__main__":
    # TODO - command line arguments
    main()
