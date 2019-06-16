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
slice_spacing = 0

downscaling_y = 0.5
downscaling_x = 0.5

"""
"""

"""
    Image slicing for NN
"""


def patch_slice(image, patch_dimensions, angle=None):
    """

    :param image:
    :param patch_dimensions:
    :param angle:
    :return:
    """

    # number_of_slices = actual amount of slices on the y and x axis
    resized_image, final_image, number_of_slices = check_dimensions(image, patch_dimensions, angle)
    y_axis, x_axis = get_gap_points(resized_image.shape, number_of_slices)

    sliced_images = deque([])

    for i in range(len(y_axis)):
        y_point = y_axis[i]
        for j in range(len(x_axis)):
            x_point = x_axis[j]
            # Y and X point is a tuple containing the mix max position so it can be used for accessing a numpy array
            sliced = image[y_point[0]:y_point[1], x_point[0]:x_point[1]]
            sliced_images.append(sliced)





def check_dimensions(image, patch_dimensions, angle):
    """
        Specific method to patch making
    :param image: Image we are working with
    :param patch_dimensions: Size of the patch, i.e. 14x14 in pixels
    :param angle: Indicates whether or not the image should be rotated at increments of 90 degrees or randomly
    :return:
    """
    height, width, _ = image.shape
    print(height, width)
    # Keeps track of the original image and updates it if there are any new changes
    original_image = image
    # Calculate the number of slices that would be able to fit it and adjust image *IF NECESSARY*
    slices_axis_y = math.floor(height / patch_dimensions)
    slices_axis_x = math.floor(width / patch_dimensions)

    # Checking if the image will be divided correctly
    # If it is not divided into equal patches, resizing is required
    new_height = int(patch_dimensions * slices_axis_y)
    new_width = int(patch_dimensions * slices_axis_x)

    if new_height != height or new_width != width:
        # print(new_height, new_width)
        original_image = openCV.resize(image, (new_width, new_height), openCV.INTER_AREA)

    if angle is "random" or angle == 90:
        # 1. Calculate the diagonal of a patch, that way we would know its max Height and max Width when rotated
        # 2. Calculate the max Height and max Width
        # 3. Create a blank image big enough to hold all rotated patches
        if angle != 90:
            diagonal = math.ceil(patch_dimensions * math.sqrt(2))
            h = int(new_height / patch_dimensions) * diagonal
            w = int(new_width / patch_dimensions) * diagonal
            number_of_slices = (slices_axis_y, slices_axis_x)
            final_image = initialize_new_image(h, w, number_of_slices)
        else:
            number_of_slices = (int(new_height / patch_dimensions), int(new_width / patch_dimensions))
            final_image = initialize_new_image(new_height, new_width, number_of_slices)
        return original_image, final_image, number_of_slices
    else:
        number_of_slices = (int(new_height / patch_dimensions), int(new_width / patch_dimensions))
        final_image = initialize_new_image(new_height, new_width, number_of_slices)
        # The resized image is the current image we will be working with
        # The final image is the one where we would be saving all the patches to
        return original_image, final_image, number_of_slices


def slice(image, slices=default_slice_value, angle=None):
    """
        Slices an image into N x N grid
    :param image: The input image
    :param slices: Into how many slices N x N the image will be cut
    :param angle: Whether or not the slices will be rotated 90 degrees or randomly up to 360 degrees
    :return:
    """
    new_image = None
    y_axis, x_axis = get_gap_points(image.shape, slices)
    height, width, colour_channels = image.shape
    sliced_images = deque([])

    """
        Series of loops that cut an image into slices ready to be saved
    """

    for i in range(int(math.sqrt(slices))):
        #  Y goes here
        y_point = y_axis[i]
        for j in range(int(math.sqrt(slices))):
            x_point = x_axis[j]
            # The sliced image
            # print("Y1 point", y_point[0], "Y2 point", y_point[1])
            sliced = image[y_point[0]:y_point[1], x_point[0]:x_point[1]]
            sliced_images.append(sliced)
            # openCV.imshow("Slice" + str(i) + str(j), sliced)

    if angle is None:
        new_image = initialize_new_image(height, width, slices)
        create_shuffled_image(new_image, sliced_images, slices)
    elif angle == 90 or angle is "random":
        sliced_and_rotated, h, w = slice_rotational(sliced_images, slices, angle)
        new_image = initialize_new_image(h, w, slices)
        create_shuffled_image(new_image, sliced_and_rotated, slices)

    # print(new_image.shape)
    openCV.imshow("Image", new_image)
    openCV.waitKey(0)
    openCV.destroyAllWindows()

    # Save the shuffled image
    # openCV.imwrite("../output/bright" + str(slices) + "pieces.png", new_image)


def initialize_new_image(height, width, slices):
    """
        Calculates the new dimensions of the image. This is done by taking into account how many slices have to be cut.
        Creates a new 'empty' image ready to be populated with actual data
    :param height:
    :param width:
    :param slices:
    :return:
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


def slice_rotational(sliced_images, slices=None, angle=None):
    """
    :param sliced_images:
    :param slices:
    :param angle:
    :return:
    """
    sliced_images_rotated = deque([])
    # Keeps track of the maximum height and width from all slices
    # Used to calculated the height and width of the new_image
    max_height = 0
    max_width = 0
    ang = 0
    while len(sliced_images) > 0:
        sliced = sliced_images.pop()
        # downscaled = openCV.resize(sliced, None, fx=downscaling_x, fy=downscaling_y, interpolation=openCV.INTER_CUBIC)
        if angle is "random":
            ang = random.randrange(30, 331, 15)
        elif angle == 90:
            ang = random.randrange(0, 361, 90)
        rotated_image = rotate_image_boundary(sliced, ang)
        shape = rotated_image.shape
        # Checks for the maximum height and width of a slice
        if max_height < shape[0]:
            max_height = shape[0]
        if max_width < shape[1]:
            max_width = shape[1]
        sliced_images_rotated.append(rotated_image)

    # print("Max_y", max_height, "Max_x", max_width, end="\n\n")
    if slices is None:
        return sliced_images_rotated
    else:
        return sliced_images_rotated, max_height * math.sqrt(slices), max_width * math.sqrt(slices)


def rotate_image_boundary(image, angle):
    """
        Supposedly it is working as intended?
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

    # rotated = openCV.warpAffine(image, matrix, (new_width, new_height), borderMode=openCV.BORDER_CONSTANT, borderValue=(255, 255, 255))
    rotated = openCV.warpAffine(image, matrix, (new_width, new_height), borderValue=(255, 255, 255))
    return rotated


def create_shuffled_image(new_image, sliced_images, slices):
    """
        Creates the shuffled image
    :param new_image:
    :param sliced_images:
    :param slices:
    :return:
    """

    # Shuffles the image slices inside the deque (Randomly)
    random.shuffle(sliced_images)
    # for i in range(len(sliced_images)):
    #     openCV.imwrite("../output/cat_" + str(i) + ".png", sliced_images[i])

    # The starting points used to determine the region in which a sliced image would be added to the shuffled image
    starting_point_y = 0
    starting_point_x = 0

    # The ending points used to determine the region which a sliced image would not pass
    ending_point_y = 0
    ending_point_x = 0

    # Keeps track of the height and width of the sliced image which is next in line(deque)
    height = 0
    width = 0

    if type(slices) is tuple:
        range_for_i = slices[0]
        range_for_j = slices[1]
    else:
        range_for_i = int(math.sqrt(slices))
        range_for_j = int(math.sqrt(slices))

    top_ending_point_y = 0

    for i in range(range_for_i):
        for j in range(range_for_j):
            # Standard loops to iterate over the slices
            # (Testing purposes) print("Fail at", str(i) + str(j))

            # Acquires the height and width of the sliced image which is next in line
            height = peek(sliced_images, 0)
            width = peek(sliced_images, 1)
            """
                This is done since no image slice is perfect all the time
                If you consider an image of size 640x640 and cut into 9 slices (3x3)
                You wouldn't be able to obtain a perfect cut
                640/3 = 213.3
                In this there would be some slices that are 213x214 | 214x213 | 214x214
                In our case:
                    -> ending_point_x is updated after each iteration on the x axi, thus eliminating any possible 
                        of by 1 errors
                    -> starting_point_x is updated after each iteration as well by knowing the point of the last slice
                        that was inserted into the new image and subtracting it by the width of the slice next in line
                    -> ending_point_y is updated after each iteration and in all fairness that is not the most efficient
                        way of doing it
                    -> starting_point_x is only updated after one row (inner loop is done) is populated 

            """
            # Updating the x axis end point
            #
            ending_point_x += width
            # Updating the x axis starting point
            #
            starting_point_x = ending_point_x - width
            # Updating the y axis end point
            #
            ending_point_y = starting_point_y + height
            # TODO comment this part
            # TODO Why was it added
            # TODO get the highest point of all images and starts from this point on th next row
            if top_ending_point_y < ending_point_y:
                top_ending_point_y = ending_point_y

            # (Testing purposes)
            # print("ending_point_y", ending_point_y, "ending_point_x", ending_point_x)
            # print("Coordinates", str(i) + str(j))
            # print("Height", height, "Width", width)
            # print("starting_point_y", starting_point_y, "ending_point_y", ending_point_y, "starting_point_x", starting_point_x, "ending_point_x", ending_point_x)
            # print(new_image.shape, end="\n\n")

            new_image[starting_point_y + ((i + 1) * slice_spacing):ending_point_y + ((i + 1) * slice_spacing),
            starting_point_x + ((j + 1) * slice_spacing): ending_point_x + (
                        (j + 1) * slice_spacing)] = sliced_images.pop()

        # Prevent the call of the peek() function after the deque has been exhausted
        if len(sliced_images) != 0:
            # starting_point_y += peek(sliced_images, 0)
            starting_point_y = top_ending_point_y
        starting_point_x = 0
        ending_point_x = 0


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


def get_gap_points(size_of_image, slices):
    """
        Gets tha gap points
        EXTENSIVELY TESTED TO ENSURE THE CORRECT POINTS ARE FOUND FUCK ME
    :param size_of_image:
    :param slices:
    :return:
    """
    # Gets the height and width of the image
    height, width, _ = size_of_image

    # Calculates at what interval the cut will be made
    # No need for rounding, ceiling or flooring, check for loops
    if type(slices) is tuple:
        range_for_y = slices[0]
        range_for_x = slices[1]
        height_interval = height / slices[
            0]  # In this case the slices variable represents the value of the patch, i.e. 14x14
        width_interval = width / slices[
            1]  # In this case the slices variable represents the value of the patch, i.e. 14x14
    else:
        range_for_y = int(math.sqrt(slices))
        range_for_x = int(math.sqrt(slices))
        height_interval = height / int(math.sqrt(slices))
        width_interval = width / int(math.sqrt(slices))

    # Used to save 2D points that would be used to crop an image
    y_axis = []
    x_axis = []

    # Used a variable outside of the loop to keep track of the 2D points, since Python is not like Java
    # (Don't be like Python kiddos)
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
        # y_axis.append((math.floor(y), int(y + height_interval - 1))) Commented due to numpy since y[0] and y[1] indexing
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
        # x_axis.append((math.floor(x), int(x + width_interval - 1))) Commented due to numpy since x[0] and x[1] indexing
        x_axis.append((math.floor(x), int(x + width_interval)))
        # Assign the width interval to x_gap
        x_gap += width_interval

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


def patches():
    # input_image = openCV.imread("../input/cat_image.jpg")
    input_image = openCV.imread("../input/green_supra.jpg")
    # input_image = openCV.imread("../input/cat_asym.jpg")
    # patch_slice(input_image, 320, 90)
    # patch_slice(input_image, 214)


def grid():
    input_image = openCV.imread("../input/cat_rectangle_image.jpg")
    slice(input_image, 16, 90)


def test():
    input_image = openCV.imread("../input/cat_image.jpg")
    arr = [[1, 2, 3, 4, 5, 6, 7, 8],
           [9, 10, 11, 12, 13, 14, 15, 16],
           [11, 12, 13, 14, 15, 16, 17, 18],
           [19, 20, 21, 22, 23, 24, 25, 26]]
    arr_num = numpy.array(arr)
    print(arr_num[0:1, 1:4])
    # The same
    """
        This is the same
        print(input_image[1, 1])
        print(input_image[1:2, 1:2])
    """
    """
        This is the same
        print(input_image[639, 639])
        print(input_image[639:640, 639:640])
        print(input_image([639: , 639: ])
    """
    """
        If you go above the last index
    """
    """
        IMPORTANT
        numpy [y1:y2, x1:x2]
        y1, y2 indexes that start from 0 to n - 1, where n is the size (in other words the starts
        x1, x2 indexes that end 
    """
    print(input_image[639:640, 639:640])
    print(input_image[640:650, 640:650])
    # print(input_image[640, 640]) Shoots an error

    # print(arr_num.shape)
    # print(arr_num[0:2, 1:4])
    # print(arr_num[0:2, 4:8])
    # print("Splitting, top left")
    # print("Splitting, top right")
    # print("Splitting, bottom left")
    # print("Splitting, bottom right")


def check(image):
    print()


def main():
    # input_image = openCV.imread("input/cat_rectangle_image.jpg")
    # input_image = openCV.imread("input/cat_image.jpg")
    # input_image = openCV.imread("input/cat_asym.jpg")
    # input_image = openCV.imread("input/crash.jpg")
    # colors, counts = numpy.unique(input_image.reshape(-1, 3), axis=0, return_counts=True)
    #
    # for color, count in zip(colors, counts):
    #     print("{} = {} pixels".format(color, count))

    # slice(input_image, 169, "random")
    # slice(input_image, 16, "random")
    # slice(input_image, 4, "random")

    patches()
    # grid()
    # test()


if __name__ == "__main__":
    main()
