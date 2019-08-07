import cv2 as openCV
import numpy

# Global variables
lower_bound = 250   #
upper_bound = 255   #

# TODO Add encapsulation because it is not good without it
# TODO Improve on the rotated extraction


def get_pieces(image):
    """
        Initiates the detection and extraction of patch pieces from an image
    :param image: Image to be processed
    :return: A list of patches with equal sizes
    """

    list_of_all_pieces = []
    og_height, og_width, _ = image.shape
    sorted_contours = detect(image)
    dimensions_of_piece = 0
    counter = []
    all_extracted = []
    for i in range(len(sorted_contours)):
        contour = sorted_contours[i]
        # Get bounding box
        x, y, w, h = openCV.boundingRect(contour)
        # print(x, y, w, h)
        # Ignores the image box contour
        if og_height == h and og_width == w:
            continue

        if h < 20 or w < 20:
            continue

        # Ge the minimum bounding box
        rectangle = openCV.minAreaRect(contour)
        # Gets the points of the contour surrounding the region of interest
        box = openCV.boxPoints(rectangle)
        # Based on the bounding rectangle around of the contour, gets ROI
        region_of_interest = image[y:(y + h), x:(x + w)]
        # Corrects the angle if necessary
        corrected_angle = correct_angle(get_angle(region_of_interest))

        if corrected_angle == -0.0:
            extracted = extract(image, box)
            dimensions_of_piece = extracted.shape
            counter.append(dimensions_of_piece)
            all_extracted.append(extracted)
            # list_of_all_pieces.append(extracted)
        else:
            rotated = rotate(region_of_interest, region_of_interest.shape[:2], corrected_angle)
            box_b = get_points(rotated)
            # Rotate the image to de-skew it
            # Finding the max and min x,y co-ordinates for the piece
            # (hh, ww) = region_of_interest.shape[:2]
            # center = (w // 2, h // 2)
            # rotation_matrix = openCV.getRotationMatrix2D(center, corrected_angle, 1.0)
            # rotated = openCV.warpAffine(region_of_interest, rotation_matrix, (ww, hh), flags=openCV.INTER_CUBIC,
            #                             borderMode=openCV.BORDER_REPLICATE)
            extracted = extract(rotated, box_b)
            # TODO The Rotated extraction will be fixed in a later stage as it is not that important
            # openCV.imshow("Extracted", extracted)
            # openCV.waitKey(0)
            # openCV.destroyAllWindows()

    # So much spaghetti but it works I guess
    cc = {}
    for item in counter:
        cc[item] = 0

    for item in counter:
        cc[item] = cc[item] + 1

    max_key = max(cc, key=lambda k: cc[k])
    for i in all_extracted:
        sh = i.shape
        if sh == max_key:
            list_of_all_pieces.append(i)

    return list_of_all_pieces, max_key, (og_height, og_width)


def rotate(region_of_interest, dimensions, angle):
    """
        Corrects the angle of a region of interest.
    :param region_of_interest: The region which contains that patch to be extracted after angle correction
    :param dimensions: A list containing the center points of the ROI
    :param angle: The corrected angle value
    :return: A rotated region of interest
    """
    height, width = region_of_interest.shape[:2]
    center = (dimensions[1] // 2, dimensions[0] // 2)
    rotational_matrix = openCV.getRotationMatrix2D(center, angle, 1.0)
    rotated = openCV.warpAffine(region_of_interest, rotational_matrix, (width, height),
                                borderMode=openCV.BORDER_REPLICATE)
    return rotated


def correct_angle(angle):
    """
        Corrects the angle
    :param angle: Old angle
    :return: New angle value
    """
    if angle < -45:
        corrected_angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make it positive
    else:
        corrected_angle = -angle

    return corrected_angle


def extract(image, box=None):
    """
        Performs the extraction of a patch
    :param image:
    :param box:
    :return:
    """
    if box is not None:
        # Get all box points
        # point_A = box[0]  # point A
        point_B = box[1]  # point B
        # point_C = box[2]  # point C
        point_D = box[3]  # point D
        # For patches
        # The points that are returned are from the pixel outer edge of the square patch
        # In this case we have to mitigate the + 1 or - 1 error that is developed since it would calculate the distances
        # of two point incorrectly
        """
            For example:
                -> the points are returned into the following fashion:
                    1. Bottom left corner
                    2. Top left corner
                    3. Top right corner
                    4. Bottom right corner
                -> the counter will be applied 1 pixel before the actual patch is found
                -> in our case if the patch starts at (10,10), the counter would be applied to (9,9) thus returning 
                    (9,9) as a starting point of the patch instead of (10,10)
        """
        # distance_AB = math.sqrt(((point_A[0] + 1) - (point_B[0] + 1)) ** 2 + (point_A[1] - (point_B[1] + 1)) ** 2)
        # distance_BC = math.sqrt(((point_B[0] + 1) - point_C[0]) ** 2 + ((point_B[1] + 1) - (point_C[1] + 1)) ** 2)
        # distance_CD = math.sqrt(((point_C[0] - 1) - (point_D[0] - 1)) ** 2 + ((point_C[1] + 1) - point_D[1]) ** 2)
        # distance_DA = math.sqrt((point_D[0] - (point_A[0] + 1)) ** 2 + ((point_D[1] - 1) - (point_A[1] - 1)) ** 2)
        # print("Distances:", distance_AB, distance_BC, distance_CD, distance_DA)
        # Gets the exact section of the image we need
        return image[int(round(point_B[1]) + 1):int(round(point_D[1])), int(round(point_B[0]) + 1):int(round(point_D[0]))]
    else:
        list_of_all_pieces = []
        # Grayscaling the image
        og_height, og_width, _ = image.shape
        gray = openCV.cvtColor(image, openCV.COLOR_BGR2GRAY)
        _, threshold = openCV.threshold(gray, lower_bound, upper_bound, openCV.THRESH_BINARY)
        _, contours, hierarchy = openCV.findContours(threshold, openCV.RETR_LIST, openCV.CHAIN_APPROX_SIMPLE)
        openCV.drawContours(image, contours, -1, (0, 255, 0), 1)
        #  Sort contours
        print("con", len(contours))
        sorted_contours = sorted(contours, key=lambda ctr: openCV.boundingRect(ctr)[0])


def get_angle(region_of_interest):
    """
        Gets the angle of region of interest
    :param region_of_interest: The ROI containing a particular patch
    :return: The angle value of the patch inside ROI
    """
    gray = openCV.cvtColor(region_of_interest, openCV.COLOR_BGR2GRAY)
    _, threshold = openCV.threshold(gray, lower_bound, upper_bound, openCV.THRESH_BINARY)
    # Computes a rotated bonding box that contains all necessary coordinates of a piece
    coordinates = numpy.column_stack(numpy.where(threshold < 255))
    angle = openCV.minAreaRect(coordinates)[-1]
    return angle


def get_points(region_of_interest):
    """
        Gets the points of a region of interest that has had its angle corrected
    :param region_of_interest: The patch that we are going to find the 4 contour points
    :return: A list points that represents the contour surrounding the patch
    """
    gray = openCV.cvtColor(region_of_interest, openCV.COLOR_BGR2GRAY)
    _, threshold = openCV.threshold(gray, lower_bound, upper_bound, openCV.THRESH_BINARY)
    _, contours, hierarchy = openCV.findContours(threshold, openCV.RETR_LIST, openCV.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda x: openCV.contourArea(x))
    # TODO Find the four corner points
    print(len(sorted_contours))
    # for i in range(len(sorted_contours)):
    #     openCV.drawContours(region_of_interest, sorted_contours, len(sorted_contours) - 2, (0, 0, 255), 1)
    #     print(sorted_contours[i].shape)
    #     openCV.imshow("t", region_of_interest)
    #     openCV.waitKey(0)
    #     openCV.destroyAllWindows()
    # for i in range(len(sorted_contours)):
    #     openCV.drawContours(region_of_interest, sorted_contours[i], -1, (0, 0, 255), 1)
    #     openCV.imshow("t", region_of_interest)
    #     openCV.waitKey(0)
    #     openCV.destroyAllWindows()
    # Ge the minimum bounding box
    rectangle = openCV.minAreaRect(sorted_contours[len(sorted_contours) - 2])
    # Extract the 4 points of this minimum bounding box
    box = openCV.boxPoints(rectangle)
    # openCV.imshow("t", region_of_interest)
    # openCV.waitKey(0)
    # openCV.destroyAllWindows()

    return box
    pass


def detect(image):
    """
        Detects patches via contours
    :param image: The image we are working with
    :return: All contours found sorted in a ascending order
    """
    # Grayscaling the image
    og_height, og_width, _ = image.shape
    gray = openCV.cvtColor(image, openCV.COLOR_BGR2GRAY)
    _, threshold = openCV.threshold(gray, lower_bound, upper_bound, openCV.THRESH_BINARY)
    _, contours, hierarchy = openCV.findContours(threshold, openCV.RETR_LIST, openCV.CHAIN_APPROX_SIMPLE)
    #  Sort contours
    sorted_contours = sorted(contours, key=lambda ctr: openCV.boundingRect(ctr)[0])

    return sorted_contours


def load_puzzle(path=None):
    # input_image = openCV.imread("../output/no_rotation/cat_asym_no.png")
    # input_image = openCV.imread("../output/no_rotation/cat_4_no.png")
    # input_image = openCV.imread("../output/cat4pieces.png")
    # input_image = openCV.imread("../output/rotated/cat_4asym_rotation.png")
    if path is not None:
        input_image = openCV.imread(path)
    else:
        input_image = openCV.imread("../output/rotated/cat_16_90.png")
        # input_image = openCV.imread("../output/rotated/cat_asym_90.png")
        # input_image = openCV.imread("../output/no_rotation/cat_4_no.png")
    return input_image


def main(path=None):
    """

    :param path: The path to the puzzle from which are going to extract the square pieces
    :return: returns a list of all extracted square pieces
    """
    # pieces = detect_pieces(load_puzzle())
    # pieces = get_pieces(load_puzzle(path))
    # for p  in range(len(pieces)):
    #     openCV.imshow("test", pieces[p])
    #     openCV.waitKey(0)
    #     openCV.destroyAllWindows()
    # Returns pieces
    pieces, dimensions, og_dimensions = get_pieces(load_puzzle(path))

    return pieces, dimensions, og_dimensions



# if __name__ == "__main__":
#     main()
