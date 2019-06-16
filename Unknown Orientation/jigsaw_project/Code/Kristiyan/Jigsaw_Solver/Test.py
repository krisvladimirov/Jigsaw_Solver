import Code.Kristiyan.Jigsaw_Solver.Compatibility as Compatibility
import Code.Kristiyan.Jigsaw_Solver.Detector as Detector
import Code.Kristiyan.Jigsaw_Solver.Piece as Piece
import Code.Kristiyan.Jigsaw_Solver.Side as Side
import Code.Kristiyan.Jigsaw_Solver.Solver as Solver
# from Code.Kristiyan.Network.PuzzleNetwork import JigNetwork

import numpy
import Code.Kristiyan.Jigsaw_Solver.Constants as Constants
import cv2 as openCV
from sympy.ntheory import factorint
from sympy.ntheory import divisors, divisor_count
from functools import reduce
from collections import deque
from operator import sub
#
# import keras
# from keras.applications.resnet50 import ResNet50


# TODO - How much to rotate i % 4
"""
    Rotation testing
"""
#
# for i in range(10):
#     print("At index: ", i, ":-: How many rotations should be done: ", i % 4)

"""
    Testing unsharp filter
"""

# Pipeline image stuff
off_set = 25
border_width = 10
extracted_pieces, dimensions, _ = Detector.main("../output/rotated/big_cat_9_90.png")
for i in range(len(extracted_pieces)):
    piece = extracted_pieces[i]
    height, width, colour = extracted_pieces[i].shape
    edged_frame = numpy.full((height + 2 * off_set, width + 2 * off_set, colour), (0, 0, 0), dtype="uint8")
    piece[border_width:height - border_width, border_width:width - border_width] = (0, 0, 0)
    edged_frame[off_set:height + off_set, off_set:width + off_set] = piece
    openCV.imwrite("weights/" + str(i) + ".png", edged_frame)

# def unsharp_mask(image, kernel_size=(9, 9), sigma=5.0, amount=1.0, threshold=0):
#     """Return a sharpened version of the image, using an unsharp mask."""
#     blurred = openCV.GaussianBlur(image, kernel_size, sigma)
#     sharpened = float(amount + 1) * image - float(amount) * blurred
#     sharpened = numpy.maximum(sharpened, numpy.zeros(sharpened.shape))
#     sharpened = numpy.minimum(sharpened, 255 * numpy.ones(sharpened.shape))
#     sharpened = sharpened.round().astype(numpy.uint8)
#     if threshold > 0:
#         low_contrast_mask = numpy.absolute(image - blurred) < threshold
#         numpy.copyto(sharpened, image, where=low_contrast_mask)
#     return sharpened
#
#
# def example():
#     image = openCV.imread('../input/cat_big.jpg')
#     sharpened_image = unsharp_mask(image)
#     openCV.imwrite("../input/Sharpened.png", sharpened_image)
#     # openCV.imshow("Sharpened", sharpened_image)
#     # openCV.waitKey(0)
#     # openCV.destroyAllWindows()
#
#
# example()


"""
    Testing the weights once more
"""
# height = 960
# width = 960
# extracted_pieces, dimensions, og_dimensions = Detector.main("../output/rotated/big_cat_v1_4_90.png")
# sol = Solver.Solver()
# piece_a = extracted_pieces[2]
# piece_b = extracted_pieces[3]
# testing_images = [numpy.rot90(piece_a, k=3), numpy.rot90(piece_b, k=3)]
# sol.start_solving(testing_images, dimensions, og_dimensions)
# x = sol.weights_0_4
# print(x.min())
# z = set()
# for i in range(4):
#     for j in range(4):
#         sol = Solver.Solver()
#         testing_images = [numpy.rot90(piece_a, k=i), numpy.rot90(piece_b, k=j)]
#         sol.start_solving(testing_images, dimensions, og_dimensions)
#         k = sol.weights_0_4
#         x = sol.weights_0_4.min()
#         # print(str(i), str(j), ":", x)
#         print(str(i) + " " + str(j) + " -> " + "piece_0: " + str(sol.weights_0_4[0][1].min()) + ", piece_1: " + str(sol.weights_0_4[1][0].min()))
#         z.add(x)
#
# print("From set:", z)

# x = {"1": 1, "2": 2}
# keys = x.keys()
# print(type(keys))
# rot_x = numpy.rot90(x, k=1)
# print(rot_x)


"""
    Network testing
"""
# height = 320
# width = 320
# piece_a = None
# piece_b = None
# piece_c = None
# piece_d = None
# combined_img = numpy.zeros((640, 640, 3), dtype="uint8")
# extracted_pieces, dimensions, _ = Detector.main("../output/non_rotated/supra_16_no.png")
# for i in range(len(extracted_pieces)):
#     if i == 1:
#         piece_a = extracted_pieces[i]
#     if i == 4:
#         piece_b = extracted_pieces[i]
#     if i == 11:
#         piece_c = extracted_pieces[i]
#     if i == 15:
#         piece_d = extracted_pieces[i]
#
#
# combined_img[0:320, 320:640] = piece_b
# combined_img[0:320, 0:320] = piece_d
# combined_img[320:640, 0:320] = piece_a
# combined_img[320:640, 320:640] = piece_c
#
# prediction_a = JigNetwork.predict_image(combined_img, "../Network/PuzzleNetwork/jignet_v4_keras_resnet.h5")
# print(prediction_a)

"""
    Testing return types
"""



# def test_func():
#     return 1, 2
#
#
# x = test_func()
# print(type(x))


"""
    Testing weight computing with rotation
"""

piece_a = None
piece_b = None
rotated_a = None
rotated_b = None
extracted_pieces, dimensions, og_dimensions = Detector.main("../output/rotated/cat_testing/cat_4_90_v2.png")
sol = Solver.Solver()
testing_images = []
for i in range(len(extracted_pieces)):
    if i == 0:
        testing_images.append(extracted_pieces[i])
    if i == 2:
        testing_images.append(extracted_pieces[i])

sol.start_solving(testing_images, dimensions, og_dimensions)


# Test 1
"""
    Done on cat_4_90_v3
    Relation: R->L
    cat_4_90_v1 where we have to rotate the image on the right so we can get the left side properly
    Pieces = { 0, 2 }
    Side = { Right, Top } after rotation:
    Side = { Right, Top -> Left }
    Expected output = 1221138666.7254162
"""
print(Compatibility.mgc_ssd_compatibility(piece_a, rotated_b, Constants.RIGHT_LEFT))


# Test 2
"""
    Done on cat_4_90_v3
    Relation: B->T
    cat_4_90_v2 where we have to rotate the image on the right so we its top side correctly
    Pieces = { 1, 2 }
    Side = { Bottom, Right } after rotation:
    side = { Bottom, Right -> Top }
    # 1221138666.7254162
"""
# print(Compatibility.mgc_ssd_compatibility(piece_a, rotated_b, Constants.BOTTOM_TOP))


# Test 3
"""
    Done on cat_4_90_v3
    Relation: R->L
    cat_4_90_v4 we do not have to rotate anything we just pass two images with right and left side
    Pieces = { 3, 2 }
    Side = { Right, Left } no rotation required
    # 1221138666.7254162     
"""
# print(Compatibility.mgc_ssd_compatibility(piece_a, piece_b, Constants.RIGHT_LEFT))


# Test 4
"""
    cat_4_90_v3 we do not have to rotate anything we just pass two images
    
    Relation: B->T
    Pieces = { 3, 2 }
    Side = { Bottom, Top }
    # 1221138666.7254162
    
    # Rotating it, to make Right -> Left relation, by 1
    Relation: R->L
    Pieces = { 3, 2 }
    Side = { Bottom, Top } after rotation:
    Side = { Right, Left }
    # 1221138666.7254162
    # 1221138666.7254162
    
    # Rotating it, to make Top -> Bottom relation, by 2
    Pieces = { 3, 2 }
    Side = { Bottom, Top } after rotation:
    Pieces = { 2, 3 }
    Side = { Top, Bottom }
    # 1277553616.00141
    # Making the rotation from B -> T to T -> B there is a difference when calculating the weight for some reason
    # Thus this should apply also to changing the direction from R -> L to L -> R
    TESTED AT NEXT SECTION
"""

# print(Compatibility.mgc_ssd_compatibility(piece_a, piece_b, Constants.BOTTOM_TOP))
# print(Compatibility.mgc_ssd_compatibility(rotated_b, rotated_a, Constants.BOTTOM_TOP))

# Test 5
"""
    Done on cat_4_90_v2
    Rotating it, to make Right -> Left, but upside down to see if it makes a difference
    Pieces = { 2, 1 }
    Side = { Right, Bottom } after rotation (k=3):
    Side = { Right, Left } (at this point both of the piece images will be upside down
    # 1277553616.0014102
"""
# print(Compatibility.mgc_ssd_compatibility(piece_a, rotated_b, Constants.RIGHT_LEFT))

# So wen comparing we can see that if the image is upside down it does affect the weight calculations
# Test 5 and Test 4 last sub-test


"""
    Done on cat_4_90_v2
    Rotating piece b to produce a Bottom -> Top relationship
    Pieces = { 1, 2 }
    Side = { Bottom, Right } after rotation (k=1):
    Side = { Bottom, Top }
    # 1221138666.7254162
"""
# print(Compatibility.mgc_ssd_compatibility(piece_a, rotated_b, Constants.BOTTOM_TOP))
# Exact same result as in Test 1, 2, 3, 4 (first)


"""
    Generic Testing
"""

#
# class Car():
#     def __init__(self, make, index):
#         self.make = make
#         self.index = index
#
#     def get_make(self):
#         return self.make
#
#
# list_of_cars = [Car('BMW', 0), Car("Toyota", 1), Car("Audi", 2)]
# x = list_of_cars[0]  # As long as there is a reference to an object it will not get deleted
# print(list_of_cars)
# print(x)
# list_of_cars[0] = None
# print(list_of_cars)
# del x
# print(x)


# dict2 = {
#     4: (2, 1)
# }


# dict1 = {
#     8: (0, 0),
#     1: (0, 1),
#     19: (1, 0),
#     10: (1, 1)
# }
#
# dict2 = {
#     4: (2, 1)
# }
#
# values1 = dict1.values()
# values2 = dict2.values()



# print(set(values1))
# print(set(values2))
# # How we check if two sets have any common values i.e. Intersection
# print(set(values1) & set(values2))
# comp = set(values1) & set(values2)
# if not comp:
#     print("Is empty")

# print(len(dict1))

#
#
# x = [1,2,3,4]
# while x:
#     print("Removed", x.pop())
#     print("New list size", len(x))
# Convert numpy array to tuple
# def totuple(a):
#     try:
#         return tuple(totuple(i) for i in a)
#     except TypeError:
#         return a
# side_a = []
# side_b = []
# side_c = []
# queue = deque(maxlen=6)
# queue.append((0, 0))
# queue.append((0, 1))
# queue.append((0, 2))
# queue.append((1, 0))
# queue.append((1, 1))
# queue.append((1, 2))
# approximated = numpy.absolute(tuple(map(sub, (1, 0), (0, 1))))
# converted = tuple(map(tuple, approximated))
# converted2 = totuple(approximated)
# print(type(approximated))
# print(type(converted))
# if converted2 in queue:
#     print("It is in")
# else :
#     print("It is not")
# side_a = []
# side_b = []
# shaped_top_side = []

# myList = [2,3,-3,-2]
# Soluton to problem in solver
# print(tuple(map(abs, tuple(map(sub, (1, 0), (0, 1))))))

# x = (1, 1)
# y = (1, 0)
# print(tuple(map(sub, x, y)))
#
# image = openCV.imread("../input/cat_rectangle_image.jpg")
# print(image.shape)
# solution = numpy.ones((250, 500, Constants.COLOUR_CHANNELS),
#                                        dtype="uint8")
# print(solution.shape)
# piece_a = 0
# piece_b = 0
# side_a = 0
# side_aa = 0
# side_b = 0
# side_bb = 0
# rotated_a = 0
# rotated_b = 0
# extracted_pieces, dimensions, _ = Detector.main("../output/no_rotation/cat_4_no.png")
# for i in range(len(extracted_pieces)):
#     # openCV.imshow("Extracted_In_Solver " + str(i), extracted_pieces[i])
#     # openCV.waitKey(0)
#     # openCV.destroyAllWindows()
#     # Instantiates a single piece with its four sides
#     # It also includes the index of the piece based on the order that was extracted
#     single_piece = Piece.Piece(extracted_pieces[i], i)
#
#     # All because the equations work on a Right-Left principle
#     if i == 3:
#         piece_b = extracted_pieces[i]
#         # Initially we are interested in the BT relationship, so we rotate the piece TOP to become the LEFT side
#         # Rotating piece_b first
#         rotated_b = numpy.rot90(piece_b, 0).astype(numpy.int16)
#         # Getting the Right side
#         side_bb = rotated_b[0:height, 0:1]
#         side_b = numpy.reshape(rotated_b[0:height, 0:1], (1, width, 3))
#         print(side_b.shape)
#     elif i == 0:
#         piece_a = extracted_pieces[i]
#         # Initially we are interested in the BT relationship, so we rotate the piece BOTTOM to become the RIGHT side
#         # Rotating piece_a first
#         rotated_a = numpy.rot90(piece_a, 0).astype(numpy.int16)
#         # Getting the BOTTOM
#         side_aa = rotated_a[0:height, (width - 1):width]
#         side_a = numpy.reshape(rotated_a[0:height, (width - 1):width], (1, width, 3))
#         print(side_a.shape)
#
# print("SSD_ROTATION_COMPATIBILITY:")
# print(Compatibility.ssd_rotation_compatibility(piece_a, piece_b, 0, 0))
# print("SSD_COMPATIBILITY(320,3):")
# print(Compatibility.ssd_compatibility(rotated_a[:, -1], rotated_b[:, 0]))
# print("SSD_COMPATIBILITY(1,320,3):")
# print(Compatibility.ssd_compatibility(side_a, side_b))
# print("SSD_COMPATIBILITY(320,1,3):")
# print(Compatibility.ssd_compatibility(side_aa, side_bb))

# # Bottom -> Top test for non rotated images
# print("MGC with adjusting rotation between Bottom and Top. This is equivalent to Right -> Left")
# print(Compatibility.mgc_rotation_compatibility(piece_a, piece_b, 1, 1))
# print("MGC with adjusting rotation between Top and Bottom. This is equivalent to Left -> Right")
# print(Compatibility.mgc_rotation_compatibility(piece_b, piece_a, 1, 1))


# Left -> Right test for non rotated images
# """
#     It would be best if we have a Left -> Right relationship to transform it to a Right -> Left relationship
#
# """
# print("MGC with adjusting rotation between Left and Right")
# print(Compatibility.mgc_rotation_compatibility(piece_a, piece_b, 2, 0))
# print("MGC without adjusting rotation between Left and Right")
# print(Compatibility.mgc_rotation_compatibility(piece_a, piece_b, 0, 0))
# print("MGC without adjusting rotation but transforming Left -> Right relationship into Right -> Left relationship")
# print(Compatibility.mgc_rotation_compatibility(piece_b, piece_a, 0, 0))


# def example(x, y, z):
#     print(x)
#     print(y)
#     print(z)



# queue.append((0, 0))
# queue.append((0, 1))
# queue.append((0, 2))
# queue.append((1, 0))
# queue.append((1, 1))
# queue.append((1, 2))
#
# print(queue[0])

# extracted_pieces, dimensions = Detector.main("../output/rotated/cars/supra_16.1_90.png")
# extracted_pieces, dimensions, _ = Detector.main("../output/non_rotated/adac_gt_40_no.png")

# x = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
# x.remove((0, 0), (0, 1))
# print(x)
# Test array
# arr = numpy.full((8, 5), fill_value=Constants.VALUE_INITIALIZER, dtype="int8")
# print(arr)
# counter = 0
# # Doing it the lazy way
# for i in range(0, 8):
#     for j in range(0, 5):
#         arr[i, j] = counter
#         counter = counter + 1
#
# print(arr)
# rotated = numpy.rot90(arr)
# print(rotated)
# rotated[0, 0] = -2
# rotated[1, 0] = -1
# print(rotated)
# print(rotated[1, 0])
# tu = (1, 0)
# if rotated[tu] == -1:
#     print("Horray!")
# # How to make the initial original positions

# rotated = numpy.rot90(np_matrix)
#
# print(matrix)
# print("")
# print(np_matrix)
# print("")
# print(rotated)
# print("")
# print(np_matrix[0][2])


# % remainder
# // round number

# def factors(n):
#     return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

# def get_factors(n):
#     factors = []
#     for heightt in range(1, n):
#         if n % heightt == 0:
#             pair = (int(n / heightt), heightt)
#             factors.append(pair)
#
#     return factors
#
#
# print(get_factors(40))
# for i in range(len(extracted_pieces)):
#     openCV.imshow("Extracted_In_Solver " + str(i), extracted_pieces[i])
#     openCV.waitKey(0)
#     openCV.destroyAllWindows()
#     # Instantiates a single piece with its four sides
#     # It also includes the index of the piece based on the order that was extracted
#     single_piece = Piece.Piece(extracted_pieces[i], i)
    # For presentation
    # img = numpy.zeros((500, 500, 3), dtype="uint8")
    # if i == 3:
    #     # Get the sides
    #     left_side = single_piece.piece[0:height, 0:4]
    #     top_side = single_piece.piece[0:4, 0:width]
    #     right_side = single_piece.piece[0:height, (width - 4):width]
    #     bottom_side = single_piece.piece[(height - 4):height, 0:width]
    #
    #     # Put in the sides (MGC)
    #     img[10:height+10, 0+10:4+10] = left_side
    #     img[0+10:4+10, 0+10:width+10] = top_side
    #     img[0+10:height+10, (width - 4)+10:width+10] = right_side
    #     img[(height - 4)+10:height+10, 0+10:width+10] = bottom_side
    #
    #     openCV.imwrite("Img/sides_cat.png", img)
    #     openCV.imshow("Sides", img)
    #     openCV.waitKey(0)
    #     openCV.destroyAllWindows()

    # if i == 2:
    #     # Bottom
    #     made_up = numpy.rot90(single_piece.piece, 1)
    #     openCV.imshow("Rotated", made_up)
    #     openCV.waitKey(0)
    #     openCV.destroyAllWindows()
    #     side_a = made_up[(height - 1):height, 0:width]
    #     shaped_a = side_a
    #
    # elif i == 1:
    #     # Right
    #     side_b = numpy.rot90(single_piece.piece[0:height, (width - 1):width], 1)
    #     shaped_b = side_b

# (0, 2, 5, ?)
# (2, 1, 2, *56*)  (1, 2, 14, *56*)
# (0, 3, 10, ?) (3, 0, 10, ?) - No rotation *1069*
#

# Left  -   [0:height, 0:1]
# Top   -   [0:1, 0:width]
# Right -   [0:height, (width - 1):width]
# Bottom  -   [(height - 1):height, 0:width]
# Result    -   [(2, 1, 5, 23), (2, 0, 5, 24), (3, 2, 0, 538)]
# (1, 320, 3)
#
# print("Shape of side_a:", side_a.shape)
# print("Shape of side_b:", side_b.shape)
# print("Dissimilarity between them without reshaping ", end="")
# print(Compatibility.ssd_compatibility(side_a, side_b))
#
# # TODO 295037 before reshaping
# # TODO 43263 After reshaping
# print("Dissimilarity between them with reshaping ", end="")
# # print(Compatibility.ssd_compatibility(shaped_top_side, side_b))
# print(Compatibility.ssd_compatibility(shaped_a, shaped_b))
# Piece_a: 3 Piece_b: 2 Orientation_a: 2 Orientation_b: 3 Dissimilarity: 72


# side_a = numpy.random.randint(0, 256, (1, 5, 3), dtype="uint8")
# side_b = numpy.random.randint(0, 256, (5, 1, 3), dtype="uint8")
# print("A: \n", side_a, end="\n\n")
# print("B: \n", side_b)
# side_c = numpy.reshape(side_b, (1, 5, 3))
# print("C: \n", side_c)
# print(Compatibility.ssd_compatibility(side_a, side_b))
# # print(side_b.shape, side_c.shape)
# print(Compatibility.ssd_compatibility(side_a, side_c))
#
# # TODO before reshaping
# # TODO 1526 (5,1,3)

"""
    Testing of python copy mechanics
"""

# positions = numpy.full((3, 3), fill_value=Constants.VALUE_INITIALIZER, dtype="int8")
# positions[0][0] = 4
# positions[1][0] = 1
# shallow_copy = positions.copy()
#
# shallow_copy[0][0] = 5
#
#
# print("Old list:")
# print(positions)
# print("Shallow copy_changed")
# print(shallow_copy)

# print("Old list type:", type(positions))
# print("Shallow copy:", type(shallow_copy))
