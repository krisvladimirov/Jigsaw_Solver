import Compatibility as Compatibility
import Detector as Detector
import Piece as Piece
import Side as Side
import numpy
import Constants as Constants
import cv2 as openCV
from sympy.ntheory import factorint
from sympy.ntheory import divisors, divisor_count
from functools import reduce
from collections import deque
from operator import sub
from scipy.io import loadmat
import h5py


"""
    Reading the .mat file
    Three ways to read by PyTables, or with Scipy.io import loadmat, or with h5py
    Using h5py
"""


# counter = 0
# t = numpy.zeros((3, 4), dtype="int8")
# for x in range(4):
#     for y in range(3 - 1, -1, -1):
#         t[y][x] = counter
#         counter = counter + 1
#
# print(t)
arr = [
    ['c', 'f', 'i', 'l'],
    ['b', 'e', 'h', 'k'],
    ['a', 'd', 'g', 'j']
]
#
# # From right to left, row by row
# for y in range(3):
#     for x in range(3, -1, -1):
#         print(arr[y][x], end=", ")
#     print("")
# print("")
#
# # From top to bottom, column by column
# for x in range(4):
#     for y in range(3):
#         print(arr[y][x], end=", ")
#     print("")
# print("")
#
# # From bottom to top, column by column
# for x in range(4):
#     for y in range(2, -1, -1):
#         print(arr[y][x], end=", ")
#     print("")

for i in range(0, len(arr[0]), 1):
    print(i)


# def read_cycle_data(path_to_file, how_many_correspondences):
#
#     file = h5py.File(path_to_file)
#     destination = file["/matching/value"]
#     total_number = len(destination) * how_many_correspondences
#
#     storage = numpy.zeros(shape=total_number, dtype="float64")
#     for index in range(total_number):
#         for val in file[destination[index][0]][()]:
#             storage[index] = val[0]
#
#     return storage
# s = read_cycle_data('matching.mat', 1)

# img = openCV.imread("../output/no_rotation/big_cat_36_no.png")
# openCV.imshow("", img)
# openCV.waitKey(0)
# openCV.destroyAllWindows()

# file = h5py.File("matching.mat")
# matching_group = file["matching"]
# value_data = matching_group["value"]
# # print(value_data)
# actual_data = value_data[
#
# print(file.get("matching/value")[()])
# But they are HDF5 object references
# print(actual_data[5, :].value)



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
height = 320
width = 320
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
# extracted_pieces, dimensions, _ = Detector.main("../output/no_rotation/adac_gt_40_no.png")

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
