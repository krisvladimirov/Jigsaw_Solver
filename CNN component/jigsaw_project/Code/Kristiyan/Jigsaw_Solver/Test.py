import Compatibility as Compatibility
import Detector as Detector
import Piece as Piece
import Side as Side
import numpy
import Constants as Constants
import cv2 as openCV
import matplotlib.pyplot as plt
from Network import JigNetwork as jn
# from sympy.ntheory import factorint
# from sympy.ntheory import divisors, divisor_count
from functools import reduce
from collections import deque
from operator import sub
import  pandas
"""
    Example on how to reshape arrays
"""


def main():
    epoch = """Epoch 1/10  - 3758s - loss: 0.3428 - acc: 0.8680 - val_loss: 0.1503 - val_acc: 0.9452
    Epoch 2/10 - 1729s - loss: 0.1327 - acc: 0.9518 - val_loss: 0.1174 - val_acc: 0.9607
    Epoch 3/10 - 1728s - loss: 0.0900 - acc: 0.9684 - val_loss: 0.0968 - val_acc: 0.9656
    Epoch 4/10 - 1727s - loss: 0.0694 - acc: 0.9756 - val_loss: 0.1058 - val_acc: 0.9601
    Epoch 5/10 - 1728s - loss: 0.0590 - acc: 0.9795 - val_loss: 0.0619 - val_acc: 0.9787
    Epoch 6/10 - 1728s - loss: 0.0512 - acc: 0.9822 - val_loss: 0.0586 - val_acc: 0.9798
    Epoch 7/10 - 1728s - loss: 0.0449 - acc: 0.9847 - val_loss: 0.0646 - val_acc: 0.9773
    Epoch 8/10 - 1727s - loss: 0.0390 - acc: 0.9867 - val_loss: 0.0952 - val_acc: 0.9631
    Epoch 9/10 - 1728s - loss: 0.0347 - acc: 0.9880 - val_loss: 0.0865 - val_acc: 0.9667
    Epoch 10/10 - 1729s - loss: 0.0316 - acc: 0.9890 - val_loss: 0.0538 - val_acc: 0.9812
    Epoch 1/10 - 1743s - loss: 0.5155 - acc: 0.8204 - val_loss: 0.3385 - val_acc: 0.8654
    Epoch 2/10 - 1733s - loss: 0.1222 - acc: 0.9555 - val_loss: 0.0943 - val_acc: 0.9666
    Epoch 3/10 - 1741s - loss: 0.0834 - acc: 0.9705 - val_loss: 0.0984 - val_acc: 0.9671
    Epoch 4/10 - 1735s - loss: 0.0683 - acc: 0.9763 - val_loss: 0.0759 - val_acc: 0.9730
    Epoch 5/10 - 1733s - loss: 0.0584 - acc: 0.9800 - val_loss: 0.0655 - val_acc: 0.9771
    Epoch 6/10 - 1732s - loss: 0.0503 - acc: 0.9828 - val_loss: 0.0824 - val_acc: 0.9727
    Epoch 7/10 - 1733s - loss: 0.0466 - acc: 0.9841 - val_loss: 0.0673 - val_acc: 0.9766
    Epoch 8/10 - 1733s - loss: 0.0403 - acc: 0.9863 - val_loss: 0.0676 - val_acc: 0.9788
    Epoch 9/10 - 1733s - loss: 0.0356 - acc: 0.9876 - val_loss: 0.0709 - val_acc: 0.9773
    Epoch 10/10 - 4567s - loss: 0.0321 - acc: 0.9888 - val_loss: 0.0609 - val_acc: 0.9799
    Epoch 1/10 - 1749s - loss: 0.3966 - acc: 0.8302 - val_loss: 0.1280 - val_acc: 0.9525
    Epoch 2/10 - 1739s - loss: 0.1091 - acc: 0.9609 - val_loss: 0.1313 - val_acc: 0.9545
    Epoch 3/10 - 1737s - loss: 0.0799 - acc: 0.9720 - val_loss: 0.0726 - val_acc: 0.9740
    Epoch 4/10 - 1739s - loss: 0.0672 - acc: 0.9767 - val_loss: 0.0656 - val_acc: 0.9777
    Epoch 5/10 - 2701s - loss: 0.0568 - acc: 0.9803 - val_loss: 0.0680 - val_acc: 0.9767
    Epoch 6/10 - 4002s - loss: 0.0493 - acc: 0.9832 - val_loss: 0.0911 - val_acc: 0.9663
    Epoch 7/10 - 3656s - loss: 0.0439 - acc: 0.9851 - val_loss: 0.0523 - val_acc: 0.9823
    Epoch 8/10 - 1740s - loss: 0.0391 - acc: 0.9866 - val_loss: 0.0549 - val_acc: 0.9817
    Epoch 9/10 - 1740s - loss: 0.0349 - acc: 0.9881 - val_loss: 0.0685 - val_acc: 0.9761
    Epoch 10/10 - 1740s - loss: 0.0312 - acc: 0.9892 - val_loss: 0.0615 - val_acc: 0.9809
    Epoch 1/10 - 1756s - loss: 0.4377 - acc: 0.8139 - val_loss: 0.1536 - val_acc: 0.9425
    Epoch 2/10 - 1744s - loss: 0.1114 - acc: 0.9604 - val_loss: 0.0967 - val_acc: 0.9636
    Epoch 3/10 - 1745s - loss: 0.0792 - acc: 0.9721 - val_loss: 0.0873 - val_acc: 0.9689
    Epoch 4/10 - 1745s - loss: 0.0654 - acc: 0.9776 - val_loss: 0.0703 - val_acc: 0.9761
    Epoch 5/10 - 1744s - loss: 0.0562 - acc: 0.9806 - val_loss: 0.0622 - val_acc: 0.9791
    Epoch 6/10 - 1743s - loss: 0.0551 - acc: 0.9811 - val_loss: 0.0761 - val_acc: 0.9737
    Epoch 7/10 - 1744s - loss: 0.0451 - acc: 0.9845 - val_loss: 0.0815 - val_acc: 0.9721
    Epoch 8/10 - 1745s - loss: 0.0412 - acc: 0.9860 - val_loss: 0.0620 - val_acc: 0.9784
    Epoch 9/10 - 1745s - loss: 0.0360 - acc: 0.9877 - val_loss: 0.0572 - val_acc: 0.9803
    Epoch 10/10 - 1745s - loss: 0.0323 - acc: 0.9889 - val_loss: 0.0613 - val_acc: 0.9811
    Epoch 1/10 - 1756s - loss: 0.7455 - acc: 0.6276 - val_loss: 0.6975 - val_acc: 0.8727
    Epoch 2/10 - 1744s - loss: 0.1586 - acc: 0.9420 - val_loss: 0.2172 - val_acc: 0.9207
    Epoch 3/10 - 1744s - loss: 0.0954 - acc: 0.9657 - val_loss: 0.0936 - val_acc: 0.9680
    Epoch 4/10 - 1744s - loss: 0.0730 - acc: 0.9746 - val_loss: 0.0927 - val_acc: 0.9684
    Epoch 5/10 - 1744s - loss: 0.0593 - acc: 0.9795 - val_loss: 0.0797 - val_acc: 0.9717
    Epoch 6/10 - 1744s - loss: 0.0533 - acc: 0.9819 - val_loss: 0.0558 - val_acc: 0.9807
    Epoch 7/10 - 1744s - loss: 0.0464 - acc: 0.9842 - val_loss: 0.0835 - val_acc: 0.9707
    Epoch 8/10 - 1745s - loss: 0.0415 - acc: 0.9860 - val_loss: 0.0535 - val_acc: 0.9815
    Epoch 9/10 - 1745s - loss: 0.0355 - acc: 0.9879 - val_loss: 0.0658 - val_acc: 0.9777
    Epoch 10/10 - 1744s - loss: 0.0327 - acc: 0.9886 - val_loss: 0.0533 - val_acc: 0.9826"""
    # convert string to csv
    # data = epoch.splitlines()
    # for i in range(len(data)):
    #     row = data[i].split("-")
    #     loss = row[2][-7:]
    #     acc = row[3][-7:]
    #     val_loss = row[4][-7:]
    #     val_acc = row[5][-7:]
    #     print("{0},jignet_v16_keras_resnet_all_cv{1}.h5,{2},{3},{4},{5}".format(i % 10, i//10, acc, loss, val_acc, val_loss))

    # file_path = "Network//training_result.csv"
    # data = pandas.read_csv(file_path)
    # resnet = data.loc[data['name'] == ' jignet_v16_keras_resnet_all_cv2.h5']
    # vgg = data.loc[data['name'] == 'jignet_v22_vgg_all_cv2.h5']
    # num_epoch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #
    # plt.figure(1)
    # plt.title("ResNet50 Training Loss vs VGG16 Training Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.plot(num_epoch, resnet['loss'], label="ResNet50 Training")
    # plt.plot(num_epoch, vgg['loss'], label="VGG16 Training")
    # plt.plot(num_epoch, resnet['val_loss'], label="ResNet50 Validation")
    # plt.plot(num_epoch, vgg['val_loss'], label="VGG16 Validation")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #
    # plt.figure(2)
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.plot(num_epoch, resnet['acc'], label="ResNet50 Training ")
    # plt.plot(num_epoch, vgg['acc'], label="VGG16 Training ")
    # plt.plot(num_epoch, resnet['val_acc'], label="ResNet50 Validation")
    # plt.plot(num_epoch, vgg['val_acc'], label="VGG16 Validation")
    # plt.title("ResNet50 Validation Loss vs VGG16 Validation Loss")
    # # Place a legend to the right of this smaller subplot.
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.savefig('validation_loss.png')
    # plt.show()

    #
    # print(vgg)
    # print(resnet['acc'])
    # import torchvision.models as models
    # from ptflops import get_model_complexity_info
    # net = models.densenet161()
    # myvgg = jn.create_vgg_model()
    # flops, params = get_model_complexity_info(myvgg, (64, 64), as_strings=True, print_per_layer_stat=True)
    # print('Flops:  ' + flops)
    # print('Params: ' + params)
    # Flops: 7.82
    # GMac
    # Params: 28.68
    # M

    # with torch.cuda.device(0):


if __name__ == "__main__":
    main()
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
