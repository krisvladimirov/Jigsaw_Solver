Author: Kristiyan Vladimirov
Version: 1.0
------------------------------------------------------------------------------------------------------------------------
Description:
    - Used to cut images into slices and shuffle them in order to create a jigsaw image to be solved
    - Two available options of slicing an image:
        1.  Slice an image into (N x N), additionally it could be rotated by 90, 180, 270 degrees or randomly
        2.  Slice an image into multiple patches of size (N x N), additionally it could be rotated by 90, 180, 270
            degrees or randomly
------------------------------------------------------------------------------------------------------------------------
How to work with it:
    0.0 Use the main() function for calling, since it will always be called

    1.1 Choose grid() if you want to slice an image into a N x N grid, i.e. 4 x 4 which is a total of 16 pieces
        -   Inside the grid function the image to be processed is loaded
        -   3 possible calls of the slice function which would start the cutting process:
            -> slice(input_image, S) or slice(input_image, S, 90) or   slice(input_image, S, "random"), where S = N * N

    1.2 Choose patches() if you want to slice an image into patches, i.e. 14x14 where each patch has 14x14 dimensions
        -   Inside the patches function the image to be processed is loaded
        -   3 possible calls of the slice function which would start the cutting process:
            -> patch_slice(input_image, S) or patch_slice(input_image, S, 90) or patch_slice(input_image, S, "random"),
            -> where S is the side of the desired patch

    IMPORTANT
    2. Both slice() and patch_slice() save the image after the process of cutting, shuffling and assembly has finished
        -   Specify the target output location of the jigsaw image and its name
