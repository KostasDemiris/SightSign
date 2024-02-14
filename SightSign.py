from numpy import asarray
import math
from PIL import Image
import matplotlib.pyplot as plt
import pygame
import pygame.camera
import time
from functools import reduce
from functools import partial
import operator


class matrix_operations:

    @staticmethod
    def identity_element(dim):
        matrix = [[0 for col in range(dim)] for row in range(dim)]
        for diagonal in range(dim):
            matrix[diagonal][diagonal] = 1
        return matrix

    @staticmethod
    def conformable_matrix(matrix_one, matrix_two):
        return len(matrix_one) == len(matrix_two)
        # When given two regular matrices (n x m and i x j), they're conformable if the num of rows of the first is
        # equal to the number of columns of the second

    @staticmethod
    def regular_matrix(matrix):
        for row in matrix:
            if len(row) != len(matrix[0]):
                return False
        return True

    @staticmethod
    def traditional_matrix_multiplication(matrix_one, matrix_two):
        if matrix_operations.conformable_matrix(matrix_one, matrix_two) and matrix_operations.regular_matrix(matrix_one) \
                and matrix_operations.regular_matrix(matrix_two):
            output_matrix = [[0 for col in range(len(matrix_two[0]))] for row in range(len(matrix_one))]
            for row in range(len(matrix_one)):
                for col in range(len(matrix_one[row])):
                    for sub_row in range(len(matrix_two)):
                        output_matrix[row][col] += matrix_one[row][sub_row] * matrix_two[sub_row][col]
            return output_matrix
        else:
            print("This operation cannot be performed")
            return False  # The operation cannot be performed

    @staticmethod
    def split(matrix):
        # this splits the matrix (which must be of order 2^n).
        mid_x, mid_y = len(matrix[0])//2, len(matrix)//2

        return [row[:mid_x] for row in matrix[:mid_y]], [row[mid_x:] for row in matrix[:mid_y]], [row[:mid_x] for
                                        row in matrix[mid_y:]], [row[mid_x:] for row in matrix[mid_y:]]

    @staticmethod
    def h_merge(source, target):
        for row in range(len(source)):
            source[row] += target[row]
        return source

    @staticmethod
    def v_merge(source, target):
        for row in range(len(target)):
            source.append(target[row])
        return source

    @staticmethod
    def strassen(source, target):
        # strassen matrix multiplication
        if len(source) == 1:
            return [[source[0][0] * target[0][0]]]
        s_q_one, s_q_two, s_q_three, s_q_four = matrix_operations.split(source)  # s_q_one is source quadrant one
        t_q_one, t_q_two, t_q_three, t_q_four = matrix_operations.split(target)
        # refer to this: https://www.geeksforgeeks.org/strassens-matrix-multiplication/
        # for the operations that are performed below.

        product_one = matrix_operations.strassen(s_q_one,
            matrix_operations.matrix_addition(t_q_two, matrix_operations.scalar_matrix_multiplication(t_q_four, -1)))
        product_two = matrix_operations.strassen(matrix_operations.matrix_addition(s_q_one, s_q_two), t_q_four)
        product_three = matrix_operations.strassen(matrix_operations.matrix_addition(s_q_three, s_q_four), t_q_one)
        product_four = matrix_operations.strassen(s_q_four,
            matrix_operations.matrix_addition(matrix_operations.scalar_matrix_multiplication(t_q_one, -1), t_q_three))
        product_five = matrix_operations.strassen(matrix_operations.matrix_addition(s_q_one, s_q_four),
            matrix_operations.matrix_addition(t_q_one, t_q_four))
        product_six = matrix_operations.strassen(matrix_operations.matrix_addition(s_q_two,
            matrix_operations.scalar_matrix_multiplication(s_q_four, -1)),
                                                 matrix_operations.matrix_addition(t_q_three, t_q_four))
        product_seven = matrix_operations.strassen(matrix_operations.matrix_addition(s_q_one,
            matrix_operations.scalar_matrix_multiplication(s_q_three, -1)),
                                                   matrix_operations.matrix_addition(t_q_one, t_q_two))

        # We use these to calculate the final four quadrants
        p_q_one = matrix_operations.matrix_addition(matrix_operations.matrix_addition(product_five, product_four),
            matrix_operations.matrix_addition(matrix_operations.scalar_matrix_multiplication(product_two, -1),
                                              product_six))
        p_q_two = matrix_operations.matrix_addition(product_one, product_two)
        p_q_three = matrix_operations.matrix_addition(product_three, product_four)
        p_q_four = matrix_operations.matrix_addition(matrix_operations.matrix_addition(product_one, product_five),
            matrix_operations.scalar_matrix_multiplication(matrix_operations.matrix_addition(product_three,
                                                                                             product_seven), -1))

        row_one = matrix_operations.h_merge(p_q_one, p_q_two)
        row_two = matrix_operations.h_merge(p_q_three, p_q_four)

        return matrix_operations.v_merge(row_one, row_two)

    @staticmethod
    def matrix_addition(matrix_one, matrix_two):  # Input matrices must be conformable to addition
        matrix_three = [[] for row in matrix_one]
        for row in range(len(matrix_one)):
            matrix_three[row] = [a + b for (a, b) in zip(matrix_one[row], matrix_two[row])]
        return matrix_three

    @staticmethod
    def gaussian_elimination(matrix):  # My implementation only works on square matrices because that's what I made it
        # for, all the library functions have checks before calling but be careful when calling alone otherwise
        for diagonal in range(len(matrix) - 1):
            # matrix[diagonal] = list(map(operator.mul(), 1 / matrix[diagonal][diagonal]), matrix[diagonal])
            for row in range(diagonal + 1, len(matrix)):

                if matrix_operations.approximately_equal(matrix[diagonal][diagonal], 0, 8):
                    # This just switches the rows if the diagonal factor is zero
                    found = False
                    for switch_row in range(len(matrix)):
                        if not matrix_operations.approximately_equal(matrix[switch_row][diagonal], 0, 8):
                            matrix[diagonal], matrix[switch_row] = matrix[switch_row], matrix[diagonal]
                            found = True
                    if not found:
                        return False
                    else:
                        return matrix_operations.gaussian_elimination(matrix)  # There's a smarter way to do this but
                    # I think I'm depreciating this function anyway so this is just a stopgap measure

                try:
                    multiplicative_factor = matrix[row][diagonal] / matrix[diagonal][diagonal]
                    matrix[row] = [a - (b * multiplicative_factor) for (a, b) in zip(matrix[row], matrix[diagonal])]

                except ZeroDivisionError:
                    return [[False for row in matrix] for column in matrix[0]]

        return matrix

    @staticmethod
    def matrix_determinant(matrix):
        if not matrix_operations.regular_matrix(matrix) and (len(matrix) == len(matrix[0])):
            return False  # The determinant only exists for a square matrix
        gaussian_matrix = matrix_operations.gaussian_elimination(matrix.copy())
        # matrix.copy because for some bizarre reason it would change the value of matrix anyway?
        if gaussian_matrix:
            return reduce(operator.mul,
                          [gaussian_matrix[diagonal][diagonal] for diagonal in range(len(gaussian_matrix))], 1)
        else:
            return 0  # The gaussian matrix operation failed because the matrix was singular, so we return 0 as the det.

    @staticmethod
    def matrix_transpose(matrix):
        output_matrix = [[0 for element in range(len(matrix[0]))] for row in range(len(matrix))]
        for row in range(len(matrix)):
            for column in range(len(matrix[0])):
                output_matrix[row][column] = matrix[column][row]
        return output_matrix

    @staticmethod
    def matrix_adjugate(matrix):
        # must have a square matrix input
        output_matrix = [[0 for element in range(len(matrix[0]))] for row in range(len(matrix))]
        for row in range(len(matrix)):
            for column in range(len(matrix[0])):
                output_matrix[row][column] = (math.pow(-1, (row + 1) + (column + 1))) * \
                                             matrix_operations.matrix_determinant(
                                                 list((row_e[:column] + row_e[column + 1:]) for row_e in
                                                      (matrix[:row] + matrix[row + 1:])))
                #                            alternates between plus and minus cause of cofactor matrix stuff...

        return matrix_operations.matrix_transpose(output_matrix)

    @staticmethod
    def scalar_matrix_multiplication(matrix, scalar):  # This is for 2d matrices
        return [[matrix[row][column] * scalar for row in range(len(matrix))] for column in range(len(matrix[0]))]

    @staticmethod
    def approximately_equal(a, b, decimal_points):  # Because vals are not exact when stored on computer
        return round(a, decimal_points) == round(b, decimal_points)

    @staticmethod
    def approximately_round(decimal_points, a):
        if matrix_operations.approximately_equal(a, round(a), 5):
            return round(a)
        return a

    @staticmethod
    def inverse_matrix(matrix):
        # This is used in reversing a dct coefficient matrix, which is implemented in the code but isn't used in
        # any of the hashing functions. This isn't the most efficient implementation but works well for the small
        # matrices we're working with post-resize.
        if not matrix_operations.approximately_equal(matrix_operations.matrix_determinant(matrix), 0, 5):

            # return [list(map(partial(matrix_operations.approximately_round(), 5), row)) for row in
            # matrix_operations.scalar_matrix_multiplication(matrix_operations.matrix_adjugate(matrix),
            # 1 / matrix_operations.matrix_determinant( matrix))]

            adj_scaled_matrix = matrix_operations.scalar_matrix_multiplication(
                matrix_operations.matrix_adjugate(matrix),
                1 / matrix_operations.matrix_determinant(
                    matrix))
            inverse_matrix = matrix_operations.matrix_transpose(adj_scaled_matrix)

            # Because of issues with rounding an' all, plus for this project especially the difference between
            # zero and 0.00000000023 is unimportant.

            return [list(map(partial(matrix_operations.approximately_round, 5), row)) for row in inverse_matrix]
        else:
            print(f"Matrix {matrix} cannot be inverted, its determinant is zero")
            return False


class user_interaction:  # This is where any interaction with the user occurs, such as taking photos, as well as where i
    # aggregate clumps of functions that deal with interaction e.g. taking a grayscale photo
    def __init__(self, settings_path):
        self.photo_in_use = None
        self.started_up = False
        self.camera_list = []
        self.settings_path = settings_path

    def set_up(self):  # this is a basic version that just initialises the camera, it should eventually also load
        # any settings provided by the user from previous sessions
        if not self.started_up:
            pygame.camera.init()
            self.camera_list = pygame.camera.list_cameras()
            time.sleep(1)
        # unClogger = u_int.take_grayscale_photo(dumpster, False)  # Long story, but the first image comes out super 
        # bright,
        self.started_up = True

    def webcamCapture(self, fileName):
        # This takes a photo using a webcam, and saves it to the provided file path as well as returning it.
        # There is error checking for no camera being detected, but I haven't implemented a safeguard for that error
        # being raised.

        if not self.started_up:
            print("starting up camera")
            self.set_up()

        if self.camera_list:
            Camera = pygame.camera.Camera(self.camera_list[0], (1920, 1080))
            # Opens the default camera (hopefully the webcam?) and has its dimensions set to the ones above

            Camera.start()  # Opens the camera
            for i in range(4):
                Photograph = Camera.get_image()  # Takes the photo and saves it to this variable
            pygame.image.save(Photograph, str(fileName))  # Saves the photo just taken as CurrentImage.jpg
            Camera.stop()
            return Photograph

        else:
            print("Error, no camera detected")
            return False

    def take_grayscale_photo(self, filePath, display):
        # aggregate function (not sure if this is the right term but this is what I call them)

        self.webcamCapture(filePath)
        grayImage = image_conv.image_as_grayscale(filePath)
        if display:
            image_conv.display_image_grayscale(grayImage)
        return grayImage


class image_conv:
    class hash_object:  # Currently unused, just a framework for potential use in later stages
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return self.value == other.value

        @staticmethod
        def calculate_hamming_distance(hash_one, hash_two):
            pass

    @staticmethod
    def XOR(bin_1, bin_2):
        if (bin_1 and not bin_2) or (bin_2 and not bin_1):
            return 1
        else:
            return 0

    @staticmethod
    def truthConversion(string):
        # Formerly used in the map function in hamming distance, until lambda functions replaced it, and then direct
        # comparisions replaced those. This is still here just in case
        return list(map((lambda s: s == "1"), string))

    @staticmethod
    def hamming_distance(bin_string_1, bin_string_2):
        if len(bin_string_1) == len(bin_string_2):  # necessary condition for the hamming distance
            # bin_string_1 = list(map((lambda x: x == "1"), bin_string_1))
            # bin_string_2 = list(map((lambda x: x == "1"), bin_string_2))
            return sum(
                [image_conv.XOR(bin_string_1[x] == "1", bin_string_2[x] == "1") for x in range(len(bin_string_1))])
        else:
            return False


    # a pretty much completely utility class for converting images to the desired type
    @staticmethod
    def image_as_grayscale(image_path):
        # Only accepts images with three or more colour channels, and will only consider
        # the first three, so preferable to not use CYMK formats or something like that. not sure why i would but yeah
        photo_to_convert = Image.open(f"{image_path}")
        photo_as_array = asarray(photo_to_convert)
        colour_array = photo_as_array.tolist()
        grayscale_array = [[None for i in range(len(colour_array[0]))] for x in range(len(colour_array))]
        for row in range(len(colour_array)):  # We will be using the luminosity formula to convert to each pixel to
            # grayscale, but this may change in the future.
            for pixel in range(len(colour_array[row])):
                gamma_corrected_pixel = image_conv.gamma_correct(colour_array[row][pixel])
                grayscale_array[row][pixel] = (
                    int((0.333 * gamma_corrected_pixel[0]) + (0.333 * gamma_corrected_pixel[1]) + (
                            0.333 * gamma_corrected_pixel[2])))
                # The average sum of the gamma corrected pixels is a greyscale conversion method called gleam,
                # which is the preferred method for performance according to Christopher Kanan and Garrison W.
                # Cottrell's paper on this exact topic. It's a short and pretty interesting paper, I recommend it.

        return grayscale_array  # a python array of integer grayscale values (no particular bounds depends on the image)

    @staticmethod
    def display_image_grayscale(image_as_grayscale):  # only takes grayscale (bitmap) version
        plt.imshow(image_as_grayscale, interpolation='nearest', cmap='gray', vmin=0, vmax=130)
        plt.show()

    @staticmethod
    def gamma_correct(colour_channel):  # takes in a one dimensional array with three integer colour values
        return [int(10 * colour ** (1 / 2.2)) for colour in colour_channel]
        # Gamma correction improves performance for image classification, apparently.

    @staticmethod
    def integral_image(image_array):  # accepts a grayscale, two-dimensional (rows with pixels in them) python array
        image_rows = len(image_array)
        row_length = len(image_array[0])
        integral_array = [[] for row in range(image_rows)]
        integral_array[0].append(image_array[0][0])  # Exception to calculations because it has no left or above pixel

        for pixel in range(1, row_length):  # These have no "above" pixel
            integral_array[0].append(image_array[0][pixel] + integral_array[0][pixel - 1])

        for row in range(1, image_rows):
            integral_array[row].append(
                integral_array[row - 1][0] + image_array[row][0])  # See the above for why we do this
            for pixel in range(1, row_length):
                integral_array[row].append(image_array[row][pixel] + integral_array[row - 1][pixel]
                                           + integral_array[row][pixel - 1] - integral_array[row - 1][pixel - 1])  # As
                # integral images, they encompass all the area to the left and above them, so we take two additional
                # integral points, we have to consider (and remove) their overlap

        return integral_array  # come back to this function and use the method outlined in the efficient integral paper

    @staticmethod
    def calculate_integral_area(integral_image, bottom_point, top_point):  # Takes in an integral image python array as
        # outputted by the integral image function, as well as two one dimensional point arrays, containing x,
        # y values in that order. This does not test the bounds of those values because it is intended that they be
        # previously validated, so this can bug out.

        if bottom_point[0] == 0:  # makes life easier
            if bottom_point[1] == 0:
                return integral_image[top_point[1]][top_point[0]]
            return integral_image[top_point[1]][top_point[0]] - integral_image[bottom_point[1]][top_point[0] - 1]

        elif bottom_point[1] == 0:
            return integral_image[top_point[1]][top_point[0]] - integral_image[top_point[1]][bottom_point[0] - 1]

        else:
            return integral_image[top_point[1]][top_point[0]] - integral_image[top_point[1]][bottom_point[0] - 1] \
                   - integral_image[bottom_point[1]][top_point[0] - 1] + integral_image[bottom_point[1]][
                       bottom_point[0]]

    @staticmethod
    def average(values):  # Takes the average of an array
        return reduce(lambda x, y: x + y, values) / len(values)

    @staticmethod
    def resize(image, width, length):  # width and length must be less than the original's, this downsizes
        # Might have messed up proportions if you go for dimensions that are different in proportion to the original
        # Also this only works on grayscale cause I didn't consider other pixels

        int_ = int  # binding global int function to a local one, in order to compute faster. Not necessary, with almost
        # no speed benefit, maybe even slower, but I saw it on a stack overflow answer and wanted to try it out (￣ー￣)ｂ

        oWidth, oLength = len(image[0]), len(image)
        if oWidth < width or oLength < length:
            return image  # This function downsizes only

        widInt, lenInt = oWidth / width, oLength / length  # These are the intervals at which we segregate pixel groups
        # for averaging

        newImage = [[] for j in range(length)]  # allocation and then assignment is faster than append,
        # with all of its checks and security and stuff (I think)
        fromLength, toLength = 0, lenInt
        # We will leapfrog between these values to separate the pixel groups, [COME BACK AND IMPROVE]

        while fromLength < oLength:
            fromWidth, toWidth = 0, widInt
            # This bit just goes through the image and averages bits of it out and assigns them to the new image
            for row in range(int_(fromLength), int_(toLength)):
                while fromWidth < oWidth:
                    newImage[int_(fromLength / lenInt)].append(
                        image_conv.average((image[row][int_(fromWidth):int_(toWidth)])))
                    fromWidth = toWidth
                    toWidth += widInt
                    if toWidth > oWidth:
                        toWidth = oWidth

            fromLength = toLength
            toLength += lenInt

            if toLength > oLength:
                toLength = oLength  # if it's not a good ratio, it might squash a bit on the end.

        return newImage

    @staticmethod
    def discrete_cosine_transformation_1d(values):  # These are not particularly fast implementations of DCT by the way
        # https://users.cs.cf.ac.uk/dave/Multimedia/node231.html and
        # https://www.educative.io/answers/what-is-the-discrete-cosine-transform-dct are where I learnt how DCTs work
        # along with a couple of youtube videos which I won't cite because there are quite a few

        dct_list = [None for val in values]
        dct_list[0] = math.sqrt(2 / len(values)) * (1 / math.sqrt(2)) * (2 / len(values)) * sum(values)
        # cos(0) = 1, thus just a sum I think
        for k in range(1, len(values)):
            dct_list[k] = math.sqrt(2 / len(values)) * sum(
                [values[n] * math.cos((2 * n + 1) * 3.14 * k) for n in range(len(values))])
        return dct_list

    @staticmethod
    def dct_2_calc(row, col, N, M, matrix):
        # The row and column of the operation  being performed, N is the length of the matrix and M is the width.
        return sum([sum([(math.cos((3.14 * row * (2 * i + 1)) / (2 * N)) * math.cos(
            (3.14 * col * (2 * j + 1)) / (2 * M)) * matrix[i][j]) for j in range(M)]) for i in range(N)])
        #  a pretty long formula, so I put it in another function. This is just the 2d DCT general equation though

    @staticmethod
    def direct_discrete_cosine_transformation_2d(matrix):
        # dct_matrix = [image_conv.discrete_cosine_transformation_1d(matrix[row]) for row in range(len(matrix))]
        # dct_matrix[0] = list(map((1/math.sqrt(2)), dct_matrix[0]))  # This is incorrect by the way
        lambda_sqrt = math.sqrt(1 / 2)  # Will use this a lot for lambda values so might as well only calculate it once
        Len_sqrt = math.sqrt(2 / len(matrix))  # same reasoning
        Wid_sqrt = math.sqrt(2 / len(matrix[0]))  # ^
        dims_x, dims_y = len(matrix[0]), len(matrix)

        dct_matrix = [[None for value in range(dims_x)] for row in range(dims_y)]
        for row in range(dims_y):
            if row == 0:
                Lam_r = lambda_sqrt
            else:
                Lam_r = 1
            for column in range(dims_x):
                if column == 0:
                    Lam_c = lambda_sqrt
                else:
                    Lam_c = 1
                dct_matrix[row][column] = Len_sqrt * Wid_sqrt * Lam_c * Lam_r * image_conv.dct_2_calc(row, column,
                                                                                                      dims_y, dims_x,
                                                                                                      matrix)
                # print(dct_matrix[row][column], row, column)
        return dct_matrix

    @staticmethod
    def direct_inverse_discrete_cosine_transformation(matrix):
        pass

    @staticmethod
    def generate_dct_transformation_matrix(dim):
        return [[math.cos(i * (j + 0.5) * (3.14 / dim)) for i in range(0, dim)] for j in range(0, dim)]

    @staticmethod
    def square_discrete_cosine_transformation_2d(matrix):  # This one works on square matrices specifically
        transformation_matrix = image_conv.generate_dct_transformation_matrix(len(matrix))
        return matrix_operations.traditional_matrix_multiplication(matrix, transformation_matrix)

    @staticmethod
    def inv_square_dct_2d(matrix):
        transformation_matrix = image_conv.generate_dct_transformation_matrix(len(matrix))
        inverse_transformation_matrix = matrix_operations.inverse_matrix(transformation_matrix)
        return matrix_operations.traditional_matrix_multiplication(matrix, inverse_transformation_matrix)

    @staticmethod
    def difference_hash(matrix, dim_x, dim_y):  # very fast and ignores lighting level since it's relative
        hash_bits = []
        for i in range(dim_y):
            for j in range(dim_x - 1):  # Because we ignore the last value
                if matrix[i][j] > matrix[i][j + 1]:
                    hash_bits.append("1")
                else:
                    hash_bits.append("0")
        return "".join(hash_bits)

    @staticmethod
    def average_hash(matrix, dim_x, dim_y):
        hash_bits = [0] * dim_x * dim_y
        avg = sum([sum(n) for n in matrix]) / (dim_x * dim_y)
        for i in range(dim_y):
            for j in range(dim_x):
                if matrix[i][j] > avg:
                    hash_bits[dim_y][dim_x] = "1"
                else:
                    hash_bits[dim_y][dim_x] = "0"
        return "".join(hash_bits)

    @staticmethod
    def perceptual_hash(matrix, dim_x, dim_y):
        # https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html -- Idea behind this
        dct_arr = image_conv.square_discrete_cosine_transformation_2d(matrix)
        imp_freq = [row[:dim_x // 4] for row in dct_arr[:dim_y // 4]]  # These represent the lowest freq in the image
        # high freq gives detail, low freq gives structure, we want to extract the image's structure
        print(imp_freq)
        avg_min_first = (sum([sum(row) for row in imp_freq]) - imp_freq[0][0]) / (((dim_x // 4) * (dim_y // 4)) - 1)
        print(avg_min_first)
        # first val is pure colours and throws off avg because it's overly large, explained in link
        hash_str = []
        for row in range(len(imp_freq)):
            for col in range(len(imp_freq[0])):
                if imp_freq[row][col] > avg_min_first:
                    hash_str.append("1")
                else:
                    hash_str.append("0")
        return "".join(hash_str)

    @staticmethod
    # https://pyimagesearch.com/2017/11/27/image-hashing-opencv-python/ --> The inspiration for my idea
    def img_Diff_Hash(image):  # This assumes the input is already as grayscale and returns a 64 bit difference hash str
        return image_conv.difference_hash(image_conv.resize(image, 9, 8), 9, 8)

    @staticmethod
    def img_Avg_Hash(image):
        return image_conv.average_hash(image_conv.resize(image, 8, 8), 8, 8)

    @staticmethod
    def img_Per_Hash(image):
        return image_conv.perceptual_hash(image_conv.resize(image, 32, 32), 32, 32)


class texture_utils:  # just a bunch of functions related to texture analysis and feature extraction, they're all static

    @staticmethod
    def XOR(bin_1, bin_2):
        if (bin_1 and not bin_2) or (bin_2 and not bin_1):
            return 1
        else:
            return 0

    @staticmethod
    def LBP_compare(threshold, value):
        return 0 if value < threshold else 1

    @staticmethod
    def spiral_join(bin_vals, dims):
        # traverses the 2d list in a spiral pattern and makes all the values one string. Also, very lazy programming.
        # Come back to this...
        x_bound = y_bound = dims - 1
        cur_x = cur_y = 0
        summed = []
        for column in range(x_bound):  # because on the first line, it has to traverse the same distance on the way back
            summed.append(str(bin_vals[cur_y][cur_x]))
            cur_x += 1

        while x_bound >= 2:
            for row in range(y_bound):
                summed.append(str(bin_vals[cur_y][cur_x]))
                cur_y += 1
            y_bound -= 1

            for column in range(x_bound):
                summed.append(str(bin_vals[cur_y][cur_x]))
                cur_x -= 1
            x_bound -= 1

            for row in range(y_bound):
                summed.append(str(bin_vals[cur_y][cur_x]))
                cur_y -= 1
            y_bound -= 1

            for column in range(x_bound):
                summed.append(str(bin_vals[cur_y][cur_x]))
                cur_x += 1
            x_bound -= 1

        # summed.append(str(bin_vals[cur_y][cur_x]))
        # for completion's sake this would also include the central pixel (we don't want this for this version though)

        return "".join(summed)

    @staticmethod
    def spiral_concatenation(vals,
                             dims):  # This takes in an array and applies the spiral concatenation operation for LBPs
        summed = []
        dirs = [+1, -1, -1, +1]  # y, x, y, x
        cur_dir = 0
        cur_x = cur_y = 0
        bounds = dims - 1  # This represents the y and x bounds respectively, which are how the spiral pattern is
        # formed, by moving bounds amount in one direction, then decrementing bounds

        for column in range(bounds):
            summed.append(str(vals[(dims * cur_y) + cur_x]))
            cur_x += 1

        while bounds >= 1:
            for row in range(bounds):
                summed.append(str(vals[(dims * cur_y) + cur_x]))
                cur_y += dirs[cur_dir]
            cur_dir = (cur_dir + 1) % 4
            for column in range(bounds):
                summed.append(str(vals[(dims * cur_y) + cur_x]))
                cur_x += dirs[cur_dir]
            cur_dir = (cur_dir + 1) % 4
            bounds -= 1

        return "".join(summed)

    @staticmethod
    def list_square(inp_list):  # This takes a one dimensional list and converts into a square matrix, list length
        # must be a square number or else you will lose the values at the end of the list
        dimensions = int(math.sqrt(len(inp_list)))
        square_matrix = [[0 for column in range(dimensions)] for row in range(dimensions)]

        for row_count in range(dimensions):
            for col_count in range(dimensions):
                square_matrix[row_count][col_count] = inp_list[dimensions * row_count + col_count]

        return square_matrix

    @staticmethod
    def localBinaryPattern(image, dims):
        # This takes in a grayscale image that has been reduced to a 2 by 2 python matrix, and the dimensions of the
        # neighbours that we take. Dimensions should be odd

        # https://aihalapathirana.medium.com/understanding-the-local-binary-pattern-lbp-a-powerful-method-for-texture-analysis-in-computer-4fb55b3ed8b8
        img_width, img_length, edge = len(image[0]), len(image), dims // 2
        lbp_list = []
        # compare = lambda thresh, val: 0 if val < thresh else (1 if thresh is not val else 2)

        for row in range(edge, img_length - edge):
            for pixel in range(edge, img_width - edge):
                # neighbourhood = flatten(image[row - edge: row + edge + 1][pixel - edge: pixel + edge + 1])
                neighbourhood = []
                for neighbour_row in image[row - edge: row + edge + 1]:  # This way it's already flattened
                    for neighbour_pixel in neighbour_row:
                        neighbourhood.append(neighbour_pixel)

                binary_vals = list(map(partial(texture_utils.LBP_compare, image[row][pixel]), neighbourhood))
                # binary_vals = [map((lambda thresh=central_pixel, val: 0 if (val < thresh) else 1, neighbourhood)]
                lbp_list.append(texture_utils.spiral_concatenation(binary_vals, dims))  # formats it to be a string
                # representing a binary num

                # list square isn't necessary if we update spiral_concatenation to work with 1d list with square
                # number of elements

        # This returns an LBP signature that identifies this and similar textures
        return lbp_list  # error case if dimensions were too large is that this is empty. Check it afterwards


image_file_path = "/Users/kostasdemiris/VS-code-programs/Project-Cinder/images/Photo on 31-10-2022 at 09.46.jpg"

test_matrix = [[3, 2, 1], [7, 6, 5], [11, 10, 12]]
test_matrix = [[1, 2], [3, 4]]
print(matrix_operations.strassen(test_matrix, matrix_operations.identity_element(2)))



# This is a display case of perceptual hashing
u_int = user_interaction("NA")  # NA as in not applicable as in no settings because that hasn't been implemented yet...

image_1 = u_int.take_grayscale_photo(image_file_path, True)
image_2 = u_int.take_grayscale_photo(image_file_path, True)

hash_1, hash_2 = image_conv.img_Per_Hash(image_1), image_conv.img_Per_Hash(image_2)
print(hash_1, hash_2)
print(f"{image_conv.hamming_distance(hash_1, hash_2)} is the hamming distance between the first and second images.")
print("PS, for this example and particular implementation, there's a VERY strong blurring effect, so any hamming "
      "distance greater than 2 or 3 means that the images are significantly different")



