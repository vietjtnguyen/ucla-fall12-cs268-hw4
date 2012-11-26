#!/usr/bin/python

'''
Copyright (c) 2012 Viet Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
'''

import math
import sys

import cv2
import numpy as np

def define_chessboard(cell_shape, cell_size):
	'''
	Defines information needed for chessboard calibration in OpenCV. Takes in the
	cell shape which is a 2-tuple containing the number of rows and columns and
	the cell size in whatever metric representation is desired (e.g. meters).

	Returns a 2-tuple where the first item is the "internal corner shape." This is
	a 2-tuple that is basically the provided cell shape except reduced by one in
	each dimension. It represents the number of internal corners in each dimension.
	The second item in the returned 2-tuple is a NumPy array of shape (*, 1, 3)
	where * is internal_corner_shape[0]*internal_corner_shape[1]. It is a list of
	row vectors in R^3 that represent world metric coordinates of the chess board.
	This array should have the same length (of the first dimension) as the found
	corners list returned from cv2.findChessboardCorners.
	'''
	internal_corner_shape = tuple([x-1 for x in cell_shape])
	corner_world_points = []
	for i in range(0, internal_corner_shape[0]):
		for j in range(0, internal_corner_shape[1]):
			corner_world_points.append([[i*cell_size, j*cell_size, 0.0]])
	corner_world_points = np.array(corner_world_points, dtype = np.float32)
	return internal_corner_shape, corner_world_points

def calibrate_camera_from_images(image_paths, image_size, cell_shape, cell_size, show_images = True):
	'''
	Returns the result of cv2.calibrateCamera applied on the images listed in
	the image_paths argument. The image_size must be specified as a 2-tuple containing
	width and height (in that order) in pixels. It is assumed that all images are of
	the same size. The cell_shape refers to the number of rows and columns on the
	chessboard being used as a calibration target. This should be passed as a 2-tuple.
	The cell_size is the size of each cell on the chessboard in whatever metric space
	you desire (e.g. meters). The function also takes an optional keyword parameter named
	show_images which defaults to True. If set to false, the chessboard find results will
	not be displayed (no windows will be created).

	The return is a 5-tuple containing the following items: reprojection error (this is
	actually uncertain, but appears to be), the intrinsic camera matrix as a R^3x3 matrix,
	the distortion coefficients as a R^5 array, a list of rotation vectors corresponding to
	each image (rotation vectors being R^3 vectors that can be used to obtain a rotation
	matrix via Rodrigues' rotation formula), and a list of translate vectors corresponding
	to each image (as R^3 vectors).

	See also:
	 * http://opencv.willowgarage.com/documentation/python/calib3d_camera_calibration_and_3d_reconstruction.html#calibratecamera2
	'''

	# define chessboard properties
	internal_corner_shape, corner_world_points = define_chessboard(cell_shape, cell_size)

	# initialize empty lists that will serve as the first two arguments for
	# cv2.calibrateCamera
	corner_world_point_sets = []
	found_corner_sets = []

	# walk through the subset of images gathering found chess board corners
	for image_path in image_paths:

		# load the image
		cv_image = cv2.imread(image_path)

		# find the chessboard corners
		chessboard_result = cv2.findChessboardCorners(
			cv_image,
			internal_corner_shape,
			flags =
				cv2.CALIB_CB_ADAPTIVE_THRESH +
				cv2.CALIB_CB_NORMALIZE_IMAGE)# +
				#cv2.CALIB_CB_FAST_CHECK) # don't use this because it causes too many false-negatives
		chessboard_was_found, found_corners = chessboard_result

		# display the found corners
		if show_images:
			cv2.namedWindow('calib frame')
			cv2.drawChessboardCorners(
				cv_image,
				internal_corner_shape,
				found_corners,
				chessboard_was_found,
			)
			cv2.imshow('calib frame', cv_image)
			cv2.waitKey(1)

		# if the chessboard was found, record it in the set that will be used by
		# cv2.calibrateCamera
		if chessboard_was_found:

			# recompose the arrays from shape (N,1,*) to just (N,*)
			corner_world_points_recomposed = np.array(
				[x[0] for x in corner_world_points],
				dtype = np.float32,
			)
			found_corners_recomposed = np.array(
				[x[0] for x in found_corners],
				dtype = np.float32,
			)

			corner_world_point_sets.append(corner_world_points_recomposed)
			found_corner_sets.append(found_corners_recomposed)

	# attempt camera calibration with the chessboards found in the subset of images
	calibrate_camera_return = \
		cv2.calibrateCamera(

			# set of corner world points to calibrate next argument on
			# a list of lists of R^3 points
			#   first dimension is per calibration image
			#   second dimension is per corner
			#   third dimension is per component of the R^3 vector
			corner_world_point_sets,

			# a set of image points corresponding to world points in first argument
 			# a list of lists of R^2 points
			#   first dimension is per calibration image
			#   second dimension is per corner
			#   third dimension is per component of the R^2 vector
			found_corner_sets,

			# 2-tuple containing width and height of the calibration images
			image_size,
		)

	return calibrate_camera_return

def hw4_calibration(show_images = True):
	'''
	Performs the camera calibration specifically for homework 4 and returns a 2-tuple
	containing the intrinsic calibration matrix and the distortion coefficients.

	The function also takes an optional keyword parameter named show_images which defaults
	to True. If set to false, the chessboard find results will not be displayed (no
	windows will be created).
	'''

	# Viet Nguyen: Note that images 930 and 931 contain occlusions.
	calibration_image_paths = ['LDWS_calibrate/IMG_0068 {0:04}.bmp'.format(x) for x in range(1, 1157)]
	
	# create a subset of the images that we'll actually use. this subset is basically
	# num_of_images_in_subset images divided evenly in the calibration image sequence
	num_of_images_in_subset = 36
	calibration_image_paths_subset = [calibration_image_paths[x] for x in range(0, len(calibration_image_paths), len(calibration_image_paths)/num_of_images_in_subset)]

	image_size = (480, 640) # pixels
	cell_shape = (9, 9) # cell shape
	cell_size = 48.0/1000.0 # 48 mm in meters

	calibrate_camera_return = calibrate_camera_from_images(calibration_image_paths_subset, image_size, cell_shape, cell_size, show_images)
	reprojection_error, intrinsic_matrix, distortion_coefficients, r_vecs, t_vecs = calibrate_camera_return

	return intrinsic_matrix, distortion_coefficients

if __name__ == '__main__':

	intrinsic_matrix, distortion_coefficients = hw4_calibration()

	print('intrinsic_matrix:')
	print(intrinsic_matrix)
	print('')
	print('distortion_coefficients:')
	print(distortion_coefficients)

	# end in an interactive shell so we can look at the values
	try:
		import IPython
		IPython.embed()
	except Exception as e:
		sys.exit()

