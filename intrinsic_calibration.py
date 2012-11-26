#!/usr/bin/python

import math
import sys

import cv2
import numpy as np

'''
The outline as specified by http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html is as follows:

 1. Read the settings.
 2. Get next input, if it fails or we have enough of them then perform calibration.
 3. Find the pattern in the current input.
 4. Show state and result for the user, plus command line control of the application.
 5. Show the distortion removal for the images too.

Those steps are perhaps better read in pseudocode as follows:

read_settings()
while( input_is_available() or input_sufficient_for_calibration() ):
	input = get_input()
	pattern = find_pattern(input)
	display(pattern)
	calibration_results = calculate_calibration() #?
display(calibration_results)
'''

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

def calibrate_camera_from_images(image_paths, image_size, cell_shape, cell_size):

	# define chessboard properties
	internal_corner_shape, corner_world_points = define_chessboard(cell_shape, cell_size)

	# initialize empty lists that will serve as the first two arguments for
	# cv2.calibrateCamera
	corner_world_point_sets = []
	found_corner_sets = []

	# walk through the subset of images gathering found chess board corners
	for image_name in image_paths:

		# load the image
		cv_image = cv2.imread(image_name)

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

if __name__ == '__main__':

	# Viet Nguyen: Note that images 930 and 931 contain occlusions.
	calibration_image_paths = ['LDWS_calibrate/IMG_0068 {0:04}.bmp'.format(x) for x in range(1, 1157)]
	
	# create a subset of the images that we'll actually use. this subset is basically
	# num_of_images_in_subset images divided evenly in the calibration image sequence
	num_of_images_in_subset = 36
	calibration_image_paths_subset = [calibration_image_paths[x] for x in range(0, len(calibration_image_paths), len(calibration_image_paths)/num_of_images_in_subset)]

	image_size = (480, 640) # pixels
	cell_shape = (9, 9) # cell shape
	cell_size = 48.0/1000.0 # 48 mm in meters

	calibrate_camera_return = calibrate_camera_from_images(calibration_image_paths_subset, image_size, cell_shape, cell_size)
	reprojection_error, intrinsic_matrix, distortion_coefficients, r_vecs, t_vecs = calibrate_camera_return
	print(calibrate_camera_return)

	# end in an interactive shell so we can look at the values
	try:
		import IPython
		IPython.embed()
	except Exception as e:
		sys.exit()

