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

internal_corner_shape, corner_world_points = \
	define_chessboard(*[
		(9, 9), # cell shape
		48.0/1000.0, # cell size = 48 mm in meters
	])

# Viet Nguyen: Note that images 930 and 931 contain occlusions.
calibration_image_names = ['LDWS_calibrate/IMG_0068 {0:04}.bmp'.format(x) for x in range(1, 1157)]

for image_name in calibration_image_names:
	#image_name = calibration_image
	cv_image = cv2.imread(image_name)
	chessboard_result = cv2.findChessboardCorners(
		cv_image,
		internal_corner_shape,
		flags =
			cv2.CALIB_CB_ADAPTIVE_THRESH +
			cv2.CALIB_CB_NORMALIZE_IMAGE)# +
			#cv2.CALIB_CB_FAST_CHECK)
	chessboard_was_found, found_corners = chessboard_result

	cv2.namedWindow('calib frame')
	cv2.drawChessboardCorners(cv_image, internal_corner_shape, found_corners, chessboard_was_found)
	cv2.imshow('calib frame', cv_image)
	cv2.waitKey(33) # assuming 30 fps

#cv2.calibrateCamera
#cv2.cvtColor
#cv2.findChessboardCorners

try:
	import IPython
	IPython.embed()
except Exception as e:
	sys.exit()
