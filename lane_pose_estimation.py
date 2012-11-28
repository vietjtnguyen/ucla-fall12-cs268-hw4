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

from helper import colvec2tuple, tuple2colvec, tuple2inttuple
import intrinsic_calibration
import line_detection
from line import Line

def hw4_lane_pose_estimation():
	'''
	TODO DOCUMENTATION
	'''

	image_paths = ['LDWS_test/LDWS_test_data {0:03}.bmp'.format(x) for x in range(1, 609)]

	intrinsic_matrix, distortion_coefficients = intrinsic_calibration.hw4_calibration(False)

	left_search_strips, right_search_strips = line_detection.define_hw4_search_strips()

	cv2.namedWindow('display')

	for image_path in image_paths:
		cv_image = cv2.imread(image_path)

		lanes_found, left_lane_line, right_lane_line, vanishing_point = \
			line_detection.detect_lanes(
				cv_image,
				left_search_strips,
				right_search_strips,
			)

		display_image = cv_image

		if lanes_found:
			vanishing_point_pixels = tuple2inttuple(colvec2tuple(vanishing_point))
			cv2.circle(display_image, vanishing_point_pixels, 10, (255, 255, 255))

			bottom_image_line = Line(np.matrix([[0],[480]]), np.matrix([[1],[0]]))
			bottom_left = Line.intersection(left_lane_line, bottom_image_line)
			bottom_right = Line.intersection(right_lane_line, bottom_image_line)
			bottom_left_pixels = tuple2inttuple(colvec2tuple(bottom_left))
			bottom_right_pixels = tuple2inttuple(colvec2tuple(bottom_right))
			cv2.circle(display_image, bottom_left_pixels, 10, (0, 0, 255))
			cv2.circle(display_image, bottom_right_pixels, 10, (255, 0, 0))

			almost_bottom_image_line = Line(np.matrix([[0],[480-150]]), np.matrix([[1],[0]]))
			almost_bottom_left = Line.intersection(left_lane_line, almost_bottom_image_line)
			almost_bottom_right = Line.intersection(right_lane_line, almost_bottom_image_line)
			almost_bottom_left_pixels = tuple2inttuple(colvec2tuple(almost_bottom_left))
			almost_bottom_right_pixels = tuple2inttuple(colvec2tuple(almost_bottom_right))
			cv2.circle(display_image, almost_bottom_left_pixels, 10, (0, 0, 255))
			cv2.circle(display_image, almost_bottom_right_pixels, 10, (255, 0, 0))

			cv2.line(display_image, vanishing_point_pixels, bottom_left_pixels, (255, 0, 255), 1, cv2.CV_AA)
			cv2.line(display_image, vanishing_point_pixels, bottom_right_pixels, (255, 0, 255), 1, cv2.CV_AA)

			object_points = np.array([
				[-1.6, 0, 0],
				[1.6, 0, 0],
				[-1.6, 4.0, 0],
				[1.6, 4.0, 0],
			])

			image_points = np.array([
				bottom_left.T.A[0],
				bottom_right.T.A[0],
				almost_bottom_left.T.A[0],
				almost_bottom_right.T.A[0],
			])
			
			solve_pnp_results = cv2.solvePnP(
				object_points,
				image_points,
				intrinsic_matrix,
				distortion_coefficients,
			)

			pnp_success, rotation_omega, translate = solve_pnp_results
			horizontal_drift = -translate[0][0]
			print(horizontal_drift)
			
			cv2.line(display_image, (320-80, 10), (320+80, 10), (0, 0, 0), 1, cv2.CV_AA)
			cv2.line(display_image, (320-80, 10), (320-80, 50), (0, 0, 0), 1, cv2.CV_AA)
			cv2.line(display_image, (320+80, 10), (320+80, 50), (0, 0, 0), 1, cv2.CV_AA)
			cv2.line(display_image, (320-80, 50), (320+80, 50), (0, 0, 0), 1, cv2.CV_AA)
			cv2.line(display_image, (320+int(40*horizontal_drift/0.4), 30), (320+int(40*horizontal_drift/0.4), 50), (0, 0, 0), 1, cv2.CV_AA)

		cv2.imshow('display', display_image)
		cv2.waitKey(1)

if __name__ == '__main__':
	hw4_lane_pose_estimation()
	
