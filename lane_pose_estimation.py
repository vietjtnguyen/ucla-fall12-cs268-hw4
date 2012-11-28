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

from helper import colvec2tuple, tuple2colvec, augcolvec, unaugcolvec, tuple2inttuple
import intrinsic_calibration
import lane_detection
from line import Line

def hw4_lane_pose_estimation():
	'''
	TODO DOCUMENTATION
	'''

	image_paths = ['LDWS_test/LDWS_test_data {0:03}.bmp'.format(x) for x in range(1, 609)]

	intrinsic_matrix, distortion_coefficients = intrinsic_calibration.hw4_calibration(False)
	calibration_matrix = np.linalg.inv(intrinsic_matrix)

	left_search_strips, right_search_strips = lane_detection.define_hw4_search_strips()

	cv2.namedWindow('display')

	for image_path in image_paths:
		cv_image = cv2.imread(image_path)

		lanes_found, left_lane_line, right_lane_line, vanishing_point = \
			lane_detection.detect_lanes(
				cv_image,
				left_search_strips,
				right_search_strips,
			)

		display_image = cv_image

		if lanes_found:

			# opencv camera coordinate system is
			# +z forward
			# +y up
			# +x left

			aug_vanishing_point = augcolvec(vanishing_point)
			calibrated_vanishing_point = calibration_matrix * aug_vanishing_point
			rotation_forward_basis = calibrated_vanishing_point / np.linalg.norm(calibrated_vanishing_point)
			elementary_up_basis = tuple2colvec((0, 1, 0))
			rotation_left_basis = np.matrix(np.cross(elementary_up_basis.T, rotation_forward_basis.T).T)
			rotation_up_basis = np.matrix(np.cross(rotation_forward_basis.T, rotation_left_basis.T).T)
			r11, r21, r31 = colvec2tuple(rotation_left_basis)
			r12, r22, r32 = colvec2tuple(rotation_up_basis)
			r13, r23, r33 = colvec2tuple(rotation_forward_basis)
			rotation = np.matrix([
				[r11, r12, r13],
				[r21, r22, r23],
				[r31, r32, r33],
			])
			
			pick_image_line = Line(np.matrix([[0],[250]]), np.matrix([[1],[0]]))
			pick_left = Line.intersection(left_lane_line, pick_image_line)
			pick_right = Line.intersection(right_lane_line, pick_image_line)

			pick = pick_left
			aug_pick = augcolvec(pick)
			calibrated_pick = calibration_matrix * aug_pick

			calibrated_up_vanishing_point = rotation_up_basis / rotation_up_basis[2][0]
			aug_up_vanishing_point = intrinsic_matrix * calibrated_up_vanishing_point
			up_vanishing_point = unaugcolvec(aug_up_vanishing_point)
			up_line = Line.from_points(colvec2tuple(pick), colvec2tuple(up_vanishing_point))
			
			calibrated_left_vanishing_point = rotation_left_basis / rotation_left_basis[2][0]
			aug_left_vanishing_point = intrinsic_matrix * calibrated_left_vanishing_point
			left_vanishing_point = unaugcolvec(aug_left_vanishing_point)
			left_line = Line.from_points(colvec2tuple(pick), colvec2tuple(left_vanishing_point))

			corresponding_pick = Line.intersection(left_line, right_lane_line)
			aug_corresponding_pick = augcolvec(corresponding_pick)
			calibrated_corresponding_pick = calibration_matrix * aug_corresponding_pick

			x_1 = calibrated_pick
			x_2 = calibrated_corresponding_pick
			q_unit = rotation_left_basis
			
			preimage_space_basis_x = x_1 / np.linalg.norm(x_1)
			preimage_space_basis_z = np.matrix(np.cross(rotation_left_basis.T, preimage_space_basis_x.T).T)
			preimage_space_basis_y = np.matrix(np.cross(preimage_space_basis_x.T, preimage_space_basis_z.T).T)

			a = np.cross(calibrated_pick.T, rotation_left_basis.T).T
			b = np.cross(calibrated_pick.T, calibrated_corresponding_pick.T).T
			a = a/np.linalg.norm(a)
			b = b/np.linalg.norm(b)

			import IPython
			IPython.embed()
			sys.exit()

			bottom_image_line = Line(np.matrix([[0],[480]]), np.matrix([[1],[0]]))
			bottom_left = Line.intersection(left_lane_line, bottom_image_line)
			bottom_right = Line.intersection(right_lane_line, bottom_image_line)

			# set up pixel coordinates for drawing
			vanishing_point_pixels = tuple2inttuple(colvec2tuple(vanishing_point))
			up_vanishing_point_pixels = tuple2inttuple(colvec2tuple(up_vanishing_point))
			left_vanishing_point_pixels = tuple2inttuple(colvec2tuple(left_vanishing_point))
			pick_left_pixels = tuple2inttuple(colvec2tuple(pick_left))
			pick_right_pixels = tuple2inttuple(colvec2tuple(pick_right))
			bottom_left_pixels = tuple2inttuple(colvec2tuple(bottom_left))
			bottom_right_pixels = tuple2inttuple(colvec2tuple(bottom_right))
			corresponding_pick_pixels = tuple2inttuple(colvec2tuple(corresponding_pick))

			# draw features
			cv2.circle(display_image, vanishing_point_pixels, 10, (255, 255, 255))
			cv2.line(display_image, up_vanishing_point_pixels, pick_left_pixels, (255, 255, 0), 1, cv2.CV_AA)
			cv2.line(display_image, left_vanishing_point_pixels, pick_left_pixels, (0, 255, 255), 1, cv2.CV_AA)
			cv2.circle(display_image, pick_left_pixels, 10, (0, 0, 255))
			cv2.circle(display_image, corresponding_pick_pixels, 10, (255, 0, 0))
			cv2.line(display_image, vanishing_point_pixels, bottom_left_pixels, (255, 0, 255), 1, cv2.CV_AA)
			cv2.line(display_image, vanishing_point_pixels, bottom_right_pixels, (255, 0, 255), 1, cv2.CV_AA)

			#object_points = np.array([
			#	[-1.6, 0, 0],
			#	[1.6, 0, 0],
			#	[-1.6, 4.0, 0],
			#	[1.6, 4.0, 0],
			#])

			#image_points = np.array([
			#	pick_left.T.A[0],
			#	pick_right.T.A[0],
			#	almost_pick_left.T.A[0],
			#	almost_pick_right.T.A[0],
			#])
			#
			#solve_pnp_results = cv2.solvePnP(
			#	object_points,
			#	image_points,
			#	intrinsic_matrix,
			#	distortion_coefficients,
			#)

			#pnp_success, rotation_omega, translate = solve_pnp_results
			#horizontal_drift = -translate[0][0]
			#print(horizontal_drift)
			#
			#cv2.line(display_image, (320-80, 10), (320+80, 10), (0, 0, 0), 1, cv2.CV_AA)
			#cv2.line(display_image, (320-80, 10), (320-80, 50), (0, 0, 0), 1, cv2.CV_AA)
			#cv2.line(display_image, (320+80, 10), (320+80, 50), (0, 0, 0), 1, cv2.CV_AA)
			#cv2.line(display_image, (320-80, 50), (320+80, 50), (0, 0, 0), 1, cv2.CV_AA)
			#cv2.line(display_image, (320+int(40*horizontal_drift/0.4), 30), (320+int(40*horizontal_drift/0.4), 50), (0, 0, 0), 1, cv2.CV_AA)

		cv2.imshow('display', display_image)
		cv2.waitKey(1)

if __name__ == '__main__':
	hw4_lane_pose_estimation()
	
