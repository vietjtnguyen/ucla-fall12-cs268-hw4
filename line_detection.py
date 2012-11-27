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

import intrinsic_calibration

class LineOfInterest():
	'''
	TODO DOCUMENTATION
	'''

	def __init__(self, left_point, width):
		'''
		TODO DOCUMENTATION
		'''
		self.left_point = left_point
		self.width = width
		self.right_point = (self.left_point[0]+width, self.left_point[1])
	
	def __str__(self):
		return str(self.left_point)+', '+self.width

def define_lines_of_interest():
	img_region = (0, 0, 640, 480)
	vertical_interval = (218, 368)
	vertical_range = vertical_interval[1]-vertical_interval[0]
	vertical_step = 10
	width_interval = (36, 120)
	width_range = width_interval[1]-width_interval[0]
	left_center_interval = (205, -15)
	left_center_range = left_center_interval[1]-left_center_interval[0]
	right_center_interval = (280, 440)
	right_center_range = right_center_interval[1]-right_center_interval[0]

	left_lines_of_interest = []
	right_lines_of_interest = []

	for y in range(vertical_interval[0], vertical_interval[1]+vertical_step, vertical_step):
		width = int(width_range*(y-vertical_interval[0])/vertical_range)+width_interval[0]
		left_center = int(left_center_range*(y-vertical_interval[0])/vertical_range)+left_center_interval[0]
		right_center = int(right_center_range*(y-vertical_interval[0])/vertical_range)+right_center_interval[0]
		

		left_line = ((left_center-width, y), (left_center+width, y))
		left_line = cv2.clipLine(img_region, left_line[0], left_line[1])[1:] # assumes lines always overlap image region
		left_lines_of_interest.append(LineOfInterest(left_line[0], left_line[1][0]-left_line[0][0]))

		right_line = ((right_center-width, y), (right_center+width, y))
		right_line = cv2.clipLine(img_region, right_line[0], right_line[1])[1:] # assumes lines always overlap image region
		right_lines_of_interest.append(LineOfInterest(right_line[0], right_line[1][0]-right_line[0][0]))
	
	return left_lines_of_interest, right_lines_of_interest

if __name__ == '__main__':

	image_paths = ['LDWS_test/LDWS_test_data {0:03}.bmp'.format(x) for x in range(1, 609)]

	intrinsic_matrix, distortion_coefficients = intrinsic_calibration.hw4_calibration(False)

	# regions of interest
	# row  left    right   region
	#      center  center  width
	# 218  194     276     298/2
	# 318  55      375     73/2

	left_interest, right_interest = define_lines_of_interest()

	cv2.namedWindow('edges')

	for image_path in image_paths:
		cv_image = cv2.imread(image_path)

		low_threshold = 100
		ratio = 3

		canny_image = cv2.Canny(cv_image, low_threshold, low_threshold*ratio)
		canny_image = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)

		for interest in left_interest:
			cv2.line(canny_image, interest.left_point, interest.right_point, (0, 0, 255), 1, cv2.CV_AA)

		for interest in right_interest:
			cv2.line(canny_image, interest.left_point, interest.right_point, (255, 0, 0), 1, cv2.CV_AA)
		
		cv2.imshow('edges', canny_image)
		cv2.waitKey(33)

	# end in an interactive shell so we can look at the values
	try:
		import IPython
		IPython.embed()
	except Exception as e:
		sys.exit()

