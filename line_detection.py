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

if __name__ == '__main__':

	image_paths = ['LDWS_test/LDWS_test_data {0:03}.bmp'.format(x) for x in range(1, 609)]

	intrinsic_matrix, distortion_coefficients = intrinsic_calibration.hw4_calibration(False)

	# regions of interest
	# row  left    right   region
	#      center  center  width
	# 218  194     276     298/2
	# 318  55      375     73/2

	cv2.namedWindow('edges')

	for image_path in image_paths:
		cv_image = cv2.imread(image_path)

		low_threshold = 100
		ratio = 3

		canny_image = cv2.Canny(cv_image, low_threshold, low_threshold*ratio)
		canny_image = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)

		for y in range(218, 318+50+10, 10):
			width = int((120.0-36.0)*(y-218.0)/100.0)+36
			left_center = int((55.0-194.0)*(y-218.0)/100.0)+194
			right_center = int((375.0-276.0)*(y-218.0)/100.0)+276
			cv2.line(canny_image, (left_center-width, y), (left_center+width, y), (0, 0, 255), 1, cv2.CV_AA)
			cv2.line(canny_image, (right_center-width, y+1), (right_center+width, y+1), (0, 255, 0), 1, cv2.CV_AA)
		
		cv2.imshow('edges', canny_image)
		cv2.waitKey(33)

	# end in an interactive shell so we can look at the values
	try:
		import IPython
		IPython.embed()
	except Exception as e:
		sys.exit()

