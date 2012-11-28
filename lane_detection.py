#!/usr/bin/python

'''
This script is meant to be imported for its functionality. However, a standalone
behavior does exist. Standalone behavior is to read a line from stdin and treat it as the
file name for an image to perform lane detection on. Said file is then read in as an
image. The lanes are detected, drawn on the image, and then displayed. The file name is read in from stdin because the test footage has a space in the file name and Pytho will
automatically split the file name apart in sys.argv.

EXAMPLE

echo -n "LDWS_test/LDWS_test_data 001.bmp" | ./cv ipython --pdb ./lane_detection.py

LICENSE

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
from line import Line, ransac_line2d

class LaneSearchStrip():
	'''
	This class represents a single-pixel tall horizontal line that acts as the window
	of search for lane edges, such as from a Canny edge detector.
	'''

	def __init__(self, left_point, width):
		'''
		Creates a lane search strip from the left-most point as a 2-tuple and the width.
		'''
		self.left_point = left_point
		self.width = width
		self.right_point = (self.left_point[0]+width, self.left_point[1])
	
	def __str__(self):
		'''
		Returns the string representation.
		'''
		return '{0!r} to {1!r}'.format(self.left_point, self.right_point)
	
	def __repr__(self):
		'''
		Returns a Python representation.
		'''
		return 'LaneSearchStrip({0!r},{1!r})'.format(self.left_point, self.width)

def create_search_strip_set(vertical_interval, vertical_step, width_interval, center_interval, clip_region):
	'''
	Returns a list of lane search strips (as LaneSearchStrip objects). There are five
	arguments. The first, vertical_interval, is a 2-tuple containing the vertical pixel
	range of the search strip set. Note the order as the other arguments are linearly
	interpolated from this range. The second argument, vertical_step, is the vertical
	pixel separation between each search strip. The width_interval and center_interval,
	arguments three and four respectively, define what the width of the strip will be at
	the start of the vertical interval and what the center of the strip will be (as a
	2-tuple) at the start of the vertical interval. The fifth argument, clip_region,
	is a 4-tuple defining the bounds of the image as (left, top, right, bottom) which is
	used to clip the search strips into the image.
	'''
	# define some convenient local variables
	vertical_range = vertical_interval[1]-vertical_interval[0]
	width_range = width_interval[1]-width_interval[0]
	center_range = center_interval[1]-center_interval[0]

	# initialize the set of search strips
	search_strips = []

	# walk through each vertical step
	for y in range(vertical_interval[0], vertical_interval[1]+vertical_step, vertical_step):
		# determine the width and center of the strip using linear interpolation
		width = int(width_range*(y-vertical_interval[0])/vertical_range)+width_interval[0]
		center = int(center_range*(y-vertical_interval[0])/vertical_range)+center_interval[0]

		# define the strip as a segment
		segment = ((center-width, y), (center+width, y))

		# clip the segment to the image
		segment_is_in_image, left_point, right_point = cv2.clipLine(clip_region, segment[0], segment[1])

		# if the segment isn't in the image, don't add it
		if not segment_is_in_image:
			continue

		# add the clipped segment to the image
		search_strips.append(LaneSearchStrip(left_point, right_point[0]-left_point[0]))
	
	return search_strips

def find_lane_points(canny_image, search_strips, lane):
	'''
	TODO DOCUMENTATION
	'''

	assert(lane == 'left' or lane == 'right')

	if lane == 'left':
		direction = -1
		start_point_name = 'right_point'
		end_point_name = 'left_point'
	elif lane == 'right':
		direction = 1
		start_point_name = 'left_point'
		end_point_name = 'right_point'
	
	lane_points = []

	for search_strip in search_strips:
		start_point = getattr(search_strip, start_point_name)
		end_point = getattr(search_strip, end_point_name)

		x, y = start_point

		while( x * direction <= end_point[0] * direction ):
			intensity_at_point = canny_image[y][x]
			if intensity_at_point == 255:
				lane_points.append((x, y))
				break
			x += direction

	return lane_points

def detect_lanes(cv_image, left_search_strips, right_search_strips):
	'''
	TODO DOCUMENTATION
	'''
	# turn canny detector knobs
	low_threshold = 100
	ratio = 3

	# apply the canny edge detector
	canny_image = cv2.Canny(cv_image, low_threshold, low_threshold*ratio)

	# find the intersections between the search strips and the edges from the canny
	# detector
	left_lane_points = find_lane_points(canny_image, left_search_strips, 'left')
	right_lane_points = find_lane_points(canny_image, right_search_strips, 'right')

	# fit line models to the intersections found
	left_lane_line = ransac_line2d(left_lane_points)
	right_lane_line = ransac_line2d(right_lane_points)

	# if either of the lane lines cannot be found, return failure
	if left_lane_line == None or right_lane_line == None:
		return False, left_lane_line, right_lane_line, None

	# find the vanishing point
	vanishing_point = Line.intersection(left_lane_line, right_lane_line)

	return True, left_lane_line, right_lane_line, vanishing_point

def define_hw4_search_strips():
	'''
	Returns a 2-tuple containing the list of left lane search strips and the lits of
	right lane search strips specific to homework 4.
	'''
	vertical_interval = (218, 368)
	vertical_step = 10
	width_interval = (36, 120)
	left_center_interval = (205, -15)
	right_center_interval = (280, 440)
	clip_region = (0, 0, 640, 480)

	left_search_strips = create_search_strip_set(
		vertical_interval, vertical_step,
		width_interval, left_center_interval,
		clip_region,
	)
	right_search_strips = create_search_strip_set(
		vertical_interval, vertical_step,
		width_interval, right_center_interval,
		clip_region,
	)

	return left_search_strips, right_search_strips

if __name__ == '__main__':
	image_path = sys.stdin.readline()

	left_search_strips, right_search_strips = define_hw4_search_strips()

	cv_image = cv2.imread(image_path)

	if cv_image == None:
		print('ERROR: Could not load file "{0}" as an image!'.format(image_path))
		sys.exit()

	lanes_found, left_lane_line, right_lane_line, vanishing_point = \
		detect_lanes(
			cv_image,
			left_search_strips,
			right_search_strips,
		)
	
	if lanes_found == False:
		print('ERROR: Could not find lanes in the image!')
		sys.exit()

	vanishing_point_pixels = tuple2inttuple(colvec2tuple(vanishing_point))
	cv2.circle(cv_image, vanishing_point_pixels, 10, (255, 255, 255))

	bottom_image_line = Line(np.matrix([[0],[480]]), np.matrix([[1],[0]]))
	bottom_left = Line.intersection(left_lane_line, bottom_image_line)
	bottom_right = Line.intersection(right_lane_line, bottom_image_line)
	bottom_left_pixels = tuple2inttuple(colvec2tuple(bottom_left))
	bottom_right_pixels = tuple2inttuple(colvec2tuple(bottom_right))
	cv2.circle(cv_image, bottom_left_pixels, 10, (0, 0, 255))
	cv2.circle(cv_image, bottom_right_pixels, 10, (255, 0, 0))

	cv2.line(cv_image, vanishing_point_pixels, bottom_left_pixels, (255, 0, 255), 1, cv2.CV_AA)
	cv2.line(cv_image, vanishing_point_pixels, bottom_right_pixels, (255, 0, 255), 1, cv2.CV_AA)

	cv2.namedWindow('display')
	cv2.imshow('display', cv_image)
	cv2.waitKey(0)

