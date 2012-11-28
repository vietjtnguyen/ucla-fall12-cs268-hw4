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

import numpy as np

from helper import *

class Line():
	'''
	Representation of an R^2 line using an origin point and a unit direction vector,
	both represented as NumPy matrix (linalg compatible) type two-item column vectors.
	'''

	def __init__(self, origin, unit_dir):
		'''
		Creates a line from the provided origin and unit direction.
		'''
		self.origin = origin
		self.unit_dir = unit_dir
	
	def __iter__(self):
		'''
		Returns an iterator representation as the 2-tuple containing the origin and
		the unit_dir for tuple unpacking convenience.
		'''
		yield self.origin
		yield self.unit_dir
	
	def __str__(self):
		'''
		Returns a string representation of the line.
		'''
		x, y = colvec2tuple(self.origin)
		a, b = colvec2tuple(self.unit_dir)
		return '({0},{1})+({2},{3})*t'.format(x, y, a, b)
	
	def __repr__(self):
		'''
		Returns a Python representatoin of the line.
		'''
		return 'Line({0!r},{1!r})'.format(self.origin, self.unit_dir).replace('\n', '').replace(' ', '')
	
	@staticmethod
	def from_points(point_a, point_b):
		'''
		Constructs a line from two 2-tuples representing two points on the line.
		'''
		# convert points into column vectors
		point_a, point_b = [tuple2colvec(point) for point in [point_a, point_b]]

		# arbitrarily choose the point_a as the origin
		origin = point_a
		
		# define a direction vector
		unit_dir = (point_b - point_a)

		# normalize the direction vector
		unit_dir = unit_dir / np.linalg.norm(unit_dir)

		return Line(origin, unit_dir)
	
	@staticmethod
	def intersection(line_a, line_b):
		'''
		Finds the intersection between to Lines analytically. Returns the intersection
		point as a Numpy matrix (linealg compatible) type two-item column vector. If
		there is no intersection (the lines are parallel) then returns None.
		'''
		# unpack origins and unit_dirs from the lines
		origin_a, unit_dir_a = line_a
		origin_b, unit_dir_b = line_b

		# unpack variables for convenience
		x1, y1 = colvec2tuple(origin_a)
		a1, b1 = colvec2tuple(unit_dir_a)
		x2, y2 = colvec2tuple(origin_b)
		a2, b2 = colvec2tuple(unit_dir_b)

		# lines are parallel, return None
		if a1*b2 == a2*b1:
			return None
		
		# find parameter for unit_dir_b
		t2 = (b1*(x2-x1)-a1*(y2-y1))/(a1*b2-a2*b1)

		# return the intersection point composed from the second parameter
		return origin_b + unit_dir_b * t2

