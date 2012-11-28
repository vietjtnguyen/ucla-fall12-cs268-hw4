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

def colvec2tuple(colvec):
	'''
	Converts a column vector (represented as a NumPy matrix) into a n-tuple where n is
	the first element in the shape of the column vector (colvec.shape[0]).
	'''
	colvec = colvec.A
	return tuple([colvec[row][0] for row in range(0, colvec.shape[0])])

def tuple2colvec(colvec):
	'''
	Converts a n-tuple into a column vector (represented as a NumPy matrix) where the
	number of rows is equal to n.
	'''
	return np.matrix([[x] for x in colvec])

def tuple2inttuple(v):
	'''
	Converts each item in a tuple to an integer.
	'''
	return tuple(map(int, v))

def tuple2floattuple(v):
	'''
	Converts each item in a tuple to a float.
	'''
	return tuple(map(float, v))
