#!/usr/bin/python

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
