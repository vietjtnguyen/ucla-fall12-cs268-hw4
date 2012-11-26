#!/usr/bin/python

import math
import sys

import cv2
import numpy as np

"""
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
"""

