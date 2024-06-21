"""
This module is responsible for detecting calibration points.
There is a separate function for thermal images and rgb images.
"""
import math

import cv2
import numpy as np
import imutils

MINIMAL_THERMAL_CONTOUR_AREA = 90
MAXIMAL_THERMAL_CONTOUR_AREA = 2000
FULFILMENT_THERMAL_THRESHOLD = 0.4

MINIMAL_RGB_CONTOUR_AREA = 20
MAXIMAL_RGB_CONTOUR_AREA = 1000
FULFILMENT_RGB_THRESHOLD = 0.4

NUMBER_OF_CALIBRATION_POINTS = 9
NUMBER_OF_CP_WITH_TRIANGLE = 10

THERMAL_THRESOLD_KERNEL_HYDRA = 171
THERMAL_THRESOLD_C_HYDRA = -20

THERMAL_THRESOLD_KERNEL_HUB = 171
THERMAL_THRESOLD_C_HUB = -30

RGB_THRESHOLD_LEVEL = 250


def validate_contour(contour, minimal_size: int, maximal_size: int, fulfilment_threshold: float):
    """Check if contour has correct shape and size

    Args:
        contour (cv2.Mat): validated contour
        minimal_size (int): minimal area of contour 
        maximal_size (int): maximal area of contour
        fulfilment_threshold (float): values <0,1> - accept shapes that are more similar to circle than this value 

    Returns:
        True if contour is valid, False otherwise
    """

    contour_area = cv2.contourArea(contour)

    _, r = cv2.minEnclosingCircle(contour)

    circle_area = math.pi*r*r

    percentage_of_fulfilment = contour_area/circle_area

    return minimal_size <= contour_area <= maximal_size and percentage_of_fulfilment >= fulfilment_threshold


def find_calibration_points_on_heatmap(image: cv2.Mat, is_hydra=True):
    """Find nine points on a thermal calibration heatmap.

    Args:
        image (cv2.Mat): calibration heatmap
        is_hydra (bool): flag indicating device type (hydra or hub) 

    Returns:
        coordinates - sorted list of calibration point coordinates
        thermal_rgb - rgb version of a image - for debugging
        thresh_rgb - rgb version of thresholded image - for debugging
    """
    image_8u = np.uint8(image)

    if image_8u.ndim == 3:
        image_rgb = image_8u.copy()
        image_8u = cv2.cvtColor(image_8u, cv2.COLOR_RGB2GRAY)
    else:
        image_rgb = cv2.cvtColor(image_8u, cv2.COLOR_GRAY2RGB)

    if is_hydra:
        thresh = cv2.adaptiveThreshold(
            image_8u, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, THERMAL_THRESOLD_KERNEL_HYDRA, THERMAL_THRESOLD_C_HYDRA)
    else:
        thresh = cv2.adaptiveThreshold(
            image_8u, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, THERMAL_THRESOLD_KERNEL_HUB, THERMAL_THRESOLD_C_HUB)   
    
    thresh_8u = np.uint8(thresh)

    if is_hydra:    
        kernel = np.ones((3,3))
        thresh_8u = cv2.erode(thresh_8u, kernel=kernel, iterations=1)
        thresh_8u = cv2.dilate(thresh_8u, kernel=kernel, iterations=1)

    thresh_rgb = cv2.cvtColor(thresh_8u, cv2.COLOR_GRAY2RGB)

    contours = cv2.findContours(
        thresh_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    contours = [contour for contour in contours]

    # In case of small noise, but large enough to be a point candidate
    if len(contours) > NUMBER_OF_CP_WITH_TRIANGLE:
        contours.sort(key=lambda contour: cv2.contourArea(contour), reverse=True)
        contours = contours[:NUMBER_OF_CP_WITH_TRIANGLE]

    coordinates = []

    # we now want to filter out noise
    for c in contours:
        correct_contour = validate_contour(c, MINIMAL_THERMAL_CONTOUR_AREA, MAXIMAL_THERMAL_CONTOUR_AREA, FULFILMENT_THERMAL_THRESHOLD)
        if correct_contour:
            ((x, y), _) = cv2.minEnclosingCircle(c)
            coordinates.append((int(x), int(y)))

    if len(coordinates) == NUMBER_OF_CALIBRATION_POINTS:
        coordinates = sort_calibration_points(coordinates)
    elif len(coordinates) == NUMBER_OF_CP_WITH_TRIANGLE:
        coordinates = sort_calibration_points_remove_triangle(coordinates)

    for x, y in coordinates:
        cv2.circle(thresh_rgb, (x, y), 0, (0, 0, 255), 10)

    return coordinates, image_rgb, thresh_rgb


def find_calibration_points_on_rgb_photo(image):
    """Find nine points on a rgb calibration photo.

    Args:
        image (cv2.Mat): calibration photo

    Returns:
        coordinates - sorted list of calibration point coordinates
        thermal_rgb - rgb version of a image - for debugging
        thresh_rgb - rgb version of thresholded image - for debugging
    """
    image_8u = np.uint8(image)

    kernel = (3, 3)
    image_8u = cv2.GaussianBlur(image_8u, kernel, 0)

    image_8u = cv2.normalize(image_8u, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    image_rgb = cv2.cvtColor(image_8u, cv2.COLOR_GRAY2RGB)

    _, thresh = cv2.threshold(image_8u, RGB_THRESHOLD_LEVEL, 255, cv2.THRESH_BINARY)

    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    contours = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    correct_contours = []

    # we now want to filter out noise
    for c in contours:
        is_correct_contour = validate_contour(c, MINIMAL_RGB_CONTOUR_AREA, MAXIMAL_RGB_CONTOUR_AREA, FULFILMENT_RGB_THRESHOLD)
        if is_correct_contour:
            correct_contours.append(c)

    # In case of small reflections on box walls
    if len(correct_contours) > NUMBER_OF_CP_WITH_TRIANGLE:
        correct_contours.sort(key=lambda contour: cv2.contourArea(contour), reverse=True)
        correct_contours = correct_contours[:NUMBER_OF_CP_WITH_TRIANGLE]

    coordinates = []

    for c in correct_contours:
        (x, y), _ = cv2.minEnclosingCircle(c)
        coordinates.append((int(x), int(y)))

    if len(coordinates) == NUMBER_OF_CALIBRATION_POINTS or len(coordinates) == NUMBER_OF_CP_WITH_TRIANGLE:

        if len(coordinates) == NUMBER_OF_CALIBRATION_POINTS:
            coordinates = sort_calibration_points(coordinates)
        elif len(coordinates) == NUMBER_OF_CP_WITH_TRIANGLE:
            coordinates = sort_calibration_points_remove_triangle(coordinates)

    for x, y in coordinates:
        cv2.circle(image_rgb, (x, y), 0, (0, 0, 255), 10)

    return coordinates, image_rgb, thresh_rgb


def sort_calibration_points(points: list[(int, int)]):
    """ Sort array so points are in a following order:
        1 2 3
        4 5 6
        7 8 9
    Args:
        points (list[(int, int)]): list of points to be sorted
    Returns:
        points - sorted points
    """
    points.sort(key=lambda coordinate: coordinate[1])
    points[0:3] = sorted(points[0:3], key=lambda coordinate: coordinate[0])
    points[3:6] = sorted(points[3:6], key=lambda coordinate: coordinate[0])
    points[6:9] = sorted(points[6:9], key=lambda coordinate: coordinate[0])
    return points


def sort_calibration_points_remove_triangle(points: list[(int, int)]):
    """ Sort array so points are in a following order:
        1 2 3
        4 5 6 7
        8 9 10
        Then remove point 5 (triangle)
    Args:
        points (list[(int, int)]): list of points to be sorted
    Returns:
        points - sorted points
    """
    points.sort(key=lambda coordinate: coordinate[1])
    points[0:3] = sorted(points[0:3], key=lambda coordinate: coordinate[0])
    points[3:7] = sorted(points[3:7], key=lambda coordinate: coordinate[0])
    points[7:10] = sorted(points[7:10], key=lambda coordinate: coordinate[0])
    points.pop(4)
    return points


def find_coordinates_by_threshold(image: cv2.Mat, threshold: float):
    """Find coordinates of all points brighter than the threshold. 

    Args:
        image (cv2.Mat): image with bright spots (usually after thresholding)
        threshold (float): threshold against which points will be selected

    Returns:
        coordinates - calibration point coordinates (to be clustered)
    """
    loc = np.where(image >= threshold)

    coordinates = []
    for pt in zip(*loc[::-1]):
        coordinates.append((int(pt[0]), int(pt[1])))

    return coordinates

