"""This module is responsible for connecting auto_calibration module into rgb-mapping module"""
import cv2
import numpy as np

from automatic_point_detection import auto_point_detection

THERMAL_RGB_METHOD_MESSAGE = "\"y\" or manual method?"
THERMAL_MESSAGE = "Press \"y\" if image is correct, any key otherwise"


def detect_calibration_points_rgb_image(image: cv2.Mat):
    """Find nine calibration points on a rgb calibration image.

    Args:
        image (cv2.Mat): rgb calibration image

    Returns:
        coordinates - sorted list of calibration point coordinates,
        None - if algorithm failed 
    """
    debug_image = image.copy()
    work_image = image.copy()

    if work_image.ndim == 3:
        work_image = cv2.cvtColor(work_image, cv2.COLOR_RGB2GRAY)
    else:
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2RGB)

    coordinates, _, _ = auto_point_detection.find_calibration_points_on_rgb_photo(work_image)

    if not coordinates: # image may have inverted colors
        work_image = 255 - work_image
        coordinates, _, _ = auto_point_detection.find_calibration_points_on_rgb_photo(work_image)

    for x, y in coordinates:
        cv2.circle(debug_image, (int(x), int(y)), 0, (0, 0, 255), 10)

    if validate_detected_points(debug_image, THERMAL_RGB_METHOD_MESSAGE):
        return convert_coordinates_to_numpy_array(coordinates)

    return None



def detect_calibration_points_thermal_image(image: cv2.Mat, is_hydra=True):
    """Find nine calibration points on a thermal calibration image

    Args:
        image (cv2.Mat): thermal calibration image
        is_hydra (bool): flag indicating device type (hydra or hub)

    Returns:
        Sorted list of calibration point coordinates
    """
    coordinates, result_image, _ = auto_point_detection.find_calibration_points_on_heatmap(image, is_hydra=is_hydra)

    test_image = result_image.copy()

    for x, y in coordinates:
        cv2.circle(test_image, (x, y), 0, (0, 0, 255), 10)

    if validate_detected_points(test_image, THERMAL_MESSAGE):
        return convert_coordinates_to_numpy_array(coordinates)
    return None


def validate_detected_points(image: cv2.Mat, text: str):
    """Display image with calibration points for user approval.

    Args:
        image (cv2.Mat): calibration image with dots representing calibration points
        text (str): text to be displayed 

    Returns:
        True if user pressed 'y' (approval), False otherwise
    """
    print(text)

    cv2.namedWindow(text, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(text, 640, 640)
    cv2.imshow(text, image)
    choice = cv2.waitKey(20000)
    cv2.destroyAllWindows()

    if choice == ord('y'):
        return True
    return False


def convert_coordinates_to_numpy_array(coordinates: list[(int, int)]):
    """Converts coordinates to the format expected by the rgb-mapping module. 

    Args:
        coordinates (list[(int, int)]): list of coordinates to be converted

    Returns:
        List of converted coordinates
    """
    numpy_arr = np.zeros((len(coordinates), 2))
    for i, (x, y) in enumerate(coordinates):
        numpy_arr[i][0] = int(x)
        numpy_arr[i][1] = int(y)
    return numpy_arr
