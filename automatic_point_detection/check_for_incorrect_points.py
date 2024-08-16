"""
This is a script to filter and validate thermal calibration points.
It has two major functions: checking for rotated images and checking for shifted images.
"""

import os
import argparse

import cv2

from calibration_utils import get_data_for_device, try_to_read_list_from_file


BASE_PATH = "/home/jacek/delta-thermal/calibration_data/"

DEBUG_HELP = "Show each image with additional data"
ROTATION_HELP = "Check which images are rotated"
SHIFT_HELP = "Check which images are shifted"
THRESHOLD_HELP = (
    "For shift: show images that have points that are closer than this number"
)
"For rotation: show images that have more rotation than this number"
DEVICE_ID_HELP = "Device ID to check, always debug mode"
FILE_HELP = "Check devices from this file"
DIRECTORY_HELP = (
    "Save each selected image to file in this directory (create if not existing)"
)

ARG_ERROR_TEXT = "You have to use either --shift or --rotatoin option\nExiting"
NO_FILE_DEVICE_ERROR = "You have to specify file or device_id\n"

IGNORE_VALUE = -1


def show_image(device_id: str, image: cv2.Mat):
    """Show an image to a user and ask him if he wants to exit program"""
    window_name = f"{device_id} - q to exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 640)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(20000)
    cv2.destroyAllWindows()

    if key == ord("q"):
        return True
    return False


def save_image_to_file(image: cv2.Mat, filename: str, directory: str):
    """Save image to file in specified directory"""
    file_path = os.path.join(directory, filename)
    print(file_path)
    cv2.imwrite(file_path, image)


def check_if_image_is_shifted(
    device_id: str, threshold: int, debug: bool, save_directory: str = ""
):
    """Check if the image is shifted enough that some of the points are not visible

    Args:
        device_id (str): id of the device
        threshold (int): how close to the edge the point must be to mark image corrupted (-1 to ignore)
        debug (bool): if True, display each image to the user
        save_directory (str): directory to save image, if empty do not save image
    Returns:
        True if user wants to close script, False otherwise
    """
    success, calibration_points, thermal_image = get_data_for_device(device_id)

    if not success:
        return False

    max_x, max_y = calibration_points.max(axis=0)

    are_points_under_threshold = calibration_points.min() <= threshold
    are_points_over_threshold = max_x >= (240 - threshold) or max_y >= (320 - threshold)

    if save_directory or debug:
        for x, y in calibration_points:
            cv2.circle(thermal_image, (int(x), int(y)), 0, (0, 0, 255), 5)

    if (
        are_points_under_threshold
        or are_points_over_threshold
        or threshold == IGNORE_VALUE
    ):
        print(device_id)
        if save_directory:
            save_image_to_file(
                thermal_image, f"{device_id}_bad_shift.png", save_directory
            )
        if debug:
            print(calibration_points)
            if show_image(device_id, thermal_image):
                return True

    return False


def check_if_image_is_rotated(
    device_id: str, threshold: int, debug: bool, save_directory: str = ""
):
    """Check if the image is rotated enough to interfere with calibration

    Args:
        device_id (str): id of the device
        threshold (int): How much the image must be rotated to mark it corrupted (-1 to ignore)
        debug (bool): if True, display each image to the user
        save_directory (str): directory to save image, if empty do not save image
    Returns:
        True if user wants to close script, False otherwise
    """
    success, calibration_points, thermal_image = get_data_for_device(device_id)

    if not success:
        return False

    thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_GRAY2RGB)

    cp = calibration_points

    a_vertical = abs(
        round((int(cp[7][0]) - int(cp[1][0])) / (int(cp[7][1]) - int(cp[1][1])) * 100)
    )
    a_horizontal = abs(
        round((int(cp[5][1]) - int(cp[3][1])) / (int(cp[5][0]) - int(cp[3][0])) * 100)
    )

    if save_directory or debug:
        for x, y in calibration_points:
            cv2.circle(thermal_image, (int(x), int(y)), 0, (0, 0, 255), 5)

            cv2.line(
                thermal_image,
                (cp[1][0], cp[1][1]),
                (cp[7][0], cp[7][1]),
                (0, 255, 0),
                2,
            )
            cv2.line(
                thermal_image,
                (cp[3][0], cp[3][1]),
                (cp[5][0], cp[5][1]),
                (0, 255, 0),
                2,
            )

    if a_vertical > threshold or a_horizontal > threshold or threshold == IGNORE_VALUE:
        print(device_id)
        if save_directory:
            save_image_to_file(
                thermal_image, f"{device_id}_bad_rotation.png", save_directory
            )
        if debug:
            print(f"vertical - {a_vertical}, horizontal - {a_horizontal}")
            if show_image(device_id, thermal_image):
                return True

    return False


def main():
    """Main function, loads data and determines how the program will be executed"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", required=False, help=DEBUG_HELP)
    parser.add_argument("--file", type=str, help=FILE_HELP)
    parser.add_argument(
        "--directory", required=False, type=str, default="", help=DIRECTORY_HELP
    )
    parser.add_argument("--rotation", action="store_true", help=ROTATION_HELP)
    parser.add_argument("--shift", action="store_true", help=SHIFT_HELP)
    parser.add_argument(
        "--threshold",
        required=False,
        type=int,
        default=IGNORE_VALUE,
        help=THRESHOLD_HELP,
    )
    parser.add_argument("--device_id", required=False, help=DEVICE_ID_HELP)
    args = parser.parse_args()

    if args.rotation == args.shift:
        parser.print_help()
        parser.error(ARG_ERROR_TEXT)

    if not args.file and not args.device_id:
        parser.print_help()
        parser.exit(1, NO_FILE_DEVICE_ERROR)

    directory_path = args.directory

    if directory_path:
        if not directory_path.startswith("/") and not directory_path.startswith("~"):
            directory_path = os.path.join(os.getcwd(), directory_path)
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

    if args.rotation:
        func = check_if_image_is_rotated
    else:
        func = check_if_image_is_shifted

    if args.device_id:
        func(
            device_id=args.device_id,
            threshold=args.threshold,
            save_directory=directory_path,
            debug=True,
        )
    else:
        calibrated_devices = try_to_read_list_from_file(args.file)

        for device_id in calibrated_devices:
            if func(
                device_id=device_id,
                threshold=args.threshold,
                save_directory=directory_path,
                debug=args.debug,
            ):
                break


if __name__ == "__main__":
    main()
