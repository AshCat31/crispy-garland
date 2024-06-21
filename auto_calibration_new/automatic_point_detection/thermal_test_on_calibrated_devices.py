"""Test thermal automatic calibration on already calibrated devices"""
import os

import math
from collections import defaultdict

import auto_point_detection

import calibration_utils

THERMAL_TESTS_DIRECTORY = "thermal_tests_dataset"
TEST_DATASET_FILENAME = "calibrated_devices.txt"
CALIBRATION_DATA_PATH = "/home/jacek/delta-thermal/calibration_data/"


def calculate_point_difference(p1: tuple[int, int], p2: tuple[int, int]):
    """Calculate distance between two points."""
    diff_x = p1[0] - p2[0]
    diff_y = p1[1] - p2[1]

    return math.sqrt(diff_x * diff_x + diff_y * diff_y)


def compare_points(expected_arr, result_arr):
    """Find maximal distance between calibration points"""
    if len(expected_arr) != len(result_arr):
        return None

    differences = [
        calculate_point_difference(expected, result)
        for expected, result in zip(expected_arr, result_arr)
    ]

    return max(differences)


def save_list_to_file(data_list, directory_path, filename):
    """Save list to a file and print confirmation"""
    file_path = os.path.join(directory_path, filename)
    with open(file_path, 'w') as file:
        for line in data_list:
            file.write(f"{line}\n")
        print(f"Test {filename} data saved to file{file_path}")


def exec_test():
    """For each calibrated device try to calibrate them"""
    results = defaultdict(int)
    failures = []
    errors = []
    correct = []
    corrupted_count = 0

    test_data_dir = os.path.join(os.getcwd(), THERMAL_TESTS_DIRECTORY)
    test_dataset_path = os.path.join(
        test_data_dir, TEST_DATASET_FILENAME)
    devices = calibration_utils.try_to_read_list_from_file(test_dataset_path)

    for device_id in devices:
        success, correct_points, thermal_image = calibration_utils.get_data_for_device(
            device_id)
        if not success:
            corrupted_count += 1
            continue
        coordinates, _, _ = auto_point_detection.find_calibration_points_on_heatmap(thermal_image)

        compare_result = compare_points(correct_points, coordinates)

        if compare_result is None:
            failures.append(device_id)

        elif compare_result > 25:
            errors.append(device_id)
        else:
            results[math.ceil(compare_result/10)] += 1
            correct.append(device_id)

    total_count = len(devices)
    failure_count = len(failures)
    correct_count = len(correct)
    error_count = len(errors)
    error_rate = round(error_count / (error_count+correct_count) * 100, 2)
    failure_rate = round(failure_count / (total_count) * 100, 2)
    print(
        f"Test results:\nTotal: {total_count}, correct: {correct_count}, "
        f"errors: {error_count}, failures: {failure_count}, corrutped: {corrupted_count}, "
        f"error rate (from errors&correct only): {error_rate}%, failure rate: {failure_rate}%")
    for key, value in results.items():
        print(f"Distance {key} - {value} cases")

    save_list_to_file(correct, test_data_dir, 'correct.txt')
    save_list_to_file(errors, test_data_dir, 'errors.txt')
    save_list_to_file(failures, test_data_dir, 'failures.txt')


if __name__ == "__main__":
    exec_test()
