# auto_calibration - Automatic detection of calibration points

Here is the repository of the algorithm to perform rgb-thermal mapping.

It consists of following files and directories:

1. `automatic_point_detection` - directory for the algorithm to automatically find calibration points on a thermal and rgb images
    1. `thermal_tests_dataset` - directory that has files for thermal functions tests
        1. `calibrated_devices.txt` - file with all calibrated devices to be tested, likely out of date.
    2. `wip_dataset` - directory for test calibration images
        1. `rgb` - current rgb test calibration images
        2. `thermal` - current thermal test heatmaps
        3. `devices.txt` - current mapping of device id to image name
    3. `auto_point_detection.py` - main file that consists of functions to detect calibration points
    4. `calibration_test.ipynb` - jupyter journal to test auto_calibration on files from `wip_dataset` 
    5. `calibration_utils.py` - module responsible for the functions of loading data, including from files or from devices 
    6. `check_for_incorrect_points.py` - script that checks if the device has rotated or offset calibration points
    7. `rgb_test_WIP.ipynb` - jupyter journal to develop detection of calibration points on a color rgb image using resources from `wip_dataset`  
    8. `thermal_print_images.py` - script to generate calibration images with marked calibration points (correct and generated). Best run after `thermal_test_on_calibrated_devices.py`.
    9. `thermal_test_on_calibrated_devices.py` - script to execute tests on a thermal images
    10. `thermal_test_wip.ipynb` - jupyter journal to develop detection of calibration points on a thermal heatmap using resources from `wip_dataset`  
2. `automatic_calibration.py` - script that allows for smooth integration of `automatic_point_detection` into `calculate_rgb_thermal_mapping.py`
3. `automatic_rgb_thermal_mapping.py` - script that performs automatic parallax calibration of specified device using data from s3 and sends results to s3
4. `calculate_rgb_thermal_mapping.py` - script that allows user to perform calibration of specified device using data from file system or s3
5. `paralax_calibrator.py` - script that calculates parallax calibration based on calibration points on rgb and thermal image
6. `README.md` - this beautiful work of art
7. `rgb_thermal_mapping_utils.py` - module responsible for the functions of generating debug images for mapping scripts
8. `s3_utils.py` - module responsible for s3 integration
                      

## I want to integrate auto_calibration into my workflow

### Thermal images && LED RGB images:

Just pass calibration image to a respective method:

`auto_point_detection.find_calibration_points_on_heatmap(image)`
or
`auto_point_detection.find_calibration_points_on_rgb_photo(image)`
