#!/usr/bin/env python3
"""
" Paralax Calibrator
"
" This module creates calibration files for paralax correction between RGB and
" Thermal images
"
"""

__author__ = 'Delta Thermal Inc.'
__version__ = "1.0"
__copyright__ = """
    Copyright 2018-2023 Delta Thermal Inc.

    All Rights Reserved.
    Covered by one or more of the Following US Patent Nos. 10,991,217,
    Other Patents Pending.
"""

import os
import math
import logging
import json

import cv2
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import tkinter as tk
from tkinter import filedialog

class ParalaxCalibrator:
    """
    This class is designed to output the calibration images
    """
    def __init__(self):
        """
        These constants were provided by Noam in main.py in Delta-Thermal-RGB-Mapping
        git repo for calibrations
        """
        # rgb camera:
        self.rgb_img_size = (320,240) 
        self.rgb_padding = (100,100)
        self.rgb_distortion_center = (self.rgb_img_size[1]/2 , self.rgb_img_size[0]/2) #(1136,934) #(1178,908) # Calculated using Matlab's script. Relevant for all rgb images
        self.rgb_c = -0.38 
        self.rgb_d = 1 - self.rgb_c
        self.rgb_radius_norm_factor = self.rgb_img_size[0]/2 # The smallest dimension divided by 2
        
        # Default emissitivy value, in case that no emissivity_matrix avilable
        self.default_emissivity = 0.9
        
        # Thermal camera:
        self.trml_img_size = (320,240) # (y,x)
        self.trml_distortion_center_shift = (-12,6)
        self.trml_distortion_center = (self.trml_img_size[1]/2 + self.trml_distortion_center_shift[1], self.trml_img_size[0]/2 + self.trml_distortion_center_shift[0])
        self.angular_mapping_corner_val = 100
        self.alpha = 47
        self.x_center_alpha = -2
        self.y_center_alpha = 2

        # Sensor sensetivity gaussian sigma
        self.sens_sigma = 250

    def __call__(self, thermal_image, color_image, thermalPoints, rgbPoints):
        """

        This class will create the calibration files for Delta Thermal, Inc.

        :param thermal_image: thermal image (must be 240x320, grayscale)
        :param color_image: color image (must be 240x320, RGB)
        :param rgbPoints: a (9,2) array containing 9 x,y coordinates of the heater elements in the rgb image
        :param thermalPoints: a (9,2) array containing 9 x,y coordinates of the heater elements in the thermal image

        
        """
        self.thermalImage = thermal_image
        self.colorImage = color_image

        if len(self.colorImage.shape) == 3:
            r, g, b = self.colorImage[:,:,0], self.colorImage[:,:,1], self.colorImage[:,:,2]
            self.colorImage = 0.2989 * r + 0.5870 * g + 0.1140 * b

        logging.debug(self.thermalImage.shape)
        logging.debug(self.colorImage.shape)

        assert self.thermalImage.shape == self.trml_img_size and self.colorImage.shape == self.rgb_img_size
        assert rgbPoints.shape == (9,2) and thermalPoints.shape == (9,2)

        if np.max(self.colorImage) <= 1:
            self.colorImage = (self.colorImage * 255).astype(np.uint8)

        #calculate undistorted array coordinates
        rgb_corner_elements = self.get_corners_coordinates(rgbPoints)
        rgb_corner_distortions = self.calculate_undistorted_coordinates_of_array(
            rgb_corner_elements,
            self.rgb_distortion_center,
            self.rgb_c,
            self.rgb_d,
            self.rgb_radius_norm_factor,
            )

        # Build corners for angular mapping (for pixels that fall in angular angle |x| & |y| <= angular_mapping_corner_val*max_angular_mapping_factor and
        max_angular_mapping_factor = 1.4

        # Setting the angular coordinates of the thermal image
        angular_mapping_corners_coordinates = np.array([
            [-self.angular_mapping_corner_val,-self.angular_mapping_corner_val],
            [self.angular_mapping_corner_val,-self.angular_mapping_corner_val],
            [-self.angular_mapping_corner_val,self.angular_mapping_corner_val],
            [self.angular_mapping_corner_val,self.angular_mapping_corner_val]
            ])

        # Calculate the affine transformation from rgb_elements_corners_undistorted_coordinates --> trml_elements_corners_undistorted_coordinates
        rgb_to_angular_mapping_perspective_transformation_matrix = self.calculate_geometrical_transformation(
            rgb_corner_distortions, 
            angular_mapping_corners_coordinates,
            )

        # Build array for thermal aligned distorted corners (fixed predefined coordinates that span the image on the frame)
        trml_aligned_distorted_corners_coordinates = self.get_trml_aligned_distorted_corners_coordinates()

        # Get the corners of the 3x3 array for calculating the transformations
        trml_elements_corners_coordinates = self.get_corners_coordinates(thermalPoints)

        # Calculate the perspective transformation from trml_aligned_distorted_corners_coordinates --> trml_elements_corners_coordinates
        trml_aligned_to_original_perspective_transformation_matrix = self.calculate_geometrical_transformation(
            trml_aligned_distorted_corners_coordinates, 
            trml_elements_corners_coordinates,
            )

        # Calculate the perspective transformation from trml_elements_corners_coordinates --> trml_aligned_distorted_corners_coordinates (inverse of prev)
        trml_original_to_aligned_perspective_transformation_matrix = self.calculate_geometrical_transformation(
            trml_elements_corners_coordinates,
            trml_aligned_distorted_corners_coordinates,
            )

        # Calculate the aligned thermal element coordinates
        trml_aligned_coordinates = np.zeros((9, 2))
        for i in range(9):
            trml_aligned_coordinates[i, :] = self.calculated_perspective_transform_coordinations(
                trml_original_to_aligned_perspective_transformation_matrix,
                thermalPoints[i,:],
                )

        # Calculate themal angular mapping coefs
        logging.debug(f'trml_aligned_coordinates:{trml_aligned_coordinates}')
        logging.debug(f'trml_distortion_center:{self.trml_distortion_center}')
        a_coefs_vector, b_coefs_vector, c_coefs_vector = self.get_trml_angular_mapping_coef(
            trml_aligned_coordinates,
            self.trml_distortion_center,
            )
        logging.debug(f'a_coefs_vector:{a_coefs_vector}')
        logging.debug(f'b_coefs_vector:{b_coefs_vector}')
        logging.debug(f'c_coefs_vector:{c_coefs_vector}')

        # Calculate shifting for the angular mapping
        parabolic_x_coef = self.calculate_parabolic_coef_for_shifted_angular_mapping(
            (self.x_center_alpha/self.alpha)*self.angular_mapping_corner_val,
            self.angular_mapping_corner_val,
            )
        parabolic_y_coef = self.calculate_parabolic_coef_for_shifted_angular_mapping(
            (self.y_center_alpha/self.alpha)*self.angular_mapping_corner_val,
            self.angular_mapping_corner_val,
            )

        # For faster process, we can sample in lower resolution and rescale the array layer
        map_low_resolution = True

        # Set the gap between sampled pixels
        if map_low_resolution:
            mapping_resolution = 2
        else:
            mapping_resolution = 1

        # Calculating the limit of the thermal data
        max_angular_mapping_value = self.angular_mapping_corner_val * max_angular_mapping_factor

        # Mapping matricies
        # For each pixel, what is the row,col of the thermal image
        mapped_coordinates = np.zeros(
                (
                    int((self.rgb_img_size[0]+2*self.rgb_padding[0]) / mapping_resolution),
                    int((self.rgb_img_size[1]+2*self.rgb_padding[1]) / mapping_resolution),
                    2,
                    )
                ).astype(int)

        # For every pixel, 0 = No trml data for this pixel, 1 = There is thermal data for this pixel
        in_trml_mask_matrix = np.zeros(
                (
                    int((self.rgb_img_size[0]+2*self.rgb_padding[0]) / mapping_resolution),
                    int((self.rgb_img_size[1]+2*self.rgb_padding[1]) / mapping_resolution),
                    )
                ).astype(int)

        # Sensitivity correction, because of the known sensitivity drop for from the center of FOX
        trml_sensitivity_correction_matrix = np.zeros(
                (
                    int((self.rgb_img_size[0]+2*self.rgb_padding[0]) / mapping_resolution),
                    int((self.rgb_img_size[1]+2*self.rgb_padding[1]) / mapping_resolution),
                    )
                )

        # DEBUG! Init the target array
        mapped_thermal_matrix = np.zeros(
                (
                    int((self.rgb_img_size[0]+2*self.rgb_padding[0]) / mapping_resolution),
                    int((self.rgb_img_size[1]+2*self.rgb_padding[1]) / mapping_resolution),
                    )
                )

        half_rgb_padding = (int(self.rgb_padding[0]/2),int(self.rgb_padding[1]/2))

        # For every pixel, calculate the value from the thremal image and fill the array
        for row in range(int((self.rgb_img_size[0]+2*self.rgb_padding[0]) / mapping_resolution)):
            for col in range(int((self.rgb_img_size[1]+2*self.rgb_padding[1]) / mapping_resolution)):
                # Get the loop pixel coordinate
                loop_rgb_coordinate = np.array(
                        [
                            (col - half_rgb_padding[1])*mapping_resolution,
                            (row - half_rgb_padding[0])*mapping_resolution
                            ]
                        ) # (x,y)

                # Preform the chain of transformations for getting the coordinates in the thermal image
                loop_undistorted_rgb_coordinate = self.calculate_undistorted_coordinates(
                        loop_rgb_coordinate,
                        self.rgb_distortion_center,
                        self.rgb_c,
                        self.rgb_d,
                        self.rgb_radius_norm_factor,
                        )

                if loop_undistorted_rgb_coordinate is not None:
                    loop_angular_mapped_coordinate = self.calculated_perspective_transform_coordinations(
                            rgb_to_angular_mapping_perspective_transformation_matrix,
                            loop_undistorted_rgb_coordinate
                            )
                    loop_angular_mapped_shifted_coordinate = self.calculated_shifted_angular_mapped_coordinate(
                            loop_angular_mapped_coordinate,
                            parabolic_x_coef,
                            parabolic_y_coef
                            )
                    # loop_angular_mapped_shifted_coordinate[0] = 100 - (100 - loop_angular_mapped_shifted_coordinate[0])*0.83
                    # loop_angular_mapped_shifted_coordinate[1] -= 27
                    # if loop_angular_mapped_shifted_coordinate[0] >= -max_angular_mapping_value and loop_angular_mapped_shifted_coordinate[0] <= max_angular_mapping_value and loop_angular_mapped_shifted_coordinate[1] >= -max_angular_mapping_value and loop_angular_mapped_shifted_coordinate[1] <= max_angular_mapping_value:
                    # Cancel skipping mapping for pixel if the pixel is out of angular range
                    loop_trml_aligned_coordinate, calc_valid_flag = self.get_trml_aligned_coordinated_from_angular_mapping(
                            loop_angular_mapped_shifted_coordinate,
                            self.angular_mapping_corner_val,
                            a_coefs_vector,
                            b_coefs_vector,
                            c_coefs_vector,
                            self.trml_distortion_center
                            )
                    # Check if found a mathematical solution for the thermal pixel finding
                    if calc_valid_flag:

                        if loop_trml_aligned_coordinate[0].imag != 0 or loop_trml_aligned_coordinate[1].imag != 0:
                            logging.warning("Imag part not zero!!")

                        loop_trml_coordinate_unbounded = self.calculated_perspective_transform_coordinations(
                                trml_aligned_to_original_perspective_transformation_matrix,
                                loop_trml_aligned_coordinate,
                                )

                        # Check that the pixel is in the thermal image range
                        if 0 <= loop_trml_coordinate_unbounded[0] < self.trml_img_size[1] and 0 <= loop_trml_coordinate_unbounded[1] < self.trml_img_size[0]:
                            trml_row,trml_col = self.get_trml_coordinates(
                                    loop_trml_coordinate_unbounded,
                                    self.trml_img_size,
                                    )

                            # Get the value from the thermal image and fill the array
                            val = self.thermalImage[trml_row, trml_col]
                            mapped_thermal_matrix[row, col] = val

                            # Save coordinates
                            mapped_coordinates[row,col,:] = np.array([trml_row,trml_col])
                            in_trml_mask_matrix[row, col] = 1
                            trml_sensitivity_correction_matrix[row, col] = self.get_sensitivity_correction_factor(
                                loop_angular_mapped_shifted_coordinate)

        if map_low_resolution:
            # Rescaling final map matrix
            mapped_thermal_matrix = np.kron(mapped_thermal_matrix, [[1, 1], [1, 1]])

            # Rescalse coordinates matrix
            row_mapped_coordinates = np.kron(mapped_coordinates[:,:,0], [[1, 1], [1, 1]])
            col_mapped_coordinates = np.kron(mapped_coordinates[:,:,1], [[1, 1], [1, 1]])
            row_mapped_coordinates = np.expand_dims(row_mapped_coordinates, axis=2)
            col_mapped_coordinates = np.expand_dims(col_mapped_coordinates, axis=2)
            mapped_coordinates = np.concatenate([row_mapped_coordinates,col_mapped_coordinates], axis=2)

            # Rescaling sensitivity matrix
            trml_sensitivity_correction_matrix = np.kron(trml_sensitivity_correction_matrix, [[1, 1], [1, 1]])

            # Rescalse mask matrix
            in_trml_mask_matrix = np.kron(in_trml_mask_matrix, [[1, 1], [1, 1]])
            in_trml_mask_matrix[in_trml_mask_matrix > 0] = 1
        
        return (in_trml_mask_matrix, mapped_coordinates, trml_sensitivity_correction_matrix)

    def calculate_undistorted_coordinates(self, original_coordinates, distortion_center, C, D, radius_norm_factor):
        # The function gets (2,) size numpy array and distortion parameters (distortion center, C distortion value, radius normalization factor)
        # and return (2,) size numpy array contains the undistorted coordinates
    
        # The un-distortion operation is done using calculating the new radius (r_dst) using the original pixel radius (r_src),
        # while maintaining the theta angle related to the distortion center:
        # r_src = r_dst (D + C * r_dst), where D = 1-C
        # which can be calculated using the closed formula (selecting the smaller square root solution):
        # r_dst = (D^2 + 4*C*r_src)^0.5 - D)/(2*C)
        # theta_dst = theta_src
    
        # Calculate r_src and normalize it
        relative_x = original_coordinates[0] - distortion_center[0]
        relative_y = original_coordinates[1] - distortion_center[1]
        rad_src = math.sqrt(relative_x**2 + relative_y**2)

        rad_src_normalized = rad_src / radius_norm_factor
    
        if (rad_src_normalized > 1):
            polynom = [C,D,-rad_src_normalized]
            roots = np.roots(polynom)
            r_dst_normalized = np.min(roots)
            r_dst = r_dst_normalized * radius_norm_factor
        else:
            # Calculate r_dst
            inner_sqrt_term = D**2 + 4*C*rad_src_normalized
            r_dst_normalized = (math.sqrt(inner_sqrt_term) - D)/(2*C)
            r_dst = r_dst_normalized*radius_norm_factor

    
        if r_dst.imag != 0:
            return None
    
        # Calculate theta
        theta = math.atan2(relative_y,relative_x)
    
        # Calculate new coordinates
        undistorted_x_val = r_dst*math.cos(theta) + distortion_center[0]
        undistorted_y_val = r_dst*math.sin(theta) + distortion_center[1]
    
        # Store the result in new array
        undistorted_coordinates_array = np.array([undistorted_x_val,undistorted_y_val])
    
        # return the result array
        return undistorted_coordinates_array

    def calculate_undistorted_coordinates_of_array(self, original_coordinates_array, distortion_center, C, D, radius_norm_factor):
        """
         The function gets (num,2) size numpy array and distortion parameters (distortion center, C distortion value, radius normalization factor)
         and return (num,2) size numpy array contains the undistorted coordinates
         The undistortion operation is calculated using 'calculate_undistorted_coordinates' function
        """
    
        coordinates_array_shape = original_coordinates_array.shape
        undistorted_arr = np.zeros(coordinates_array_shape)
    
        # For every coordinate (x,y) in the array, calculate the undistorted values
        for i in range(coordinates_array_shape[0]):
            loop_undistorted_coordinates = self.calculate_undistorted_coordinates(
                    original_coordinates_array[i,:],
                    distortion_center,
                    C,
                    D,
                    radius_norm_factor,
                    )
            undistorted_arr[i,:] = loop_undistorted_coordinates
    
        return undistorted_arr

    def get_corners_coordinates(self, full_coordinates, corner_indices = [0,2,6,8]):
        """
        The function gets (9,2) size numpy array (9 (x,y) coordinates)
        and returns (4,2) size numpy array, which contains only the corner 
        elements in the following order (top left, top right, bottom left, bottom right)
        """
        corners_arr = np.zeros((4,2))
        for i in range(4):
            corners_arr[i,:] = full_coordinates[corner_indices[i],:]
    
        return corners_arr

    def calculate_geometrical_transformation(self,from_coordinates,to_coordinates):
        """
        Calculate the affine transformation matrix that maps 'from_coordinates' to 'to_coordinates'
        """
        transformation_matrix = cv2.getPerspectiveTransform(
                from_coordinates.astype(np.float32),
                to_coordinates.astype(np.float32),
                )
        return transformation_matrix

    def get_trml_aligned_distorted_corners_coordinates(self):
        height_width_factor = 1.921
        target_height = 250
        target_width = target_height/height_width_factor
    
        half_target_height = target_height/2
        half_target_width = target_width/2
    
        corners_array = np.array(
                [
                    [-half_target_width, -half_target_height],
                    [half_target_width, -half_target_height],
                    [-half_target_width, half_target_height],
                    [half_target_width, half_target_height]
                    ]
                )
    
        corners_array[:,0] += self.trml_img_size[1]/2
        corners_array[:,1] += self.trml_img_size[0]/2
    
        return corners_array

    def calculated_perspective_transform_coordinations(self,perspective_transformation_matrix, vector):
        extended_vector = np.array([vector[0],vector[1],1])
        transformed_vector = perspective_transformation_matrix.dot(extended_vector)
    
        # extract [x',y'] from result vector [x't,y't,t]
        transformed_vector[0]/=transformed_vector[2]
        transformed_vector[1]/=transformed_vector[2]
    
        return transformed_vector[0:2]

    def get_trml_angular_mapping_coef(self, trml1_proj_points, trml_distortion_center):
        # b_bottom, b_left, b_right, b_top = -0.001156049761200, 0.001247092149514, -9.210053762133429e-04, 0.001548588440226
        # c_bottom, c_left, c_right, c_top = 1.298948774670116e+02, -84.556090734119180, 79.460984901303260, -1.315569415055895e+02
    
        row_center = trml_distortion_center[1]
        col_center = trml_distortion_center[0]
    
        # ----------- Left lines - ----------
    
        p_left = trml1_proj_points[6, 1] - row_center
        t_left = trml1_proj_points[6, 0] - col_center
    
        k_left = trml1_proj_points[0, 1] - row_center
        n_left = trml1_proj_points[0, 0] - col_center
    
        z_left = trml1_proj_points[3, 1] - row_center
        m_left = trml1_proj_points[3, 0] - col_center
    
        mat_left = np.array([[p_left ** 2, p_left, 1],[k_left ** 2, k_left, 1],[z_left ** 2, z_left, 1]])
        vec_left = np.array([[t_left],[n_left],[m_left]])
    
        res_left = inv(mat_left).dot(vec_left)
        a_left = float(res_left[0])
        b_left = float(res_left[1])
        c_left = float(res_left[2])
        # ----------- Right lines - ----------
    
        p_right = trml1_proj_points[8, 1] - row_center
        t_right = trml1_proj_points[8, 0] - col_center
    
        k_right = trml1_proj_points[2, 1] - row_center
        n_right = trml1_proj_points[2, 0] - col_center
    
        z_right = trml1_proj_points[5, 1] - row_center
        m_right = trml1_proj_points[5, 0] - col_center
    
        mat_right = np.array([[p_right ** 2, p_right, 1],[k_right ** 2, k_right, 1],[z_right ** 2, z_right, 1]])
        vec_right = np.array([[t_right],[n_right],[m_right]])
    
        res_right = inv(mat_right).dot(vec_right)
        a_right = float(res_right[0])
        b_right = float(res_right[1])
        c_right = float(res_right[2])
    
        # ----------- Up lines - ----------
    
        p_top = trml1_proj_points[0, 0] - col_center
        t_top = trml1_proj_points[0, 1] - row_center
    
        k_top = trml1_proj_points[2, 0] - col_center
        n_top = trml1_proj_points[2, 1] - row_center
    
        z_top = trml1_proj_points[1, 0] - col_center
        m_top = trml1_proj_points[1, 1] - row_center
    
        mat_top = np.array([[p_top ** 2, p_top, 1],[k_top ** 2, k_top, 1],[z_top ** 2, z_top, 1]])
        vec_top = np.array([[t_top],[n_top],[m_top]])
    
        res_top = inv(mat_top).dot(vec_top)
        a_top = float(res_top[0])
        b_top = float(res_top[1])
        c_top = float(res_top[2])
    
        # ----------- Down lines - ----------
    
        p_bottom = trml1_proj_points[6, 0] - col_center
        t_bottom = trml1_proj_points[6, 1] - row_center
    
        k_bottom = trml1_proj_points[8, 0] - col_center
        n_bottom = trml1_proj_points[8, 1] - row_center
    
        z_bottom = trml1_proj_points[7, 0] - col_center
        m_bottom = trml1_proj_points[7, 1] - row_center
    
        mat_bottom = np.array([[p_bottom ** 2, p_bottom, 1],[k_bottom ** 2, k_bottom, 1],[z_bottom ** 2, z_bottom, 1]])
        vec_bottom = np.array([[t_bottom],[n_bottom],[m_bottom]])
    
        res_bottom = inv(mat_bottom).dot(vec_bottom)
        a_bottom = float(res_bottom[0])
        b_bottom = float(res_bottom[1])
        c_bottom = float(res_bottom[2])
    
        # -------------------------------------
    
        # Pack to tuples
        a_coefs_vector = (a_bottom, a_left, a_right, a_top)
        b_coefs_vector = (b_bottom, b_left, b_right, b_top)
        c_coefs_vector = (c_bottom, c_left, c_right, c_top)
    
        thermal_coef_matrix = [[a_bottom, a_left, a_right, a_top],
                               [b_bottom, b_left, b_right, b_top],
                               [c_bottom, c_left, c_right, c_top]]
        thermal_coef_matrix = np.array(thermal_coef_matrix)
        if not os.path.isdir("numpy_files"):
            os.mkdir("numpy_files")
        np.save("numpy_files/thermal_coef_matrix.npy",thermal_coef_matrix)
    
        return a_coefs_vector, b_coefs_vector,c_coefs_vector

    def get_trml_aligned_coordinated_from_angular_mapping(self, angular_coordinate, corner_max_val, a_coefs_vector, b_coefs_vector, c_coefs_vector, distortion_center):
    
        row_center = distortion_center[1]
        col_center = distortion_center[0]
    
        a_bottom, a_left, a_right, a_top = a_coefs_vector
        b_bottom, b_left, b_right, b_top = b_coefs_vector
        c_bottom, c_left, c_right, c_top = c_coefs_vector
    
        alpha_max = math.radians(self.alpha)
    
        alpha_x_degrees = (angular_coordinate[0]/corner_max_val)*self.alpha
        alpha_x = math.radians(alpha_x_degrees)
    
        alpha_y_degrees = (angular_coordinate[1]/corner_max_val)*self.alpha
        alpha_y = math.radians(alpha_y_degrees)
    
        if abs(alpha_x_degrees) > 90 or abs(alpha_y_degrees) > 90 :
            return None,False
    
    
    
        # Avoid division by zero
        if alpha_x == 0:
            alpha_x += 0.001
        if alpha_y == 0:
            alpha_y += 0.001
    
        fx = abs(math.sin(alpha_max) / math.sin(alpha_x))
        fy = abs(math.sin(alpha_max) / math.sin(alpha_y))
    
        if alpha_x <= 0:
            ax, bx, cx = a_left, b_left, c_left
        else:
            ax, bx, cx = a_right, b_right, c_right
    
        if alpha_y <= 0:
            ay, by, cy = a_top, b_top, c_top
        else:
            ay, by, cy = a_bottom, b_bottom, c_bottom
    
        # Solve the equations for getting [x,y] on the aligned image from the angular
        # x_pow_4_coef = (bx*by*by)/(fy**2)
        # x_pow_3_coef = 0
        # x_pow_2_coef = (bx*2*by*cy)/(fy**2)
        # x_pow_1_coef = -fx
        # x_pow_0_coef = (bx*cy*cy)/(fy**2) + cx
    
        x_pow_4_coef = (ax*ay*ay)/(fy**2)
        x_pow_3_coef = (2*ax*ay*by)/(fy**2)
        x_pow_2_coef = (ax*by*by + 2*ax*ay*cy)/(fy**2) + (bx*ay)/fy
        x_pow_1_coef = -fx + (2*ax*by*cy)/(fy**2) + (bx*by)/fy
        x_pow_0_coef = (ax*cy*cy)/(fy**2) + (bx*cy)/fy + cx
    
        polylon = np.array([x_pow_4_coef, x_pow_3_coef, x_pow_2_coef, x_pow_1_coef, x_pow_0_coef])
    
        polynom_roots_x_vals = np.roots(polylon)
    
        found_valid_xy = False
        valid_x = 0
        valid_y = 0
        valid_xy_radius_sqrd = None
    
        for loop_x_idx in range(len(polynom_roots_x_vals)):
            loop_x = polynom_roots_x_vals[loop_x_idx]
    
            if loop_x.imag == 0:
                loop_y = ((ay/fy)*(loop_x**2) + (by/fy)*loop_x + cy)/fy
    
                x = (loop_x + col_center).real
                y = (loop_y + row_center).real
    
                if not found_valid_xy:
                    found_valid_xy = True
                    valid_x = x
                    valid_y = y
                    valid_xy_radius_sqrd = x**2 + y**2
                else:
                    new_xy_radius_sqrd = x**2 + y**2
                    if new_xy_radius_sqrd < valid_xy_radius_sqrd:
                        valid_x = x
                        valid_y = y
                        valid_xy_radius_sqrd = new_xy_radius_sqrd
    
    
    
        return np.array([valid_x,valid_y]),found_valid_xy

    def calculate_parabolic_coef_for_shifted_angular_mapping(self, center_angle, corner_max_val):
        a = -center_angle / (corner_max_val**2)
        c = center_angle
        b = 1
    
        return a,b,c

    def calculated_shifted_angular_mapped_coordinate(self, original_coordinates, parabolic_x_coef, parabolic_y_coef):
        return np.array(
                [
                    self.calculated_shifted_angular_mapped_single_coordinate(
                        original_coordinates[0],
                        parabolic_x_coef
                        ),
                    self.calculated_shifted_angular_mapped_single_coordinate(
                        original_coordinates[1],
                        parabolic_y_coef,
                        ),
                    ]
                )

    def calculated_shifted_angular_mapped_single_coordinate(self, original_angle, parabolic_coef):
        a,b,c = parabolic_coef
        return a*(original_angle**2) + b*original_angle + c

    def get_trml_coordinates(self, unbounded_coordinates, trml_img_size, round = True):
        trml_row = np.clip(np.round(unbounded_coordinates[1]),0,trml_img_size[0]-1)
        trml_col = np.clip(np.round(unbounded_coordinates[0]),0,trml_img_size[1]-1)
    
        if round:
            trml_row = int(trml_row)
            trml_col = int(trml_col)
        return trml_row, trml_col

    def get_sensitivity_correction_factor(self,angular_coordinates):
        approx_true_angle_x = (angular_coordinates[0]*75)/100
        approx_true_angle_y = (angular_coordinates[1]*110)/100
        angular_diff_distnace = pow((pow(approx_true_angle_x,2) + pow(approx_true_angle_y,2)),0.5)
    
        # Get gaussian value
        gaussian_val = math.exp(-0.5*pow(angular_diff_distnace/self.sens_sigma,2))
    
        sensitivity_correction_factor = 1/gaussian_val
    
        return sensitivity_correction_factor

#################################################################
#
# TEST FUNCTIONALITY
#
#################################################################
def main():
    # -----------------------------------------------------------
    ##### Grab Thermal and RGB Images
    # -----------------------------------------------------------
    img_idx = input("Enter device id: ")
    unit_type = {"1":"hydra_","2":"mosaic_"}[input("Unit Type (1: hydra, 2: hub/mosaic):")]
    basePath = os.path.join("/home/jake/calibration-data/",img_idx)
    rgb_img_path = os.path.join(basePath,"6_inch.png")
    trml_img_path = os.path.join(basePath,"6_inch.npy")
    folder_path = basePath
    outputDir = os.path.join(basePath,'calculated_transformations')

    if unit_type != "hydra_":
        #find camera ID
        with open(os.path.join(basePath,'data.json')) as f:
            data = json.load(f)
            if data['camera_id'] is None:
                print(f"no camera id found for {img_idx}")
            img_idx = data['camera_id']

    # -----------------------------------------------------------
    # Grab points of interest
    # -----------------------------------------------------------
    rgb_coordinates_file_path = folder_path+'/rgb_'+img_idx+'_9element_coord.npy'
    trml_coordinates_file_path = folder_path +'/trml_'+img_idx+'_9element_coord.npy'
    coordinatFiles = os.path.isfile(rgb_coordinates_file_path) and os.path.isfile(trml_coordinates_file_path)
    if not coordinatFiles:
        # Read thermal image
        trml_arr = np.load(trml_img_path)
        # Make sure that thermal image is 2d (single thermal shot) or 3d (a set of thermal shots)
        assert trml_arr.ndim == 2 or trml_arr.ndim == 3
        # In case of set of thermal shots, mean their pixels (along 'time' axe)
        if trml_arr.ndim == 3:
            trml_arr = np.mean(trml_arr, axis=0)
        trml_arr = np.transpose(trml_arr)
        # Scale the thermal image by a factor or 10 (from 32x24 to 320x240) for better smoothness of the heatmap
        trml_matrix_scaled = get_scaled_trml_image_optimized(trml_arr)
        trml_matrix_scaled = np.array((trml_matrix_scaled - np.min(trml_matrix_scaled))/(np.max(trml_matrix_scaled) - np.min(trml_matrix_scaled))*255).astype(np.uint8)
        # Save the scaled image for opening it
        cv2.imwrite('trml_matrix_scaled.jpg', trml_matrix_scaled)
        # Read thermal image
        trml_img = mpimg.imread('trml_matrix_scaled.jpg')
        # Get the 9 (x,y) coordinates of the 9 elements and save them
        trml_elements_coordinates = sample_coordinate_of_corners(trml_img)  # thermal element coordinates
        if trml_elements_coordinates is not None:
            save_sampled_coordinates(trml_elements_coordinates, trml_coordinates_file_path)

        # Read rgb image
        rgb_img = mpimg.imread(rgb_img_path)

        # If image loaded in [0,1] values, convert it to [0,255] uint8 format
        if np.max(rgb_img) <= 1:
            rgb_img = (rgb_img * 255).astype(np.uint8)

        # Get the 9 (x,y) coordinates of the 9 elements and save them
        rgb_elements_coordinates = sample_coordinate_of_corners(rgb_img)  # rgb element coordinates

        if rgb_elements_coordinates is not None:
            save_sampled_coordinates(rgb_elements_coordinates, rgb_coordinates_file_path)
    else:
        trml_elements_coordinates = np.load(trml_coordinates_file_path)
        rgb_elements_coordinates = np.load(rgb_coordinates_file_path)
    # -----------------------------------------------------------
    # Create Cal Files
    # -----------------------------------------------------------
    myCal = Calibrator()
    maskMartix, coordinateMap, sensitivityMatrix = myCal(trml_img_path,rgb_img_path,trml_elements_coordinates,rgb_elements_coordinates)
    logging.debug(maskMartix.shape)
    logging.debug(coordinateMap.shape)
    logging.debug(sensitivityMatrix.shape)
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    if not os.path.exists(outputDir+'/'+img_idx):
        os.mkdir(outputDir+'/'+img_idx)
    np.save(outputDir+'/'+img_idx+'/mapped_coordinates_matrix_'+unit_type+img_idx+'.npy',coordinateMap)
    np.save(outputDir+'/'+img_idx+'/mapped_mask_matrix_'+unit_type+img_idx+'.npy',maskMartix)
    np.save(outputDir+'/'+img_idx+'/sensitivity_correction_matrix_'+unit_type+img_idx+'.npy',sensitivityMatrix)

def save_sampled_coordinates(coord_arr, file_name):
    # Saving the sampled coordinates
    np.save(file_name,coord_arr)

def sample_coordinate_of_corners(img, num_corners = 9, padding = 0):
    # The function plots the image and enable sampling (using mouse clicks) 9 (x,y) values
    # The sampling should be preformed in the following order:
    #   0     1     2
    #
    #   3     4     5
    #
    #   6     7     8
    # The function returns numpy array of size (9,2) for 9 samples of x,y values
    # If the sampling process failed, the function returns None
    print("sample ",num_corners,"in image using ginput")

    if padding > 0:
        if np.ndim(img) == 2:
            img_height,img_width = np.shape(img)
            padded_img = np.zeros((img_height+2*padding, img_width + 2*padding)).astype(np.uint8)
            padded_img[padding:padding+img_height, padding:padding+img_width] = img[:,:]
        elif np.ndim(img) == 3:
            img_height, img_width, n_cnl = np.shape(img)
            padded_img = np.zeros((img_height + 2 * padding, img_width + 2 * padding, n_cnl)).astype(np.uint8)
            padded_img[padding:padding + img_height, padding:padding + img_width,:] = img[:, :, :]
        else:
            logging.warning("Can't pad image, ndim not standard")
            return None
        img = padded_img

    plt.imshow(img, cmap='gray')
    samples_coord = plt.ginput(num_corners,timeout=90)
    plt.show()

    if len(samples_coord) == num_corners:
        # Convert to np array
        coord_arr = np.zeros((num_corners, 2))
        for i in range(num_corners):
            coord_arr[i][0] = samples_coord[i][0] - padding
            coord_arr[i][1] = samples_coord[i][1] - padding

        return coord_arr

    else:
        logging.warning("Failed to sample "+str(num_corners)+" coordinates")
        return None

if __name__ == '__main__':
    main()
