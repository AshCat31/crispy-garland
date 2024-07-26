__author__ = 'Delta Thermal Inc.'
__copyright__ = """
    Copyright 2018-2023 Delta Thermal Inc.

    All Rights Reserved.
    Covered by one or more of the Following US Patent Nos. 10,991,217,
    Other Patents Pending.
"""

import os
import statistics

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PImage
from s3_setup import S3Setup


def main():
    device_list = []
    failures = []
    doc_path = '/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt'
    s3c = S3Setup()
    s3client, bucket_name = s3c()
    show_plot = True
    with open(doc_path, 'r') as file:
        for line in file:
            device_list.append(line.split()[0])
    for i, device_id in enumerate(device_list):
        rc = ROIChecker(device_id, i, s3client, bucket_name, failures, show_plot)
        rc.start()

    # if len(failures) > 0:
    # print("Avg failures:", statistics.mean(failures))
    if show_plot:
        counts, edges, bars = plt.hist(failures, bins=15, edgecolor='black')
        plt.bar_label(bars)
        # plt.show()

        
class ROIChecker:
    def __init__(self, device_id, i, s3client, bucket_name, failures, show_plot=False) -> None:
        self.s3client = s3client
        self.bucket_name = bucket_name
        self.device_id = device_id
        self.show_plot = show_plot
        self.failures = failures
        self.i = i
        self.x_trans = self.y_trans = 0

        hub_base_image = ['/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_one.jpeg',]
        hub_rois = [[
            '/home/canyon/Test_Equipment/hub_alignment_test/roi1.npy',  # 10000000eed77a0e
            '/home/canyon/Test_Equipment/hub_alignment_test/roi3.npy',  # 1000000000eb7857
            '/home/canyon/Test_Equipment/hub_alignment_test/roi4.npy',  # 10000000011e44c9
            '/home/canyon/Test_Equipment/hub_alignment_test/roi5.npy',  # 1000000002b88c87
            '/home/canyon/Test_Equipment/hub_alignment_test/roi7.npy',  # 1000000003676bf1
            '/home/canyon/Test_Equipment/hub_alignment_test/roi8.npy',  # 10000000037c5199
            '/home/canyon/Test_Equipment/hub_alignment_test/roi9.npy',  # 1000000003a42eec
        ]]
        head_base_image = [
            '/home/canyon/Test_Equipment/head_alignment_test/port0_one.jpeg',
            '/home/canyon/Test_Equipment/head_alignment_test/port1_one.jpeg',
            '/home/canyon/Test_Equipment/head_alignment_test/port2_one.jpeg',
        ]
        head_rois = [
            ['/home/canyon/Test_Equipment/head_alignment_test/auto_port0_one.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port0_two.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port0_three.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port0_four.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port0_five.npy',
            ],
            ['/home/canyon/Test_Equipment/head_alignment_test/auto_port1_one.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port1_two.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port1_three.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port1_four.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port1_five.npy',
            ],
            ['/home/canyon/Test_Equipment/head_alignment_test/auto_port2_one.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port2_two.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port2_three.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port2_four.npy',
            '/home/canyon/Test_Equipment/head_alignment_test/auto_port2_five.npy',
            ],
        ]
        self.WIDTH, self.HEIGHT = 440, 520
        self.device_type_dict = {"100": ("_mosaic", hub_base_image, hub_rois, 1), "E66": ("_hydra", head_base_image, head_rois, 3)}
        self.device_type, self.base_image, self.roi_files, self.num_ports = self.device_type_dict[self.device_id[:3]]
        self.failures = []
        self.device_rois = []
        self.images = []
        for port in range(self.num_ports):
            if self.show_plot:
                self.images.append(self.pad_image(PImage.open(self.base_image[port])))
            port_rois = []
            for roifile in self.roi_files[port]:
                rois = np.load(roifile)
                port_rois.append(rois)
            self.device_rois.append(port_rois)

    def start(self):
        try:
            self.get_mask("2")
        except Exception as e:
            try:
                self.get_mask("")
            except Exception as e2:
                print(e2, self.device_id)
                return
        self.initialize_plot()
        self.x_trans = self.y_trans = 0
        if self.show_plot:
            for port in range(self.num_ports):
                self.axs[port].clear()

    def get_mask(self, ct):
        key = f'{self.device_id}/calculated_transformations{ct}/{self.device_id}/mapped_mask_matrix{self.device_type}_{self.device_id}.npy'
        local_directory = '/home/canyon/S3bucket/'
        try:
            self.mask_map = np.load(os.path.join(local_directory, key)).astype(np.uint8) * 255
        except FileNotFoundError:
            os.makedirs(os.path.join(local_directory, f'{self.device_id}/calculated_transformations{ct}/{self.device_id}'))
            self.s3client.download_file(Bucket=self.bucket_name, Key=key, Filename=os.path.join(local_directory, key))
            self.mask_map = np.load(os.path.join(local_directory, key)).astype(np.uint8) * 255
        self.mask_edges_contours, _ = cv2.findContours(cv2.Canny(self.mask_map, 30, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def sliders_on_changed(self, _):
        self.x_trans = self.x_slider.val
        self.y_trans = self.y_slider.val
        self.check_rois()
        self.update_plot()

    def roi_pass(self, roi):
        roi_x, roi_y = self.WIDTH + self.x_trans - roi[:, :, 0], self.HEIGHT - self.y_trans - roi[:, :, 1]
        masked_roi_check = self.mask_map[np.uint16(roi_y), np.uint16(roi_x)]
        return np.count_nonzero(masked_roi_check) >= 50
    
    def pad_image(self, rgb_img):
        rgb_img = np.asarray(rgb_img.rotate(180))
        img_padded = np.zeros((520, 440)).astype(np.uint8)
        if self.device_type == '_mosaic':
            rgb_img = rgb_img[:,:,0]*1.3
        img_padded[100 - self.y_trans:420 - self.y_trans, 100 + self.x_trans:340 + self.x_trans] = rgb_img
        return img_padded

    def check_rois(self):
        self.roi_color = []
        fail_ct = 0
        for port in range(self.num_ports):                
            cidx = 0
            for rois in self.device_rois[port]:
                for roi in rois:
                    roi_x, roi_y = self.WIDTH + self.x_trans - roi[:, :, 0], self.HEIGHT - self.y_trans - roi[:, :, 1]
                    if self.roi_pass(roi):
                        self.roi_color.append((-1, roi_x, roi_y))
                    else:
                        self.roi_color.append((cidx % 8, roi_x, roi_y))
                        fail_ct += 1
                cidx += 1
        print(fail_ct)
        self.failures.append(fail_ct)


    def update_plot(self):
        for port in range(self.num_ports):
            self.axs[port].clear()
            rgb_img = self.images[port]
            cv2.drawContours(rgb_img, self.mask_edges_contours, -1, (255, 255, 255), 1)

            self.axs[port].plot(self.therm_x, self.therm_y, 'o', markersize=10, color='magenta', zorder=8888)
            self.axs[port].plot([self.rgb_cen[0], self.therm_x], [self.rgb_cen[1], self.therm_y], markersize=5, color='red', zorder=9999)
            self.axs[port].plot(self.rgb_cen[0] + self.x_trans, self.rgb_cen[1] - self.y_trans, 'o', markersize=10, color='orange', zorder=8888)

            self.axs[port].set_title("Port " + str(port))
            self.axs[port].imshow(rgb_img, cmap='gray')
            self.axs[port].axis('off')
            for roi in self.roi_color:
                self.axs[port].plot(roi[1], roi[2], 'o', color=self.colors[roi[0]], markersize=5, zorder=roi[0]+5)
        self.fig.canvas.draw_idle()

    def initialize_plot(self):
        self.x_trans = self.y_trans = 0
        self.therm_x = statistics.mean(np.nonzero(self.mask_map)[1])
        self.therm_y = statistics.mean(np.nonzero(self.mask_map)[0])
        self.rgb_cen = (220, 260)
        self.x_trans = self.therm_x-self.rgb_cen[0]  # to center rgb with thermal
        self.y_trans = self.rgb_cen[1]-self.therm_y  # "
        self.check_rois()

        if self.show_plot:
            self.colors = ['brown', 'cyan', 'magenta', 'blue', 'green', 'yellow', 'orange', 'red', 'lightgrey']
            self.fig, self.axs = plt.subplots(1, self.num_ports)
            self.fig.suptitle(str(self.i + 1) + ": " + self.device_id)
            if self.device_type != '_hydra':
                    self.axs = [self.axs]
            axis_color = "grey"
            slider_min, slider_max = -50, 50
            x_slider_ax = self.fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor=axis_color)
            self.x_slider = plt.Slider(x_slider_ax, 'x', slider_min, slider_max, valinit=self.x_trans, valstep=1)
            y_slider_ax = self.fig.add_axes([0.02, 0.15, 0.03, 0.65], facecolor=axis_color)
            self.y_slider = plt.Slider(y_slider_ax, 'y', slider_min, slider_max, valinit=self.y_trans, valstep=1,
                                orientation="vertical")
            self.x_slider.on_changed(self.sliders_on_changed)
            self.y_slider.on_changed(self.sliders_on_changed)
            self.update_plot()


        # Update with initial data

        # if self.show_plot:
            self.fig.canvas.manager.window.wm_geometry("+0+0")
            self.fig.canvas.manager.window.geometry("1910x1000")
            plt.subplots_adjust(wspace=0, hspace=0.01, bottom=0, top=0.95, left=.05, right=.98)
            plt.show()


if __name__ == "__main__":
    main()
