__author__ = 'Delta Thermal Inc.'
__copyright__ = """
    Copyright 2018-2023 Delta Thermal Inc.

    All Rights Reserved.
    Covered by one or more of the Following US Patent Nos. 10,991,217,
    Other Patents Pending.
"""

import io
import boto3

import cv2
from PIL import Image as PImage
import matplotlib.pyplot as plt
import numpy as np
import statistics
import os

def main():
    device_list = []
    doc_path = '/home/canyon/Test_Equipment/QA_ids.txt'
    with open(doc_path, 'r') as file:
        for line in file:
            device_list.append(line.split()[0])

    cred = boto3.Session().get_credentials()
    ACCESS_KEY = cred.access_key 
    SECRET_KEY = cred.secret_key 
    SESSION_TOKEN = cred.token 
    global s3client
    s3client = boto3.client('s3',
                            aws_access_key_id = ACCESS_KEY,
                            aws_secret_access_key = SECRET_KEY,
                            aws_session_token = SESSION_TOKEN,
                            )
    global axs, x_trans, y_trans, base_image, device_rois, mask_edges_contours, xy_adjustments, show_plot, failures, device_type
    x_trans = y_trans = 0
    bucket_name = 'kcam-calibration-data'

    hub_base_image = [[ '/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_one.jpeg',
                        '/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_two.jpeg',
                        '/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_three.jpeg',
                        '/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_four.jpeg',
                        '/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_five.jpeg',
                        '/home/canyon/Test_Equipment/hub_alignment_test/breaker9_10_six.jpeg']]
    hub_rois = [ [
                '/home/canyon/Test_Equipment/hub_alignment_test/roi1.npy',  # 10000000eed77a0e
                # '/home/canyon/Test_Equipment/hub_alignment_test/roi2.npy',  # 100000006e0c4dcf  # doesn't have rois??
                '/home/canyon/Test_Equipment/hub_alignment_test/roi3.npy',  # 1000000000eb7857
                '/home/canyon/Test_Equipment/hub_alignment_test/roi4.npy',  # 10000000011e44c9
                '/home/canyon/Test_Equipment/hub_alignment_test/roi5.npy',  # 1000000002b88c87
                # '/home/canyon/Test_Equipment/hub_alignment_test/roi6.npy',  # 1000000002be265d  # more of the random dots
                '/home/canyon/Test_Equipment/hub_alignment_test/roi7.npy',  # 1000000003676bf1
                '/home/canyon/Test_Equipment/hub_alignment_test/roi8.npy',  # 10000000037c5199
                '/home/canyon/Test_Equipment/hub_alignment_test/roi9.npy',  # 1000000003a42eec
                # '/home/canyon/Test_Equipment/hub_alignment_test/roi10.npy',
                ]]
    head_base_image = [ 
                        ['/home/canyon/Test_Equipment/head_alignment_test/port0_one.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port0_two.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port0_three.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port0_four.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port0_five.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port0_six.jpeg',],
                        ['/home/canyon/Test_Equipment/head_alignment_test/port1_one.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port1_two.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port1_three.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port1_four.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port1_five.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port1_six.jpeg',],
                        ['/home/canyon/Test_Equipment/head_alignment_test/port2_one.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port2_two.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port2_three.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port2_four.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port2_five.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port2_six.jpeg',],
                        ['/home/canyon/Test_Equipment/head_alignment_test/port3_one.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port3_two.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port3_three.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port3_four.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port3_five.jpeg',
                        '/home/canyon/Test_Equipment/head_alignment_test/port3_six.jpeg',]
                        ]
    head_rois = [  # automatic, #29
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
       ['/home/canyon/Test_Equipment/head_alignment_test/auto_port3_one.npy',
        '/home/canyon/Test_Equipment/head_alignment_test/auto_port3_two.npy',
        '/home/canyon/Test_Equipment/head_alignment_test/auto_port3_three.npy',
        '/home/canyon/Test_Equipment/head_alignment_test/auto_port3_four.npy',
        '/home/canyon/Test_Equipment/head_alignment_test/auto_port3_five.npy',
        ],
    ]
    # head_rois = [
    #     ['/home/canyon/Test_Equipment/head_alignment_test/port0_one.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port0_two.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port0_three.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port0_four.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port0_five.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port0_six.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port0_seven.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port0_eight.npy',],
    #     ['/home/canyon/Test_Equipment/head_alignment_test/port1_one.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port1_two.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port1_three.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port1_four.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port1_five.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port1_six.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port1_seven.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port1_eight.npy',],
    #     ['/home/canyon/Test_Equipment/head_alignment_test/port2_one.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port2_two.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port2_three.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port2_four.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port2_five.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port2_six.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port2_seven.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port2_eight.npy',],
    #     ['/home/canyon/Test_Equipment/head_alignment_test/port3_one.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port3_two.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port3_three.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port3_four.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port3_five.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port3_six.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port3_seven.npy',
    #     '/home/canyon/Test_Equipment/head_alignment_test/port3_eight.npy'],
    # ]
    show_plot = True
    device_type_dict = {"100": ("_mosaic", hub_base_image, hub_rois), "E66": ("_hydra", head_base_image, head_rois)}
    xy_adjustments = []
    failures = []
    for i, device_id in enumerate(device_list):
        device_type, base_image, device_rois = device_type_dict[device_id[:3]]
        try:
            mask_edges_contours = get_mask("2", device_id, device_type, bucket_name)
        except Exception as e:
            try:
                mask_edges_contours = get_mask("", device_id, device_type, bucket_name)
            except Exception as e2:
                print(e2, device_id)
                continue
        initialize_plot(device_id, i)
        xy_adjustments.append([x_trans, y_trans])
        x_trans = y_trans = 0
        if show_plot:
            for port in range(num_ports):
                axs[port].clear()  # Clear previous plot  
    if len(failures) >0:
        print("Avg failures:", statistics.mean(failures))
        counts, edges, bars = plt.hist(failures,bins=15,edgecolor='black')
        plt.bar_label(bars)
        plt.show()
    # print("mean x:", statistics.mean(np.asarray(xy_adjustments)[:, 0]),"mean y:", statistics.mean(np.asarray(xy_adjustments)[:, 1]))
    # print("SD x:", statistics.stdev(np.asarray(xy_adjustments)[:, 0]),"SD y:", statistics.stdev(np.asarray(xy_adjustments)[:, 1]))

def get_mask(ct, device_id, device_type, bucket_name):
    global mask_map
    key = f'{device_id}/calculated_transformations{ct}/{device_id}/mapped_mask_matrix{device_type}_{device_id}.npy'
    local_directory='/home/canyon/S3bucket/'
    # mask_edges_contours = s3client.get_object(Bucket=bucket_name, Key=key)
    # mask_bytes = io.BytesIO(mask_edges_contours["Body"].read())
    # mask_bytes.seek(0)
    try:
        mask_map = np.load(os.path.join(local_directory, key)).astype(np.uint8) * 255
    except:  # currently not working?? now?
        try:
            os.mkdir(os.path.join(local_directory, f'{device_id}'))
            os.mkdir(os.path.join(local_directory, f'{device_id}/calculated_transformations{ct}'))
            os.mkdir(os.path.join(local_directory, f'{device_id}/calculated_transformations{ct}/{device_id}'))
        except FileExistsError:
            try:
                os.mkdir(os.path.join(local_directory, f'{device_id}/calculated_transformations{ct}'))
                os.mkdir(os.path.join(local_directory, f'{device_id}/calculated_transformations{ct}/{device_id}'))
            except FileExistsError:
                try:
                    os.mkdir(os.path.join(local_directory, f'{device_id}/calculated_transformations{ct}/{device_id}'))
                except:
                    pass
        s3client.download_file(Bucket=bucket_name,Key=key, 
                               Filename=os.path.join(local_directory, key))
    mask_map = np.load(os.path.join(local_directory, key)).astype(np.uint8) * 255
    mask_edges = cv2.Canny(mask_map, 30, 200)
    mask_edges_contours, _ = cv2.findContours(mask_edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return mask_edges_contours

def sliders_on_changed(val):
    x_trans = x_slider.val
    y_trans = y_slider.val
    update_plot()

def roi_pass(roi):
    global x_trans, y_trans
    width = 440
    height = 520
    is_passing = True
    if device_type == '_hydra':
        roi_x, roi_y = width + x_trans - roi[:,:,0], height - y_trans - roi[:,:,1]
    else:
        # roi_x, roi_y = roi[:,:,0],  roi[:,:,1]
        roi_x, roi_y = width + x_trans - roi[:,:,0], height - y_trans - roi[:,:,1]
    masked_roi_check = mask_map[ np.uint16(roi_y),  np.uint16(roi_x)]
    if np.count_nonzero(masked_roi_check) < 50:
        is_passing = False
    return is_passing

def update_plot():
    global x_trans, y_trans
    rgb_cen=(220,260)
    width = 440
    height = 520
    fail_ct = 0
    therm_x = statistics.mean(np.nonzero(mask_map)[1])
    therm_y = statistics.mean(np.nonzero(mask_map)[0])
    # x_trans = therm_x-rgb_cen[0]  # to center rgb with thermal
    # y_trans = rgb_cen[1]-therm_y
    for port in range(num_ports):
        if show_plot:
            axs[port].clear()
        rgb_img = PImage.open(base_image[port][0])
        # Plot ROIs with new translation
        colors = ['brown', 'cyan', 'magenta', 'blue', 'green', 'yellow', 'orange', 'red']
        cidx = 0
        for roifile in device_rois[port]:
            rois = np.load(roifile)
            for indx, roi in enumerate(rois):
                if device_type == '_hydra':
                    roi_x, roi_y = width + x_trans - roi[:,:,0], height - y_trans - roi[:,:,1]
                else:
                    # roi_x, roi_y = roi[:,:,0],  roi[:,:,1]
                    roi_x, roi_y = width + x_trans - roi[:,:,0], height - y_trans - roi[:,:,1]
                if roi_pass(roi):
                    if show_plot:
                        axs[port].plot(roi_x, roi_y, 'o', color="lightgrey", markersize=5)
                    # axs[port].plot(roi[:,:,0],  roi[:,:,1], 'o', color=colors[cidx%8], markersize=5, zorder=9999)
                    # axs[port].plot(width + x_trans - roi[:,:,0], height - y_trans - roi[:,:,1], 'o', color=colors[cidx%8], markersize=5, zorder=9999)
                else:
                    # axs[port].plot(width + x_trans - roi[:,:,0], height - y_trans - roi[:,:,1], 'o', color=colors[cidx%8], markersize=5, zorder=9999)
                    if show_plot:
                        axs[port].plot(roi_x, roi_y, 'o', color=colors[cidx%8], markersize=5, zorder=9999)
                        print("fail:", port, " file ".join(roifile.split('/')[-1].split('_')[-2:]), 'roi', indx)
                    fail_ct +=1

            cidx += 1
        if show_plot:
            axs[port].plot([rgb_cen[0], therm_x], [rgb_cen[1], therm_y], markersize=5, color='black', zorder=9999)
        
        # Draw contours
        img_padded = np.zeros((520, 440)).astype(np.uint8)
        if device_type == '_mosaic':
            img_padded[100-y_trans:420-y_trans, 100+x_trans:340+x_trans] = np.asarray(rgb_img.rotate(180))[:, :, 0]
            # img_padded[100:420, 100:340] = np.asarray(rgb_img)[:, :, 0]
        else:
            img_padded[100-y_trans:420-y_trans, 100+x_trans:340+x_trans] = np.asarray(rgb_img.rotate(180))
        rgb_img = img_padded
        if show_plot:
            cv2.drawContours(rgb_img, mask_edges_contours, -1, (255, 255, 255), 1)

            # axs[port].plot(440-therm_x, 520-therm_y, 'o', markersize=10, color='yellow', zorder=8888)
            axs[port].plot(therm_x, therm_y,'o', markersize=10, color='magenta', zorder=8888)
            axs[port].plot(rgb_cen[0]+x_trans,rgb_cen[1]-y_trans,'o', markersize=10, color='orange', zorder=8888)
            # print(440-therm_x-statistics.mean(np.nonzero(mask_map)[1]), 520-therm_y-statistics.mean(np.nonzero(mask_map)[0]))
            # print(statistics.mean(np.nonzero(mask_map)[1]), statistics.mean(np.nonzero(mask_map)[0]),)

            # Set plot properties
            axs[port].set_title("Port " + str(port))
            axs[port].imshow(rgb_img, cmap='gray')
            # axs[port].imshow(rgb_img[20:500,20:420], cmap='gray')
            axs[port].axis('off')  

    print(fail_ct)
    failures.append(fail_ct)
    if show_plot:
        fig.canvas.draw_idle()

def initialize_plot(device_id, i):
    global fig, axs, x_slider, y_slider, num_ports, x_trans, y_trans
    x_trans = y_trans = 0
    if device_type == '_hydra':
        num_ports = 3
        if show_plot:
            fig, axs = plt.subplots(1, num_ports)
            fig.suptitle(str(i+1) + ": " + device_id)
    else:
        num_ports = 1
        if show_plot:
            fig, axs = plt.subplots(1, num_ports)
            fig.suptitle(str(i+1) + ": " + device_id)
            axs = [axs]
    # Create sliders
    if show_plot:
        axis_color = "grey"
        slider_min = -50
        slider_max = 50
        x_slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03], facecolor=axis_color)
        x_slider = plt.Slider(x_slider_ax, 'x', slider_min, slider_max, valinit=x_trans, valstep=1)
        y_slider_ax = fig.add_axes([0.02, 0.15, 0.03, 0.65], facecolor=axis_color)
        y_slider = plt.Slider(y_slider_ax, 'y', slider_min, slider_max, valinit=y_trans, valstep=1, orientation="vertical")
        x_slider.on_changed(sliders_on_changed)
        y_slider.on_changed(sliders_on_changed)
    
    # Initialize the plot with initial data
    update_plot()

    if show_plot:
        fig.canvas.manager.window.wm_geometry("+0+0")
        fig.canvas.manager.window.geometry("1910x1000")
        plt.subplots_adjust(wspace=0, hspace=0.01, bottom=0, top=0.95, left=.05, right=.98)
        plt.show()



if __name__ == "__main__":
    main()