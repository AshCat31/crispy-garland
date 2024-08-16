import time
import os

i = 0
while True:
    os.system(
        f"curl http://192.168.1.178/api/v1/gateways/178/mosaics/178/hydras/E66138528316872D/color_image -o /home/canyon/Test_Equipment/crispy-garland/mag_test_imgs/color{i}.jpeg"
    )
    i += 1
    time.sleep(120)
