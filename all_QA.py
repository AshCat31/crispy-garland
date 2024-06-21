import getCal as get
import paralax_calibrator as cal
import putCal as put
import paralax_check as check
from auto_calibration_new.automatic_rgb_thermal_mapping import do_automatic_rgb_calibration_mapping as darcm
import QA_Check_Auto as qa

def main():
    # print("Getting files...")
    # get.main()
    # print("Calibrating...")
    # cal.main()
    # print("Uploading files...")
    # put.main()
    print("Auto calibrating...")
    with open("/home/canyon/Test_Equipment/QA_ids.txt", "r") as id_file:
        for line in id_file:
            device = line.split()
            success = darcm(device[0])
            if not success:
                print("Auto failed, doing manual")
                get.download_device(device)
                cal.cal_device(device)
                put.upload_device(device)
    print("Reviewing parallax...")
    check.main()
    print("Performing QA...")
    qa.main()

if __name__ == "__main__":
    main()
