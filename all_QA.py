import QA_Check_Auto as qa
import getCal as get
import paralax_calibrator as cal
import paralax_check as check
import putCal as put
from auto_calibration_new.automatic_rgb_thermal_mapping import Mapper

def main():
    print("Auto calibrating...")
    with open("/home/canyon/Test_Equipment/crispy-garland/QA_ids.txt", "r") as id_file:
        for line in id_file:
            device = line.split("    ")
            dev_id = device[0]
            mp = Mapper()
            success = mp.do_automatic_rgb_calibration_mapping(device[0])
            if not success:
                print("Auto failed, doing manual")
                get.download_device(dev_id)
                cal.cal_device(dev_id)
                put.upload_device(dev_id)
    print("Reviewing parallax...")
    check.main()
    print("Performing QA...")
    qa.main()


if __name__ == "__main__":
    main()
