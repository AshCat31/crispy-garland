import json
import logging

from s3_setup import S3Setup


def main(s3c=None, bkn=None):
    logger = logging.getLogger(__name__)
    log_format = "%(levelname)-6s: %(message)s"
    logging.addLevelName(
        logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING)
    )
    logging.addLevelName(
        logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR)
    )
    logging.basicConfig(level=logging.WARN, format=log_format)

    if s3c is None:
        s3c = S3Setup()
        s3c, _bucket_name = s3c()
    else:
        s3client, _bucket_name = s3c, bkn
    with open("QA_ids.txt", "r") as file:
        for line in file:
            values = line.split()
            device_id = values[0]
            try:
                json_response = s3client.get_object(
                    Bucket=_bucket_name, Key=f"{device_id}/data.json"
                )
                json_file_content = (
                    json_response["Body"].read().decode("utf-8")
                )  # downloading the json
                data_content = json.loads(json_file_content)
                key = "serial_number"
                print(f"{device_id}'s current value for {key} is {data_content[key]}")
                new_value = input("New value, or enter to keep existing:\n")
                if new_value:
                    data_content[key] = new_value
                    updated_json_content = json.dumps(data_content)
                    s3client.put_object(
                        Bucket=_bucket_name,
                        Key=f"{device_id}/data.json",
                        Body=updated_json_content.encode("utf-8"),
                    )
                    print(f"New value for {key} has been saved to S3.")
                else:
                    print("Keeping existing value for", key)
            except Exception as e:
                print(device_id, e)


if __name__ == "__main__":
    main()
