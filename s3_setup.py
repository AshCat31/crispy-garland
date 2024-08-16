import boto3


class S3Setup:
    """Returns s3client, bucket_name"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cred = boto3.Session().get_credentials()
            ACCESS_KEY = cred.access_key
            SECRET_KEY = cred.secret_key
            SESSION_TOKEN = cred.token

            cls._instance.s3client = boto3.client(
                "s3",
                aws_access_key_id=ACCESS_KEY,
                aws_secret_access_key=SECRET_KEY,
                aws_session_token=SESSION_TOKEN,
            )
            cls._instance.bucket_name = "kcam-calibration-data"
        return cls._instance

    def __call__(self):
        return self.s3client, self.bucket_name
