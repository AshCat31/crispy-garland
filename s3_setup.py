import boto3


def setup_s3():
    cred = boto3.Session().get_credentials()
    ACCESS_KEY = cred.access_key 
    SECRET_KEY = cred.secret_key 
    SESSION_TOKEN = cred.token 

    s3client = boto3.client('s3',
                            aws_access_key_id = ACCESS_KEY,
                            aws_secret_access_key = SECRET_KEY,
                            aws_session_token = SESSION_TOKEN,
                            )
    bucket_name = 'kcam-calibration-data'
    return s3client, bucket_name