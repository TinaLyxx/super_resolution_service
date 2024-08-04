import boto3
import json
import sagemaker
from sagemaker import Model

sess = sagemaker.Session()

# Specify your AWS Region
aws_region='us-west-2'

# Role to give SageMaker permission to access AWS services.
sagemaker_role= "arn:aws:iam::573944535954:role/Super_Resolution"

ecr_image = "573944535954.dkr.ecr.us-west-2.amazonaws.com/super-resolution:latest"
model_name = "superresolution-demofusion"
instance_type = "ml.g6.xlarge"
endpoint_name = 'superresolution-demofusion' 

estimator = Model(
    name=model_name,
    image_uri=ecr_image,
    role=sagemaker_role,
    source_dir="/opt/ml/code",
    entry_point="inference.py",
    sagemaker_session=sess
)

predictor = estimator.deploy(1, instance_type, endpoint_name=endpoint_name)

sm_client = sess.sagemaker_runtime_client

payload = { 
    "prompt": "a satellite image",
    "negative_prompt": "blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
    "width": 2048,
    "height": 2048,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "cosine_scale_1": 3,
    "cosine_scale_2": 1,
    "cosine_scale_3": 1,
    "sigma": 0.8,
    "view_batch_size": 16,
    "stride": 64,
    "seed": 2013,
    "bucket": "test-aws-mybucket",
    "region": "us-west-2",
    "key": "data/sample.png",
}
response = sm_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload),
)

r = response["Body"]
print("RESULT r.read().decode():", r.read().decode())

