import boto3
import json


# Specify your AWS Region
aws_region='us-west-2'

# Create a low-level SageMaker service client.
sagemaker_client = boto3.client('sagemaker', region_name=aws_region)

# Role to give SageMaker permission to access AWS services.
sagemaker_role= "IAM: arn:aws:iam::573944535954:role/Super_Resolution"

ecr_image = ""
model_name = "superresolution-demofusion"

create_model_response = sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': ecr_image,
        'Environment': {
            'SAGEMAKER_PROGRAM': 'inference.py',
            'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
        }
    },
    ExecutionRoleArn = sagemaker_role
)

# The name of the endpoint configuration associated with this endpoint.
endpoint_config_name = "superresolution-demofusion-config"

create_endpoint_config_response = sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "variant1", # The name of the production variant.
            "ModelName": 'DemoFusion', 
            "InstanceType": "ml.g6.xlarge", # Specify the compute instance type.
            "InitialInstanceCount": 1 # Number of instances to launch initially.
        }
    ],
    AsyncInferenceConfig={
        "OutputConfig": {
            # Location to upload response outputs when no location is provided in the request.
            "S3OutputPath": "s3://test-aws-mybucket/results/"
            },        
        }
)

# The name of the endpoint.The name must be unique within an AWS Region in your AWS account.
endpoint_name = 'superresolution-demofusion' 



create_endpoint_response = sagemaker_client.create_endpoint(
                                            EndpointName=endpoint_name, 
                                            EndpointConfigName=endpoint_config_name) 


# Create a low-level client representing Amazon SageMaker Runtime
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=aws_region)

# Specify the location of the input. Here, a single SVM sample
input_location = "s3://test-aws-mybucket/data/"


# After you deploy a model into production using SageMaker hosting 
# services, your client applications use this API to get inferences 
# from the model hosted at the specified endpoint.
response = sagemaker_runtime.invoke_endpoint_async(
                            EndpointName=endpoint_name, 
                            InputLocation=input_location,
                            InvocationTimeoutSeconds=3600)

