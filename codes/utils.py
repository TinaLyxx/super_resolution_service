
from diffusers import AutoencoderKL
from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
import torch, gc
from torchvision import transforms
from PIL import Image
import os
import json
import boto3

PREVIEW_EXT = ["png", "jpeg", "jpg"]
GEOTIFF_EXT = ["tiff", "tif", "geotiff"]

def load_and_process_image(pil_image):
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transform(pil_image)
    image = image.unsqueeze(0).half()
    return image


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        pad_w = 0
        pad_h = (w - h) // 2
        new_image.paste(image, (0, pad_h))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        pad_w = (h - w) // 2
        pad_h = 0
        new_image.paste(image, (pad_w, 0))
        return new_image

def generate_images(prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed, image_path):
    input_image = Image.open(image_path)
    w, h = input_image.size
    padded_image = pad_image(input_image).resize((1024, 1024)).convert("RGB")
    image_lr = load_and_process_image(padded_image).to('cuda')
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = DemoFusionSDXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16)
    print("Finish pretrained parameters loading!")
    pipe = pipe.to("cuda")
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(int(seed))
    images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                  height=int(height), width=int(width), view_batch_size=int(view_batch_size), stride=int(stride),
                  num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                  cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3, sigma=sigma,
                  multi_decoder=True, show_image=False, lowvram=False, image_lr=image_lr
                 )    
    images_path = list()
    for i, image in enumerate(images):
        image_path = './tmp/image_'+str(i)+'.png' 
        images_path.append(image_path)
        height, width = image.size
        if w < h:
            resize_w = int(w * (height / h))
            edge = (width - resize_w) // 2
            crop_area = (edge, 0, width-edge, height)
            cropped_image = image.crop(crop_area)
        elif w > h:
            resize_h = int(h * (width / w))
            edge = (height - resize_h) // 2
            crop_area = (0, edge, width, height-edge)
            cropped_image = image.crop(crop_area)
        else:
            cropped_image = image
        cropped_image.save(image_path)
    pipe = None
    gc.collect()
    torch.cuda.empty_cache()
    return images_path


def download_from_s3(file_path, bucket, key, region):
    # if file_path exists, no need to download
    if os.path.exists(file_path):
        print("{} exists already".format(file_path))
        return
    s3 = boto3.client("s3", region_name=region)

    print("Downloading s3://{}/{} to {}...".format(bucket, key, file_path))
    try:
        s3.download_file(bucket, key, file_path)
        print("S3 download successful! \n")
        return
    except Exception as e:
        error = "an error occurred:{}".format(e)
        return error



def upload_to_s3(file_path, bucket, key, region):
    s3 = boto3.client("s3", region_name=region)
    _extension = file_path.split(".")[-1]
    if _extension == "png":
        content_type = "image/png"
    elif _extension in ["jpeg", "jpg"]:
        content_type = "image/jpeg"
    else:
        content_type = "image/tiff"
    print("Uploading to s3://{}/{}...".format(bucket, key))
    try:
        s3.upload_file(file_path, bucket, key, ExtraArgs={"ContentType": content_type})
        print("S3 upload successful! \n")
        return
    except Exception as e:
        error = "an error occurred:{}".format(e)
        return error




def process_input(data):
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")

    if isinstance(data, str):
        model_input = json.loads(data)

    print("Body:", model_input)

    prompt = model_input["prompt"]
    negative_prompt = model_input["negative_prompt"]
    width = model_input["width"]
    height = model_input["height"]
    num_inference_steps = model_input["num_inference_steps"]
    guidance_scale = model_input["guidance_scale"]
    cosine_scale_1 = model_input["cosine_scale_1"]
    cosine_scale_2 = model_input["cosine_scale_2"]
    cosine_scale_3 = model_input["cosine_scale_3"]
    sigma = model_input["sigma"]
    view_batch_size = model_input["view_batch_size"]
    stride = model_input["stride"]
    seed = model_input["seed"]
    bucket = model_input["bucket"]
    key = model_input["key"]
    region = model_input["region"]


    filename = key.split("/")[-1]
    local_path ="./tmp/"+ filename
    download_from_s3(local_path, bucket, key, region)

    images_path = generate_images(prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed, local_path)

    return images_path, model_input


def process_output(model_input, images_path):
    response = {}
    response["predictions"] = []
    bucket = model_input["bucket"]
    region = model_input["region"]
    for image_path in images_path:
        image_name = image_path.split("/")[-1]
        key = 'results/' + image_name
        message = upload_to_s3(image_path, bucket, key, region)
        if message is not None:
            return message
        single_response = {
            "image_s3_path" : {
                "bucket" : bucket,
                "region" : region,
                "key" : key,

            },
        }
        response["predictions"].append(single_response)
    return response


