# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import json
import os
import flask
import psutil
import GPUtil
import logging
import time
import threading

from utils import*

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

def ping_test():
    test = "pass!"
    return test

# The flask app for serving predictions
app = flask.Flask(__name__)

def monitor_memory_usage(interval=1):
    try:
        while True:
            cpu_usage = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            print(f"CPU Usage: {cpu_usage}%, Total memory: {mem.total / (1024**3):.2f} GB, Used memory: {mem.used / (1024**3):.2f} GB, Memory usage: {mem.percent}%")
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"GPU {gpu.id}: Name={gpu.name}, Total={gpu.memoryTotal} MB, Used={gpu.memoryUsed} MB, Free={gpu.memoryFree} MB, Load={gpu.load*100}%")
            time.sleep(interval)
    except Exception as e:
        logger.error(f"Error monitoring memory usage: {e}")


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ping_test() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()

    data = None
    data = flask.request.get_json()

    if data is None:
        return flask.Response(
            response="Data Load Error", status=415, mimetype="text/plain"
        )

    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")

    model_input = data
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

    message = download_from_s3(local_path, bucket, key, region)
    if message is not None:
        return flask.Response(
            response=message, status=415, mimetype="text/plain"
        )
    
    print("Begin to generate images!")
    images_path = generate_images(prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed, local_path)
    response = process_output(model_input, images_path)
    if isinstance(response, str):
        return flask.Response(
            response=message, status=415, mimetype="text/plain"
        )

    result = json.dumps(response, indent=2)

    return flask.Response(response=result, status=200, mimetype="application/json")
