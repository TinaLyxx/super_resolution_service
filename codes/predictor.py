# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import json
import os
import flask

from utils import*

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

def ping_test():
    test = "pass!"
    return test

# The flask app for serving predictions
app = flask.Flask(__name__)


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
    data = None

    if flask.request.content_type == "application/json":
        data = flask.request.get_json()
    else:
        return flask.Response(
            response="This predictor only supports JSON data", status=415, mimetype="text/plain"
        )

    images_path, model_input = process_input(data)
    response = process_output(model_input, images_path)
    result = json.dumps(response, indent=2)

    return flask.Response(response=result, status=200, mimetype="application/json")
