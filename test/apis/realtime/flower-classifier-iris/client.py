"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model

The SavedModel must be one that can take JPEG images as inputs.

Typical usage example:

    python client.py <http://host:port>
"""

import sys
import os
import base64
import requests
import json
import time

# the image URL is the location of the image we should send to the server
labels = ["setosa", "versicolor", "virginica"]

input_file = os.environ.get('input_file', './sample.json')

def main():
    # parse arg
    if len(sys.argv) != 2:
        print("usage: python client.py <http://host:port>")
        sys.exit(1)
    address = sys.argv[1]
    server_url = f"{address}/v1/models/iris:predict"

    with open(input_file) as json_file:
        data = json.load(json_file)

    # compose a JSON Predict request (send JPEG image in base64).
    model_input = json.dumps({"signature_name":"predict","inputs": data})

    # send few requests to warm-up the model.
    start = time.time()
    response = requests.post(server_url, model_input)
    end = time.time()
    prediction = response.json()["outputs"]
    predicted_class_id = int(prediction["class_ids"][0][0])
    predicted_class = labels[predicted_class_id]

    #    response.raise_for_status()
    print(f"{predicted_class} time consuming:{int((end - start) * 1000)}ms")

if __name__ == "__main__":
    main()
