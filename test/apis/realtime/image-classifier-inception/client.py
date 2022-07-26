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
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import json
import time

# the image URL is the location of the image we should send to the server
IMAGE_URL = "https://tensorflow.org/images/blogs/serving/cat.jpg"


def main():
    # parse arg
    if len(sys.argv) != 2:
        print("usage: python client.py <http://host:port>")
        sys.exit(1)
    address = sys.argv[1]
    server_url = f"{address}/v1/models/inception:predict"

    # download labels
    labels = requests.get(
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    ).text.split("\n")[1:]

    # download the image
    dl_request = requests.get(IMAGE_URL, stream=True)
    dl_request.raise_for_status()

    # compose a JSON Predict request (send JPEG image in base64).
    image = dl_request.content
    decoded_image = np.asarray(Image.open(BytesIO(image)), dtype=np.float32) / 255
    model_input = json.dumps({"signature_name":"predict","inputs": np.expand_dims(decoded_image, axis=0).tolist()})

    # send few requests to warm-up the model.
    start = time.time()
    response = requests.post(server_url, model_input)
    end = time.time()
    predicted_class = labels[np.argmax(response.json()["outputs"])]

    #    response.raise_for_status()
    print(f"{predicted_class} time consuming:{int((end - start) * 1000)}ms")

if __name__ == "__main__":
    main()
