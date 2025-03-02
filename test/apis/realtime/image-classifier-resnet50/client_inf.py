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
import json

import io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from PIL import Image
import requests
import numpy as np

# the image URL is the location of the image we should send to the server
# IMAGE_URL = "https://tensorflow.org/images/blogs/serving/cat.jpg"
#IMAGE_URL = "https://en.wikipedia.org/wiki/Macaw#/media/File:Blue-and-Yellow-Macaw.jpg"
#IMAGE_URL = "https://wja300-cortex.s3.amazonaws.com/image/cat.jpg"
#IMAGE_URL = "https://wja300-cortex.s3.amazonaws.com/image/macaw.jpg"
IMAGE_URL = ["https://wja300-cortex.s3.amazonaws.com/image/butterfly1.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/butterfly2.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/cat1.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/cat2.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/chicken1.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/chicken2.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/cow1.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/cow2.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/dog1.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/dog2.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/elephant1.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/elephant2.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/horse1.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/horse2.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/sheep1.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/sheep2.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/spider1.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/spider2.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/squirrel1.jpg",
                "https://wja300-cortex.s3.amazonaws.com/image/squirrel2.jpg"]



def processing(f, IMAGE_URL, index, server_url, labels):
    
    # download the image
    response = requests.get(IMAGE_URL, stream=True)
    img = Image.open(io.BytesIO(response.content))
    img = img.resize((224, 224))

    # process the image
    img_arr = image.img_to_array(img)
    img_arr2 = np.expand_dims(img_arr, axis=0)
    img_arr3 = resnet50.preprocess_input(np.repeat(img_arr2, 1, axis=0))
    img_list = img_arr3.tolist()
    request_payload = {"signature_name": "serving_default", "inputs": img_list}

    # send few requests to warm-up the model.
    for _ in range(3):
        response = requests.post(
            server_url,
            data=json.dumps(request_payload),
            headers={"content-type": "application/json"},
        )
        response.raise_for_status()

    # send few actual requests and report average latency.
    total_time = 0
    num_requests = 10
    for _ in range(num_requests):
        response = requests.post(
            server_url,
            data=json.dumps(request_payload),
            headers={"content-type": "application/json"},
        )
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()

        label_idx = np.argmax(response.json()["outputs"][0])
        prediction = labels[label_idx]

    print(
            "{}: Prediction class: {}, avg latency: {} ms".format(
            index, prediction, (total_time * 1000) / num_requests
        )
    )

    f.write(prediction + '\n')


def main():
    # parse arg
    if len(sys.argv) != 2:
        print("usage: python client.py <http://host:port>")
        sys.exit(1)
    address = sys.argv[1]
    server_url = f"{address}/v1/models/resnet50_neuron:predict"

    f = open('i1_R_result.txt', 'w')
 
    # download labels
    labels = requests.get(
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    ).text.split("\n")[1:]
    for i in range(5):
        for index, i in enumerate(IMAGE_URL):
            processing(f, i, index, server_url, labels)

    f.close()


if __name__ == "__main__":
    main()
