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

# the image URL is the location of the image we should send to the server

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
    dl_request = requests.get(IMAGE_URL, stream=True)
    dl_request.raise_for_status()

    # compose a JSON Predict request (send JPEG image in base64).
    jpeg_bytes = base64.b64encode(dl_request.content).decode("utf-8")
    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes

    # send few requests to warm-up the model.
    for _ in range(3):
        response = requests.post(server_url, data=predict_request)
        response.raise_for_status()

    # send few actual requests and report average latency.
    total_time = 0
    num_requests = 10
    for _ in range(num_requests):
        response = requests.post(server_url, data=predict_request)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = labels[response.json()["predictions"][0]["classes"]]

    print(
        "Prediction class: {}, avg latency: {} ms".format(
            prediction, (total_time * 1000) / num_requests
        )
    )

    f.write(prediction + '\n')



def main():
    # parse arg
    if len(sys.argv) != 2:
        print("usage: python client.py <http://host:port>")
        sys.exit(1)
    address = sys.argv[1]
    server_url = f"{address}/v1/models/resnet50:predict"

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

