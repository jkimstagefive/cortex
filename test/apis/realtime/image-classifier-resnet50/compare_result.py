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
from difflib import SequenceMatcher


def main():

    text1 = open('i1_R_result.txt').read()
    text2 = open('p2_R_result.txt').read()
    m = SequenceMatcher(None, text1, text2)
    print(m.ratio())
 
if __name__ == "__main__":
    main()
