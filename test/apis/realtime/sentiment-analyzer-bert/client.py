import tokenization
from extract_features import InputExample, convert_examples_to_features
import numpy as np
import requests
import os
import time
import json
import sys

labels = ["negative", "positive"]
vocab_file = os.environ.get('vocab_file', './vocab.txt')
input_file = os.environ.get('input_file', './sample.json')
max_token_len = os.environ.get('max_token_len', 128)


def preprocess(text):
    text_a = text
    example = InputExample(unique_id=None, text_a=text_a, text_b=None)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    feature = convert_examples_to_features([example], max_token_len, tokenizer)[0]
    input_ids = np.reshape([feature.input_ids], (1, max_token_len))
    return {
        "inputs": {"input_ids": input_ids.tolist()}
    }


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: python client.py <http://host:port>")
        sys.exit(1)
    address = sys.argv[1]
    server_url = f"{address}/v1/models/bert:predict"

    with open(input_file) as json_file:
        text = json.load(json_file)['input']
    start = time.time()
    resp = requests.post(server_url, json=preprocess(text))
    end = time.time()
    pro_0, pro_1 = resp.json()['outputs']['probabilities'][0]
    result = labels[resp.json()['outputs']['labels']]
    print(f"{result} neg_pro:{pro_0} pos_pro:{pro_1} time consuming:{int((end - start) * 1000)}ms")
