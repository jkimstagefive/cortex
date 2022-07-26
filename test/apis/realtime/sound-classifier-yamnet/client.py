import requests
import os
import soundfile as sf
import numpy as np
from scipy.io.wavfile import read
import time
import json
import sys
import io
import csv
import base64

csv_file = os.environ.get('class_names', './class_names.csv')
input_file = os.environ.get('input_file', './silence.wav')

def class_names_from_csv():
    class_names = []
    with open(csv_file, "r", newline="") as f:
        for row in csv.reader(f, delimiter=","):
            class_names.append(row[2])
    return class_names



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: python client.py <http://host:port>")
        sys.exit(1)
    address = sys.argv[1]
    server_url = f"{address}/v1/models/yamnet:predict"


    wav_data, sr = sf.read(input_file, dtype=np.int16)
    waveform = wav_data / 32768.0
    data = json.dumps({"inputs" : waveform.tolist()})

    start = time.time()
    resp = requests.post(server_url, data)
    end = time.time()
    scores = np.array(resp.json()["outputs"]["output_0"]).reshape((-1, 521))
    class_names = class_names_from_csv()
    predicted_class = class_names[scores.mean(axis=0).argmax() + 1]

    print(f"{predicted_class} time consuming:{int((end - start) * 1000)}ms")
