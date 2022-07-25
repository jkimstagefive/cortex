#!/bin/bash
URL=`cortex get --output pretty sentiment-analyzer-bert | grep endpoint | awk '{ print $2'}`
export PYTHONPATH=$(pwd)/google-bert
python3 ./client.py $URL
