#!/bin/bash
# Create a virtual environment for installing dependencies.
python3 -m virtualenv .
source ./bin/activate
# Install required dependencies.
pip3 install tensorflow==1.15.5
pip3 install tf_slim==1.1.0
git clone https://github.com/tensorflow/models
mv models/research/slim/nets nets
mv models/research/lstm_object_detection lstm_object_detection
# Run unit tests.
python3 -m unittest networks_test.py