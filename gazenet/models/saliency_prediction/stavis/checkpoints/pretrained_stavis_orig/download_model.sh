#!/bin/bash

wget -O pretrained_models.tar.gz http://cvsp.cs.ntua.gr/research/stavis/data/pretrained_models.tar.gz
tar -xzf pretrained_models.tar.gz
rm pretrained_models.tar.gz
mv pretrained_models/* .
rm -rf pretrained_models/