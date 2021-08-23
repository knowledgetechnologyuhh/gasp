#!/bin/bash

wget -O coutrot1.tar.gz https://www2.informatik.uni-hamburg.de/WTM/corpora/GASP/datasets/processed/Grouped_frames/coutrot1.tar.gz

mv coutrot1/* ./
rm -rf coutrot1/
tar -xf coutrot1.tar.gz
rm coutrot1.tar.gz
