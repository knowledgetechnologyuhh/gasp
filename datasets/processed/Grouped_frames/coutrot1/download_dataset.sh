#!/bin/bash

wget -O coutrot1.tar.gz https://www2.informatik.uni-hamburg.de/WTM/corpora/GASP/datasets/processed/Grouped_frames/coutrot1.tar.gz

tar -xf coutrot1.tar.gz
mv coutrot1/* ./
rm -rf coutrot1/
rm coutrot1.tar.gz
