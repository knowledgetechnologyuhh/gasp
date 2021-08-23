#!/bin/bash

wget -O coutrot2.tar.gz https://www2.informatik.uni-hamburg.de/WTM/corpora/GASP/datasets/processed/Grouped_frames/coutrot2.tar.gz

mv coutrot2/* ./
rm -rf coutrot2/
tar -xf coutrot2.tar.gz
rm coutrot2.tar.gz
