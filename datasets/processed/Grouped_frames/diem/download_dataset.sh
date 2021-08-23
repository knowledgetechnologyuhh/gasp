#!/bin/bash

wget -O diem.tar.gz https://www2.informatik.uni-hamburg.de/WTM/corpora/GASP/datasets/processed/Grouped_frames/diem.tar.gz

mv diem/* ./
rm -rf diem/
tar -xf diem.tar.gz
rm diem.tar.gz
