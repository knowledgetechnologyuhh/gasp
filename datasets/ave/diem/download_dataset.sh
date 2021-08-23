#!/bin/bash

# need to  manually download the files from http://www.mediafire.com/?mpu3ot0m2o384
wget -O diem.tar.gz https://www2.informatik.uni-hamburg.de/WTM/corpora/GASP/datasets/ave/diem.tar.gz
tar -xvf diem.tar.gz
mv informatik3/wtm/datasets/External\ Datasets/DIEM/* ./
rm -rf informatik3
rm diem.tar.gz
7za x "*.7z" -o*
rm *.7z
