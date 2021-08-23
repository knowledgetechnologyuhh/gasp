#!/bin/bash

wget http://cvsp.cs.ntua.gr/research/stavis/data/annotations/DIEM.tar.gz
wget http://cvsp.cs.ntua.gr/research/stavis/data/annotations/Coutrot_db1.tar.gz
wget http://cvsp.cs.ntua.gr/research/stavis/data/annotations/Coutrot_db2.tar.gz

tar -xf DIEM.tar.gz
rm DIEM.tar.gz
mv DIEM diem
tar -xf Coutrot_db1.tar.gz
rm Coutrot_db1.tar.gz
mv Coutrot_db1 coutrot1
cd coutrot1
for d in ./*/; do mv -v "$d" "${d/clip/clip_}"; done;
cd ../
tar -xf Coutrot_db2.tar.gz
rm Coutrot_db2.tar.gz
mv Coutrot_db2 coutrot2
cd coutrot2
for d in ./*/; do mv -v "$d" "${d/clip/clip_}"; done;
cd ../



