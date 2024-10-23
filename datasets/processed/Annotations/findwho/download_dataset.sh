#!/bin/bash

wget -O findwho.tar.xz https://www2.informatik.uni-hamburg.de/WTM/corpora/GASP/datasets/processed/Annotations/findwho.tar.xz

tar -xf findwho.tar.xz
mv findwho/* ./
rm -rf findwho/
rm findwho.tar.xz
