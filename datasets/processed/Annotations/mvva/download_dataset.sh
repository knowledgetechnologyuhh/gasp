#!/bin/bash

wget -O mvva.tar.xz https://www2.informatik.uni-hamburg.de/WTM/corpora/GASP/datasets/processed/Annotations/mvva.tar.xz

tar -xf mvva.tar.xz
mv mvva/* ./
rm -rf mvva/
rm mvva.tar.xz
