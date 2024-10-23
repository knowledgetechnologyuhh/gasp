#!/bin/bash

wget -O mvva.zip "https://www2.informatik.uni-hamburg.de/WTM/corpora/GASP/datasets/mvva.zip"
unzip mvva.zip
mv mvva_database_v1/* ./
rm -rf mvva_database_v1
rm mvva.zip

mkdir raw_videos
mv our_database_video/* raw_videos
mv raw_videos our_database_video
cd our_database_video
mkdir raw_audios
cd raw_videos
for i in *.avi; do ffmpeg -i "$i" "${i%.*}.mp4"; done
for i in *.mp4; do ffmpeg -i "$i" "../raw_audios/${i%.*}.wav"; done
cd ../../
