#!/bin/bash

wget --no-check-certificate --content-disposition -O - http://github.com/yufanLIU/find/archive/master.tar.gz | tar xz --strip=2 "find-master/Our_database"
mkdir raw_audios
cd raw_videos
for i in *.mp4; do ffmpeg -i "$i" "../raw_audios/${i%.*}.wav"; done
cd ../
