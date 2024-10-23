#!/bin/bash

wget http://dl.fbaipublicfiles.com/FAIR-Play/videos.tar.gz
wget http://dl.fbaipublicfiles.com/FAIR-Play/audios.tar.gz
wget http://dl.fbaipublicfiles.com/FAIR-Play/splits.tar.gz
wget -O maps.zip "https://www.dropbox.com/s/xhec895hc9qubhx/gt-FAIR-Play.zip?dl=1"

tar -xf videos.tar.gz
if [ $? -ne 0 ]; then  # Check if the extraction was successful
    echo "Failed to extract videos.tar.gz"
    exit 1
fi
rm videos.tar.gz

tar -xf audios.tar.gz && rm audios.tar.gz
tar -xf splits.tar.gz && rm splits.tar.gz
unzip maps.zip -d maps && mv maps/gt-FAIR-Play/maps/* ./maps && rm -rf ./maps/gt-FAIR-Play && rm maps.zip

