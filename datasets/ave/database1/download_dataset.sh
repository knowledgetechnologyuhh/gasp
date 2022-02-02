#!/bin/bash

./megadown.sh 'https://mega.nz/#!At8DWR7L!yf5k0jVwL961-jI4FJ2DGUAUqAu-yNbq3s3i6b52M2I'
wget -O coutrot_database1.mat "http://antoinecoutrot.magix.net/public/assets/coutrot_database1.mat"
7za x "ERB3_Stimuli.zip"
rm "ERB3_Stimuli.zip"
