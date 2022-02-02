#!/bin/bash

./megadown 'https://mega.nz/#!58dARTDR!AII7nCEktkeMqbZ2XqXBFyDsAeMKqBf9reXR-MHPuKk'
wget -O coutrot_database2.mat "http://antoinecoutrot.magix.net/public/assets/coutrot_database2.mat"
7za x "ERB4_Stimuli.zip"
rm "ERB4_Stimuli.zip"
