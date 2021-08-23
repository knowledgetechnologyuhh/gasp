#!/bin/bash

# Here , the downloaded DAVE weights refer to the avinet model trained on the AVE dataset and **NOT** the DAVE model
# https://iiitaphyd-my.sharepoint.com/personal/samyak_j_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsamyak%5Fj%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FVideo%20Saliency%2FAViNet%20Pretrained%2Ezip&parent=%2Fpersonal%2Fsamyak%5Fj%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FVideo%20Saliency&originalPath=aHR0cHM6Ly9paWl0YXBoeWQtbXkuc2hhcmVwb2ludC5jb20vOnU6L2cvcGVyc29uYWwvc2FteWFrX2pfcmVzZWFyY2hfaWlpdF9hY19pbi9FWFlxNVdpU2JoOUtxOVJfbi1HcjN5QUJSeUtQU2t4TTdST0xnLXpQRFhWX3FBP3J0aW1lPU5XN3BjZzdHMkVn

wget -O AViNet_Dave.pt https://www2.informatik.uni-hamburg.de/WTM/corpora/GASP/gazenet/models/saliency_prediction/avinet/checkpoints/pretrained_avinet_orig/AViNet_Dave.pt
wget -O ViNet_Dave.pt https://www2.informatik.uni-hamburg.de/WTM/corpora/GASP/gazenet/models/saliency_prediction/avinet/checkpoints/pretrained_avinet_orig/ViNet_Dave.pt
