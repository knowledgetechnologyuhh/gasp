#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1hVf2PKp9UQNYMeG-tyT__0qJDHkj79sV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hVf2PKp9UQNYMeG-tyT__0qJDHkj79sV" -O model.pth.tar && rm -rf /tmp/cookies.txt
