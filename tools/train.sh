#!/bin/bash

cd /content/st-gcn || exit
python main.py recognition -c config/st_gcn/ntu-xsub/train.yaml --device 0 --batch_size 32 --num_epoch 3