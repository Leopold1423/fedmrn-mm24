#!/bin/bash

com_type="fedavg"; test_frequency=1; lr=0.05; log_name="$com_type"; gpu="6";

python -u main.py \
--com_type $com_type \
--model="bisenetv2" --proj_dim=256 --rand_init=False --losstype="back" \
--dataset="voc" --root_dir='./voc' --data="train" --use_erase_data=True \
--num_classes=20 --num_users=60 --frac_num=10 --test_frequency=$test_frequency \
--lr=$lr --momentum=0.0 --weight_decay=0.0 \
--epochs=400 --local_ep=2 --local_bs=8 \
--log_dir="./log" --log_name=$log_name --num_workers=4 --gpu=$gpu 
