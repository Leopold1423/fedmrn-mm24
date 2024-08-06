#!/bin/bash

com_type="fedsrn"; model="srn_lstm"; dataset="shakespeare"; frac=0.1; 
rounds=100; epochs=1; batch_size=128; lr=3.0; momentum=0.0;
log_dir="./log/"; log_name="${com_type}"; gpu=5;
mask_type="binary"; noise_type="uniform_0.003";

python ./main.py \
--mask_type ${mask_type} --noise_type ${noise_type} \
--com_type ${com_type} --model ${model} --dataset ${dataset} --frac ${frac} \
--rounds ${rounds} --epochs ${epochs} --batch_size ${batch_size} --lr ${lr} --momentum ${momentum} \
--gpu ${gpu} --log_dir ${log_dir} --log_name ${log_name}

