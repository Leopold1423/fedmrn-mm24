#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

part_strategy="iid"; part_strategy_list=("iid" "labeldir0.3" "labelcnt0.3")
model_client="srn_cnn4"; model="cnn4"; dataset="svhn"; 
lr=0.1; momentum=0; l2=0; rounds=100; epochs=10; batch_size=64; save_round=0; num_per_round=10; num_client=100; val_ratio=0.0;
com_type="fedsrn"; noise_type="uniform_0.005"; mask_type="signed"; 

gpu=5
gpu_clients=(5 5 5 5 5 5 5 5 5 5) 
ip="0.0.0.0:11230"

for b in 0 1 2; do  # part_strategy
    part_strategy=${part_strategy_list[${b}]}
    dir="../log/${dataset}+${num_client}/${dataset}+${part_strategy}/${model}+${com_type}+${noise_type}+${mask_type}+lr=${lr}/"
    name="${dataset}+${part_strategy}+${model}+${com_type}"
    python ../train/server.py \
    --com_type ${com_type} --noise_type ${noise_type} --mask_type ${mask_type} \
    --model ${model} --dataset ${dataset} --lr ${lr} --momentum ${momentum} --l2 ${l2} \
    --rounds ${rounds} --epochs ${epochs} --batch_size ${batch_size} --save_round ${save_round} \
    --num_per_round ${num_per_round} --num_client ${num_client} \
    --gpu ${gpu} --ip ${ip} --log_dir ${dir} --log_name ${name} &

    for a in $(seq 0 $(($num_per_round-1))); do # client
        python ../train/client.py \
        --com_type ${com_type} \
        --model ${model_client} --dataset ${dataset} --part_strategy ${part_strategy} --num_client ${num_client} --id ${a} --val_ratio ${val_ratio} \
        --gpu ${gpu_clients[${a}]} --ip ${ip} --log_dir ${dir} --log_name ${name} &
    done
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM 
    wait
done 

