## FedMRN

**Masked Random Noise for Communication Efficient Federaetd Learning [MM 2024]**

## Environment Setup
Please install the necessary dependencies first:
```
pip install -r requirements.txt
```

## Data Partition
Please run the following code to download and partition datasets:
```
python ./dataloader/datapartition.py 
```

## Run Experiments
Please use the scripts to run the experiments, for example:
```
./run/1.1-fmnist+fedavg.sh
./run/1.2-fmnist+fedsrn.sh
```

## Citation
```
@inproceedings{li2024fedmrn,
author = {Li, Shiwei and Cheng, Yingyi and Wang, Haozhao and Tang, Xing and Xu, Shijie and Luo, Weihong and Li, Yuhua and Liu, Dugang and He, Xiuqiang and Li, Ruixuan},
title = {Masked Random Noise for Communication-Efficient Federated Learning},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia (MM 24)},
pages = {3686–3694},
numpages = {9},
year = {2024},
}
```