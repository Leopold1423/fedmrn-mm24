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
  title={Masked Random Noise for Communication Efficient Federaetd Learning},
  author={Shiwei Li and Yingyi Cheng and Haozhao Wang and Xing Tang and Shijie Xu and Weihong Luo and Yuhua Li and Dugang Liu and Xiuqiang He and and Ruixuan Li},
  booktitle={Proceedings of the 32nd ACM International conference on Multimedia},
  pages={xxx--xxx},
  year={2024}
}
```