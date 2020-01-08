## 1. Introduction
This repo. is the PyTorch implementation of multi-object tracker OTCD.
The paper is [real-time online multi-object tracking in compressed domain](https://ieeexplore.ieee.org/abstract/document/8734056).
There maybe a slight gap between the performance obtained by this script and the performance reported in the paper.

## 2. Citation
```
@article{liu2019real,
  title={Real-Time Online Multi-Object Tracking in Compressed Domain},
  author={Liu, Qiankun and Liu, Bin and Wu, Yue and Li, Weihai and Yu, Nenghai},
  journal={IEEE Access},
  volume={7},
  pages={76489--76499},
  year={2019},
  publisher={IEEE}
}
```

## 3. Requirements

- [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))
- [Docker-Compose](https://pypi.org/project/docker-compose/)

## 4. Quickstart

1) Download this repo
```
git clone https://github.com/liuqk3/OTCD.git
cd OTCD
```

2) Build docker image
```
sudo docker-compose build
```

3) Download the pretrained model from [BaiduYunPan](https://pan.baidu.com/s/1-Enzek8SQpnr1BY1uzde4w). Then put all models to ```./save```. If you have any problems with the download process, please email me.

4) Download [MOT Challenge dataset](https://motchallenge.net/) and place into a sub directory of OTCD. If placed outside of the OTCD directory, edit the volume mapping in `docker-compose.yml`.

5) When finished start docker container
```
sudo docker-compose up -d
```

6) Enter the container
```
sudo docker exec -it otcd bash
```

7) Start tracker
```
python tracking_on_mot.py --mot_dir path/to/MOT-dataset
```

## 5. Code for training
The training scripts are also published in ```useful_scripts```. You can train all the models by the given scripts.
