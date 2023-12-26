# RiFeNet
This is a PyTorch implementation of AAAI2024 paper "[Relevant Intrinsic Feature Enhancement Network for  Few-Shot Semantic Segmentation]((https://arxiv.org/abs/2312.06474))".



# Usage

### Requirements
```
Python==3.8
GCC==5.4
torch==1.6.0
torchvision==0.7.0
cython
tensorboardX
tqdm
PyYaml
opencv-python
pycocotools
```

#### Build Dependencies
```
cd model/ops/
bash make.sh
cd ../../
```

### Data Preparation

+ PASCAL-5^i: Please refer to [PFENet](https://github.com/dvlab-research/PFENet) to prepare the PASCAL dataset for few-shot segmentation. 

+ COCO-20^i: Please download COCO2017 dataset from [here](https://cocodataset.org/#download). Put or link the dataset to ```YOUR_PROJ_PATH/data/coco```. And make the directory like this:

```
${YOUR_PROJ_PATH}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- instances_train2017.json
        |   `-- instances_val2017.json
        |-- train2017
        |   |-- 000000000009.jpg
        |   |-- 000000000025.jpg
        |   |-- 000000000030.jpg
        |   |-- ... 
        `-- val2017
            |-- 000000000139.jpg
            |-- 000000000285.jpg
            |-- 000000000632.jpg
            |-- ... 
```

Then, run  
```
python prepare_coco_data.py
```
to prepare COCO-20^i data.

### Train
Download the ImageNet pretrained [**backbones**](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) and put them into the `initmodel` directory.

Then, run this command: 
```
    sh train.sh {*dataset*} {*model_config*}
```
For example, 
```
    sh train.sh pascal split0_resnet50
```

### Test Only
+ Modify `config` file (specify checkpoint path)
+ Run the following command: 
```
    sh test.sh {*dataset*} {*model_config*}
```

For example, 
```
    sh test.sh pascal split0_resnet50
```


# Acknowledgement

This project is built upon [CyCTR](https://github.com/YangFangCS/CyCTR-Pytorch) and [PFENet](https://github.com/dvlab-research/PFENet), thanks for their great works!

# Citation

If you find our codes or models useful, please consider to give us a star or cite with:
```
@article{bao2023relevant,
  title={Relevant Intrinsic Feature Enhancement Network for Few-Shot Semantic Segmentation},
  author={Bao, Xiaoyi and Qin, Jie and Sun, Siyang and Zheng, Yun and Wang, Xingang},
  journal={arXiv preprint arXiv:2312.06474},
  year={2023}
}
```
