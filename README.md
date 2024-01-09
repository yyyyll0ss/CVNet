# CVNet
This is Pytorch implementation for "When Vectorization Meets Change Detection". If you have any questions, please contact yanyl@hnu.edu.cn

## Overview
The overall framework for CVNet.
![teaser](Framework.png)
## Running the code
1. Run train.py to train a new model.

- Please put the  change vectorization datasets into datasets folder.
- The datasets folder is constructed as follows:

  -- indian_pines
  
  ---- indian_pines.mat
  
  ---- indian_pines_gt.mat
  
  ---- indian_pines_coco.json

2. Run inference.py to inference the vectorization results.

3. Run evaluation.py to evaluate the vectorization performance.

- The datasets can be downloaded from: 
- WHU-Vector-CD: https://pan.baidu.com/s/1j1hu1gI4yWHQHNWobwQEYA   code：ikls
- LEVIR-Vector-CD: https://pan.baidu.com/s/13loTooaG0hK5zukgVe1sVw   code：ikls

## BibTeX
```
@article{yan2023vectorization,
  title={When Vectorization Meets Change Detection},
  author={Yan, Yinglong and Yue, Jun and Lin, Jiaxing and Guo, Zhengyang and Fang, Yi and Li, Zhenhao and Xie, Weiying and Fang, Leyuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE}
}
```
