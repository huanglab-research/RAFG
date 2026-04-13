# Towards Region-Aware Finer Self-Supervised Learning for Fine-Grained Visual Recognition

This repository is an official implementation of the paper "Towards Region-Aware Finer Self-Supervised Learning for Fine-Grained Visual Recognition".

By Yao Xu, Lei Huang Jie Nie, Yadong Huo and Zhiqiang We.

> **Abstract:** 
Self-supervised visual representation learning has achieved remarkable success across a variety of conventional computer vision tasks. However, their reliance on generic global feature extraction mechanisms fundamentally limits their ability to capture subtle inter-class variations, which are crucial for fine-grained visual recognition. Specifically, the model lacks the ability to perceive semantically meaningful local regions within images, making it difficult to effectively identify and
utilize these regions to focus on category-specific details. Although recent studies have attempted to incorporate region-level information through random cropping strategies, such methods often capture only the shared features among similar categories, thereby weakening the discriminative power of the learned representations in fine-grained recognition tasks. To address this limitation, we propose a novel region-aware self-supervised learning framework that suppresses redundant shared representations while enhancing the learning of fine-grained class-specific cues without relying on manual annotations. The framework consists of two synergistic components. At the inter-branch level, we introduce a region-aware self-supervised proxy task designed to enhance key local fine-grained features. It encourages cross-branch learning between semantically salient local regions and the global view, guiding the model to focus on more informative parts of the image. At the intra-branch
level, we introduce a Local-Global Mutual Learning module that integrates local details with holistic semantic context, thereby promoting greater representational diversity and improving the model’s discriminative performance on fine-grained categories. Extensive experiments on three benchmark datasets CUB-200-2011, Stanford Cars and FGVC-Aircraft show that our region-aware self-supervised framework achieves superior performance compared to existing state-of-the-art methods.



## Contents

1. [Datasets](#Datasets)
1. [Environment](#Environment)
1. [Training](#Training)
1. [Test](#Test)
1. [Acknowledgements](#Acknowledgment)

## Datasets
Experiments on **3 image datasets**:
FGVC-Aircraft，Stanford Cars, CUB-200-2011

|#|Datasets|Download|
|---|----|-----|
|1|FGVC-Aircraft|[Link](https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft)|
|2|Stanford Cars|[Link](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)
|3|CUB-200-2011|[Link](https://www.kaggle.com/datasets/wenewone/cub2002011)  |


 ## Environment
```python
pip install -r requirement.txt
```
#### Pretrained Model
You can download  pre-trained  models: [Swin-T, Pretrain on ImageNet-1k](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)

## Training

```python
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 main_RAFG.py --arch swin --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 64 --epochs 300 --teacher_temp 0.07 --warmup_epochs 10 --warmup_teacher_temp_epochs 30 --norm_last_layer false --use_dense_prediction True --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml
```

## Test

```python
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 eval_linear.py --data_path $DATA_PATH --output_dir $OUT_PATH/lincls/epoch1 --pretrained_weights  /data1/output1/checkpoint.pth --checkpoint_key teacher --batch_size_per_gpu 128 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --n_last_blocks 4 --num_labels 200
```

## Acknowledgment

We are very grateful for these excellent works: [LCR](https://github.com/GANPerf/LCR),[EsViT](https://github.com/microsoft/esvit),[DINO]( https://github.com/facebookresearch/dino). Please follow their respective licenses for usage and redistribution. Thanks for their awesome works.

## 📬 Contact

Feel free to contact me if there is any question. (Yao Xu: [xuyao8809@stu.ouc.edu.cn](mailto:xuyao8809@stu.ouc.edu.cn), Lei Huang: [huangl@ouc.edu.cn](mailto:huangl@ouc.edu.cn))

---
