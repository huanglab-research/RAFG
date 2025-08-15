# Towards Region-Aware Finer Self-Supervised Learning for Fine-Grained Visual Recognition

This is the  implementation of  paper: Towards Region-Aware Finer Self-Supervised Learning for Fine-Grained Visual Recognition

> ⚠ **Note:** The source code is currently incomplete and will be fully released once the manuscript is accepted by the journal.

## Datasets
Experiments on **3 image datasets**:
FGVC-Aircraft，Stanford Cars, CUB-200-2011

|#|Datasets|Download|
|---|----|-----|
|1|FGVC-Aircraft|[Link](https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft)|
|2|Stanford Cars|[Link](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)
|3|CUB-200-2011|[Link](https://www.kaggle.com/datasets/wenewone/cub2002011)  |


## Pretrained Model
You can download  pre-trained  models: [Swin-T, Pretrain on ImageNet-1k](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)

 ## Environment
```python
pip install -r requirement.txt
```

## Training/Resume Training

```python
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 main_RAFG.py --arch swin --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 64 --epochs 300 --teacher_temp 0.07 --warmup_epochs 10 --warmup_teacher_temp_epochs 30 --norm_last_layer false --use_dense_prediction True --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml
```

## Test/Evaluation

```python
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 eval_linear.py --data_path $DATA_PATH --output_dir $OUT_PATH/lincls/epoch1 --pretrained_weights  /data1/output1/checkpoint.pth --checkpoint_key teacher --batch_size_per_gpu 128 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --n_last_blocks 4 --num_labels 200
```

## License & Acknowledgment

We are very grateful for these excellent works: [LCR](https://github.com/GANPerf/LCR),[EsViT](https://github.com/microsoft/esvit),[DINO]( https://github.com/facebookresearch/dino). Please follow their respective licenses for usage and redistribution. Thanks for their awesome works.

## 📬 Contact

Feel free to contact me if there is any question. (Yao Xu: [xuyao8809@stu.ouc.edu.cn](mailto:xuyao8809@stu.ouc.edu.cn), Lei Huang: [huangl@ouc.edu.cn](mailto:huangl@ouc.edu.cn))

---
