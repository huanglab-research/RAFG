
# Towards Region-Aware Finer Self-Supervised Learning for Fine-Grained Visual Recognition

This is the  implementation of  paper: Towards Region-Aware Finer Self-Supervised Learning for Fine-Grained Visual Recognition

## the Framework of the Proposed RAFG
<table border=0 >
	<tbody>
    <tr>
		<tr>
			<td width="40%" > <img src="https://github.com/huanglab-research/RAFG/blob/master/ff.png"> </td>
		</tr>
	</tbody>
</table>


## Datasets
Experiments on **3 image datasets**:
FGVC-Aircraft，Stanford Cars, CUB-200-2011

### Datasets
|#|Datasets|Download|
|---|----|-----|
|1|FGVC-Aircraft|[Link](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)|
|2|Stanford Cars|[Link](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
|3|CUB-200-2011|[Link](https://authors.library.caltech.edu/27452/)  |

 ### Environment
```python
pip install -r requirement.txt
```

### Training/Resume Training

```python
CUDA_VISIBLE_DEVICES=4,6 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 main_esvit.py --arch swin --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 64 --epochs 300 --teacher_temp 0.07 --warmup_epochs 10 --warmup_teacher_temp_epochs 30 --norm_last_layer false --use_dense_prediction True --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml
```

### Test/Evaluation

```python
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 eval_linear.py --data_path $DATA_PATH --output_dir $OUT_PATH/lincls/epoch0020328 --pretrained_weights /home//data1/hl/xy/esvitt6/outputest328//checkpoint0020.pth --checkpoint_key teacher --batch_size_per_gpu 128 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --n_last_blocks 4 --num_labels 200
```


### Pretrained Model
You can download  pre-trained  models: ([here](https://github.com/huanglab-research/RAFG/blob/master/swin_tiny_patch4_window7_224.pth)).   

------

 
 

 
 
 
 


