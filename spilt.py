"""
读取label_map.txt文件,获得每个图像的标签
读取mat2txt.txt文件,获取每个图像的train, test标签.其中1为训练,0为测试.
"""
 
import os
import shutil
import numpy as np
import time
 
time_start = time.time()
 
path = '/home/ouc/fbj/stanfordcar/'
 
# 文件路径
path_images = path + 'label_map.txt'
path_split = path + 'mat2txt.txt'
trian_save_path = path + 'dataset/train/'
test_save_path = path + 'dataset/val/'
 
# 读取label_map.txt文件，总类数
images = []
with open(path_images, 'r') as f:
    for line in f:
        images.append(list(line.strip('\n').split(',')))
 
# 读取mat2txt.txt文件，总图数
split = []
with open(path_split, 'r') as f_:
    for line in f_:
        split.append(list(line.strip('\n').split()))
 
# 划分
num = len(images)  # 图像的总类别数
photo_num = len(split)  # 图像总个数
 
for k in range(num):
    split_name0 = images[k][0].split('.')[0].strip()
    split_name1 = images[k][0].split('.')[1].strip()
    split_name = split_name0 + '.' + split_name1  
 
    val_save_paths = "/home/ouc/fbj/stanfordcar/dataset/train/" + split_name
    train_save_paths = "/home/ouc/fbj/stanfordcar/dataset/val/" + split_name
 
    for i in range(photo_num):
        photo_name = str(split[i]).split()[1].split('/')[1]
        copy_photo_paths = path + 'images/' + split_name + '/' + photo_name
        
       
        if k == class_index:
            if int(split[i][0][-1]) == 1:  # 划分到训练集
                if os.path.isdir(trian_save_path + split_name):
                    shutil.copy(copy_photo_paths, train_save_paths)
                else:
                    os.makedirs(trian_save_path + split_name)
                    shutil.copy(copy_photo_paths, train_save_paths)
                print('%s处理完毕!' % photo_name)
            else:
                # 判断文件夹是否存在
                if os.path.isdir(test_save_path + split_name):
                    shutil.copy(copy_photo_paths, val_save_paths)
                else:
                    os.makedirs(test_save_path + split_name)
                    shutil.copy(copy_photo_paths, val_save_paths)
                print('%s处理完毕!' % photo_name)
        else:
            continue
    print("第{}类分类完毕".format(k))
 
time_end = time.time()
print('划分结束, 耗时%s!!' % (time_end - time_start))