# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:52:05 2024

@author: qiao_
"""

import os
from torch.utils.data import Dataset, DataLoader  #自定义的母类，必须的
from torchvision.transforms import transforms
from PIL import Image 
import torch
import glob
import csv
import random

class YoloData(Dataset):
    def __init__(self, root, resize, stage, mode="uni-classifier"):
        super(YoloData, self).__init__()
        self.root = root
        self.stage = stage
        self.resize = resize
        self.name2label = {}  # "sq...":0
        for name in sorted(os.listdir(os.path.join(root, 'images', stage))):
            label = ["-1 0 0 0 0 0"]
            labelfilepath = os.path.join(root, 'labels', stage, name.split('.')[0]+'.txt')
            if os.path.exists(labelfilepath) and not os.path.isdir(labelfilepath):
                ff = open(labelfilepath)
                label = ff.readlines()
                ff.close()

            self.name2label[name] = label
        # print(self.name2label)
        # image, label
        self.images, self.labels = self.load_csv(mode+'-'+stage+'-images.csv', mode)  #csv文件存在 直接读取

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0
        img, label = self.images[idx], self.labels[idx]
        
        tf = transforms.Compose([   #常用的数据变换器
					            lambda x:Image.open(x).convert('RGB'),  # string path= > image data 
					            #这里开始读取了数据的内容了
					            transforms.Resize(   #数据预处理部分
					                (int(self.resize * 1.25), int(self.resize * 1.25))), 
					            transforms.RandomRotation(15), 
					            transforms.CenterCrop(self.resize), #防止旋转后边界出现黑框部分
					            transforms.ToTensor(),
					            transforms.Normalize(mean=[0.485, 0.456, 0.406],
					                                 std=[0.229, 0.224, 0.225])
       							 ])
        img = tf(img)
        label = torch.tensor(label + 1)  #转化tensor
        return img, label       #返回当前的数据内容和标签
    
    def load_csv(self, filename, mode):
        if not os.path.exists(os.path.join(self.root, filename)): 
        	     #如果没有保存csv文件，那么我们需要写一个csv文件，如果有了直接读取csv文件
            images = []
            for name in self.name2label.keys():   
            	            # 'pokemon\\mewtwo\\00001.png
                images += glob.glob(os.path.join(os.path.join(self.root, 'images', self.stage), name))
                #images += glob.glob(os.path.join(os.path.join(self.root, 'images'), name, '*.jpg'))
                #images += glob.glob(os.path.join(os.path.join(self.root, 'images'), name, '*.jpeg'))
                	
            	        
            random.shuffle(images)
            # print(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-1]        #从名字就可以读取标签
                    label = self.name2label[name][0].split()[0]
                    	                # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])  #写进csv文件
        	
        	    # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
            	            # 'pokemon\\bulbasaur\\00000000.png', 0
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels
    
    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean
        return x
if __name__=='__main__':
    db = YoloData('pokeman', 224, 'train')
    loader = DataLoader(db, batch_size=32, shuffle=True)
    for x, y in loader: #此时x,y是批量的数据
        print(x.shape)
