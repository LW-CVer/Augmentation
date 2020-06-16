#_*_coding:utf-8_*_
import tensorflow as tf
import cv2
import random
from augmentations import *
import pandas as pd
class RandomApply(object):
    def __init__(self,transforms,p=0.5):
        self.p=p
        self.transforms=transforms
    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img
class Shuffle(object):
    def __init__(self,transforms):
        self.transforms=transforms
    def __call__(self,img):
        random.shuffle(self.transforms)
        for i in range(len(self.transforms)):
            img=self.transforms[i](img)
        return img
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
#读取CSV文件
def get_data(file_path):
        file_path=file_path
        data=pd.read_csv(file_path,encoding="ISO-8859-1")
        imgs_name=[]
        labels=[]
        for i in data['filename']:
            imgs_name.append(i)
        for i in data['label']:
            labels.append(i)
            
        return imgs_name,labels
#数据增强        
def make_transform(w,h):
    transform1=Compose([
                    Resize(w,h)     
            ])

    transform2=Compose([
                   Resize(w,h),
                   random.choice([RandomRotation(),ColorJitter(),RandomFlip()])
                 
            ])

    transform3=Compose([
                   Resize(w,h),
                   RandomCrop(w,h)
                   
            ])

    transform4=Compose([
                   Resize(w,h),
                   random.choice([CenterCrop(w,h),RandomCrop(w,h)])
                  
            ])

    transform5=Compose([
                   RandomResizedCrop(w,h),
                   RandomApply([random.choice([RandomRotation(),ColorJitter(),RandomFlip()])],p=0.5)
                   
            ])

    transform6=Compose([
                Shuffle([RandomRotation(),
                RandomResizedCrop(w,h)]),
                ColorJitter()
              
            ])
           
    return [transform1,transform2,transform3,transform4,transform5,transform6]      

    
class Dataset(object):
    '''
        imgs_path:图像数据存储路径
        labels:编码后的标签
        batch_size:batch中的图像数
        num_parallel_calls:读取数据的进程数
        type:训练或测试
        output_size:处理后的图像尺度
    '''
    def __init__(self,imgs_path,labels,batch_size=1,num_parallel_calls=1,prefetch_num=1,output_size=(224,224),type='train',):
        
        self.imgs_path=imgs_path
        self.labels=labels
        self.batch_size=batch_size
        self.num_parallel_calls=num_parallel_calls
        self.type=type
        self.output_size=output_size
        self.prefetch_num=prefetch_num
        assert len(self.imgs_path)==len(self.labels)
        
    #获取迭代器      
    def get_iterator(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.imgs_path,self.labels))
        dataset = dataset.shuffle(len(self.imgs_path))
        dataset = dataset.map(lambda img,label:tf.py_func(self.train_preprocess,[img,label],[tf.float64,tf.int32]),num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_num)
        iterator = dataset.make_initializable_iterator()
        return iterator
    #Map函数
    def train_preprocess(self,img_path,label):
       
        transform=make_transform(self.output_size[0],self.output_size[1])
        
        if 'str' not in str(type(img_path)):
            img_path = img_path.decode()
            img_path=str(img_path.strip())
        
        img=cv2.imread(img_path).astype(numpy.float64)
       
        if self.type=="train":
            temp=random.randint(0,9)
            if temp==0:
                img=transform[0](img)
            elif temp==1:
                img=transform[1](img)
            elif temp==2:
                img=transform[2](img)
            elif 3<=temp<=4:
                img=transform[3](img)  
            elif 5<=temp<=6:
                img=transform[4](img)
            else:
                img=transform[5](img)
        else:
            img=Resize(self.output_size[0],self.output_size[1])(img)
        #opencv读取的默认为BGR，需要调整对应通道的mean和std
        #img=Normalize((0.406,0.456,0.485,), (0.225,0.224, 0.229))(img)
        return img,label
    def __len__(self):
        return len(self.imgs_path)

