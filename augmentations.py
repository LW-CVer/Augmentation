#_*_ coding:utf-8_*_
import cv2
import random
import numpy
import math
#缩放
class Resize(object):
    '''
    输入目标尺度w,h
    interpolation:插值方法，随机选择
    '''
    def __init__(self,width,height):
        self.width=width
        self.height=height
    def __call__(self,img):
        interpolation=[cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_AREA,cv2.INTER_CUBIC]
        img=cv2.resize(img,(self.width,self.height),interpolation=interpolation[random.randint(0,3)])
        return img

#随机中心点裁剪
class RandomCrop(object):
    '''
        在获取图像基础上裁剪出固定大小图像，中心点随机。
        
    padding:最大填充数
    mode。应当是‘constant’，‘edge’，‘reflect’或‘symmetric’之一。随机选择。
    constant：用常数扩展，这个值由fill参数指定。
    edge：用图像边缘上的指填充。
    reflect：以边缘为对称轴进行轴对称填充（边缘值不重复）。
    > 例如，在[1, 2, 3, 4]的两边填充2个元素会得到[3, 2, 1, 2, 3, 4, 3, 2]。
    symmetric：用图像边缘的反转进行填充（图像的边缘值需要重复）。
    > 例如，在[1, 2, 3, 4]的两边填充2个元素会得到[2, 1, 1, 2, 3, 4, 4, 3]。


    ''' 
    def __init__(self,width,height,padding=0):
        self.width=width
        self.height=height
        self.padding=padding
    @staticmethod    
    def Crop(img,width,height):
        w, h = img.shape[1],img.shape[0]
        
        tw, th = width,height
        if w == tw and h == th:
            return (0, 0, w, h)
        if w<tw or h<th:
            raise Exception("裁剪尺度大于原图")
        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)
        return (i,j,i+tw,j+th)
    def __call__(self,img):
        assert img.shape[1]>=self.width and img.shape[0]>=self.height
        mode=['constant','edge','reflect','symmetric']
        temp=random.randint(0,3)
        
        if self.padding!=0:
            if mode[temp]=='constant':
                padding=random.randint(0,self.padding)
                img = numpy.pad(img, ((padding, padding), (padding, padding), (0,0)), 'constant', constant_values=0)
            else :
                img = numpy.pad(img, ((padding, padding), (padding, padding), (0,0)), mode[temp])
        
        point=self.Crop(img,self.width,self.height)
        img=img[point[1]:point[3],point[0]:point[2]]
        return img
     
#中心点裁剪
class CenterCrop(object):
    '''
        以获取的图像的中心点为基础进行裁剪
    '''
    def __init__(self,width,height):
        self.width=width
        self.height=height
    def __call__(self,img):
        assert img.shape[1]>=self.width and img.shape[0]>=self.height
        x,y=img.shape[1]//2,img.shape[0]//2
        w=self.width
        h=self.height
        img=img[y-h//2:y+h//2,x-w//2:x+w//2]
        return img
         
#随机长宽比裁剪
class RandomResizedCrop(object):
    '''
        
    '''
    def __init__(self,width,height):
        self.width=width
        self.height=height
    @staticmethod
    def Crop(img):
        area = img.shape[0] * img.shape[1]

        for attempt in range(20):
            target_area = random.uniform(*(0.65,1.0)) * area
            log_ratio = (math.log(0.65), math.log(1.33333))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[1] - w)
                j = random.randint(0, img.shape[0] - h)
                return i, j, i+w, j+h

        # 中心裁剪
        in_ratio = img.shape[1] / img.shape[0]
        if (in_ratio < 0.65):
            w = img.shape[1]
            h = int(round(w / 0.65))
        elif (in_ratio > 1.33333):
            h = img.shape[0]
            w = int(round(h * 1.33333))
        else:  
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[1] - w) // 2
        j = (img.shape[0] - h) // 2
        return i, j, i+w, j+h
       
    def __call__(self,img):
        
        point=self.Crop(img)
    
        img=img[point[1]:point[3],point[0]:point[2]].astype(numpy.float64)
        
        img=Resize(self.width,self.height)(img)
        return img
 
#随机对图像进行翻转 
class RandomFlip(object):
    '''
        依据概率对图像进行翻转
        
    '''
    def __call__(self,img):
        
        if(random.randint(0,9)<5): 
            img = cv2.flip(img, 1)   
        return img
       
   
#随机旋转
class RandomRotation(object):
    '''
       
    '''
    def __call__(self,img):
        
        interpolation=[cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_AREA,cv2.INTER_CUBIC]
        temp=random.randint(0,2)
        width=img.shape[1]
        height=img.shape[0]
        #旋转10°-15°以内
        if temp==0:
            degrees=10
            center=(0,0)
        elif temp==1:
            degrees=15
            center=(width//2,height//2)
        else:
            degrees=10
            center=(random.randint(0,width),random.randint(0,height))
            
        M = cv2.getRotationMatrix2D(center, random.randint(0,degrees), 1.0)
        img = cv2.warpAffine(img, M, (width, height),flags=interpolation[random.randint(0,3)])
        if (img.shape[1]!=width and img.shape[0]!=height):
            img=Resize(width,height)(img)
        return img  
     
#添加亮度、对比度、饱和度、色相扰动
class ColorJitter(object):
    '''    
    

    '''
    def __call__(self,img):
        
        transform=[self.Brightness,self.Contrast,self.Value,self.Saturation,self.Hue]
        random.shuffle(transform)
        for i in range(5):
            if random.randint(0,1):
                img=transform[i](img)
                print
        return img
        
       
    def Brightness(self,img):
        
        h, w, ch = img.shape
        img_temp = numpy.zeros([h, w, ch], img.dtype)
        img = cv2.addWeighted(img, 1, img_temp, 0, random.randint(-32,32))
        return img
    
    def Contrast(self,img):
        a=random.uniform(0.5,1.5)
        h, w, ch = img.shape
        img_temp = numpy.zeros([h, w, ch], img.dtype)
        img = cv2.addWeighted(img, a, img_temp, 1-a, 0)
        return img
    #透明度,图片存在像素溢出情况，需要测试效果
    def Value(self,img):
        img=img.astype(numpy.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        delta = 1 + random.uniform(-0.5, 0.5)
        img=img.astype(numpy.float64)
        img[:,:,2] *= delta
        img=img.astype(numpy.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR).astype(numpy.float64)
        return img
    def Saturation(self,img):
        img=img.astype(numpy.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        delta = 1 + random.uniform(-0.5, 0.5)
        img=img.astype(numpy.float64)
        img[:,:,1] *= delta
        
        img=img.astype(numpy.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR).astype(numpy.float64)
        return img
         
    def Hue(self,img):
        
        
        img=img.astype(numpy.uint8)
        if random.randint(0,1):
        #颜色空间转换必须转换为整数
            img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
            #按经验值（-36，36）调整
            img[:,:,0] = (img[:,:,0]+random.randint(-36,36))%180
            #按比例调整
            #hue=random.uniform(-0.5,0.5)
            #img[:,:,0] = (img[:,:,0]*(1+hue))%180
            img[:,:,0]=img[:,:,0].astype(numpy.uint8)
            img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
            img=img.astype(numpy.float64)
            return img
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            img[:,:,0] = (img[:,:,0]+random.randint(-36,36))%180
            #hue=random.uniform(-0.5,0.5)
            #img[:,:,0] = (img[:,:,0]*(1+hue))%180
            img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
            img=img.astype(numpy.float64)
            return img
  

#使用后模型易于训练，效果会稍微降低，使用前需要获取数据集平均值和标准差
class Normalize(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
    def __call__(self,img):
        img=img/255
        for i in range(img.shape[2]):
            img[:,:,i]=(img[:,:,i]-self.mean[i])/self.std[i]
        return img

       

      
        
        
        
        
        
        