一、mydataset.py:封装了tensorflow数据加载pipeline、图像数据增强
-----    
    包含的类：
        Dataset：封装数据管道pipeline，返回一个可初始化迭代器对象
        
        RandomApply：使数据增强操作随机产生作用
        
        Shuffle：随机打乱数据增强操作
        
        Compose：组合数据增强操作，使其顺序执行
    
    包含的方法：
        get_data：读取CSV文件，将标签编码
        
        make_transform：组合各类有效的数据增强方法
        
二、augmentations.py:基于opencv实现的数据增强方法
---    
    包含的类：
        Resize：尺度缩放
        
        RandomCrop：随机裁剪
        
        CenterCrop：中心点裁剪
        
        RandomResizedCrop：随机尺度裁剪
        
        RandomFlip：随机翻转
        
        RandomRotation：随机旋转
        
        ColorJitter：颜色扰动

三、使用方法：
---
    1.导入mydataset模块

    2.调用Dataset类，生成并返回一个可初始化迭代器对象。参数列表如下：
        imgs_path:list对象，图像数据存储路径
        labels:list对象，编码后的标签
        batch_size:一个batch中的图像数，默认为1
        num_parallel_calls:读取数据的进程数,默认为1
        prefetch_num:预加载的图像batch数，默认为1
        output_size:处理后的图像尺度（即网络的输入），默认为（224，224）
        type:训练或测试，如果是测试，只对图片进行Resize操作
        
    3.调用迭代器进行模型训练
    
四、测试效果对比
---
    测试平台：pytorch
    测试网络：ResNet-50
    
    1.不进行数据增强，只进行Resize。损失和验证效果图如下：
![](http://git.yuntongxun.com/liwei11/Data_augmentations/raw/master/img/1.png)
![](http://git.yuntongxun.com/liwei11/Data_augmentations/raw/master/img/2.png)
   
    2.进行随机尺度裁剪、随机旋转、随机颜色扰动。损失、验证以及和1对比效果图如下：
![](http://git.yuntongxun.com/liwei11/Data_augmentations/raw/master/img/3.png)
![](http://git.yuntongxun.com/liwei11/Data_augmentations/raw/master/img/4.png)
![](http://git.yuntongxun.com/liwei11/Data_augmentations/raw/master/img/5.png)
    
    以上组合为测试效果最好的，验证分类准确率最高为63.4%

    3.将不同的操作组合并行，为不同组合分配不同触发概率。损失、验证以及和1、2对比效果图如下：
![](http://git.yuntongxun.com/liwei11/Data_augmentations/raw/master/img/6.png)
![](http://git.yuntongxun.com/liwei11/Data_augmentations/raw/master/img/7.png)
![](http://git.yuntongxun.com/liwei11/Data_augmentations/raw/master/img/8.png)
    
    将不同的能提高分类准确率的数据增强组合进行并行，每个组合触发的概率根据提升的效果来分配，实现了目前最好的分类效果66.7%



