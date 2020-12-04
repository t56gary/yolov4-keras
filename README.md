## YOLOV4：You Only Look Once 目標檢測模型在Keras當中的實現
---

### 目錄
1. [性能情況 Performance](#性能情況)
2. [實現的內容 Achievement](#實現的內容)
3. [所需環境 Environment](#所需環境)
4. [注意事項 Attention](#注意事項)
5. [小技巧的設置 TricksSet](#小技巧的設置)
6. [文件下載 Download](#文件下載)           
7. [預測步驟 How2predict](#預測步驟)
8. [訓練步驟 How2train](#訓練步驟)
9. [參考資料 Reference](#Reference)

### 性能情况
| 訓練數據集 | 權值文件名稱 | 測試數據集 | 輸入圖片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12+COCO | [yolo4_voc_weights.h5](https://github.com/bubbliiiing/yolov4-keras/releases/download/v1.0/yolo4_voc_weights.h5) | VOC-Test07 | 416x416 | - | 84.1
| COCO-Train2017 | [yolo4_weight.h5](https://github.com/bubbliiiing/yolov4-keras/releases/download/v1.0/yolo4_weight.h5) | COCO-Val2017 | 416x416 | 43.1 | 66.0

### YOLOV4實現的內容
- [x] 主要特徵擷取(backcone)：DarkNet53 => CSPDarkNet53
- [x] 特徵整合(Neck)：SPP，PAN
- [x] 訓練用到的小技巧：Mosaic數據增強、Label Smoothing平滑、CIOU、learningrate Cosine Annealing
- [x] 激活函數：使用Mish激活函數
- [ ] ……balabla

### 所需環境
tensorflow-gpu==1.13.1  
keras==2.1.5  

### 注意事項
code中的yolo4_weights.h5是基於608x608的圖片訓練的，但是由於顯存原因。我將代碼中的圖片大小修改成了416x416。有需要的可以修改回來。code中的默認anchors是基於608x608的圖片的。
**注意不要使用中文標籤，文件夾中不要有空格！ **
**在訓練前需要務必在model_data下新建一個txt文檔，文檔中輸入需要分的類，在train.py中將classes_path指向該文件**。

### 小技巧的設置
在train.py文件下：   
1、mosaic參數可用於控制是否實現Mosaic數據增強。
2、Cosine_scheduler可用於控制是否使用learningrate Cosine Annealing。
3、label_smoothing可用於控制是否Label Smoothing平滑。

### 文件下载
訓練所需的yolo4_weights.h5可在百度網盤中下載。
鏈接: https://pan.baidu.com/s/1FF79PmRc8BzZk8M_ARdMmw 提取碼: dc2j
yolo4_weights.h5是coco數據集的權重。
yolo4_voc_weights.h5是voc數據集的權重。

### 預測步驟
#### 1、使用pre-train權重
a、下載完庫後解壓，在百度網盤下載yolo4_weights.h5或者yolo4_voc_weights.h5，放入model_data，運行predict.py，輸入
```python
img/street.jpg
```
可完成预测。  
b、利用video.py可进行摄像头检测。  
#### 2、使用自己训练的权重
a、按照训练步骤训练。  
b、在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    "model_path": 'model_data/yolo4_weight.h5',
    "anchors_path": 'model_data/yolo_anchors.txt',
    "classes_path": 'model_data/coco_classes.txt,
    "score" : 0.5,
    "iou" : 0.3,
    # 显存比较小可以使用416x416
    # 显存比较大可以使用608x608
    "model_image_size" : (416, 416)
}

```
c、运行predict.py，输入  
```python
img/street.jpg
```
可完成预测。  
d、利用video.py可进行摄像头检测。  

### 训练步骤
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4、在训练前利用voc2yolo4.py文件生成对应的txt。  
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6、此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  
7、**在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**，示例如下：   
```python
classes_path = 'model_data/new_classes.txt'    
```
model_data/new_classes.txt文件内容为：   
```python
cat
dog
...
```
8、运行train.py即可开始训练。         
9、2020.12.03更新,目前所使用的annotation檔案路徑為相對位置,因此在使用時須先將utils.py裡的Image.OPEN中的路徑做修改

### mAP目标检测精度计算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

### Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  
