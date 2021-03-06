import os
import numpy as np
import copy
import colorsys
from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from nets.yolo4 import yolo_body,yolo_eval
from utils.utils import letterbox_image
from tensorflow.compat.v1.keras.backend import get_session
#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path"        : '/content/drive/MyDrive/Colab Notebooks/mango/h5/ep044-loss12.058-val_loss10.655.h5',
        "anchors_path"      : '/content/drive/MyDrive/Colab Notebooks/mango/anchor.txt',
        "classes_path"      : '/content/drive/MyDrive/Colab Notebooks/mango/classes.txt',
        "score"             : 0.01,
        "iou"               : 0.1,
        "max_boxes"         : 100,
        # 显存比较小可以使用416x416
        # 显存比较大可以使用608x608
        "model_image_size"  : (416, 416)
    }
#     _defaults = {
#         "model_path"        : '/content/drive/MyDrive/AI鯉魚王/Yolo_v4/h5/2020_12_05_15:00:00/ep133-loss2.364-val_loss2.243.h5',
#         "anchors_path"      : '/content/drive/MyDrive/AI鯉魚王/Yolo_v4/yolo_anchors.txt',
#         "classes_path"      : '/content/drive/MyDrive/AI鯉魚王/Yolo_v4/model_data/bccd_classes.txt',
#         "score"             : 0.4,
#         "iou"               : 0.6,
#         "max_boxes"         : 100,
#         # 显存比较小可以使用416x416
#         # 显存比较大可以使用608x608
#         "model_image_size"  : (416, 416)
#     }
#鯉魚王用
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = get_session()
        self.boxes, self.scores, self.classes = self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2, ))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                num_classes, self.input_image_shape, max_boxes = self.max_boxes,
                score_threshold = self.score, iou_threshold = self.iou)
        return boxes, scores, classes

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        start = timer()

        #加入不重複標框判斷式的宣告
        s1=0
        s2=0
        s3=0
        c1=[]
        c2=[]
        c3=[]

        # 调整图片使其符合输入要求
        new_image_size = (self.model_image_size[1],self.model_image_size[0])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 预测结果
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 设置字体
        font = ImageFont.truetype(font='/content/yolov4-keras/font/simhei.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        small_pic=[]

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

#             if predicted_class=="clownfish1" and score>s1:
#                 top, left, bottom, right = box
#                 top = max(0, np.floor(top + 0.5).astype('int32'))
#                 left = max(0, np.floor(left + 0.5).astype('int32'))
#                 bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
#                 right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
#                 s1=score
#                 predicted_class = "小丑魚1"
#                 c1 = [predicted_class, score, left, top, right, bottom]
#             elif predicted_class=="clownfish2" and score>s2:
#                 top, left, bottom, right = box
#                 top = max(0, np.floor(top + 0.5).astype('int32'))
#                 left = max(0, np.floor(left + 0.5).astype('int32'))
#                 bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
#                 right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
#                 s2=score
#                 predicted_class = "小丑魚2"
#                 c2 = [predicted_class, score, left, top, right, bottom]
#             if predicted_class=="clownfish3" and score>s3:
#                 top, left, bottom, right = box
#                 top = max(0, np.floor(top + 0.5).astype('int32'))
#                 left = max(0, np.floor(left + 0.5).astype('int32'))
#                 bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
#                 right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
#                 s3=score
#                 predicted_class = "小丑魚3"
#                 c3 = [predicted_class, score, left, top, right, bottom]
            

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
                
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

#         if c1!=[]:
#             print("c1")
#             label = '{} '.format(c1[0])
#             for i in range(thickness):
#             # print(i)
#                 draw.rectangle(
#                     [c1[2] + i, c1[3] + i, c1[4] - i, c1[5] - i],
#                     outline=(255,0,0))

#                 draw.text([c1[2]+10,c1[3]-75],label, fill=(255,0,0), font=font)
#         if c2!=[]:
#             print("c2")
#             label = '{} '.format(c2[0])
#             for i in range(thickness):
#             # print(i)
#                 draw.rectangle(
#                     [c2[2] + i, c2[3] + i, c2[4] - i, c2[5] - i],
#                     outline=(0,255,0))
#                 draw.text([c2[2]+10,c2[3]-75],label, fill=(0,255,0), font=font)
#         if c3!=[]:
#             print("c3")
#             label = '{} '.format(c3[0])
#             for i in range(thickness):
#             # print(i)
#                 draw.rectangle(
#                     [c3[2] + i, c3[3] + i, c3[4] - i, c3[5] - i],
#                     outline=(0,0,255))
#                 draw.text([c3[2]+10,c3[3]-75],label, fill=(0,0,255), font=font)

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()
