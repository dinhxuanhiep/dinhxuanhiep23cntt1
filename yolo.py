import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox

class YOLO(object):
    _defaults = {
        "model_path"        : 'logs/best_epoch_weights.pth',
        "classes_path"      : 'model_data/my_classes.txt',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "input_shape"       : [416, 416],
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
        "letterbox_image"   : False,
        "cuda"              : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        # --- GHI CỨNG ANCHORS CHUẨN ---
        self.anchors = np.array([
            [10, 14],  [23, 27],   [37, 58], 
            [81, 82],  [135, 169], [344, 319]
        ], dtype='float32')
        self.num_anchors = 6
        
    
        self.anchors_mask = [[3, 4, 5], [0, 1, 2]]

        self.class_names, self.num_classes = get_classes(self.classes_path)
        

        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_path)
        self.bbox_util.anchors_mask = [[3, 4, 5], [0, 1, 2]]

        self.generate()

    def generate(self):
        self.net = YoloBody(self.anchors_mask, self.num_classes)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading weights from: {self.model_path}")
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()
        print('--> Load weights thành công!')

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image, crop = False, count = False):
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image, [], [] 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // 416, 1)

        out_boxes = []
        out_scores = []
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[c]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(np.shape(image)[0], np.floor(bottom).astype('int32'))
            right   = min(np.shape(image)[1], np.floor(right).astype('int32'))

            out_boxes.append([top, left, bottom, right])
            out_scores.append(score)

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textbbox((0, 0), label, font)
            label_size = (label_size[2], label_size[3])
            
            label = label.encode('utf-8')
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image, out_boxes, out_scores

    @property
    def colors(self):
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)

        return colors
