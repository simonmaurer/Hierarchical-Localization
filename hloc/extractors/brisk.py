from ..utils.base_model import BaseModel

import cv2
import torch
import numpy as np


class BRISK(BaseModel):
    default_conf = {
        'model_name': 'brisk',
        'detection_threshold': None,
        'min_size': 320,
        'max_size': 1024,
        'desc_uint8': False,
    }
    required_inputs = ['image']

    def _init(self, conf):
        detection_threshold = conf['detection_threshold']
        if detection_threshold:
            self.detector = cv2.BRISK_create(thresh=detection_threshold)
        else:
            self.detector = cv2.BRISK_create()
    
    def _forward(self, data):
        img = data['image']
        device = img.device
        img = img.squeeze()
        img = img.transpose(1, 2, 0)
        if img.dtype != torch.uint8:
            img = (img*255.).type(torch.uint8)
        img = img.numpy()
        
        keypoints, descriptors = self.detector.detectAndCompute(img, None)
        keypoints = np.asarray([k.pt for k in keypoints])

        keypoints = torch.from_numpy(keypoints).to(device)
        if not self.conf['desc_uint8']:
            descriptors = np.unpackbits(descriptors, axis = -1)
            descriptors = torch.FloatTensor(descriptors).to(device).t()
        else:
            descriptors = torch.from_numpy(descriptors).to(device).t()

        pred = {'keypoints': keypoints[None],
                'descriptors': descriptors[None]}
        
        return pred
