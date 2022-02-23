from ..utils.base_model import BaseModel

import cv2
import torch
import numpy as np


class ORB(BaseModel):
    default_conf = {
        'model_name': 'orb',
        'max_keypoints': None,
        'min_size': 320,
        'max_size': 1024,
        'desc_uint8': False,
    }
    required_inputs = ['image']

    def _init(self, conf):
        num_features = conf['max_keypoints']
        if num_features:
            self.detector = cv2.ORB_create(nfeatures=num_features)
        else:
            self.detector = cv2.ORB_create()
    
    def _forward(self, data):
        img = data['image']
        device = img.device
        img = img.squeeze()
        print("Shape: ",img.shape)
        print("Dtype:", img.dtype)
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
            descriptors = torch.from_numpy(descriptors),to(device).t()

        pred = {'keypoints': keypoints[None],
                'descriptors': descriptors[None]}
        
        return pred
