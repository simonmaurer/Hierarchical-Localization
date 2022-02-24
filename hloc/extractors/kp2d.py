import sys
from pathlib import Path

from ..utils.base_model import BaseModel

import tensorflow as tf
import torch

muri_path = Path(__file__).parent / "../../third_party/mnet"
sys.path.append(str(muri_path))
from muri.detector import MURIDetector


class KP2D(BaseModel):
    default_conf = {
        'model_name': 'kp2d',
        'keep_topk_or_threshold': 0.7,
        'min_size': 320,
        'max_size': 1024,
        'gpu': False,
    }
    required_inputs = ['image']

    def _init(self, conf):
        model_path = muri_path / "experiments" / conf['model_name']
        self.detector = MURIDetector(str(model_path.absolute()), input_shape=(240,320,3), keep_topk_or_threshold=conf['keep_topk_or_threshold'], gpu=conf['gpu'])

    def _normalize(self, img):
        img -= 0.5
        img *= 0.225
        return img
    
    def _forward(self, data):
        img = data['image']
        device = img.device
        img = img.permute((0,2,3,1))
        img = self._normalize(img)
        img = tf.convert_to_tensor(img.cpu()numpy())
        
        scores, keypoints, descriptors = self.detector.detectAndCompute(img)

        scores = torch.from_numpy(scores).to(device)
        keypoints = torch.from_numpy(keypoints).to(device)
        descriptors = torch.from_numpy(descriptors).to(device).t()

        pred = {'keypoints': keypoints[None],
                'descriptors': descriptors[None],
                'scores': scores[None]}
        
        return pred
