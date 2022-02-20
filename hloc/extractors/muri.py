import sys
from pathlib import Path

from ..utils.base_model import BaseModel

import tensorflow as tf
import torch
import larq_compute_engine as lce

muri_path = Path(__file__).parent / "../../third_party/muri"
sys.path.append(str(muri_path))
from muri.detector import MURIDetector


class MURI(BaseModel):
    default_conf = {
        'model_name': 'muri',
        'keep_topk_or_threshold': 0.7,
        'min_size': 320,
        'max_size': 1024,
    }
    required_inputs = ['image']

    def _init(self, conf):
        model_path = muri_path / "experiments" / conf['model_name']
        self.detector = MURIDetector(str(model_path.absolute()), input_shape=(240,320,3), keep_topk_or_threshold=conf['keep_topk_or_threshold'])
        #tflite_path = ""
        #lce_model = None
        #with open(tflite_path, 'rb') as file:
        #    lce_model = file.read()
        #self.net = lce.Interpreter(lce_model, num_threads=4)

    def _normalize(self, img):
        img -= 0.5
        img *= 0.225
        return img
    
    def _forward(self, data):
        img = data['image']
        img = tf.convert_to_tensor(img.numpy())
        img = tf.transpose(img, (0,2,3,1))
        img = self._normalize(img)
        
        #descriptors, keypoints, scores = self.net.predict(img)
        #descriptors = descriptors.squeeze()
        #keypoints = keypoints.squeeze()
        #scores = scores.squeeze()
        #descriptors = descriptors.reshape((descriptors.shape[0]*descriptors.shape[1], descriptors.shape[2]))
        #keypoints = keypoints.reshape((keypoints.shape[0]*keypoints.shape[1], keypoints.shape[2]))
        #scores = scores.reshape((scores.shape[0]*scores.shape[1], ))
        
        scores, keypoints, descriptors = self.net.detectAndCompute(img)

        scores = torch.from_numpy(scores)
        keypoints = torch.from_numpy(keypoints)
        descriptors = torch.from_numpy(descriptors).t()

        pred = {'keypoints': keypoints[None],
                'descriptors': descriptors[None],
                'scores': scores[None]}
        
        return pred
