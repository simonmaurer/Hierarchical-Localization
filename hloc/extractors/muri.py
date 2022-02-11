import sys
from pathlib import Path

from ..utils.base_model import BaseModel

import larq_compute_engine as lce

r2d2_path = Path(__file__).parent / "../../third_party/r2d2"
sys.path.append(str(r2d2_path))
from extract import load_network, NonMaxSuppression, extract_multiscale


class MURI(BaseModel):
    default_conf = {
        'model_name': 'r2d2_WASF_N16.pt',
        'max_keypoints': 5000,
        'scale_factor': 2**0.25,
        'min_size': 256,
        'max_size': 1024,
        'min_scale': 0,
        'max_scale': 1,
        'reliability_threshold': 0.7,
        'repetability_threshold': 0.7,
    }
    required_inputs = ['image']

    def _init(self, conf):
        model_fn = r2d2_path / "models" / conf['model_name']
        #self.norm_rgb = tvf.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        tflite_path = ""
        lce_model = None
        with open(tflite_path, 'rb') as file:
            lce_model = file.read()
        self.net = lce.Interpreter(lce_model, num_threads=4)

    def _normalize(self, img):
        img -= 0.5
        img *= 0.225
        return img
    
    def _forward(self, data):
        img = data['image']
        img = self._normalize(img)
        
        descriptors, keypoints, scores = self.net.predict(img)
        descriptors = descriptors.squeeze()
        keypoints = keypoints.squeeze()
        scores = scores.squeeze()
        descriptors = descriptors.reshape((descriptors.shape[0]*descriptors.shape[1], descriptors.shape[2]))
        keypoints = keypoints.reshape((keypoints.shape[0]*keypoints.shape[1], keypoints.shape[2]))
        scores = scores.reshape((scores.shape[0]*scores.shape[1], ))

        xys, desc, scores = extract_multiscale(
            self.net, img, self.detector,
            scale_f=self.conf['scale_factor'],
            min_size=self.conf['min_size'],
            max_size=self.conf['max_size'],
            min_scale=self.conf['min_scale'],
            max_scale=self.conf['max_scale'],
        )
        idxs = scores.argsort()[-self.conf['max_keypoints'] or None:]
        xy = xys[idxs, :2]
        desc = desc[idxs].t()
        scores = scores[idxs]

        pred = {'keypoints': keypoints[None],
                'descriptors': descriptors[None],
                'scores': scores[None]}
        
        return pred
