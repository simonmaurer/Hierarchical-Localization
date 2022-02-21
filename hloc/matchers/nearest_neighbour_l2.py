import torch
import cv2

from ..utils.base_model import BaseModel


class NearestNeighborL2(BaseModel):
    default_conf = {
        'ratio_threshold': None,
        'distance_threshold': None,
        'do_mutual_check': True,
    }
    required_inputs = ['descriptors0', 'descriptors1']

    def _init(self, conf):
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=conf['do_mutual_check'])

    def _forward(self, data):
        descriptors0 = data['descriptors0']
        descriptors1 = data['descriptors1']
        if descriptors0.size(-1) == 0 or descriptors1.size(-1) == 0:
            matches0 = torch.full(
                descriptors0.shape[:2], -1,
                device=descriptors0.device)
            return {
                'matches0': matches0,
                'matching_scores0': torch.zeros_like(matches0)
            }
        ratio_threshold = self.conf['ratio_threshold']
        if descriptors0.size(-1) == 1 or descriptors1.size(-1) == 1:
            ratio_threshold = None
        
        descriptors0 = descriptors0.numpy()
        descriptors1 = descriptors1.numpy()
        
        if ratio_threshold:
            knn_matches = self.matcher.knnMatch(descriptors0, descriptors1, k=2)
            matches = []
            for m,n in knn_matches:
                if m.distance < ratio_threshold*n.distance:
                    good.append([m])
        else:
            matches = self.matcher.match(descriptors0, descriptors1)
            matches_idx = np.array([m.queryIdx for m in matches])
            #m_keypoints = keypoints[matches_idx, :]
            matches_idx = np.array([m.trainIdx for m in matches])
            #m_warped_keypoints = warped_keypoints[matches_idx, :]
            
        
        return {
            'matches0': matches0,
            'matching_scores0': scores0,
        }
