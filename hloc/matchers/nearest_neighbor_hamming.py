import torch
import numpy as np

from ..utils.base_model import BaseModel


def find_nn(dist, ratio_thresh, distance_thresh):
    dist_nn, ind_nn = dist.topk(2 if ratio_thresh else 1, dim=-1, largest=False)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=dist.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh)*dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    return matches


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new


class NearestNeighborHamming(BaseModel):
    default_conf = {
        'ratio_threshold': None,
        'distance_threshold': None,
        'do_mutual_check': True,
    }
    required_inputs = ['descriptors0', 'descriptors1']

    def _init(self, conf):
        pass

    def _forward(self, data):
        if data['descriptors0'].size(-1) == 0 or data['descriptors1'].size(-1) == 0:
            matches0 = torch.full(
                data['descriptors0'].shape[:2], -1,
                device=data['descriptors0'].device)
            return {
                'matches0': matches0,
            }
        ratio_threshold = self.conf['ratio_threshold']
        if data['descriptors0'].size(-1) == 1 or data['descriptors1'].size(-1) == 1:
            ratio_threshold = None
            
        d0, d1 = data['descriptors0'], data['descriptors1']
        if d0.dtype == torch.uint8:
            device = d0.device
            d0 = torch.from_numpy(np.unpackbits(d0.numpy(), axis=1).astype(np.float32)).to(device)
        if d1.dtype == torch.uint8:
            device = d1.device
            d1 = torch.from_numpy(np.unpackbits(d1.numpy(), axis=1).astype(np.float32)).to(device)
            
        dist = torch.einsum('bdn,bmd->bnm', 1-d0, torch.einsum('bij->bji', d1)) + torch.einsum('bdn,bmd->bnm', d0, torch.einsum('bij->bji', 1-d1))
        matches0 = find_nn(dist, ratio_threshold, self.conf['distance_threshold'])
        
        if self.conf['do_mutual_check']:
            matches1 = find_nn(dist.transpose(1, 2), ratio_threshold, self.conf['distance_threshold'])
            matches0 = mutual_check(matches0, matches1)
            
        return {
            'matches0': matches0,
        }
