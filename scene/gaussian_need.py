
import torch
from scene.gaussian_model import GaussianModel
from train import save_training_vis

normal = save_training_vis.render_pkg["normal"]
pseudo_normal = save_training_vis.render_pkg["pseudo_normal"]

def flip_align_view(normal, pseudo_normal):
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = torch.sum(
        normal * -pseudo_normal, dim=-1, keepdims=True) # (N, 1)
    non_flip = dotprod>=0 # (N, 1)
    normal_flipped = normal*torch.where(non_flip, 1, -1) # (N, 3)
    return normal_flipped, non_flip

Gaussian = GaussianModel

scales = Gaussian.get_scaling
rotations = GaussianModel.get_rotation

def get_minimum_axis(scales, rotations):
        sorted_idx = torch.argsort(scales, descending=False, dim=-1)
        R = rotations
        R_sorted = torch.gather(R, dim=2, index=sorted_idx[:,None,:].repeat(1, 3, 1)).squeeze()
        x_axis = R_sorted[:,0,:] # normalized by defaut
        
        return x_axis